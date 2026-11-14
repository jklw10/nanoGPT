"""
nodediagnostic.py
High-throughput, fully-vectorized multi-island deliberation diagnostic.
Batches all islands, seeds, and forward modes into a single GPU graph.
Saves complete diagnostic telemetry to JSON and PT for post-hoc analysis.
"""

import argparse
import datetime
import importlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. CHESS UTILITIES & NOTATION
# ==========================================

def sq_to_name(sq):
    return f"{chr(ord('a') + (sq % 8))}{(sq // 8) + 1}"

def action_to_str(act):
    return f"{sq_to_name(act // 64)}{sq_to_name(act % 64)}"

def uci_to_action(uci_str):
    if not uci_str or len(uci_str) < 4: return None
    try:
        fr = (int(uci_str[1]) - 1) * 8 + (ord(uci_str[0]) - ord('a'))
        to = (int(uci_str[3]) - 1) * 8 + (ord(uci_str[2]) - ord('a'))
        return fr * 64 + to
    except Exception:
        return None

def buntanh(x, slope, bound):
    return torch.tanh(x - torch.tanh(x) * slope) * bound

def make_circulant_matrix(dim: int, center_offset=1.0, sigma_val=0.5, device="cuda", dtype=torch.float32):
    k_sz = min(31, dim if dim % 2 == 1 else dim - 1)
    if k_sz < 3: return torch.eye(dim, device=device, dtype=dtype)
    half_k = k_sz // 2
    t = torch.arange(-half_k, half_k + 1, device=device, dtype=dtype)
    kernel = (torch.exp(-0.5 * (t * max(sigma_val * math.pi, 1e-4)) ** 2) * torch.cos(center_offset * math.pi * t))
    kernel = (kernel / (torch.norm(kernel, p=2) + 1e-8)).view(1, 1, k_sz)
    eye = torch.eye(dim, device=device, dtype=dtype).unsqueeze(1)
    return F.conv1d(F.pad(eye, (half_k, half_k), mode='circular'), kernel).squeeze(1)

# ==========================================
# 1. VECTORIZED MULTI-ISLAND MODEL
# ==========================================

class BatchedIslandNet(nn.Module):
    """
    Unified multi-island architecture:
    Executes I parallel islands with distinct seeds, init schemes, and deliberation modes
    within a single 4D tensor computation graph.
    """
    def __init__(self, num_islands, sub_batch_size, num_nodes=32, dim=32, input_dim=64, 
                 island_modes=None, init_schemes=None, seeds=None, device="cuda"):
        super().__init__()
        self.I = num_islands
        self.sub_B = sub_batch_size
        self.B = num_islands * sub_batch_size
        self.N = num_nodes
        self.D = dim
        self.device = device
        self.num_input_nodes = 2
        self.motor_nodes_count = 4
        self.lp_alpha = 0.45

        mode_map = {"viscous": 0, "prekick": 1, "raw": 2}
        self.island_modes = [m.lower() for m in (island_modes or ["viscous"] * num_islands)]
        self.mode_tensor = torch.tensor([mode_map[m] for m in self.island_modes], device=device).view(self.I, 1, 1, 1)

        self.init_schemes = [s.lower() for s in (init_schemes or ["baseline"] * num_islands)]
        self.seeds = seeds or [1000 * (i + 1) + 42 for i in range(num_islands)]

        self.sensory_proj_weight = nn.Parameter(torch.zeros(self.I, input_dim, self.num_input_nodes * dim, device=device), requires_grad=False)
        self.w1_weight = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim * 2, device=device), requires_grad=False)
        self.w_rx = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim, device=device), requires_grad=False)
        self.w_tx = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim, device=device), requires_grad=False)
        
        self.action_logit_scale = nn.Parameter(torch.full((self.I, 1, 1, 1), 6.0, device=device), requires_grad=False)
        self.register_buffer("spatial_affinity", torch.zeros(self.I, 64, 64, device=device))
        self.register_buffer("normalized_square_enc", torch.zeros(self.I, 64, 64, device=device))

        self.register_buffer("circ_1_0", make_circulant_matrix(dim, 1.0, 1.0, device=device))
        self.register_buffer("circ_0_25", make_circulant_matrix(dim, 1.0, 0.25, device=device))

        self._initialize_all_islands()

    def _initialize_all_islands(self):
        for i in range(self.I):
            torch.manual_seed(self.seeds[i])
            torch.cuda.manual_seed_all(self.seeds[i])
            scheme = self.init_schemes[i]

            # 1. Sensory Projection
            if scheme == "spatial":
                spatial_mat = torch.zeros(64, self.num_input_nodes * self.D, device=self.device)
                for sq in range(64):
                    r, c = sq // 8, sq % 8
                    for d in range((self.num_input_nodes * self.D) // 4):
                        freq = 2.0 ** (d / 4.0)
                        spatial_mat[sq, 4 * d + 0] = math.sin(r * freq)
                        spatial_mat[sq, 4 * d + 1] = math.cos(r * freq)
                        spatial_mat[sq, 4 * d + 2] = math.sin(c * freq)
                        spatial_mat[sq, 4 * d + 3] = math.cos(c * freq)
                self.sensory_proj_weight.data[i] = F.normalize(spatial_mat, p=2, dim=0)
            else:
                raw_sens = torch.empty(64, self.num_input_nodes * self.D, device=self.device)
                nn.init.orthogonal_(raw_sens, gain=1.0)
                self.sensory_proj_weight.data[i] = F.normalize(raw_sens, p=2, dim=0)

            sq_enc = self.sensory_proj_weight.data[i].clone()
            self.normalized_square_enc[i] = sq_enc
            self.spatial_affinity[i] = torch.matmul(sq_enc, sq_enc.T)

            # 2. Internal Transition Weights
            if scheme in ("quasizero", "spatial"):
                eye_split = torch.cat([torch.eye(self.D, device=self.device), torch.zeros(self.D, self.D, device=self.device)], dim=1)
                noise = torch.randn(self.N, self.D, self.D * 2, device=self.device) * 0.01
                self.w1_weight.data[i] = (eye_split.unsqueeze(0).expand(self.N, -1, -1) + noise) * 0.5
                self.action_logit_scale.data[i].fill_(1.0)
            else:
                raw_w1 = torch.randn(self.N, self.D, self.D * 2, device=self.device) / math.sqrt(self.D)
                self.w1_weight.data[i] = F.normalize(raw_w1, p=2, dim=-2) * 0.8

            # 3. Transceivers
            if scheme in ("quasizero", "spatial"):
                dft = torch.fft.fft(torch.eye(self.D, device=self.device)).real
                dft_norm = F.normalize(dft, p=2, dim=-1) * 0.8
                self.w_rx.data[i] = dft_norm.unsqueeze(0).expand(self.N, -1, -1)
                self.w_tx.data[i] = dft_norm.unsqueeze(0).expand(self.N, -1, -1)
            else:
                raw_rx = torch.randn(self.N, self.D, self.D, device=self.device) / math.sqrt(self.D)
                raw_tx = torch.randn(self.N, self.D, self.D, device=self.device) / math.sqrt(self.D)
                self.w_rx.data[i] = F.normalize(raw_rx, p=2, dim=-2) * 0.8
                self.w_tx.data[i] = F.normalize(raw_tx, p=2, dim=-2) * 0.8

    def init_states(self, batch_size):
        sub_B = batch_size // self.I
        s = 1e-4 * torch.randn(self.I, self.N, sub_B, self.D, device=self.device)
        o = 1e-4 * torch.randn(self.I, self.N, sub_B, self.D, device=self.device)
        err = torch.full((self.I, self.N, sub_B), 0.05, device=self.device)
        lp = torch.zeros((self.I, self.N, sub_B), device=self.device)
        return s, o, err, lp

    def get_action_logits(self, state):
        motor = state[:, -self.motor_nodes_count:, :, :].permute(0, 2, 1, 3).reshape(self.I, self.sub_B, self.motor_nodes_count * self.D)
        nf = F.normalize(motor[..., :64], p=2, dim=-1)
        nt = F.normalize(motor[..., 64:], p=2, dim=-1)

        lf = torch.matmul(nf, self.normalized_square_enc)
        lt = torch.matmul(nt, self.normalized_square_enc)
        cohere = torch.sum(nf * nt, dim=-1, keepdim=True).unsqueeze(-1)

        logits = (lf.unsqueeze(-1) + lt.unsqueeze(-2) + cohere * self.spatial_affinity.unsqueeze(1)) * self.action_logit_scale
        return logits.view(self.B, 4096)

    def forward(self, env_in, threat, state, out, err_ema, lp_ema, thinking_steps=32, act_jitter=None):
        e_in = env_in.view(self.I, self.sub_B, 64)
        th = threat.view(self.I, 1, self.sub_B, 1)

        raw_s = torch.matmul(e_in, self.sensory_proj_weight)
        s_lat = F.softsign(raw_s.view(self.I, self.sub_B, self.num_input_nodes, self.D).permute(0, 2, 1, 3))
        bus = s_lat.mean(dim=1)

        c_s, c_o, c_e, c_p = state.clone(), out.clone(), err_ema.clone(), lp_ema.clone()
        latched_s = c_s.clone()
        prev_s = c_s.clone()

        latched_em = buntanh(torch.matmul(c_s, self.w_tx), 0.5, 1.0)
        cooldown = torch.zeros(self.I, 1, self.sub_B, 1, dtype=torch.int16, device=self.device)
        settled_counter = torch.zeros(self.I, 1, self.sub_B, 1, dtype=torch.int16, device=self.device)

        for step in range(thinking_steps):
            # FIXED: average over the 4 motor nodes (dim 1) so shape is [I, 1, sub_B, 1]
            motor_vel = torch.norm(
                c_s[:, -self.motor_nodes_count:] - prev_s[:, -self.motor_nodes_count:], 
                dim=-1, 
                keepdim=True
            ).mean(dim=1, keepdim=True)
            is_motor_calm = (motor_vel < 0.025) & (self.mode_tensor == 0)
            latched_s = torch.where(is_motor_calm, c_s, latched_s)

            overall_vel = torch.norm(c_s - prev_s, dim=-1, keepdim=True).mean(dim=1, keepdim=True)
            is_settled = (overall_vel < 0.018) & (cooldown == 0)
            settled_counter = torch.where(is_settled, settled_counter + 1, torch.zeros_like(settled_counter))

            trigger_pulse = settled_counter >= 2
            latch_prekick = trigger_pulse & (self.mode_tensor == 1)
            latched_s = torch.where(latch_prekick, c_s, latched_s)

            if trigger_pulse.any():
                pulse = torch.randn_like(c_s) * 0.35
                c_s = torch.where(trigger_pulse, F.softsign(c_s + pulse), c_s)
                cooldown = torch.where(trigger_pulse, 8, cooldown)
                settled_counter = torch.where(trigger_pulse, 0, settled_counter)

            is_coasting = (cooldown > 0) | (overall_vel > 0.025)
            noise_gate = torch.where(is_coasting, 0.0, 1.0)
            cooldown = torch.clamp(cooldown - 1, min=0)
            prev_s = c_s.clone()

            b_exp = bus.unsqueeze(1).expand(-1, self.N, -1, -1)
            perceived = buntanh(torch.matmul(b_exp, self.w_rx), 0.5, math.sqrt(2.0 + math.sqrt(2.0)))

            raw_boredom = torch.pow(1.0 - torch.clamp(F.relu(c_p) / (c_e + 1e-6), 0.0, 1.0), 1.0).unsqueeze(-1)
            boredom = raw_boredom * noise_gate
            tgt = perceived + math.sqrt(0.05) * torch.matmul(torch.randn_like(c_o), self.circ_1_0) * boredom - (0.02) * c_s * (1.0 - boredom)
            tgt = torch.cat([s_lat, F.softsign(tgt)[:, self.num_input_nodes:, :, :]], dim=1)

            w1_in = torch.matmul(c_s, self.w1_weight)
            if act_jitter is not None: w1_in = w1_in + act_jitter
            mean, err_p = torch.chunk(buntanh(w1_in, 0.5, math.sqrt(2.0 + math.sqrt(2.0))), 2, dim=-1)

            loss_w1 = (F.mse_loss(mean, tgt, reduction='none') + 0.1 * F.mse_loss(err_p, torch.abs(tgt - mean), reduction='none')) * (1.0 + 2.0 * th)
            cur_err = loss_w1.mean(dim=-1)
            c_e = c_e * (1.0 - self.lp_alpha) + cur_err * self.lp_alpha
            c_p = c_p * (1.0 - self.lp_alpha) + torch.log((c_e + 1e-6) / (cur_err + 1e-6)) * self.lp_alpha

            raw_omega = th * torch.pow(1.0 - torch.clamp(F.relu(c_p) / (c_e + 1e-6), 0.0, 1.0), 1.0).unsqueeze(-1)
            omega = raw_omega * noise_gate
            settled = F.softsign(c_s + 0.25 * (tgt - c_s) + 0.125 * torch.matmul(torch.randn_like(tgt), self.circ_0_25) * omega * 0.40)
            c_s = torch.cat([s_lat, settled[:, self.num_input_nodes:, :, :]], dim=1)

            raw_em = buntanh(torch.matmul(c_s, self.w_tx), 0.5, 1.0)
            node_vel = torch.norm(c_s - prev_s, dim=-1, keepdim=True)
            latched_em = torch.where(node_vel < 0.025, raw_em, latched_em)
            firing_mask = (latched_em.abs().mean(dim=-1, keepdim=True) > 0.05).float()
            swarm_bcast = (latched_em[:, self.num_input_nodes:] * firing_mask[:, self.num_input_nodes:]).sum(dim=1) / (firing_mask[:, self.num_input_nodes:].sum(dim=1) + 1.0)
            bus = F.softsign(0.65 * bus + 0.35 * (0.60 * s_lat.mean(dim=1) + 0.40 * swarm_bcast))

        # FIXED: average over the 4 motor nodes (dim 1)
        final_motor_vel = torch.norm(
            c_s[:, -self.motor_nodes_count:] - prev_s[:, -self.motor_nodes_count:], 
            dim=-1, 
            keepdim=True
        ).mean(dim=1, keepdim=True)
        latched_s = torch.where((self.mode_tensor == 0) & (final_motor_vel < 0.025), c_s, latched_s)
        latched_s = torch.where(self.mode_tensor == 2, c_s, latched_s)

        return loss_w1.mean(), latched_s, c_o, c_e, c_p

    def forward_island(self, isl, env_in, threat, state, out, err_ema, lp_ema, thinking_steps=8):
        """Single-island forward pass for arbitrary evaluation batch sizes."""
        if state.dim() == 4:
            state = state.squeeze(0)
            out = out.squeeze(0)
            err_ema = err_ema.squeeze(0)
            lp_ema = lp_ema.squeeze(0)
        PB = env_in.shape[0]

        raw_s = torch.matmul(env_in, self.sensory_proj_weight[isl])
        s_lat = F.softsign(raw_s.view(PB, self.num_input_nodes, self.D).permute(1, 0, 2))
        bus = s_lat.mean(dim=0)

        c_s, c_o, c_e, c_p = state.clone(), out.clone(), err_ema.clone(), lp_ema.clone()
        latched_s = c_s.clone()
        prev_s = c_s.clone()

        w_rx = self.w_rx[isl]
        w_tx = self.w_tx[isl]
        w1_w = self.w1_weight[isl]
        mode = self.island_modes[isl]

        cooldown = torch.zeros(PB, dtype=torch.int16, device=self.device)
        settled_counter = torch.zeros(PB, dtype=torch.int16, device=self.device)
        th = threat.view(1, PB, 1)

        latched_em = buntanh(torch.bmm(c_s, w_tx), 0.5, 1.0)

        for step in range(thinking_steps):
            motor_vel = torch.norm(c_s[-self.motor_nodes_count:] - prev_s[-self.motor_nodes_count:], dim=-1).mean(dim=0)
            if mode == "viscous" and (motor_vel < 0.025).any():
                calm_idx = torch.nonzero(motor_vel < 0.025).squeeze(-1)
                latched_s[:, calm_idx, :] = c_s[:, calm_idx, :]

            overall_vel = torch.norm(c_s - prev_s, dim=-1).mean(dim=0)
            is_settled = (overall_vel < 0.018) & (cooldown == 0)
            settled_counter = torch.where(is_settled, settled_counter + 1, torch.zeros_like(settled_counter))

            trigger_pulse = settled_counter >= 2
            if mode == "prekick" and trigger_pulse.any():
                p_idx = torch.nonzero(trigger_pulse).squeeze(-1)
                latched_s[:, p_idx, :] = c_s[:, p_idx, :]

            if trigger_pulse.any():
                p_idx = torch.nonzero(trigger_pulse).squeeze(-1)
                pulse = torch.randn_like(c_s[:, p_idx, :]) * 0.35
                c_s[:, p_idx, :] = F.softsign(c_s[:, p_idx, :] + pulse)
                cooldown[p_idx] = 8
                settled_counter[p_idx] = 0

            is_coasting = (cooldown > 0) | (overall_vel > 0.025)
            noise_gate = torch.where(is_coasting, 0.0, 1.0).view(1, PB, 1)
            cooldown = torch.clamp(cooldown - 1, min=0)
            prev_s = c_s.clone()

            b_exp = bus.unsqueeze(0).expand(self.N, -1, -1)
            perceived = buntanh(torch.bmm(b_exp, w_rx), 0.5, math.sqrt(2.0 + math.sqrt(2.0)))

            raw_boredom = torch.pow(1.0 - torch.clamp(F.relu(c_p) / (c_e + 1e-6), 0.0, 1.0), 1.0).unsqueeze(-1)
            boredom = raw_boredom * noise_gate
            tgt = perceived + math.sqrt(0.05) * torch.matmul(torch.randn_like(c_o), self.circ_1_0) * boredom - (0.02) * c_s * (1.0 - boredom)
            tgt = torch.cat([s_lat, F.softsign(tgt)[self.num_input_nodes:]], dim=0)

            w1_in = torch.bmm(c_s, w1_w)
            mean, err_p = torch.chunk(buntanh(w1_in, 0.5, math.sqrt(2.0 + math.sqrt(2.0))), 2, dim=-1)

            loss_w1 = (F.mse_loss(mean, tgt, reduction='none') + 0.1 * F.mse_loss(err_p, torch.abs(tgt - mean), reduction='none')) * (1.0 + 2.0 * th)
            cur_err = loss_w1.mean(dim=-1)
            c_e = c_e * (1.0 - self.lp_alpha) + cur_err * self.lp_alpha
            c_p = c_p * (1.0 - self.lp_alpha) + torch.log((c_e + 1e-6) / (cur_err + 1e-6)) * self.lp_alpha

            raw_omega = th * torch.pow(1.0 - torch.clamp(F.relu(c_p) / (c_e + 1e-6), 0.0, 1.0), 1.0).unsqueeze(-1)
            omega = raw_omega * noise_gate
            settled = F.softsign(c_s + 0.25 * (tgt - c_s) + 0.125 * torch.matmul(torch.randn_like(tgt), self.circ_0_25) * omega * 0.40)
            c_s = torch.cat([s_lat, settled[self.num_input_nodes:]], dim=0)

            raw_em = buntanh(torch.bmm(c_s, w_tx), 0.5, 1.0)
            node_vel = torch.norm(c_s - prev_s, dim=-1, keepdim=True)
            latched_em = torch.where(node_vel < 0.025, raw_em, latched_em)
            firing_mask = (latched_em.abs().mean(dim=-1, keepdim=True) > 0.05).float()
            swarm_bcast = (latched_em[self.num_input_nodes:] * firing_mask[self.num_input_nodes:]).sum(dim=0) / (firing_mask[self.num_input_nodes:].sum(dim=0) + 1.0)
            bus = F.softsign(0.65 * bus + 0.35 * (0.60 * s_lat.mean(dim=0) + 0.40 * swarm_bcast))

        final_motor_vel = torch.norm(c_s[-self.motor_nodes_count:] - prev_s[-self.motor_nodes_count:], dim=-1).mean(dim=0)
        if mode == "viscous" and (final_motor_vel < 0.025).any():
            c_idx = torch.nonzero(final_motor_vel < 0.025).squeeze(-1)
            latched_s[:, c_idx, :] = c_s[:, c_idx, :]
        if mode == "raw":
            latched_s = c_s

        return loss_w1.mean(), latched_s, c_o, c_e, c_p

    def get_action_logits_island(self, isl, state):
        if state.dim() == 4:
            state = state.squeeze(0)
        PB = state.shape[1]
        motor = state[-self.motor_nodes_count:, :, :].permute(1, 0, 2).reshape(PB, self.motor_nodes_count * self.D)
        nf = F.normalize(motor[:, :64], p=2, dim=-1)
        nt = F.normalize(motor[:, 64:], p=2, dim=-1)
        sq_enc = self.normalized_square_enc[isl]
        lf = torch.matmul(nf, sq_enc)
        lt = torch.matmul(nt, sq_enc)
        cohere = torch.sum(nf * nt, dim=-1, keepdim=True).unsqueeze(-1)
        logits = ((lf.unsqueeze(2) + lt.unsqueeze(1) + cohere * self.spatial_affinity[isl].unsqueeze(0)) * self.action_logit_scale[isl].squeeze()).view(PB, 4096)
        return logits

# ==========================================
# 2. VECTORIZED ENGINE & STEPPING
# ==========================================

def step_unified_swarm(model, env, states, thinking_steps=32):
    """Executes a unified ply step across all 4,096 games concurrently."""
    s_w, o_w, e_w, l_w, s_b, o_b, e_b, l_b, noise = states
    B = env.B
    active = ~env.dones
    l_mask = env.compute_legal_masks()
    actions = torch.zeros(B, dtype=torch.long, device=model.device)

    w_turn = active & (env.players == 0)
    b_turn = active & (env.players == 1)
    is_agent = (w_turn & (env.match_modes != 1)) | (b_turn & (env.match_modes != 3))
    is_bot = active & (~is_agent)

    # 4D Reshaped Views for Broadcasting: [I, 1, sub_B, 1]
    is_white_4d = (env.players == 0).view(model.I, 1, model.sub_B, 1)
    is_agent_4d = is_agent.view(model.I, 1, model.sub_B, 1)
    is_white_3d = (env.players == 0).view(model.I, 1, model.sub_B)
    is_agent_3d = is_agent.view(model.I, 1, model.sub_B)

    if is_agent.any():
        e_in, th = env.get_obs(noise)
        curr_s = torch.where(is_white_4d, s_w, s_b)
        curr_o = torch.where(is_white_4d, o_w, o_b)
        curr_e = torch.where(is_white_3d, e_w, e_b)
        curr_l = torch.where(is_white_3d, l_w, l_b)

        loss, ns, no, ne, nl = model(e_in, th, curr_s, curr_o, curr_e, curr_l, thinking_steps=thinking_steps)
        acts = torch.argmax(model.get_action_logits(ns) + l_mask, dim=-1)
        actions = torch.where(is_agent, acts, actions)

        # In-place buffer updates for callers
        w_upd = is_white_4d & is_agent_4d
        b_upd = (~is_white_4d) & is_agent_4d
        w_upd_3d = is_white_3d & is_agent_3d
        b_upd_3d = (~is_white_3d) & is_agent_3d

        states[0] = torch.where(w_upd, ns, s_w)
        states[1] = torch.where(w_upd, no, o_w)
        states[2] = torch.where(w_upd_3d, ne, e_w)
        states[3] = torch.where(w_upd_3d, nl, l_w)

        states[4] = torch.where(b_upd, ns, s_b)
        states[5] = torch.where(b_upd, no, o_b)
        states[6] = torch.where(b_upd_3d, ne, e_b)
        states[7] = torch.where(b_upd_3d, nl, l_b)

    if is_bot.any():
        bot_idx = torch.where(is_bot)[0]
        bot_acts = env.nc.select_cuda_engine_move(
            env.boards[bot_idx], l_mask[bot_idx],
            (env.players[bot_idx] == 0), model.device, env=env, b_idx=bot_idx
        )
        actions[bot_idx] = bot_acts

    env.dones, env.winners = env.step(actions, env.dones, env.winners, legal_mask=l_mask)
    return actions

# ==========================================
# 3. TELEMETRY & PERSISTENCE
# ==========================================

def save_diagnostic_dump(report_data, full_pt_tensors, out_dir="out/diagnostics"):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(out_dir, f"report_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    pt_path = os.path.join(out_dir, f"tensors_{stamp}.pt")
    torch.save(full_pt_tensors, pt_path)

    print("\n" + "=" * 80)
    print(f" [DIAGNOSTIC PERSISTENCE COMPLETE]")
    print(f"  -> JSON Telemetry Report: {json_path}")
    print(f"  -> Raw Tensor Checkpoint: {pt_path}")
    print("=" * 80 + "\n")

# ==========================================
# 4. DIAGNOSTIC HARNESS RUNNER
# ==========================================

def run_stability_diagnostic(base_module_name="models.nodechess20", num_islands=16, total_batch_size=4096,
                             diagnostic_batches=3, test_modes=["viscous", "prekick"], 
                             init_schemes=["baseline", "quasizero"],
                             thinking_steps=32, noise_cap=0.03, sf_gauntlet_games=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Diagnostic Loader] Importing base architecture from: {base_module_name}...")
    nc = importlib.import_module(base_module_name)

    sub_B = total_batch_size // num_islands
    seeds = [1000 * (i + 1) + 42 for i in range(num_islands)]

    assigned_modes = [test_modes[i % len(test_modes)] for i in range(num_islands)]
    assigned_inits = [init_schemes[(i // len(test_modes)) % len(init_schemes)] for i in range(num_islands)]

    print("=" * 102)
    print(f" HIGH-THROUGHPUT MULTI-ISLAND DIAGNOSTIC (SYNCHRONOUS BATCHED GRAPH)")
    print(f" Total Batch Budget: {total_batch_size} | Islands: {num_islands} ({sub_B} games/island)")
    print(f" Mode Allocations:   {dict(zip(range(num_islands), zip(assigned_inits, assigned_modes)))}")
    print("=" * 102)

    model = BatchedIslandNet(
        num_islands=num_islands, sub_batch_size=sub_B,
        island_modes=assigned_modes, init_schemes=assigned_inits,
        seeds=seeds, device=device
    )

    env = nc.GuardrailedTorchChess(total_batch_size, device, halfmove_limit=100)
    env.nc = nc
    env.dones = torch.zeros(total_batch_size, dtype=torch.bool, device=device)
    env.winners = torch.zeros(total_batch_size, dtype=torch.int8, device=device)
    env.plies = torch.zeros(total_batch_size, dtype=torch.long, device=device)
    env.match_modes = (torch.arange(total_batch_size, device=device) % 4).to(torch.int8)

    s, o, err, lp = model.init_states(total_batch_size)
    noise = torch.full((total_batch_size,), noise_cap * 0.50, device=device)
    states = [s.clone(), o.clone(), err.clone(), lp.clone(), s.clone(), o.clone(), err.clone(), lp.clone(), noise]

    island_sp_games = [0] * num_islands
    island_sp_draws = [0] * num_islands
    island_bot_games = [0] * num_islands
    island_bot_draws = [0] * num_islands
    island_bot_wins = [0] * num_islands
    island_plies = [[] for _ in range(num_islands)]

    print("\n---> Commencing Synchronous Batched Deliberation...")
    for b_idx in range(1, diagnostic_batches + 1):
        batch_completed = 0
        batch_start = time.time()

        while batch_completed < total_batch_size:
            step_unified_swarm(model, env, states, thinking_steps=thinking_steps)
            env.plies[~env.dones] += 1

            if env.dones.any():
                fin_i = torch.where(env.dones)[0]
                batch_completed += len(fin_i)

                for idx in fin_i.tolist():
                    isl = idx // sub_B
                    p = env.plies[idx].item()
                    w = env.winners[idx].item()
                    m = env.match_modes[idx].item()
                    island_plies[isl].append(p)

                    if m in (0, 2):
                        island_sp_games[isl] += 1
                        if w == 3: island_sp_draws[isl] += 1
                    else:
                        island_bot_games[isl] += 1
                        if (m == 3 and w == 1) or (m == 1 and w == 2): island_bot_wins[isl] += 1
                        elif w == 3: island_bot_draws[isl] += 1

                # Clean reset of completed games
                env.match_modes[env.dones] = (env.match_modes[env.dones] + 1) % 4
                env.reset(mask=env.dones)
                env.plies[env.dones] = 0
                env.dones[env.dones] = False
                env.winners[env.dones] = 0

        dt = time.time() - batch_start
        gps = total_batch_size / max(1e-3, dt)
        print(f"  [Batch {b_idx:02d}/{diagnostic_batches:02d}] Throughput: {gps:.1f} games/sec | Total Games: {b_idx * total_batch_size:,}")

    # ==========================================
    # 5. STOCKFISH EVALUATION & FORENSICS
    # ==========================================
    sf_exec = shutil.which("stockfish")
    island_sf_draws = [0] * num_islands

    print(f"\n---> Evaluating Stockfish Gauntlet across all {num_islands} islands ({sf_gauntlet_games} games each)...")
    if sf_exec:
        for isl in range(num_islands):
            sf = nc.StockfishEngine(sf_exec)
            env_sf = nc.GuardrailedTorchChess(sf_gauntlet_games, device)
            sf_is_black = torch.arange(sf_gauntlet_games, device=device) < (sf_gauntlet_games // 2)

            # Slice single agent from island and expand to evaluation batch size
            isl_s = states[0][isl, :, 0:1, :].expand(-1, sf_gauntlet_games, -1).clone()
            isl_o = states[1][isl, :, 0:1, :].expand(-1, sf_gauntlet_games, -1).clone()
            isl_e = states[2][isl, :, 0:1].expand(-1, sf_gauntlet_games).clone()
            isl_l = states[3][isl, :, 0:1].expand(-1, sf_gauntlet_games).clone()

            winners = torch.zeros(sf_gauntlet_games, dtype=torch.int8, device=device)
            dones = torch.zeros(sf_gauntlet_games, dtype=torch.bool, device=device)
            plies = torch.zeros(sf_gauntlet_games, dtype=torch.long, device=device)

            while not dones.all() and (plies < 200).any():
                act_mask = ~dones
                l_m = env_sf.compute_legal_masks()
                acts = torch.zeros(sf_gauntlet_games, dtype=torch.long, device=device)

                is_white = (env_sf.players == 0)
                is_sf = torch.where(is_white, ~sf_is_black, sf_is_black)

                if (act_mask & is_sf).any():
                    for gi in torch.where(act_mask & is_sf)[0].tolist():
                        legals = torch.nonzero(l_m[gi] > -1e8).squeeze(-1).tolist()
                        if not legals:
                            dones[gi], winners[gi] = True, 3
                            continue
                        fen = nc.board_to_fen(
                            env_sf.boards[gi], int(env_sf.players[gi].item()), 
                            env_sf.castling_rights[gi].tolist(), int(env_sf.en_passant_sq[gi].item()),
                            int(env_sf.halfmove_clock[gi].item()), int(plies[gi].item() // 2 + 1)
                        )
                        bm = sf.get_move(fen, nodes=100)
                        act = uci_to_action(bm) if bm else None
                        acts[gi] = act if act in legals else legals[0]

                if (act_mask & (~is_sf)).any():
                    e_in, th = env_sf.get_obs(torch.full((sf_gauntlet_games,), 0.001, device=device))
                    _, isl_s, isl_o, isl_e, isl_l = model.forward_island(isl, e_in, th, isl_s, isl_o, isl_e, isl_l, thinking_steps=8)
                    agent_acts = torch.argmax(model.get_action_logits_island(isl, isl_s) + l_m, dim=-1)
                    acts = torch.where(act_mask & (~is_sf), agent_acts, acts)

                dones, winners = env_sf.step(acts, dones, winners, legal_mask=l_m)
                plies[act_mask] += 1
                over = (plies >= 200) & (~dones)
                if over.any(): dones = dones | over; winners = torch.where(over, 3, winners)

            sf.close()
            draws = (winners == 3).sum().item()
            island_sf_draws[isl] = draws
            print(f"  Island {isl:02d} ({assigned_inits[isl]}+{assigned_modes[isl]}): SF@100 Draws: {draws:2d}/{sf_gauntlet_games}")

    # ==========================================
    # 6. METRIC COMPILATION & SCORECARDS
    # ==========================================
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_batch_size": total_batch_size,
        "num_islands": num_islands,
        "thinking_steps": thinking_steps,
        "islands": {}
    }

    print("\n" + "=" * 102)
    print("                     ARCHITECTURAL TELEMETRY SCORECARD")
    print("=" * 102)
    print(f"{'Isl':<3} | {'Config':<20} | {'Seed':<5} | {'SP Rep%':<8} | {'Bot W%':<7} | {'Bot D%':<7} | {'SF Draws':<10} | {'Status'}")
    print("-" * 102)

    for i in range(num_islands):
        sp_rep = (island_sp_draws[i] / max(1, island_sp_games[i])) * 100.0
        bot_w = (island_bot_wins[i] / max(1, island_bot_games[i])) * 100.0
        bot_d = (island_bot_draws[i] / max(1, island_bot_games[i])) * 100.0
        sf_d = island_sf_draws[i]

        status = "SOLID" if sf_d >= 2 and sp_rep < 80 else ("LOTTERY" if sf_d > 0 and sp_rep >= 80 else "COLLAPSED")
        cfg_str = f"{assigned_inits[i][:8]}+{assigned_modes[i][:8]}"
        print(f"{i:02d}  | {cfg_str:<20} | {seeds[i]:<5} | {sp_rep:6.1f}% | {bot_w:5.1f}% | {bot_d:5.1f}% | {sf_d:2d}/{sf_gauntlet_games:<6} | {status}")

        report["islands"][i] = {
            "config": cfg_str, "init": assigned_inits[i], "mode": assigned_modes[i], "seed": seeds[i],
            "sp_rep_rate": sp_rep, "bot_win_rate": bot_w, "bot_draw_rate": bot_d,
            "sf_draws": sf_d, "avg_ply": statistics.mean(island_plies[i]) if island_plies[i] else 0.0,
            "status": status
        }

    tensor_payload = {
        "model_state_dict": model.state_dict(),
        "states": [st.cpu() for st in states],
        "seeds": seeds,
        "assigned_modes": assigned_modes,
        "assigned_inits": assigned_inits
    }
    save_diagnostic_dump(report, tensor_payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Throughput Multi-Island Diagnostic")
    parser.add_argument("--base_model", type=str, default="models.nodechess20")
    parser.add_argument("--islands", type=int, default=16)
    parser.add_argument("--total_batch_size", type=int, default=4096)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--modes", type=str, nargs="+", default=["viscous", "prekick"])
    parser.add_argument("--init_schemes", type=str, nargs="+", default=["baseline", "quasizero"])
    parser.add_argument("--thinking_steps", type=int, default=32)
    parser.add_argument("--sf_games", type=int, default=20)
    args = parser.parse_args()

    torch.cuda.init()
    torch.set_float32_matmul_precision('high')
    run_stability_diagnostic(
        base_module_name=args.base_model,
        num_islands=args.islands,
        total_batch_size=args.total_batch_size,
        diagnostic_batches=args.batches,
        test_modes=args.modes,
        init_schemes=args.init_schemes,
        thinking_steps=args.thinking_steps,
        sf_gauntlet_games=args.sf_games
    )