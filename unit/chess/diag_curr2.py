"""
nodediagnostic_isolated_ideas.py
Ablation Diagnostic Testing 4 Architectural Fixes in Strict Isolation:
  Group 0: baseline (Status Quo Control)
  Group 1: +geom_affinity (Idea 1 Isolated: 2D Board Euclidean Distance Affinity)
  Group 2: +clean_quasizero (Idea 2 Isolated: True Uncorrupted Quasizero Noise Block)
  Group 3: +bus_diversity (Idea 3 Isolated: Multi-Harmonic Stigmergic Bus Routing)
  Group 4: +motor_decode (Idea 4 Isolated: Decoupled Motor-Cortex Coordinate Heads)
  Group 5: +all_fixes (Full Stack: All 4 Fixes Combined)
"""

import argparse
import datetime
import importlib
import json
import math
import os
import shutil
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. CHESS UTILITIES & NOTATION
# ==========================================

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

def compute_chessboard_euclidean_affinity(sigma=2.5, device="cuda"):
    """Creates a true 64x64 geometric distance matrix based on 2D chessboard coordinates."""
    r = torch.arange(8, device=device).unsqueeze(1).expand(8, 8).reshape(64)
    c = torch.arange(8, device=device).unsqueeze(0).expand(8, 8).reshape(64)
    coords = torch.stack([r, c], dim=-1).float()  # (64, 2)
    dist = torch.cdist(coords, coords, p=2)       # (64, 64)
    
    # Gaussian spatial decay: nearby squares have strong affinity, distant squares decay smoothly
    affinity = torch.exp(-dist / sigma)
    # Zero out diagonal: moving to the exact same square is illegal in chess
    affinity.fill_diagonal_(0.0)
    return affinity

# ==========================================
# 1. MUON RIEMANNIAN OPERATORS
# ==========================================

def batched_newtonschulz5(G, steps=5, eps=1e-7):
    """Batched Muon polar decomposition for [Batch, D_in, D_out]."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    norm = torch.linalg.matrix_norm(X, keepdim=True) + eps
    X = X / norm

    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(-2, -1)
        transposed = True

    for _ in range(steps):
        A = torch.bmm(X, X.transpose(-2, -1))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)

    if transposed:
        X = X.transpose(-2, -1)
    return X.type_as(G)

def project_to_sphere_tangent(G, W, dim=-2):
    """Projects gradient matrix G onto tangent space of spherical weights W."""
    dot_prod = (G * W).sum(dim=dim, keepdim=True)
    w_norm_sq = (W * W).sum(dim=dim, keepdim=True) + 1e-8
    return G - (dot_prod / w_norm_sq) * W

# ==========================================
# 2. VECTORIZED ABLATION MULTI-ISLAND MODEL
# ==========================================

class AblationIslandNet(nn.Module):
    def __init__(self, num_islands, sub_batch_size, group_configs, seeds, num_nodes=32, dim=32, input_dim=64, device="cuda"):
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
        self.group_configs = group_configs
        self.seeds = seeds

        # All islands run prekick mode with Muon learning enabled
        self.dust_lr = torch.full((self.I, 1, 1, 1), 0.02, device=device)

        self.sensory_proj_weight = nn.Parameter(torch.zeros(self.I, input_dim, self.num_input_nodes * dim, device=device), requires_grad=False)
        self.w1_weight = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim * 2, device=device), requires_grad=False)
        self.w_rx = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim, device=device), requires_grad=False)
        self.w_tx = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim, device=device), requires_grad=False)
        
        # Dedicated motor readout matrices (used when fix_motor is active)
        self.motor_enc_from = nn.Parameter(torch.zeros(self.I, 64, 64, device=device), requires_grad=False)
        self.motor_enc_to = nn.Parameter(torch.zeros(self.I, 64, 64, device=device), requires_grad=False)

        self.action_logit_scale = nn.Parameter(torch.full((self.I, 1, 1, 1), 2.5, device=device), requires_grad=False)
        self.register_buffer("spatial_affinity", torch.zeros(self.I, 64, 64, device=device))
        self.register_buffer("normalized_square_enc", torch.zeros(self.I, 64, 64, device=device))

        self.register_buffer("circ_1_0", make_circulant_matrix(dim, 1.0, 1.0, device=device))
        self.register_buffer("circ_0_25", make_circulant_matrix(dim, 1.0, 0.25, device=device))

        self._initialize_ablation_islands()

    def _initialize_ablation_islands(self):
        geom_aff_matrix = compute_chessboard_euclidean_affinity(sigma=2.5, device=self.device)

        for i in range(self.I):
            torch.manual_seed(self.seeds[i])
            torch.cuda.manual_seed_all(self.seeds[i])
            cfg = self.group_configs[i]

            # ----------------------------------------------------
            # 1. SENSORY ENCODING PROJECTION
            # ----------------------------------------------------
            raw_sens = torch.empty(64, self.num_input_nodes * self.D, device=self.device)
            nn.init.orthogonal_(raw_sens, gain=1.0)
            self.sensory_proj_weight.data[i] = F.normalize(raw_sens, p=2, dim=0)
            sq_enc = self.sensory_proj_weight.data[i].clone()
            self.normalized_square_enc[i] = sq_enc

            # ----------------------------------------------------
            # 2. SPATIAL AFFINITY: Idea 1 Isolation
            # ----------------------------------------------------
            if cfg["fix_spatial"]:
                self.spatial_affinity[i] = geom_aff_matrix.clone()
            else:
                # Baseline flaw: Orthogonal collapse -> Identity matrix
                self.spatial_affinity[i] = torch.matmul(sq_enc, sq_enc.T)

            # ----------------------------------------------------
            # 3. W1 WEIGHT & ERROR CHANNEL: Idea 2 Isolation
            # ----------------------------------------------------
            if cfg["fix_qz"]:
                # True Quasizero: Identity block scaled to 0.8, micro-noise block untouched
                eye_part = torch.eye(self.D, device=self.device).unsqueeze(0).expand(self.N, -1, -1) * 0.8
                micro_noise = torch.randn(self.N, self.D, self.D, device=self.device) * 0.01
                self.w1_weight.data[i] = torch.cat([eye_part, micro_noise], dim=-1)
            else:
                # Baseline flaw: Column-wise normalization blows up noise block to 0.8
                eye_split = torch.cat([torch.eye(self.D, device=self.device), torch.zeros(self.D, self.D, device=self.device)], dim=1)
                noise = torch.randn(self.N, self.D, self.D * 2, device=self.device) * 0.01
                w1_cand = (eye_split.unsqueeze(0).expand(self.N, -1, -1) + noise) * 0.5
                self.w1_weight.data[i] = F.normalize(w1_cand, p=2, dim=-2) * 0.8

            # ----------------------------------------------------
            # 4. STIGMERGIC BUS DIVERSITY: Idea 3 Isolation
            # ----------------------------------------------------
            if cfg["fix_bus"]:
                # Harmonic phase-shifting: Each node gets a distinct frequency channel
                for n in range(self.N):
                    phase = 2.0 * math.pi * n / self.N
                    freq = 1.0 + 0.5 * math.sin(phase)
                    t = torch.arange(self.D, device=self.device).float()
                    dft_phase = torch.cos(2.0 * math.pi * freq * torch.ger(t, t) / self.D + phase)
                    self.w_rx.data[i, n] = F.normalize(dft_phase, p=2, dim=-1) * 0.8
                    self.w_tx.data[i, n] = F.normalize(dft_phase.T, p=2, dim=-1) * 0.8
            else:
                # Baseline flaw: All 32 nodes are identical clones of base DFT
                dft = torch.fft.fft(torch.eye(self.D, device=self.device)).real
                dft_norm = F.normalize(dft, p=2, dim=-1) * 0.8
                self.w_rx.data[i] = dft_norm.unsqueeze(0).expand(self.N, -1, -1)
                self.w_tx.data[i] = dft_norm.unsqueeze(0).expand(self.N, -1, -1)

            # ----------------------------------------------------
            # 5. MOTOR-CORTEX DECODING HEAD: Idea 4 Isolation
            # ----------------------------------------------------
            if cfg["fix_motor"]:
                raw_m_from = torch.empty(64, 64, device=self.device)
                raw_m_to = torch.empty(64, 64, device=self.device)
                nn.init.orthogonal_(raw_m_from, gain=1.0)
                nn.init.orthogonal_(raw_m_to, gain=1.0)
                self.motor_enc_from.data[i] = F.normalize(raw_m_from, p=2, dim=0)
                self.motor_enc_to.data[i] = F.normalize(raw_m_to, p=2, dim=0)
            else:
                # Baseline flaw: Motor cortex piggybacks on sensory node projections
                self.motor_enc_from.data[i] = sq_enc.clone()
                self.motor_enc_to.data[i] = sq_enc.clone()

    def normalize_w1(self):
        # Apply normalization respecting the active configuration per island
        for i in range(self.I):
            if self.group_configs[i]["fix_qz"]:
                # Normalize only mean-prediction block (cols 0..31), leave error block intact
                self.w1_weight.data[i, :, :, :self.D].copy_(
                    F.normalize(self.w1_weight.data[i, :, :, :self.D], p=2, dim=-2) * 0.8
                )
            else:
                self.w1_weight.data[i].copy_(F.normalize(self.w1_weight.data[i], p=2, dim=-2) * 0.8)

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

        lf = torch.matmul(nf, self.motor_enc_from)
        lt = torch.matmul(nt, self.motor_enc_to)
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

        cooldown = torch.zeros(self.I, 1, self.sub_B, 1, dtype=torch.int16, device=self.device)
        settled_counter = torch.zeros(self.I, 1, self.sub_B, 1, dtype=torch.int16, device=self.device)

        for step in range(thinking_steps):
            overall_vel = torch.norm(c_s - prev_s, dim=-1, keepdim=True).mean(dim=1, keepdim=True)
            is_settled = (overall_vel < 0.018) & (cooldown == 0)
            settled_counter = torch.where(is_settled, settled_counter + 1, torch.zeros_like(settled_counter))

            trigger_pulse = settled_counter >= 2
            latched_s = torch.where(trigger_pulse, c_s, latched_s)

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
            latched_em = torch.where(node_vel < 0.025, raw_em, raw_em)
            firing_mask = (latched_em.abs().mean(dim=-1, keepdim=True) > 0.05).float()
            swarm_bcast = (latched_em[:, self.num_input_nodes:] * firing_mask[:, self.num_input_nodes:]).sum(dim=1) / (firing_mask[:, self.num_input_nodes:].sum(dim=1) + 1.0)
            bus = F.softsign(0.65 * bus + 0.35 * (0.60 * s_lat.mean(dim=1) + 0.40 * swarm_bcast))

        return loss_w1.mean(), latched_s, c_o, c_e, c_p, s_lat

    def step_muon_dust(self, e_in_flat, th_flat, latched_s, s_lat, c_o, c_e, c_p):
        """Vectorized Muon-Sphere Tangent DUST update."""
        DUST_SIGMA = 0.04
        PB = self.sub_B

        s_magnitude = latched_s.abs().sum(dim=(2, 3), keepdim=True)
        hunger_switch = torch.randn_like(latched_s) / (1.0 + s_magnitude)

        xi = torch.randn(self.I, self.N, PB, self.D * 2, device=self.device)
        w1_in_p = torch.matmul(latched_s, self.w1_weight) + (DUST_SIGMA * xi)
        w1_in_n = torch.matmul(latched_s, self.w1_weight) - (DUST_SIGMA * xi)

        mean_p, err_p_pos = torch.chunk(buntanh(w1_in_p, 0.5, math.sqrt(2.0 + math.sqrt(2.0))), 2, dim=-1)
        mean_n, err_p_neg = torch.chunk(buntanh(w1_in_n, 0.5, math.sqrt(2.0 + math.sqrt(2.0))), 2, dim=-1)

        b_exp = s_lat.mean(dim=1).unsqueeze(1).expand(-1, self.N, -1, -1)
        perceived = buntanh(torch.matmul(b_exp, self.w_rx), 0.5, math.sqrt(2.0 + math.sqrt(2.0)))
        tgt = torch.cat([s_lat, F.softsign(perceived)[:, self.num_input_nodes:, :, :]], dim=1)

        th = th_flat.view(self.I, 1, self.sub_B, 1)
        loss_p = (F.mse_loss(mean_p, tgt, reduction='none') + 0.1 * F.mse_loss(err_p_pos, torch.abs(tgt - mean_p), reduction='none')) * (1.0 + 2.0 * th)
        loss_n = (F.mse_loss(mean_n, tgt, reduction='none') + 0.1 * F.mse_loss(err_p_neg, torch.abs(tgt - mean_n), reduction='none')) * (1.0 + 2.0 * th)

        lp = loss_p.mean(dim=-1, keepdim=True)
        ln = loss_n.mean(dim=-1, keepdim=True)

        gh = (((lp - ln) / (2.0 * DUST_SIGMA)) * xi)
        gh_demand = gh.std(dim=-1, keepdim=True, correction=0)
        s_hungry = latched_s + hunger_switch + (torch.randn_like(latched_s) * gh_demand * 0.10)

        raw_grad = torch.matmul(s_hungry.permute(0, 1, 3, 2), gh) / PB
        ortho_grad = batched_newtonschulz5(raw_grad.view(self.I * self.N, self.D, self.D * 2)).view(self.I, self.N, self.D, self.D * 2)

        tangent_grad = project_to_sphere_tangent(ortho_grad, self.w1_weight, dim=-2)
        self.w1_weight.data.sub_(self.dust_lr * tangent_grad)
        self.normalize_w1()

    def forward_island(self, isl, env_in, threat, state, out, err_ema, lp_ema, thinking_steps=8):
        if state.dim() == 4:
            state = state.squeeze(0); out = out.squeeze(0); err_ema = err_ema.squeeze(0); lp_ema = lp_ema.squeeze(0)
        PB = env_in.shape[0]

        raw_s = torch.matmul(env_in, self.sensory_proj_weight[isl])
        s_lat = F.softsign(raw_s.view(PB, self.num_input_nodes, self.D).permute(1, 0, 2))
        bus = s_lat.mean(dim=0)

        c_s, c_o, c_e, c_p = state.clone(), out.clone(), err_ema.clone(), lp_ema.clone()
        latched_s = c_s.clone(); prev_s = c_s.clone()

        w_rx = self.w_rx[isl]; w_tx = self.w_tx[isl]; w1_w = self.w1_weight[isl]
        cooldown = torch.zeros(PB, dtype=torch.int16, device=self.device)
        settled_counter = torch.zeros(PB, dtype=torch.int16, device=self.device)
        th = threat.view(1, PB, 1)

        for step in range(thinking_steps):
            overall_vel = torch.norm(c_s - prev_s, dim=-1).mean(dim=0)
            is_settled = (overall_vel < 0.018) & (cooldown == 0)
            settled_counter = torch.where(is_settled, settled_counter + 1, torch.zeros_like(settled_counter))

            trigger_pulse = settled_counter >= 2
            if trigger_pulse.any():
                p_idx = torch.nonzero(trigger_pulse).squeeze(-1)
                latched_s[:, p_idx, :] = c_s[:, p_idx, :]
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
            firing_mask = (raw_em.abs().mean(dim=-1, keepdim=True) > 0.05).float()
            swarm_bcast = (raw_em[self.num_input_nodes:] * firing_mask[self.num_input_nodes:]).sum(dim=0) / (firing_mask[self.num_input_nodes:].sum(dim=0) + 1.0)
            bus = F.softsign(0.65 * bus + 0.35 * (0.60 * s_lat.mean(dim=0) + 0.40 * swarm_bcast))

        return loss_w1.mean(), latched_s, c_o, c_e, c_p

    def get_action_logits_island(self, isl, state):
        if state.dim() == 4:
            state = state.squeeze(0)
        PB = state.shape[1]
        motor = state[-self.motor_nodes_count:, :, :].permute(1, 0, 2).reshape(PB, self.motor_nodes_count * self.D)
        nf = F.normalize(motor[:, :64], p=2, dim=-1)
        nt = F.normalize(motor[:, 64:], p=2, dim=-1)
        
        m_from = self.motor_enc_from[isl]
        m_to = self.motor_enc_to[isl]
        lf = torch.matmul(nf, m_from)
        lt = torch.matmul(nt, m_to)
        cohere = torch.sum(nf * nt, dim=-1, keepdim=True).unsqueeze(-1)
        logits = ((lf.unsqueeze(2) + lt.unsqueeze(1) + cohere * self.spatial_affinity[isl].unsqueeze(0)) * self.action_logit_scale[isl].squeeze()).view(PB, 4096)
        return logits

# ==========================================
# 3. UNIFIED DELIBERATION & TRAIN STEP
# ==========================================

def step_unified_swarm(model, env, states, thinking_steps=32):
    s_w, o_w, e_w, l_w, s_b, o_b, e_b, l_b, noise = states
    B = env.B
    active = ~env.dones
    l_mask = env.compute_legal_masks()
    actions = torch.zeros(B, dtype=torch.long, device=model.device)

    w_turn = active & (env.players == 0)
    b_turn = active & (env.players == 1)
    is_agent = (w_turn & (env.match_modes != 1)) | (b_turn & (env.match_modes != 3))
    is_bot = active & (~is_agent)

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

        loss, ns, no, ne, nl, s_lat = model(e_in, th, curr_s, curr_o, curr_e, curr_l, thinking_steps=thinking_steps)
        acts = torch.argmax(model.get_action_logits(ns) + l_mask, dim=-1)
        actions = torch.where(is_agent, acts, actions)

        # Muon Tangent DUST step on active deliberating agents
        model.step_muon_dust(e_in, th, ns, s_lat, no, ne, nl)

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
# 4. EXPERIMENT ORCHESTRATOR
# ==========================================

def run_ablation_diagnostic():
    parser = argparse.ArgumentParser(description="Strict Isolation Diagnostic of 4 Architectural Ideas")
    parser.add_argument("--sub_b", type=int, default=160, help="Batch size per island (total games = 24 * sub_b)")
    parser.add_argument("--batches", type=int, default=3, help="Number of training batches")
    parser.add_argument("--sf_games", type=int, default=20, help="Stockfish games per island")
    parser.add_argument("--sf_nodes", type=int, default=100, help="Stockfish node budget")
    parser.add_argument("--thinking_steps", type=int, default=32, help="Internal deliberation steps (must be >= 24)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = importlib.import_module("models.nodechess20")

    # 6 Isolated Experimental Arms
    arms = [
        {"name": "0. baseline (Control)",      "fix_spatial": False, "fix_qz": False, "fix_bus": False, "fix_motor": False},
        {"name": "1. +geom_affinity (Idea 1)", "fix_spatial": True,  "fix_qz": False, "fix_bus": False, "fix_motor": False},
        {"name": "2. +clean_qz (Idea 2)",      "fix_spatial": False, "fix_qz": True,  "fix_bus": False, "fix_motor": False},
        {"name": "3. +bus_diversity (Idea 3)", "fix_spatial": False, "fix_qz": False, "fix_bus": True,  "fix_motor": False},
        {"name": "4. +motor_decode (Idea 4)",  "fix_spatial": False, "fix_qz": False, "fix_bus": False, "fix_motor": True},
        {"name": "5. +all_fixes (Full Stack)", "fix_spatial": True,  "fix_qz": True,  "fix_bus": True,  "fix_motor": True},
    ]

    seeds_per_arm = 4
    num_islands = len(arms) * seeds_per_arm  # 6 * 4 = 24 islands
    sub_B = args.sub_b
    total_batch_size = num_islands * sub_B   # e.g., 24 * 160 = 3840

    island_configs = [arms[i // seeds_per_arm] for i in range(num_islands)]
    seeds = [1000 + i * 47 for i in range(num_islands)]

    print("=" * 115)
    print(f" STRICT ISOLATION DIAGNOSTIC OF 4 ARCHITECTURAL IDEAS ({len(arms)} ARMS x {seeds_per_arm} SEEDS)")
    print(f" Total Batch Budget: {total_batch_size:,} | Islands: {num_islands} ({sub_B} games/island)")
    print(" Arm Configurations:")
    for a_idx, arm in enumerate(arms):
        print(f"   [{a_idx}] {arm['name']:<30} -> "
              f"Spatial={arm['fix_spatial']!s:<5} | QZ={arm['fix_qz']!s:<5} | "
              f"BusDiv={arm['fix_bus']!s:<5} | MotorDec={arm['fix_motor']!s:<5}")
    print("=" * 115)

    model = AblationIslandNet(
        num_islands=num_islands, sub_batch_size=sub_B,
        group_configs=island_configs, seeds=seeds, device=device
    )

    env = nc.GuardrailedTorchChess(total_batch_size, device, halfmove_limit=100)
    env.nc = nc
    env.dones = torch.zeros(total_batch_size, dtype=torch.bool, device=device)
    env.winners = torch.zeros(total_batch_size, dtype=torch.int8, device=device)
    env.plies = torch.zeros(total_batch_size, dtype=torch.long, device=device)
    env.match_modes = (torch.arange(total_batch_size, device=device) % 4).to(torch.int8)

    s, o, err, lp = model.init_states(total_batch_size)
    noise = torch.full((total_batch_size,), 0.015, device=device)
    states = [s.clone(), o.clone(), err.clone(), lp.clone(), s.clone(), o.clone(), err.clone(), lp.clone(), noise]

    island_sp_games = [0] * num_islands
    island_sp_draws = [0] * num_islands
    island_bot_games = [0] * num_islands
    island_bot_draws = [0] * num_islands
    island_bot_wins = [0] * num_islands
    island_plies = [[] for _ in range(num_islands)]

    print("\n---> Deliberating & Learning across all 24 Islands in Parallel...")
    for b_idx in range(1, args.batches + 1):
        batch_completed = 0
        batch_start = time.time()

        while batch_completed < total_batch_size:
            step_unified_swarm(model, env, states, thinking_steps=args.thinking_steps)
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

                env.match_modes[env.dones] = (env.match_modes[env.dones] + 1) % 4
                env.reset(mask=env.dones)
                env.plies[env.dones] = 0
                env.dones[env.dones] = False
                env.winners[env.dones] = 0

        dt = time.time() - batch_start
        gps = total_batch_size / max(1e-3, dt)
        print(f"  [Batch {b_idx:02d}/{args.batches:02d}] Throughput: {gps:5.1f} games/sec | Total Finished: {b_idx * total_batch_size:,}")

    # ==========================================
    # 5. STOCKFISH STANDARDIZED GAUNTLET
    # ==========================================
    sf_exec = shutil.which("stockfish")
    island_sf_draws = [0] * num_islands
    island_sf_losses = [0] * num_islands
    island_sf_surv_plies = [0.0] * num_islands

    print(f"\n---> Evaluating Standardized Stockfish Gauntlet across all {num_islands} islands ({args.sf_games} games each @ {args.sf_nodes} nodes)...")
    if sf_exec:
        for isl in range(num_islands):
            sf = nc.StockfishEngine(sf_exec)
            env_sf = nc.GuardrailedTorchChess(args.sf_games, device)
            sf_is_black = torch.arange(args.sf_games, device=device) < (args.sf_games // 2)

            isl_s = states[0][isl, :, 0:1, :].expand(-1, args.sf_games, -1).clone()
            isl_o = states[1][isl, :, 0:1, :].expand(-1, args.sf_games, -1).clone()
            isl_e = states[2][isl, :, 0:1].expand(-1, args.sf_games).clone()
            isl_l = states[3][isl, :, 0:1].expand(-1, args.sf_games).clone()

            winners = torch.zeros(args.sf_games, dtype=torch.int8, device=device)
            dones = torch.zeros(args.sf_games, dtype=torch.bool, device=device)
            plies = torch.zeros(args.sf_games, dtype=torch.long, device=device)

            while not dones.all() and (plies < 200).any():
                act_mask = ~dones
                l_m = env_sf.compute_legal_masks()
                acts = torch.zeros(args.sf_games, dtype=torch.long, device=device)

                is_white = (env_sf.players == 0)
                is_sf = torch.where(is_white, ~sf_is_black, sf_is_black)

                if (act_mask & is_sf).any():
                    for gi in torch.where(act_mask & is_sf)[0].tolist():
                        legals = torch.nonzero(l_m[gi] > -1e8).squeeze(-1).tolist()
                        if not legals: dones[gi], winners[gi] = True, 3; continue
                        fen = nc.board_to_fen(
                            env_sf.boards[gi], int(env_sf.players[gi].item()), 
                            env_sf.castling_rights[gi].tolist(), int(env_sf.en_passant_sq[gi].item()),
                            int(env_sf.halfmove_clock[gi].item()), int(plies[gi].item() // 2 + 1)
                        )
                        bm = sf.get_move(fen, nodes=args.sf_nodes)
                        act = uci_to_action(bm) if bm else None
                        acts[gi] = act if act in legals else legals[0]

                if (act_mask & (~is_sf)).any():
                    e_in, th = env_sf.get_obs(torch.full((args.sf_games,), 0.001, device=device))
                    _, isl_s, isl_o, isl_e, isl_l = model.forward_island(isl, e_in, th, isl_s, isl_o, isl_e, isl_l, thinking_steps=args.thinking_steps)
                    agent_acts = torch.argmax(model.get_action_logits_island(isl, isl_s) + l_m, dim=-1)
                    acts = torch.where(act_mask & (~is_sf), agent_acts, acts)

                dones, winners = env_sf.step(acts, dones, winners, legal_mask=l_m)
                plies[act_mask] += 1
                over = (plies >= 200) & (~dones)
                if over.any(): dones = dones | over; winners = torch.where(over, 3, winners)

            sf.close()
            draws = (winners == 3).sum().item()
            losses = (((winners == 2) & sf_is_black) | ((winners == 1) & (~sf_is_black))).sum().item()
            island_sf_draws[isl] = draws
            island_sf_losses[isl] = losses
            island_sf_surv_plies[isl] = plies.float().mean().item()

            arm_name = island_configs[isl]["name"]
            print(f"  Island {isl:02d} [{arm_name:<28}] -> SF@{args.sf_nodes} Draws: {draws:2d}/{args.sf_games} | SurvPlies: {island_sf_surv_plies[isl]:4.1f}")

    # ==========================================
    # 6. ABLATION SCORECARD & STATISTICAL VERDICT
    # ==========================================
    print("\n" + "=" * 115)
    print("                     ARCHITECTURAL ABLATION SCORECARD (IDEA BY IDEA)")
    print("=" * 115)
    print(f"{'Isl':<3} | {'Experimental Arm':<30} | {'Seed':<5} | {'SP Rep%':<8} | {'Bot Win%':<8} | {'SF Draws':<10} | {'SurvPly':<8} | {'Status'}")
    print("-" * 115)

    arm_draws = {a: [] for a in range(len(arms))}
    arm_surv = {a: [] for a in range(len(arms))}
    arm_sp_rep = {a: [] for a in range(len(arms))}

    for i in range(num_islands):
        arm_idx = i // seeds_per_arm
        sp_rep = (island_sp_draws[i] / max(1, island_sp_games[i])) * 100.0
        bot_w = (island_bot_wins[i] / max(1, island_bot_games[i])) * 100.0
        sf_d = island_sf_draws[i]
        surv = island_sf_surv_plies[i]

        arm_draws[arm_idx].append(sf_d)
        arm_surv[arm_idx].append(surv)
        arm_sp_rep[arm_idx].append(sp_rep)

        status = "VIABLE" if sf_d >= 2 and sp_rep < 80 else ("LOTTERY" if sf_d > 0 else "COLLAPSED")
        name = arms[arm_idx]["name"]
        print(f"{i:02d}  | {name:<30} | {seeds[i]:<5} | {sp_rep:6.1f}% | {bot_w:6.1f}% | {sf_d:2d}/{args.sf_games:<6} | {surv:6.1f}   | {status}")

    print("\n" + "=" * 115)
    print("                       STATISTICAL COMPARISON & VERDICT")
    print("=" * 115)
    baseline_mean_draws = statistics.mean(arm_draws[0])
    baseline_mean_surv = statistics.mean(arm_surv[0])

    for a_idx, arm in enumerate(arms):
        draws = arm_draws[a_idx]
        survs = arm_surv[a_idx]
        reps = arm_sp_rep[a_idx]

        m_draws = statistics.mean(draws)
        floor_draws = min(draws)
        m_surv = statistics.mean(survs)
        m_rep = statistics.mean(reps)

        delta_draws = m_draws - baseline_mean_draws
        delta_surv = m_surv - baseline_mean_surv

        print(f"  Arm {a_idx} [{arm['name']}]:")
        print(f"    -> SF Draws:  Mean = {m_draws:4.2f}/{args.sf_games} (Floor = {floor_draws:2d}) [Δ vs Base: {delta_draws:+4.2f}]")
        print(f"    -> Surv Plies: Mean = {m_surv:4.1f} plies [Δ vs Base: {delta_surv:+4.1f}]")
        print(f"    -> SP Repetition Rate: {m_rep:4.1f}%")
        print()

    os.makedirs("out/diagnostics", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"out/diagnostics/ablation_study_{stamp}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "arm_draws": arm_draws,
        "arm_surv": arm_surv,
        "arm_sp_rep": arm_sp_rep
    }, out_file)
    print(f"  [Ablation Results Saved] -> {out_file}\n")

if __name__ == "__main__":
    torch.cuda.init()
    torch.set_float32_matmul_precision('high')
    run_ablation_diagnostic()