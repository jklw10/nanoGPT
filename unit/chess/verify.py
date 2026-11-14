"""
verify_fullstack_scale.py
Scale-Verification Benchmark:
  Architecture: Refined Clean Full Stack (2D Spatial + Clean QZ + Coherent Bus + Harmonic Motor + Motor Muon)
  Scale: 24 Independent Seeds x 100 Stockfish Games = 2,400 Engine Matches
  Deliberation: Strictly 32 thinking steps preserved everywhere.
"""

import argparse
import datetime
import importlib
import math
import os
import shutil
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. CHESS & HARMONIC UTILITIES
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
    r = torch.arange(8, device=device).unsqueeze(1).expand(8, 8).reshape(64)
    c = torch.arange(8, device=device).unsqueeze(0).expand(8, 8).reshape(64)
    coords = torch.stack([r, c], dim=-1).float()
    dist = torch.cdist(coords, coords, p=2)
    affinity = torch.exp(-dist / sigma)
    affinity.fill_diagonal_(0.0)
    return affinity

def compute_geometric_motor_basis(device="cuda"):
    r = torch.arange(8, device=device).float() - 3.5
    c = torch.arange(8, device=device).float() - 3.5
    grid_r, grid_c = torch.meshgrid(r, c, indexing='ij')
    coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=-1)

    features = [
        coords[:, 0:1] / 3.5,
        coords[:, 1:2] / 3.5,
        coords.norm(p=2, dim=-1, keepdim=True) / 4.95,
        coords.abs() / 3.5,
        (coords[:, 0:1] + coords[:, 1:2]) / 7.0,
        (coords[:, 0:1] - coords[:, 1:2]) / 7.0,
    ]
    for freq in [1.0, 2.0, 3.0]:
        features.append(torch.cos(coords * math.pi * freq / 4.0))
        features.append(torch.sin(coords * math.pi * freq / 4.0))

    harmonics = torch.cat(features, dim=-1)
    g = torch.Generator(device=device).manual_seed(42)
    proj = torch.randn(harmonics.shape[1], 64, generator=g, device=device)
    basis = torch.matmul(harmonics, proj)
    return F.normalize(basis, p=2, dim=0)

# ==========================================
# 1. BATCHED MUON TANGENT OPERATORS
# ==========================================

def batched_newtonschulz5(G, steps=5, eps=1e-7):
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
    dot_prod = (G * W).sum(dim=dim, keepdim=True)
    w_norm_sq = (W * W).sum(dim=dim, keepdim=True) + 1e-8
    return G - (dot_prod / w_norm_sq) * W

# ==========================================
# 2. CLEAN FULL STACK NET
# ==========================================

class CleanFullStackNet(nn.Module):
    def __init__(self, num_islands, sub_batch_size, seeds, num_nodes=32, dim=32, input_dim=64, device="cuda"):
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
        self.seeds = seeds

        self.dust_lr = torch.full((self.I, 1, 1, 1), 0.02, device=device)
        self.motor_dust_lr = 0.015

        self.sensory_proj_weight = nn.Parameter(torch.zeros(self.I, input_dim, self.num_input_nodes * dim, device=device), requires_grad=False)
        self.w1_weight = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim * 2, device=device), requires_grad=False)
        self.w_rx = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim, device=device), requires_grad=False)
        self.w_tx = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim, device=device), requires_grad=False)

        self.motor_enc_from = nn.Parameter(torch.zeros(self.I, 64, 64, device=device), requires_grad=False)
        self.motor_enc_to = nn.Parameter(torch.zeros(self.I, 64, 64, device=device), requires_grad=False)

        self.action_logit_scale = nn.Parameter(torch.full((self.I, 1, 1, 1), 2.5, device=device), requires_grad=False)
        self.register_buffer("spatial_affinity", torch.zeros(self.I, 64, 64, device=device))
        self.register_buffer("circ_1_0", make_circulant_matrix(dim, 1.0, 1.0, device=device))
        self.register_buffer("circ_0_25", make_circulant_matrix(dim, 1.0, 0.25, device=device))

        self._initialize_clean_fullstack()

    def _initialize_clean_fullstack(self):
        geom_aff = compute_chessboard_euclidean_affinity(sigma=2.5, device=self.device)
        geom_motor = compute_geometric_motor_basis(device=self.device)

        dft = torch.fft.fft(torch.eye(self.D, device=self.device)).real
        dft_norm = F.normalize(dft, p=2, dim=-1) * 0.8

        for i in range(self.I):
            torch.manual_seed(self.seeds[i])
            torch.cuda.manual_seed_all(self.seeds[i])

            # 1. Sensory Projection
            raw_sens = torch.empty(64, self.num_input_nodes * self.D, device=self.device)
            nn.init.orthogonal_(raw_sens, gain=1.0)
            self.sensory_proj_weight.data[i] = F.normalize(raw_sens, p=2, dim=0)

            # 2. 2D Euclidean Board Affinity
            self.spatial_affinity[i] = geom_aff.clone()

            # 3. Clean Quasizero W1
            eye_part = torch.eye(self.D, device=self.device).unsqueeze(0).expand(self.N, -1, -1) * 0.8
            micro_noise = torch.randn(self.N, self.D, self.D, device=self.device) * 0.01
            self.w1_weight.data[i] = torch.cat([eye_part, micro_noise], dim=-1)

            # 4. Uniform Coherent DFT Bus (Arm 4 Finding)
            self.w_rx.data[i] = dft_norm.unsqueeze(0).expand(self.N, -1, -1)
            self.w_tx.data[i] = dft_norm.unsqueeze(0).expand(self.N, -1, -1)

            # 5. Decoupled 2D Harmonic Motor Heads
            self.motor_enc_from.data[i] = geom_motor.clone()
            self.motor_enc_to.data[i] = geom_motor.clone()

    def normalize_w1(self):
        for i in range(self.I):
            self.w1_weight.data[i, :, :, :self.D].copy_(
                F.normalize(self.w1_weight.data[i, :, :, :self.D], p=2, dim=-2) * 0.8
            )

    def normalize_motor_heads(self):
        for i in range(self.I):
            self.motor_enc_from.data[i].copy_(F.normalize(self.motor_enc_from.data[i], p=2, dim=0))
            self.motor_enc_to.data[i].copy_(F.normalize(self.motor_enc_to.data[i], p=2, dim=0))

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

    def step_muon_dust(self, e_in_flat, th_flat, latched_s, s_lat, c_o, c_e, c_p, l_mask):
        DUST_SIGMA = 0.04
        PB = self.sub_B

        # 1. W1 Muon Update
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

        gh = (((loss_p.mean(dim=-1, keepdim=True) - loss_n.mean(dim=-1, keepdim=True)) / (2.0 * DUST_SIGMA)) * xi)
        gh_demand = gh.std(dim=-1, keepdim=True, correction=0)
        s_hungry = latched_s + hunger_switch + (torch.randn_like(latched_s) * gh_demand * 0.10)

        raw_grad_w1 = torch.matmul(s_hungry.permute(0, 1, 3, 2), gh) / PB
        ortho_grad_w1 = batched_newtonschulz5(raw_grad_w1.view(self.I * self.N, self.D, self.D * 2)).view(self.I, self.N, self.D, self.D * 2)

        tangent_grad_w1 = project_to_sphere_tangent(ortho_grad_w1, self.w1_weight, dim=-2)
        self.w1_weight.data.sub_(self.dust_lr * tangent_grad_w1)
        self.normalize_w1()

        # 2. Motor Readout Tangent Muon Update
        xi_mf = torch.randn(self.I, 64, 64, device=self.device)
        xi_mt = torch.randn(self.I, 64, 64, device=self.device)

        motor = latched_s[:, -self.motor_nodes_count:, :, :].permute(0, 2, 1, 3).reshape(self.I, PB, self.motor_nodes_count * self.D)
        nf = F.normalize(motor[..., :64], p=2, dim=-1)
        nt = F.normalize(motor[..., 64:], p=2, dim=-1)

        lf_p = torch.matmul(nf, self.motor_enc_from + DUST_SIGMA * xi_mf)
        lt_p = torch.matmul(nt, self.motor_enc_to + DUST_SIGMA * xi_mt)
        lf_n = torch.matmul(nf, self.motor_enc_from - DUST_SIGMA * xi_mf)
        lt_n = torch.matmul(nt, self.motor_enc_to - DUST_SIGMA * xi_mt)

        cohere = torch.sum(nf * nt, dim=-1, keepdim=True).unsqueeze(-1)
        logits_p = ((lf_p.unsqueeze(-1) + lt_p.unsqueeze(-2) + cohere * self.spatial_affinity.unsqueeze(1)) * self.action_logit_scale).view(self.B, 4096)
        logits_n = ((lf_n.unsqueeze(-1) + lt_n.unsqueeze(-2) + cohere * self.spatial_affinity.unsqueeze(1)) * self.action_logit_scale).view(self.B, 4096)

        p_log_sm = F.log_softmax(logits_p + l_mask, dim=-1)
        n_log_sm = F.log_softmax(logits_n + l_mask, dim=-1)
        h_p = -(p_log_sm.exp() * p_log_sm).sum(dim=-1).view(self.I, PB).mean(dim=1, keepdim=True).unsqueeze(-1)
        h_n = -(n_log_sm.exp() * n_log_sm).sum(dim=-1).view(self.I, PB).mean(dim=1, keepdim=True).unsqueeze(-1)

        grad_mf = (((h_p - h_n) / (2.0 * DUST_SIGMA)) * xi_mf)
        grad_mt = (((h_p - h_n) / (2.0 * DUST_SIGMA)) * xi_mt)

        ortho_mf = batched_newtonschulz5(grad_mf)
        ortho_mt = batched_newtonschulz5(grad_mt)

        tan_mf = project_to_sphere_tangent(ortho_mf, self.motor_enc_from, dim=0)
        tan_mt = project_to_sphere_tangent(ortho_mt, self.motor_enc_to, dim=0)

        self.motor_enc_from.data.sub_(self.motor_dust_lr * tan_mf)
        self.motor_enc_to.data.sub_(self.motor_dust_lr * tan_mt)
        self.normalize_motor_heads()

    def forward_island(self, isl, env_in, threat, state, out, err_ema, lp_ema, thinking_steps=32):
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
# 3. UNIFIED DELIBERATION STEP
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

        model.step_muon_dust(e_in, th, ns, s_lat, no, ne, nl, l_mask)

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
# 4. LARGE-SCALE BENCHMARK RUNNER
# ==========================================

def run_scale_verification():
    parser = argparse.ArgumentParser(description="Clean Full Stack 24-Seed Scale Verification")
    parser.add_argument("--num_seeds", type=int, default=24, help="Number of independent seeds (islands)")
    parser.add_argument("--sub_b", type=int, default=160, help="Batch size per island")
    parser.add_argument("--batches", type=int, default=3, help="Number of training batches")
    parser.add_argument("--sf_games", type=int, default=100, help="Stockfish games per island")
    parser.add_argument("--sf_nodes", type=int, default=100, help="Stockfish node budget")
    parser.add_argument("--thinking_steps", type=int, default=32, help="Internal deliberation steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = importlib.import_module("models.nodechess20")

    num_islands = args.num_seeds
    sub_B = args.sub_b
    total_batch_size = num_islands * sub_B

    # 24 widely dispersed seeds across PRNG space
    seeds = [1000 + i * 137 for i in range(num_islands)]

    print("=" * 115)
    print(f" CLEAN FULL STACK SCALE VERIFICATION ({num_islands} INDEPENDENT SEEDS)")
    print(f" Total Parallel Games: {total_batch_size:,} ({sub_B} games/seed) | Thinking Steps: {args.thinking_steps}")
    print(f" Target Engine Games: {num_islands * args.sf_games:,} Stockfish Matches @ {args.sf_nodes} nodes")
    print("=" * 115)

    model = CleanFullStackNet(num_islands=num_islands, sub_batch_size=sub_B, seeds=seeds, device=device)

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
                    w = env.winners[idx].item()
                    m = env.match_modes[idx].item()

                    if m in (0, 2):
                        island_sp_games[isl] += 1
                        if w == 3: island_sp_draws[isl] += 1

                env.match_modes[env.dones] = (env.match_modes[env.dones] + 1) % 4
                env.reset(mask=env.dones)
                env.plies[env.dones] = 0
                env.dones[env.dones] = False
                env.winners[env.dones] = 0

        dt = time.time() - batch_start
        gps = total_batch_size / max(1e-3, dt)
        print(f"  [Batch {b_idx:02d}/{args.batches:02d}] Throughput: {gps:5.1f} games/sec | Completed: {b_idx * total_batch_size:,}")

    # ==========================================
    # 5. STOCKFISH ENGINE GAUNTLET
    # ==========================================
    sf_exec = shutil.which("stockfish")
    island_sf_draws = [0] * num_islands
    island_sf_surv_plies = [0.0] * num_islands
    island_draw_mat_deficit = [[] for _ in range(num_islands)]

    print(f"\n---> Evaluating Stockfish Gauntlet ({args.sf_games} games/seed @ {args.sf_nodes} nodes)...")
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
                    _, isl_s, isl_o, isl_e, isl_l = model.forward_island(
                        isl, e_in, th, isl_s, isl_o, isl_e, isl_l, thinking_steps=args.thinking_steps
                    )
                    agent_acts = torch.argmax(model.get_action_logits_island(isl, isl_s) + l_m, dim=-1)
                    acts = torch.where(act_mask & (~is_sf), agent_acts, acts)

                dones, winners = env_sf.step(acts, dones, winners, legal_mask=l_m)
                plies[act_mask] += 1
                over = (plies >= 200) & (~dones)
                if over.any(): dones = dones | over; winners = torch.where(over, 3, winners)

            sf.close()
            draws = (winners == 3).sum().item()
            island_sf_draws[isl] = draws
            island_sf_surv_plies[isl] = plies.float().mean().item()

            # Record final material balance in drawn games
            wm, bm = env_sf.get_material()
            draw_indices = torch.where(winners == 3)[0]
            if len(draw_indices) > 0:
                for di in draw_indices.tolist():
                    agent_mat = wm[di].item() if not sf_is_black[di] else bm[di].item()
                    sf_mat = bm[di].item() if not sf_is_black[di] else wm[di].item()
                    island_draw_mat_deficit[isl].append(agent_mat - sf_mat)

            mean_draw_mat = statistics.mean(island_draw_mat_deficit[isl]) if island_draw_mat_deficit[isl] else 0.0
            print(f"  Seed #{isl:02d} ({seeds[isl]:5d}) -> SF@{args.sf_nodes} Draws: {draws:2d}/{args.sf_games} | "
                  f"SurvPlies: {island_sf_surv_plies[isl]:4.1f} | DrawMatDeficit: {mean_draw_mat:+5.0f} cp")

    # ==========================================
    # 6. COMPREHENSIVE STATISTICAL SUMMARY
    # ==========================================
    print("\n" + "=" * 115)
    print(f"              CLEAN FULL STACK 24-SEED DISTRIBUTION SUMMARY ({num_islands * args.sf_games:,} GAMES)")
    print("=" * 115)

    all_draws = island_sf_draws
    all_surv = island_sf_surv_plies

    mean_d = statistics.mean(all_draws)
    med_d = statistics.median(all_draws)
    stdev_d = statistics.stdev(all_draws)
    min_d = min(all_draws)
    max_d = max(all_draws)

    mean_s = statistics.mean(all_surv)
    min_s = min(all_surv)
    max_s = max(all_surv)

    all_deficits = [item for sublist in island_draw_mat_deficit for item in sublist]
    mean_def = statistics.mean(all_deficits) if all_deficits else 0.0

    print(f"  * Draw Rate Distribution:  Mean = {mean_d:4.2f} / 100 ({mean_d:4.2f}%) | Median = {med_d:4.1f} | StDev = {stdev_d:4.2f}")
    print(f"  * Draw Range:              Floor (Worst Seed) = {min_d:2d} draws | Ceiling (Best Seed) = {max_d:2d} draws")
    print(f"  * Survival Plies:          Mean = {mean_s:4.1f} plies | Floor = {min_s:4.1f} | Ceiling = {max_s:4.1f}")
    print(f"  * Mean Material in Draws:  {mean_def:+5.0f} cp (Values close to 0 = tactical equality; -1500+ = parasitic stalls)")
    print("-" * 115)

    if min_d >= 4:
        print("  >>> VERDICT: CATASTROPHIC COLLAPSE ELIMINATED. TRUE FLOOR IS STABLE ACROSS ALL SEEDS. <<<")
    elif min_d > 0:
        print("  >>> VERDICT: RESILIENT (Floor > 0, No Total Failures, Mild Variance Remains). <<<")
    else:
        print("  >>> VERDICT: COLLAPSE DETECTED (At least one seed hit 0 draws). <<<")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    torch.cuda.init()
    torch.set_float32_matmul_precision('high')
    run_scale_verification()