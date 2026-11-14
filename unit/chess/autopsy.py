"""
seed_brute_100.py
High-throughput 100-seed brute force diagnostic (CUDA-native).
Eliminates parasitic 'Golden Tickets' (spurious repetition draws & timeout stalls)
and accurately clusters worst-case failure modes against Stockfish.
"""

import collections
import datetime
import importlib
import json
import math
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. CHESS UTILITIES & OPERATORS
# ==========================================

def uci_to_action(uci_str):
    if not uci_str or len(uci_str) < 4:
        return None
    try:
        fr = (int(uci_str[1]) - 1) * 8 + (ord(uci_str[0]) - ord('a'))
        to = (int(uci_str[3]) - 1) * 8 + (ord(uci_str[2]) - ord('a'))
        return fr * 64 + to
    except Exception:
        return None

def action_to_str(act):
    f_c, f_r = act // 64 % 8, act // 64 // 8
    t_c, t_r = act % 64 % 8, act % 64 // 8
    return f"{chr(ord('a') + f_c)}{f_r + 1}{chr(ord('a') + t_c)}{t_r + 1}"

def buntanh(x, slope, bound):
    return torch.tanh(x - torch.tanh(x) * slope) * bound

def make_circulant_matrix(dim: int, center_offset=1.0, sigma_val=0.5, device="cuda", dtype=torch.float32):
    k_sz = min(31, dim if dim % 2 == 1 else dim - 1)
    if k_sz < 3:
        return torch.eye(dim, device=device, dtype=dtype)
    half_k = k_sz // 2
    t = torch.arange(-half_k, half_k + 1, device=device, dtype=dtype)
    kernel = (torch.exp(-0.5 * (t * max(sigma_val * math.pi, 1e-4)) ** 2) * torch.cos(center_offset * math.pi * t))
    kernel = (kernel / (torch.norm(kernel, p=2) + 1e-8)).view(1, 1, k_sz)
    eye = torch.eye(dim, device=device, dtype=dtype).unsqueeze(1)
    return F.conv1d(F.pad(eye, (half_k, half_k), mode='circular'), kernel).squeeze(1)

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
# 1. 100-ISLAND BATCHED ARCHITECTURE
# ==========================================

class BruteIslandNet(nn.Module):
    def __init__(self, num_islands=100, sub_batch_size=32, num_nodes=32, dim=32, input_dim=64, seeds=None, device="cuda"):
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

        self.seeds = seeds or [1000 + i * 47 for i in range(num_islands)]
        self.dust_lr = torch.full((self.I, 1, 1, 1), 0.02, device=device)

        self.sensory_proj_weight = nn.Parameter(torch.zeros(self.I, input_dim, self.num_input_nodes * dim, device=device), requires_grad=False)
        self.w1_weight = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim * 2, device=device), requires_grad=False)
        self.w_rx = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim, device=device), requires_grad=False)
        self.w_tx = nn.Parameter(torch.zeros(self.I, num_nodes, dim, dim, device=device), requires_grad=False)
        
        self.action_logit_scale = nn.Parameter(torch.full((self.I, 1, 1, 1), 2.5, device=device), requires_grad=False)
        self.register_buffer("spatial_affinity", torch.zeros(self.I, 64, 64, device=device))
        self.register_buffer("normalized_square_enc", torch.zeros(self.I, 64, 64, device=device))

        self.register_buffer("circ_1_0", make_circulant_matrix(dim, 1.0, 1.0, device=device))
        self.register_buffer("circ_0_25", make_circulant_matrix(dim, 1.0, 0.25, device=device))

        self._initialize_all_islands()

    def _initialize_all_islands(self):
        for i in range(self.I):
            torch.manual_seed(self.seeds[i])
            torch.cuda.manual_seed_all(self.seeds[i])

            raw_sens = torch.empty(64, self.num_input_nodes * self.D, device=self.device)
            nn.init.orthogonal_(raw_sens, gain=1.0)
            self.sensory_proj_weight.data[i] = F.normalize(raw_sens, p=2, dim=0)

            sq_enc = self.sensory_proj_weight.data[i].clone()
            self.normalized_square_enc[i] = sq_enc
            self.spatial_affinity[i] = torch.matmul(sq_enc, sq_enc.T)

            eye_split = torch.cat([torch.eye(self.D, device=self.device), torch.zeros(self.D, self.D, device=self.device)], dim=1)
            noise = torch.randn(self.N, self.D, self.D * 2, device=self.device) * 0.01
            self.w1_weight.data[i] = (eye_split.unsqueeze(0).expand(self.N, -1, -1) + noise) * 0.5
            self.w1_weight.data[i].copy_(F.normalize(self.w1_weight.data[i], p=2, dim=-2) * 0.8)

            dft = torch.fft.fft(torch.eye(self.D, device=self.device)).real
            dft_norm = F.normalize(dft, p=2, dim=-1) * 0.8
            self.w_rx.data[i] = dft_norm.unsqueeze(0).expand(self.N, -1, -1)
            self.w_tx.data[i] = dft_norm.unsqueeze(0).expand(self.N, -1, -1)

    def normalize_w1(self):
        self.w1_weight.data.copy_(F.normalize(self.w1_weight.data, p=2, dim=-2) * 0.8)

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

        lf = torch.matmul(nf, self.normalized_square_enc.transpose(-1, -2))
        lt = torch.matmul(nt, self.normalized_square_enc.transpose(-1, -2))
        cohere = torch.sum(nf * nt, dim=-1, keepdim=True).unsqueeze(-1)

        logits = (lf.unsqueeze(-1) + lt.unsqueeze(-2) + cohere * self.spatial_affinity.unsqueeze(1)) * self.action_logit_scale
        return logits.view(self.B, 4096)

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

        latched_em = buntanh(torch.bmm(c_s, w_tx), 0.5, 1.0)

        for _ in range(thinking_steps):
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

            smooth_gate = (1.0 / (1.0 + overall_vel * 15.0)).view(1, PB, 1)
            cooldown = torch.clamp(cooldown - 1, min=0)
            prev_s = c_s.clone()

            b_exp = bus.unsqueeze(0).expand(self.N, -1, -1)
            perceived = buntanh(torch.bmm(b_exp, w_rx), 0.5, math.sqrt(2.0 + math.sqrt(2.0)))

            raw_boredom = torch.pow(1.0 - torch.clamp(F.relu(c_p) / (c_e + 1e-6), 0.0, 1.0), 1.0).unsqueeze(-1)
            boredom = raw_boredom * smooth_gate
            tgt = perceived + math.sqrt(0.05) * torch.matmul(torch.randn_like(c_o), self.circ_1_0) * boredom - (0.02) * c_s * (1.0 - boredom)
            tgt = torch.cat([s_lat, F.softsign(tgt)[self.num_input_nodes:]], dim=0)

            w1_in = torch.bmm(c_s, w1_w)
            mean, err_p = torch.chunk(buntanh(w1_in, 0.5, math.sqrt(2.0 + math.sqrt(2.0))), 2, dim=-1)
            c_o = mean

            loss_w1 = (F.mse_loss(mean, tgt, reduction='none') + 0.1 * F.mse_loss(err_p, torch.abs(tgt - mean), reduction='none')) * (1.0 + 2.0 * th)
            cur_err = loss_w1.mean(dim=-1)
            c_e = c_e * (1.0 - self.lp_alpha) + cur_err * self.lp_alpha
            c_p = c_p * (1.0 - self.lp_alpha) + torch.log((c_e + 1e-6) / (cur_err + 1e-6)) * self.lp_alpha

            raw_omega = th * torch.pow(1.0 - torch.clamp(F.relu(c_p) / (c_e + 1e-6), 0.0, 1.0), 1.0).unsqueeze(-1)
            omega = raw_omega * smooth_gate
            settled = F.softsign(c_s + 0.25 * (tgt - c_s) + 0.125 * torch.matmul(torch.randn_like(tgt), self.circ_0_25) * omega * 0.40)
            c_s = torch.cat([s_lat, settled[self.num_input_nodes:]], dim=0)

            raw_em = buntanh(torch.bmm(c_s, w_tx), 0.5, 1.0)
            node_vel = torch.norm(c_s - prev_s, dim=-1, keepdim=True)
            latched_em = torch.where(node_vel < 0.025, raw_em, latched_em)
            firing_mask = (latched_em.abs().mean(dim=-1, keepdim=True) > 0.05).float()
            swarm_bcast = (latched_em[self.num_input_nodes:] * firing_mask[self.num_input_nodes:]).sum(dim=0) / (firing_mask[self.num_input_nodes:].sum(dim=0) + 1.0)
            bus = F.softsign(0.65 * bus + 0.35 * (0.60 * s_lat.mean(dim=0) + 0.40 * swarm_bcast))

        return loss_w1.mean(), latched_s, c_o, c_e, c_p

    def forward(self, env_in, threat, state, out, err_ema, lp_ema, thinking_steps=32, act_jitter=None):
        e_in = env_in.view(self.I, self.sub_B, 64)
        th = threat.view(self.I, 1, self.sub_B, 1)

        raw_s = torch.matmul(e_in, self.sensory_proj_weight)
        s_lat = F.softsign(raw_s.view(self.I, self.sub_B, self.num_input_nodes, self.D).permute(0, 2, 1, 3))
        bus = s_lat.mean(dim=1)

        c_s, c_o, c_e, c_p = state.clone(), out.clone(), err_ema.clone(), lp_ema.clone()
        latched_s = c_s.clone(); prev_s = c_s.clone()

        latched_em = buntanh(torch.matmul(c_s, self.w_tx), 0.5, 1.0)
        cooldown = torch.zeros(self.I, 1, self.sub_B, 1, dtype=torch.int16, device=self.device)
        settled_counter = torch.zeros(self.I, 1, self.sub_B, 1, dtype=torch.int16, device=self.device)

        for _ in range(thinking_steps):
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

            smooth_gate = 1.0 / (1.0 + overall_vel * 15.0)
            cooldown = torch.clamp(cooldown - 1, min=0)
            prev_s = c_s.clone()

            b_exp = bus.unsqueeze(1).expand(-1, self.N, -1, -1)
            perceived = buntanh(torch.matmul(b_exp, self.w_rx), 0.5, math.sqrt(2.0 + math.sqrt(2.0)))

            raw_boredom = torch.pow(1.0 - torch.clamp(F.relu(c_p) / (c_e + 1e-6), 0.0, 1.0), 1.0).unsqueeze(-1)
            boredom = raw_boredom * smooth_gate
            tgt = perceived + math.sqrt(0.05) * torch.matmul(torch.randn_like(c_o), self.circ_1_0) * boredom - (0.02) * c_s * (1.0 - boredom)
            tgt = torch.cat([s_lat, F.softsign(tgt)[:, self.num_input_nodes:, :, :]], dim=1)

            w1_in = torch.matmul(c_s, self.w1_weight)
            if act_jitter is not None:
                w1_in = w1_in + act_jitter
            mean, err_p = torch.chunk(buntanh(w1_in, 0.5, math.sqrt(2.0 + math.sqrt(2.0))), 2, dim=-1)
            c_o = mean

            loss_w1 = (F.mse_loss(mean, tgt, reduction='none') + 0.1 * F.mse_loss(err_p, torch.abs(tgt - mean), reduction='none')) * (1.0 + 2.0 * th)
            cur_err = loss_w1.mean(dim=-1)
            c_e = c_e * (1.0 - self.lp_alpha) + cur_err * self.lp_alpha
            c_p = c_p * (1.0 - self.lp_alpha) + torch.log((c_e + 1e-6) / (cur_err + 1e-6)) * self.lp_alpha

            raw_omega = th * torch.pow(1.0 - torch.clamp(F.relu(c_p) / (c_e + 1e-6), 0.0, 1.0), 1.0).unsqueeze(-1)
            omega = raw_omega * smooth_gate
            settled = F.softsign(c_s + 0.25 * (tgt - c_s) + 0.125 * torch.matmul(torch.randn_like(tgt), self.circ_0_25) * omega * 0.40)
            c_s = torch.cat([s_lat, settled[:, self.num_input_nodes:, :, :]], dim=1)

            raw_em = buntanh(torch.matmul(c_s, self.w_tx), 0.5, 1.0)
            node_vel = torch.norm(c_s - prev_s, dim=-1, keepdim=True)
            latched_em = torch.where(node_vel < 0.025, raw_em, latched_em)
            firing_mask = (latched_em.abs().mean(dim=-1, keepdim=True) > 0.05).float()
            swarm_bcast = (latched_em[:, self.num_input_nodes:] * firing_mask[:, self.num_input_nodes:]).sum(dim=1) / (firing_mask[:, self.num_input_nodes:].sum(dim=1) + 1.0)
            bus = F.softsign(0.65 * bus + 0.35 * (0.60 * s_lat.mean(dim=1) + 0.40 * swarm_bcast))

        return loss_w1.mean(), latched_s, c_o, c_e, c_p, s_lat

    def step_muon_dust(self, e_in_flat, th_flat, latched_s, s_lat, c_o, c_e, c_p, active_agent_mask=None):
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

        if active_agent_mask is not None:
            a_mask = active_agent_mask.view(self.I, 1, self.sub_B, 1).float()
            gh = gh * a_mask
            denom = a_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        else:
            denom = float(PB)

        gh_demand = gh.std(dim=-1, keepdim=True, correction=0)
        s_hungry = latched_s + hunger_switch + (torch.randn_like(latched_s) * gh_demand * 0.10)

        raw_grad = torch.matmul(s_hungry.permute(0, 1, 3, 2), gh) / denom
        ortho_grad = batched_newtonschulz5(raw_grad.view(self.I * self.N, self.D, self.D * 2)).view(self.I, self.N, self.D, self.D * 2)

        tangent_grad = project_to_sphere_tangent(ortho_grad, self.w1_weight, dim=-2)
        self.w1_weight.data.sub_(self.dust_lr * tangent_grad)
        self.normalize_w1()

    def get_action_logits_island(self, isl, state):
        if state.dim() == 4:
            state = state.squeeze(0)
        PB = state.shape[1]
        motor = state[-self.motor_nodes_count:, :, :].permute(1, 0, 2).reshape(PB, self.motor_nodes_count * self.D)
        nf = F.normalize(motor[:, :64], p=2, dim=-1)
        nt = F.normalize(motor[:, 64:], p=2, dim=-1)

        sq_enc = self.normalized_square_enc[isl]
        lf = torch.matmul(nf, sq_enc.T)
        lt = torch.matmul(nt, sq_enc.T)
        cohere = torch.sum(nf * nt, dim=-1, keepdim=True).unsqueeze(-1)
        logits = ((lf.unsqueeze(2) + lt.unsqueeze(1) + cohere * self.spatial_affinity[isl].unsqueeze(0)) * self.action_logit_scale[isl].squeeze()).view(PB, 4096)
        return logits

# ==========================================
# 2. RUNNER & ROOT CAUSE CLUSTERING
# ==========================================
def run_seed_brute(num_seeds=100, sub_b=32, train_batches=2, sf_games_per_seed=10, resume=True):
    device = torch.device("cuda")
    nc = importlib.import_module("models.nodechess20")

    seeds = [1000 + i * 47 for i in range(num_seeds)]
    total_batch = num_seeds * sub_b
    os.makedirs("out/diagnostics/inits", exist_ok=True)

    trained_cache_file = "out/diagnostics/trained_island_cache.pt"
    gauntlet_ckpt_file = "out/diagnostics/gauntlet_progress.pt"

    print("=" * 110)
    print(" 100-SEED BRUTE FORCE FAILURE TAXONOMY (RESUMABLE + EXTREME INIT EXPORTER)")
    print(f" Total Seeds: {num_seeds} | Games/Island: {sub_b} | Deliberation Budget: {total_batch}")
    print(f" Evaluation: {sf_games_per_seed} Stockfish @ 100 Nodes per Seed ({num_seeds * sf_games_per_seed:,} Total Engine Games)")
    print("=" * 110)

    # 1. INITIALIZE ARCHITECTURE
    model = BruteIslandNet(num_islands=num_seeds, sub_batch_size=sub_b, seeds=seeds, device=device)

    # 2. CHECKPOINTED TRAINING (SKIPS IF ALREADY TRAINED)
    states = None
    if resume and os.path.exists(trained_cache_file):
        print(f"\n[Resume Cache Found] Loading trained islands from -> {trained_cache_file}")
        cached_data = torch.load(trained_cache_file, map_location=device, weights_only=False)
        model.load_state_dict(cached_data["model_state_dict"])
        states = [s.to(device) for s in cached_data["states"]]
        noise = cached_data["noise"].to(device)
    else:
        env = nc.GuardrailedTorchChess(total_batch, device, halfmove_limit=100)
        env.nc = nc
        env.dones = torch.zeros(total_batch, dtype=torch.bool, device=device)
        env.winners = torch.zeros(total_batch, dtype=torch.int8, device=device)
        env.plies = torch.zeros(total_batch, dtype=torch.long, device=device)
        env.match_modes = (torch.arange(total_batch, device=device) % 4).to(torch.int8)

        noise = torch.empty(total_batch, device=device)
        for isl in range(num_seeds):
            start, end = isl * sub_b, (isl + 1) * sub_b
            noise[start:end] = torch.linspace(0.04, 0.08, sub_b, device=device)

        s, o, err, lp = model.init_states(total_batch)
        states = [s.clone(), o.clone(), err.clone(), lp.clone(), s.clone(), o.clone(), err.clone(), lp.clone(), noise]

        print(f"\n---> Training across all {num_seeds} seeds concurrently ({train_batches} batches)...")
        for b_idx in range(1, train_batches + 1):
            completed = 0
            t0 = time.time()
            while completed < total_batch:
                active = ~env.dones
                l_mask = env.compute_legal_masks()
                actions = torch.zeros(total_batch, dtype=torch.long, device=device)

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
                    curr_s = torch.where(is_white_4d, states[0], states[4])
                    curr_o = torch.where(is_white_4d, states[1], states[5])
                    curr_e = torch.where(is_white_3d, states[2], states[6])
                    curr_l = torch.where(is_white_3d, states[3], states[7])

                    loss, ns, no, ne, nl, s_lat = model(e_in, th, curr_s, curr_o, curr_e, curr_l, thinking_steps=32)
                    acts = torch.argmax(model.get_action_logits(ns) + l_mask, dim=-1)
                    actions = torch.where(is_agent, acts, actions)

                    model.step_muon_dust(e_in, th, ns, s_lat, no, ne, nl, active_agent_mask=is_agent)

                    w_upd = is_white_4d & is_agent_4d
                    b_upd = (~is_white_4d) & is_agent_4d
                    w_upd_3d = is_white_3d & is_agent_3d
                    b_upd_3d = (~is_white_3d) & is_agent_3d

                    states[0] = torch.where(w_upd, ns, states[0])
                    states[1] = torch.where(w_upd, no, states[1])
                    states[2] = torch.where(w_upd_3d, ne, states[2])
                    states[3] = torch.where(w_upd_3d, nl, states[3])

                    states[4] = torch.where(b_upd, ns, states[4])
                    states[5] = torch.where(b_upd, no, states[5])
                    states[6] = torch.where(b_upd_3d, ne, states[6])
                    states[7] = torch.where(b_upd_3d, nl, states[7])

                if is_bot.any():
                    bot_idx = torch.where(is_bot)[0]
                    actions[bot_idx] = env.nc.select_cuda_engine_move(
                        env.boards[bot_idx], l_mask[bot_idx], (env.players[bot_idx] == 0), device, env=env, b_idx=bot_idx
                    )

                env.dones, env.winners = env.step(actions, env.dones, env.winners, legal_mask=l_mask)
                env.plies[active] += 1

                if env.dones.any():
                    completed += env.dones.sum().item()
                    env.match_modes[env.dones] = (env.match_modes[env.dones] + 1) % 4
                    env.reset(mask=env.dones)
                    env.plies[env.dones] = 0
                    env.dones[env.dones] = False
                    env.winners[env.dones] = 0

            dt = time.time() - t0
            print(f"  [Batch {b_idx:02d}/{train_batches:02d}] {total_batch / max(1e-3, dt):.1f} games/sec")

        # Save training cache so re-runs never repeat training
        torch.save({
            "model_state_dict": model.state_dict(),
            "states": [s.cpu() for s in states],
            "noise": noise.cpu()
        }, trained_cache_file)
        print(f"[Training Complete] Checkpoint saved -> {trained_cache_file}")

    # 3. RESUMABLE STOCKFISH GAUNTLET
    sf_exec = shutil.which("stockfish")
    if not sf_exec:
        print("Stockfish not found in PATH.")
        return

    results = []
    failure_counts = collections.Counter()
    opening_moves_rotten = collections.Counter()
    opening_moves_golden = collections.Counter()
    completed_islands = set()

    # Load progressive gauntlet checkpoint if available
    if resume and os.path.exists(gauntlet_ckpt_file):
        try:
            ckpt_data = torch.load(gauntlet_ckpt_file, map_location="cpu", weights_only=False)
            results = ckpt_data.get("results", [])
            for r in results:
                completed_islands.add(r["island"])
                failure_counts[r["failure_mode"]] += 1
                if "Mode E" in r["failure_mode"]:
                    opening_moves_golden[r["opening_move"]] += 1
                else:
                    opening_moves_rotten[r["opening_move"]] += 1
            print(f"[Gauntlet Checkpoint Found] Resuming: {len(completed_islands)}/{num_seeds} seeds already evaluated.")
        except Exception as e:
            print(f"[Warning] Could not read gauntlet checkpoint: {e}. Starting fresh.")

    if len(completed_islands) < num_seeds:
        print(f"\n---> Running Stockfish Gauntlet across remaining seeds ({sf_games_per_seed} games/seed)...")

    for isl in range(num_seeds):
        if isl in completed_islands:
            continue

        sf = nc.StockfishEngine(sf_exec)
        env_sf = nc.GuardrailedTorchChess(sf_games_per_seed, device, halfmove_limit=150)
        sf_is_black = torch.arange(sf_games_per_seed, device=device) < (sf_games_per_seed // 2)

        s_w = states[0][isl, :, 0:1, :].expand(-1, sf_games_per_seed, -1).clone()
        o_w = states[1][isl, :, 0:1, :].expand(-1, sf_games_per_seed, -1).clone()
        e_w = states[2][isl, :, 0:1].expand(-1, sf_games_per_seed).clone()
        l_w = states[3][isl, :, 0:1].expand(-1, sf_games_per_seed).clone()

        s_b = states[4][isl, :, 0:1, :].expand(-1, sf_games_per_seed, -1).clone()
        o_b = states[5][isl, :, 0:1, :].expand(-1, sf_games_per_seed, -1).clone()
        e_b = states[6][isl, :, 0:1].expand(-1, sf_games_per_seed).clone()
        l_b = states[7][isl, :, 0:1].expand(-1, sf_games_per_seed).clone()

        winners = torch.zeros(sf_games_per_seed, dtype=torch.int8, device=device)
        dones = torch.zeros(sf_games_per_seed, dtype=torch.bool, device=device)
        plies = torch.zeros(sf_games_per_seed, dtype=torch.long, device=device)

        first_moves = []
        min_material_deltas = [0] * sf_games_per_seed

        while not dones.all() and (plies < 150).any():
            active = ~dones
            l_m = env_sf.compute_legal_masks()
            acts = torch.zeros(sf_games_per_seed, dtype=torch.long, device=device)
            w_turn, b_turn = active & (env_sf.players == 0), active & (env_sf.players == 1)

            for turn_mask, is_white, sb, ob, eb, lb in [
                (w_turn, True, s_w, o_w, e_w, l_w),
                (b_turn, False, s_b, o_b, e_b, l_b)
            ]:
                if not turn_mask.any():
                    continue

                sf_turn = (~sf_is_black) if is_white else sf_is_black
                active_sf = turn_mask & sf_turn
                active_agent = turn_mask & (~sf_turn)

                if active_sf.any():
                    for gi in torch.where(active_sf)[0].tolist():
                        legals = torch.nonzero(l_m[gi] > -1e8).squeeze(-1).tolist()
                        if not legals:
                            if env_sf.is_in_check(int(env_sf.players[gi].item()), env_sf.boards[gi:gi+1]):
                                dones[gi], winners[gi] = True, (1 if env_sf.players[gi] == 1 else 2)
                            else:
                                dones[gi], winners[gi] = True, 3
                            continue
                        fen = nc.board_to_fen(
                            env_sf.boards[gi], int(env_sf.players[gi].item()),
                            env_sf.castling_rights[gi].tolist(), int(env_sf.en_passant_sq[gi].item()),
                            int(env_sf.halfmove_clock[gi].item()), int(plies[gi].item() // 2 + 1)
                        )
                        bm = sf.get_move(fen, nodes=100)
                        act = uci_to_action(bm) if bm else None
                        acts[gi] = act if (act is not None and act in legals) else legals[0]

                if active_agent.any():
                    a_idx = torch.where(active_agent)[0]
                    e_in, th = env_sf.get_obs(torch.full((sf_games_per_seed,), 0.001, device=device))
                    _, ns, no, ne, nl = model.forward_island(
                        isl, e_in[a_idx], th[a_idx], sb[:, a_idx, :], ob[:, a_idx, :], eb[:, a_idx], lb[:, a_idx], thinking_steps=8
                    )
                    sb[:, a_idx, :], ob[:, a_idx, :], eb[:, a_idx], lb[:, a_idx] = ns, no, ne, nl
                    agent_acts = torch.argmax(model.get_action_logits_island(isl, ns) + l_m[a_idx], dim=-1)
                    acts[a_idx] = agent_acts

                    if len(first_moves) == 0 and is_white and (a_idx == 0).any():
                        first_moves.append(action_to_str(acts[0].item()))

            wm, bm = env_sf.get_material()
            for gi in range(sf_games_per_seed):
                ag_delta = (wm[gi] - bm[gi]).item() if sf_is_black[gi] else (bm[gi] - wm[gi]).item()
                if ag_delta < min_material_deltas[gi]:
                    min_material_deltas[gi] = ag_delta

            dones, winners = env_sf.step(acts, dones, winners, legal_mask=l_m)
            plies[active] += 1
            over = (plies >= 150) & (~dones)
            if over.any():
                dones = dones | over
                winners = torch.where(over, torch.tensor(3, dtype=torch.int8, device=device), winners)

        sf.close()

        ag_wins = (((winners == 1) & sf_is_black) | ((winners == 2) & (~sf_is_black))).sum().item()
        clean_draws, fluke_draws = 0, 0
        for gi in range(sf_games_per_seed):
            if winners[gi] == 3:
                if plies[gi].item() < 30 or min_material_deltas[gi] < -150:
                    fluke_draws += 1
                else:
                    clean_draws += 1

        loss_plies = [plies[gi].item() for gi in range(sf_games_per_seed) if winners[gi] != 3 and winners[gi] != 0]
        mean_loss_ply = (sum(loss_plies) / len(loss_plies)) if loss_plies else 150.0
        worst_mat = int(min(min_material_deltas))  # Cast explicitly to int
        w_move = first_moves[0] if first_moves else "none"

        if ag_wins > 0 or clean_draws >= 1:
            failure_mode = "Mode E: Viable / Resilient (Genuine)"
            opening_moves_golden[w_move] += 1
        elif fluke_draws >= 1:
            failure_mode = "Mode D: Parasitic Golden Ticket (Spurious Draw)"
            opening_moves_rotten[w_move] += 1
        elif mean_loss_ply <= 16:
            failure_mode = "Mode A: Rapid King Suicide"
            opening_moves_rotten[w_move] += 1
        elif worst_mat <= -300:
            failure_mode = "Mode B: Tactical Piece Blindness"
            opening_moves_rotten[w_move] += 1
        else:
            failure_mode = "Mode C: Passive Asphyxiation"
            opening_moves_rotten[w_move] += 1

        failure_counts[failure_mode] += 1
        results.append({
            "island": isl, "seed": seeds[isl], "ag_wins": ag_wins,
            "clean_draws": clean_draws, "fluke_draws": fluke_draws,
            "mean_loss_ply": round(mean_loss_ply, 1), "worst_mat": worst_mat,
            "opening_move": w_move, "failure_mode": failure_mode
        })
        completed_islands.add(isl)

        # Progressive checkpoint save so progress is never lost
        torch.save({"results": results}, gauntlet_ckpt_file)

        if (isl + 1) % 10 == 0 or isl == 0:
            print(f"  Processed {isl+1:03d}/{num_seeds} seeds | Wins: {ag_wins} | CleanD: {clean_draws} | FlukeD: {fluke_draws} | Mode: {failure_mode} (Move: {w_move})")

    # 4. RANK & EXPORT EXTREME INITIALIZATIONS (NO FLOAT FORMAT CRASH)
    def seed_score(r):
        if "Mode E" in r["failure_mode"]:
            return 1000 + r["ag_wins"] * 50 + r["clean_draws"] * 10
        elif "Mode D" in r["failure_mode"]:
            return 0
        elif "Mode A" in r["failure_mode"]:
            return -1000 + r["mean_loss_ply"]
        elif "Mode B" in r["failure_mode"]:
            return -500 + float(r["worst_mat"])
        else:
            return -200 + r["mean_loss_ply"]

    sorted_results = sorted(results, key=seed_score)
    k_extremes = min(5, len(sorted_results))
    worst_k = sorted_results[:k_extremes]
    best_k = sorted_results[-k_extremes:][::-1]

    print("\n" + "=" * 110)
    print("                     EXPORTING EXTREME INITIALIZATION ARTIFACTS")
    print("=" * 110)

    for tag, group in [("WORST", worst_k), ("BEST", best_k)]:
        print(f"\n  [{tag} SEED INITIALIZATIONS]:")
        for rank, item in enumerate(group):
            isl = item["island"]
            seed_val = item["seed"]
            mat_int = int(item["worst_mat"])
            init_dump_path = f"out/diagnostics/inits/{tag.lower()}_rank{rank+1}_seed_{seed_val}.pt"

            torch.save({
                "island": isl,
                "seed": seed_val,
                "failure_mode": item["failure_mode"],
                "opening_move": item["opening_move"],
                "ag_wins": item["ag_wins"],
                "clean_draws": item["clean_draws"],
                "fluke_draws": item["fluke_draws"],
                "worst_mat": mat_int,
                "mean_loss_ply": item["mean_loss_ply"],
                "sensory_proj": model.sensory_proj_weight[isl].detach().cpu(),
                "normalized_square_enc": model.normalized_square_enc[isl].detach().cpu(),
                "spatial_affinity": model.spatial_affinity[isl].detach().cpu(),
                "w1_weight": model.w1_weight[isl].detach().cpu(),
                "w_rx": model.w_rx[isl].detach().cpu(),
                "w_tx": model.w_tx[isl].detach().cpu(),
            }, init_dump_path)

            print(f"    * {tag} #{rank+1} | Seed: {seed_val:5d} | {item['failure_mode']:<42} | Mat: {mat_int:+5d} | Move: {item['opening_move']} -> {init_dump_path}")

    # 5. TAXONOMY REPORT
    print("\n" + "=" * 110)
    print("                     100-SEED BRUTE FORCE FAILURE TAXONOMY REPORT")
    print("=" * 110)

    n_viable = failure_counts["Mode E: Viable / Resilient (Genuine)"]
    n_golden_fluke = failure_counts["Mode D: Parasitic Golden Ticket (Spurious Draw)"]
    n_collapsed = num_seeds - n_viable

    print(f"Total Seeds Evaluated: {num_seeds}")
    print(f"  -> Genuinely Viable Seeds:             {n_viable:3d} / {num_seeds} ({n_viable / num_seeds * 100:5.1f}%)")
    print(f"  -> Parasitic Golden Tickets (Purged):  {n_golden_fluke:3d} / {num_seeds} ({n_golden_fluke / num_seeds * 100:5.1f}%)")
    print(f"  -> Total Collapsed / Flawed Seeds:     {n_collapsed:3d} / {num_seeds} ({n_collapsed / num_seeds * 100:5.1f}%)")

    print("\nFAILURE CLUSTER BREAKDOWN:")
    for mode, count in failure_counts.most_common():
        pct = (count / num_seeds) * 100.0
        print(f"  * {mode:<45}: {count:2d} seeds ({pct:5.1f}%)")

    print("\nMOST COMMON OPENING MOVES AMONG COLLAPSED & PARASITIC SEEDS:")
    for mv, cnt in opening_moves_rotten.most_common(5):
        print(f"    Move [{mv}]: {cnt:2d} seeds ({cnt / max(1, n_collapsed) * 100:5.1f}%)")

    if n_viable > 0:
        print("\nMOST COMMON OPENING MOVES AMONG GENUINELY VIABLE SEEDS:")
        for mv, cnt in opening_moves_golden.most_common(5):
            print(f"    Move [{mv}]: {cnt:2d} seeds ({cnt / max(1, n_viable) * 100:5.1f}%)")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"out/diagnostics/seed_brute_100_report_{stamp}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)

    # Clean up gauntlet checkpoint on successful completion
    if os.path.exists(gauntlet_ckpt_file):
        os.remove(gauntlet_ckpt_file)

    print(f"\n[Detailed JSON Report Saved] -> {report_file}\n" + "=" * 110)   
if __name__ == "__main__":
    torch.cuda.init()
    torch.set_float32_matmul_precision('high')
    run_seed_brute()