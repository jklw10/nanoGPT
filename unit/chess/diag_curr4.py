# =====================================================================
# nodediagnostic_full_factorial.py
# Full 2^3 Factorial Ablation (8 Arms x 4 Seeds = 32 Islands)
# Trains and evaluates all combinations of:
#   1. Anti-Bait Bot
#   2. Draw Culling Gate
#   3. R^2 History Loss
# ==========================================

import argparse
import datetime
import importlib
import math
import os
import random
import shutil
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. CHESS UTILITIES & BLUNDERING ENGINE
# ==========================================
PIECE_TO_FEN = {0: '', 1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K', 7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'}

def board_to_fen(board, turn, rights, ep=-1, halfc=0, fulln=1):
    rows = []
    for r in range(7, -1, -1):
        empty, r_s = 0, ""
        for c in range(8):
            p = int(board[r, c].item())
            if p == 0: empty += 1
            else:
                if empty: r_s += str(empty); empty = 0
                r_s += PIECE_TO_FEN.get(p, '')
        if empty: r_s += str(empty)
        rows.append(r_s)
    c_s = "".join([s for s, has in zip("KQkq", rights) if has]) or "-"
    ep_s = f"{chr(ord('a') + (ep % 8))}{(ep // 8) + 1}" if ep >= 0 else "-"
    return f"{'/'.join(rows)} {'w' if turn == 0 else 'b'} {c_s} {ep_s} {halfc} {fulln}"

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

def make_circulant_matrix(dim: int, center_offset=1.0, sigma_val=0.5, device="cuda"):
    k_sz = min(31, dim if dim % 2 == 1 else dim - 1)
    if k_sz < 3: return torch.eye(dim, device=device)
    half_k = k_sz // 2
    t = torch.arange(-half_k, half_k + 1, device=device, dtype=torch.float32)
    kernel = (torch.exp(-0.5 * (t * max(sigma_val * math.pi, 1e-4)) ** 2) * torch.cos(center_offset * math.pi * t))
    kernel = (kernel / (torch.norm(kernel, p=2) + 1e-8)).view(1, 1, k_sz)
    eye = torch.eye(dim, device=device).unsqueeze(1)
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
        coords[:, 0:1] / 3.5, coords[:, 1:2] / 3.5,
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
    return F.normalize(torch.matmul(harmonics, proj), p=2, dim=0)

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

class BlunderingStockfishEngine:
    def __init__(self, stockfish_path="stockfish", skill_level=0):
        import subprocess
        self.proc = subprocess.Popen([stockfish_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for cmd in [
            "uci", "setoption name Threads value 1", "setoption name Hash value 16",
            "setoption name UCI_LimitStrength value true", "setoption name UCI_Elo value 1320",
            f"setoption name Skill Level value {skill_level}", "isready"
        ]:
            self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        while "readyok" not in self.proc.stdout.readline(): pass

    def get_move_with_blunder(self, fen, legal_actions, blunder_prob=0.25):
        if random.random() < blunder_prob and len(legal_actions) > 1:
            return random.choice(legal_actions)

        self.proc.stdin.write(f"position fen {fen}\ngo nodes 100\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line: return random.choice(legal_actions)
            if line.startswith("bestmove"):
                p = line.split()
                bm = p[1] if len(p) > 1 and p[1] not in ("(none)", "0000") else None
                act = uci_to_action(bm) if bm else None
                return act if act in legal_actions else random.choice(legal_actions)

    def close(self):
        try: self.proc.stdin.write("quit\n"); self.proc.stdin.flush(); self.proc.terminate()
        except Exception: pass

# ==========================================
# 1. ANTI-BAIT TACTICAL CUDA ENGINE
# ==========================================
_ENG_CONST = {}

def select_cuda_engine_move_repetition_aware(boards, legal_mask, is_engine_white, device, env=None, b_idx=None, enable_anti_bait=True):
    B = boards.shape[0]
    if device not in _ENG_CONST:
        to_s, fr_s = torch.arange(4096, device=device) % 64, torch.arange(4096, device=device) // 64
        dr, dc = to_s // 8, to_s % 8
        pb = (7.0 - ((dr.float() - 3.5).abs() + (dc.float() - 3.5).abs())) * 2.5 + torch.where(((dr == 3) | (dr == 4)) & ((dc == 3) | (dc == 4)), 15.0, 0.0)
        _ENG_CONST[device] = {'to_s': to_s, 'fr_s': fr_s, 'dr': dr, 'dc': dc, 'fr_r': fr_s // 8, 'pb': pb}
    c = _ENG_CONST[device]

    flat = boards.view(B, 64)
    tp, mp = flat.gather(1, c['to_s'].unsqueeze(0).expand(B, -1)).long().clamp(0, 12), flat.gather(1, c['fr_s'].unsqueeze(0).expand(B, -1)).long().clamp(0, 12)
    p_val = torch.tensor([0, 100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000], dtype=torch.float32, device=device)
    is_w = is_engine_white.unsqueeze(1)
    is_enemy = torch.where(is_w, (tp >= 7) & (tp <= 12), (tp >= 1) & (tp <= 6))
    cap_scores = torch.where(is_enemy, p_val[tp] * 2.0 - p_val[mp] * 0.1 + 100.0, 0.0)

    dr_f, dc_f = c['dr'].unsqueeze(0).expand(B, -1), c['dc'].unsqueeze(0).expand(B, -1)
    att_r = torch.where(is_w, (dr_f + 1).clamp(0, 7), (dr_f - 1).clamp(0, 7))
    e_pawn = torch.where(is_w, 7, 1)
    sub_b = torch.arange(B, device=device).unsqueeze(1)
    p_threat = (((dc_f > 0) & (boards[sub_b, att_r, (dc_f - 1).clamp(0, 7)] == e_pawn)) | ((dc_f < 7) & (boards[sub_b, att_r, (dc_f + 1).clamp(0, 7)] == e_pawn))) & torch.where(is_w, dr_f < 7, dr_f > 0)
    safety_pen = torch.where(p_threat, torch.where(torch.where(is_w, (mp >= 1) & (mp <= 6), (mp >= 7) & (mp <= 12)), p_val[mp], 0.0) * 0.9, 0.0)

    is_p = (mp == 1) | (mp == 7)
    p_dir = torch.where(is_w, c['dr'].float() - c['fr_r'].float(), c['fr_r'].float() - c['dr'].float())
    pawn_push = torch.where(is_p & (p_dir > 0), p_dir * 12.0 + torch.where(is_w, c['dr'].float(), 7.0 - c['dr'].float()) * 6.0, 0.0) + torch.where(is_p & torch.where(is_w, c['dr'] == 7, c['dr'] == 0), 800.0, 0.0)

    is_m = (mp == 2) | (mp == 3) | (mp == 8) | (mp == 9)
    dev_bonus = torch.where(is_m & torch.where(is_w, (c['fr_r'] == 0) & (c['dr'] > 0), (c['fr_r'] == 7) & (c['dr'] < 7)), 30.0, 0.0)
    is_cast = (is_w & (mp == 6) & (c['fr_s'] == 4) & ((c['to_s'] == 6) | (c['to_s'] == 2))) | ((~is_w) & (mp == 12) & (c['fr_s'] == 60) & ((c['to_s'] == 62) | (c['to_s'] == 58)))
    castle_bonus = torch.where(is_cast, 80.0, 0.0) - torch.where(((mp == 6) | (mp == 12)) & (~is_cast), 25.0, 0.0)

    scores = cap_scores - safety_pen + c['pb'] + pawn_push + dev_bonus + castle_bonus + legal_mask

    if enable_anti_bait and env is not None and b_idx is not None:
        curr_hl = env.history_len[b_idx].clamp(min=1).long()
        h_hashes = env.history_hashes[b_idx]
        latest_hash = h_hashes.gather(1, (curr_hl - 1).unsqueeze(1).clamp(0, 127)).squeeze(1)

        idx_range = torch.arange(128, device=device).unsqueeze(0)
        valid_history = idx_range < curr_hl.unsqueeze(1)
        rep_count = ((h_hashes == latest_hash.unsqueeze(1)) & valid_history).sum(dim=-1)

        pen_mask = (rep_count >= 2).unsqueeze(1)
        is_quiet = (~is_enemy) & (~is_p)
        scores = scores - torch.where(pen_mask & is_quiet, 2000.0, 0.0)

    return torch.argmax(scores + torch.randn(B, 4096, device=device) * 0.5, dim=-1)

# Track bot's previous move for inertia damping
_BOT_PREV_FROM = None
_BOT_PREV_TO = None

def select_cuda_engine_move_v2(boards, legal_mask, is_engine_white, device, env=None, b_idx=None):
    global _BOT_PREV_FROM, _BOT_PREV_TO
    B = boards.shape[0]
    if device not in _ENG_CONST:
        to_s, fr_s = torch.arange(4096, device=device) % 64, torch.arange(4096, device=device) // 64
        dr, dc = to_s // 8, to_s % 8
        pb = (7.0 - ((dr.float() - 3.5).abs() + (dc.float() - 3.5).abs())) * 2.5 + torch.where(((dr == 3) | (dr == 4)) & ((dc == 3) | (dc == 4)), 15.0, 0.0)
        _ENG_CONST[device] = {'to_s': to_s, 'fr_s': fr_s, 'dr': dr, 'dc': dc, 'fr_r': fr_s // 8, 'pb': pb}
    c = _ENG_CONST[device]

    if _BOT_PREV_FROM is None or _BOT_PREV_FROM.shape[0] != B:
        _BOT_PREV_FROM = torch.full((B,), -1, dtype=torch.long, device=device)
        _BOT_PREV_TO = torch.full((B,), -1, dtype=torch.long, device=device)

    flat = boards.view(B, 64)
    tp = flat.gather(1, c['to_s'].unsqueeze(0).expand(B, -1)).long().clamp(0, 12)
    mp = flat.gather(1, c['fr_s'].unsqueeze(0).expand(B, -1)).long().clamp(0, 12)
    p_val = torch.tensor([0, 100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000], dtype=torch.float32, device=device)
    is_w = is_engine_white.unsqueeze(1)
    is_enemy = torch.where(is_w, (tp >= 7) & (tp <= 12), (tp >= 1) & (tp <= 6))
    
    # 1. Tactical Captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
    cap_scores = torch.where(is_enemy, p_val[tp] * 2.0 - p_val[mp] * 0.15 + 120.0, 0.0)

    # 2. Pawn Threat Detection
    dr_f, dc_f = c['dr'].unsqueeze(0).expand(B, -1), c['dc'].unsqueeze(0).expand(B, -1)
    att_r = torch.where(is_w, (dr_f + 1).clamp(0, 7), (dr_f - 1).clamp(0, 7))
    e_pawn = torch.where(is_w, 7, 1)
    sub_b = torch.arange(B, device=device).unsqueeze(1)
    p_threat = (((dc_f > 0) & (boards[sub_b, att_r, (dc_f - 1).clamp(0, 7)] == e_pawn)) | 
                ((dc_f < 7) & (boards[sub_b, att_r, (dc_f + 1).clamp(0, 7)] == e_pawn))) & torch.where(is_w, dr_f < 7, dr_f > 0)
    safety_pen = torch.where(p_threat, torch.where(torch.where(is_w, (mp >= 1) & (mp <= 6), (mp >= 7) & (mp <= 12)), p_val[mp], 0.0) * 0.9, 0.0)

    # 3. Positional Development & Pawn Pushes
    is_p = (mp == 1) | (mp == 7)
    p_dir = torch.where(is_w, c['dr'].float() - c['fr_r'].float(), c['fr_r'].float() - c['dr'].float())
    pawn_push = torch.where(is_p & (p_dir > 0), p_dir * 14.0 + torch.where(is_w, c['dr'].float(), 7.0 - c['dr'].float()) * 6.0, 0.0) + \
                torch.where(is_p & torch.where(is_w, c['dr'] == 7, c['dr'] == 0), 900.0, 0.0)

    is_m = (mp == 2) | (mp == 3) | (mp == 8) | (mp == 9)
    dev_bonus = torch.where(is_m & torch.where(is_w, (c['fr_r'] == 0) & (c['dr'] > 0), (c['fr_r'] == 7) & (c['dr'] < 7)), 35.0, 0.0)
    is_cast = (is_w & (mp == 6) & (c['fr_s'] == 4) & ((c['to_s'] == 6) | (c['to_s'] == 2))) | \
              ((~is_w) & (mp == 12) & (c['fr_s'] == 60) & ((c['to_s'] == 62) | (c['to_s'] == 58)))
    castle_bonus = torch.where(is_cast, 90.0, 0.0) - torch.where(((mp == 6) | (mp == 12)) & (~is_cast), 30.0, 0.0)

    # 4. FIX 1: ANTI-OSCILLATION INERTIA (Damp immediate undos)
    is_undo = (c['fr_s'].unsqueeze(0) == _BOT_PREV_TO.unsqueeze(1)) & \
              (c['to_s'].unsqueeze(0) == _BOT_PREV_FROM.unsqueeze(1)) & \
              (~is_enemy)  # Captures are always exempt
    inertia_pen = torch.where(is_undo, 65.0, 0.0)

    # 5. FIX 2: 50-MOVE CLOCK PROGRESS DRIVE
    if env is not None and b_idx is not None:
        clock = env.halfmove_clock[b_idx].float().unsqueeze(1)
        # Quiet moves lose 1.5 cp per halfmove past ply 10, nudging pawn breaks/trades
        stagnation_pen = torch.where((~is_p) & (~is_enemy), torch.clamp(clock - 10.0, min=0.0) * 1.5, 0.0)
        
        # FIX 3: MATERIAL CONTEMPT (Avoid draws when winning)
        wm, bm = env.get_material()
        bot_lead = torch.where(is_engine_white, wm - bm, bm - wm).unsqueeze(1)
        # If ahead by > 200 cp, penalize non-progressive shuffling
        contempt_pen = torch.where((bot_lead > 200.0) & (~is_p) & (~is_enemy), 25.0, 0.0)
    else:
        stagnation_pen = 0.0
        contempt_pen = 0.0

    scores = (cap_scores - safety_pen + c['pb'] + pawn_push + dev_bonus + castle_bonus
              - inertia_pen - stagnation_pen - contempt_pen + legal_mask)

    chosen_actions = torch.argmax(scores + torch.randn(B, 4096, device=device) * 0.4, dim=-1)

    # Update previous moves for inertia tracking
    _BOT_PREV_FROM.copy_(chosen_actions // 64)
    _BOT_PREV_TO.copy_(chosen_actions % 64)

    return chosen_actions

# ==========================================
# 2. FACTORIAL ABLATION NETWORK
# ==========================================
class FactorialAblationNet(nn.Module):
    def __init__(self, num_islands, sub_batch_size, group_configs, seeds, num_nodes=32, dim=32, input_dim=64, device="cuda"):
        super().__init__()
        self.I, self.sub_B, self.B = num_islands, sub_batch_size, num_islands * sub_batch_size
        self.N, self.D, self.device = num_nodes, dim, device
        self.num_input_nodes, self.motor_nodes_count = 2, 4
        self.lp_alpha = 0.45
        self.group_configs = group_configs
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

        self._init_network()

    def _init_network(self):
        geom_aff = compute_chessboard_euclidean_affinity(sigma=2.5, device=self.device)
        geom_motor = compute_geometric_motor_basis(device=self.device)
        dft = torch.fft.fft(torch.eye(self.D, device=self.device)).real
        dft_norm = F.normalize(dft, p=2, dim=-1) * 0.8

        for i in range(self.I):
            torch.manual_seed(self.seeds[i])
            torch.cuda.manual_seed_all(self.seeds[i])

            raw_sens = torch.empty(64, self.num_input_nodes * self.D, device=self.device)
            nn.init.orthogonal_(raw_sens, gain=1.0)
            self.sensory_proj_weight.data[i] = F.normalize(raw_sens, p=2, dim=0)

            self.spatial_affinity[i] = geom_aff.clone()

            eye_part = torch.eye(self.D, device=self.device).unsqueeze(0).expand(self.N, -1, -1) * 0.8
            micro_noise = torch.randn(self.N, self.D, self.D, device=self.device) * 0.01
            self.w1_weight.data[i] = torch.cat([eye_part, micro_noise], dim=-1)

            self.w_rx.data[i] = dft_norm.unsqueeze(0).expand(self.N, -1, -1)
            self.w_tx.data[i] = dft_norm.unsqueeze(0).expand(self.N, -1, -1)

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
        return ((lf.unsqueeze(-1) + lt.unsqueeze(-2) + cohere * self.spatial_affinity.unsqueeze(1)) * self.action_logit_scale).view(self.B, 4096)

    def forward(self, env_in, threat, state, out, err_ema, lp_ema, thinking_steps=32):
        e_in = env_in.view(self.I, self.sub_B, 64)
        th = threat.view(self.I, 1, self.sub_B, 1)

        raw_s = torch.matmul(e_in, self.sensory_proj_weight)
        s_lat = F.softsign(raw_s.view(self.I, self.sub_B, self.num_input_nodes, self.D).permute(0, 2, 1, 3))
        bus = s_lat.mean(dim=1)

        c_s, c_o, c_e, c_p = state.clone(), out.clone(), err_ema.clone(), lp_ema.clone()
        latched_s, prev_s = c_s.clone(), c_s.clone()
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
        DUST_SIGMA, PB = 0.04, self.sub_B

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

        xi_mf, xi_mt = torch.randn(self.I, 64, 64, device=self.device), torch.randn(self.I, 64, 64, device=self.device)
        motor = latched_s[:, -self.motor_nodes_count:, :, :].permute(0, 2, 1, 3).reshape(self.I, PB, self.motor_nodes_count * self.D)
        nf, nt = F.normalize(motor[..., :64], p=2, dim=-1), F.normalize(motor[..., 64:], p=2, dim=-1)

        lf_p = torch.matmul(nf, self.motor_enc_from + DUST_SIGMA * xi_mf)
        lt_p = torch.matmul(nt, self.motor_enc_to + DUST_SIGMA * xi_mt)
        lf_n = torch.matmul(nf, self.motor_enc_from - DUST_SIGMA * xi_mf)
        lt_n = torch.matmul(nt, self.motor_enc_to - DUST_SIGMA * xi_mt)

        cohere = torch.sum(nf * nt, dim=-1, keepdim=True).unsqueeze(-1)
        logits_p = ((lf_p.unsqueeze(-1) + lt_p.unsqueeze(-2) + cohere * self.spatial_affinity.unsqueeze(1)) * self.action_logit_scale).view(self.B, 4096)
        logits_n = ((lf_n.unsqueeze(-1) + lt_n.unsqueeze(-2) + cohere * self.spatial_affinity.unsqueeze(1)) * self.action_logit_scale).view(self.B, 4096)

        p_log = F.log_softmax(logits_p + l_mask, dim=-1)
        n_log = F.log_softmax(logits_n + l_mask, dim=-1)
        obj_p = -(p_log.exp() * p_log).sum(dim=-1).view(self.I, PB).mean(dim=1, keepdim=True).unsqueeze(-1)
        obj_n = -(n_log.exp() * n_log).sum(dim=-1).view(self.I, PB).mean(dim=1, keepdim=True).unsqueeze(-1)

        grad_mf = (((obj_p - obj_n) / (2.0 * DUST_SIGMA)) * xi_mf)
        grad_mt = (((obj_p - obj_n) / (2.0 * DUST_SIGMA)) * xi_mt)

        tan_mf = project_to_sphere_tangent(batched_newtonschulz5(grad_mf), self.motor_enc_from, dim=0)
        tan_mt = project_to_sphere_tangent(batched_newtonschulz5(grad_mt), self.motor_enc_to, dim=0)

        self.motor_enc_from.data.sub_(self.motor_dust_lr * tan_mf)
        self.motor_enc_to.data.sub_(self.motor_dust_lr * tan_mt)
        self.normalize_motor_heads()

    def forward_island(self, isl, env_in, threat, state, out, err_ema, lp_ema, thinking_steps=32):
        if state.dim() == 4: state, out, err_ema, lp_ema = state.squeeze(0), out.squeeze(0), err_ema.squeeze(0), lp_ema.squeeze(0)
        PB = env_in.shape[0]

        raw_s = torch.matmul(env_in, self.sensory_proj_weight[isl])
        s_lat = F.softsign(raw_s.view(PB, self.num_input_nodes, self.D).permute(1, 0, 2))
        bus = s_lat.mean(dim=0)

        c_s, c_o, c_e, c_p = state.clone(), out.clone(), err_ema.clone(), lp_ema.clone()
        latched_s, prev_s = c_s.clone(), c_s.clone()
        w_rx, w_tx, w1_w = self.w_rx[isl], self.w_tx[isl], self.w1_weight[isl]
        cooldown, settled_counter = torch.zeros(PB, dtype=torch.int16, device=self.device), torch.zeros(PB, dtype=torch.int16, device=self.device)
        th = threat.view(1, PB, 1)

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

        return latched_s, c_o, c_e, c_p

    def get_action_logits_island(self, isl, state):
        if state.dim() == 4: state = state.squeeze(0)
        PB = state.shape[1]
        motor = state[-self.motor_nodes_count:, :, :].permute(1, 0, 2).reshape(PB, self.motor_nodes_count * self.D)
        nf, nt = F.normalize(motor[:, :64], p=2, dim=-1), F.normalize(motor[:, 64:], p=2, dim=-1)

        m_from, m_to = self.motor_enc_from[isl], self.motor_enc_to[isl]
        lf, lt = torch.matmul(nf, m_from), torch.matmul(nt, m_to)
        cohere = torch.sum(nf * nt, dim=-1, keepdim=True).unsqueeze(-1)
        return ((lf.unsqueeze(2) + lt.unsqueeze(1) + cohere * self.spatial_affinity[isl].unsqueeze(0)) * self.action_logit_scale[isl].squeeze()).view(PB, 4096)

# ==========================================
# 3. SELECTIVE CULLING & ALLOSTASIS
# ==========================================
def resolve_island_agent_states(env, states, accum_losses, accum_variances, loss_steps, model):
    fin_mask = env.dones
    if not fin_mask.any(): return

    self_play = (env.match_modes == 0) | (env.match_modes == 2)
    sp_draw = fin_mask & self_play & (env.winners == 3)

    for isl in range(model.I):
        isl_start = isl * model.sub_B
        isl_end = isl_start + model.sub_B
        isl_sp_draws = sp_draw[isl_start:isl_end]
        draw_ratio = isl_sp_draws.float().mean().item()
        cfg = model.group_configs[isl]

        if cfg["fix_draw_culling_gate"] and draw_ratio >= 0.85:
            draw_idx = torch.nonzero(isl_sp_draws).squeeze(-1) + isl_start
            num_rescued = max(1, int(draw_idx.numel() * 0.10))
            perm = torch.randperm(draw_idx.numel(), device=env.device)[:num_rescued]
            sp_draw[draw_idx[perm]] = False

    mse = accum_losses / torch.clamp(loss_steps, min=1.0)
    var = accum_variances / torch.clamp(loss_steps, min=2.0)

    use_r2 = torch.tensor([cfg["fix_r2_history_loss"] for cfg in model.group_configs], device=env.device).repeat_interleave(model.sub_B)
    fitness_metric = torch.where(use_r2, mse / (var + 1e-4), mse)

    if sp_draw.any():
        dead_idx = torch.nonzero(sp_draw).squeeze(-1)
        alive_idx = torch.nonzero(~sp_draw).squeeze(-1)
        if alive_idx.numel() > 0:
            top_k = max(1, alive_idx.numel() // 4)
            _, best_sub_idx = torch.topk(fitness_metric[alive_idx], k=top_k, largest=False)
            donors = alive_idx[best_sub_idx][torch.randint(0, top_k, (dead_idx.numel(),), device=env.device)]

            for s_idx in [0, 1, 4, 5]:
                states[s_idx].view(model.I * model.N, model.sub_B, model.D)[:, dead_idx % model.sub_B, :] = \
                    states[s_idx].view(model.I * model.N, model.sub_B, model.D)[:, donors % model.sub_B, :]
            for e_idx in [2, 3, 6, 7]:
                states[e_idx].view(model.I * model.N, model.sub_B)[:, dead_idx % model.sub_B] = \
                    states[e_idx].view(model.I * model.N, model.sub_B)[:, donors % model.sub_B]

def step_unified_swarm(model, env, states, accum_losses, accum_variances, loss_steps, thinking_steps=32):
    s_w, o_w, e_w, l_w, s_b, o_b, e_b, l_b, noise = states
    active = ~env.dones
    l_mask = env.compute_legal_masks()
    actions = torch.zeros(env.B, dtype=torch.long, device=model.device)

    w_turn = active & (env.players == 0)
    b_turn = active & (env.players == 1)
    is_agent = (w_turn & (env.match_modes != 1)) | (b_turn & (env.match_modes != 3))
    is_bot = active & (~is_agent)

    is_white_4d = (env.players == 0).view(model.I, 1, model.sub_B, 1)
    is_agent_4d = is_agent.view(model.I, 1, model.sub_B, 1)
    is_white_3d = (env.players == 0).view(model.I, 1, model.sub_B)
    is_agent_3d = is_agent.view(model.I, 1, model.sub_B)

    e_in, base_th = env.get_obs(noise)

    curr_mean = e_in.mean(dim=-1, keepdim=True)
    step_var = ((e_in - curr_mean) ** 2).mean(dim=-1)
    accum_variances[active] += step_var[active]

    if is_agent.any():
        curr_s = torch.where(is_white_4d, s_w, s_b)
        curr_o = torch.where(is_white_4d, o_w, o_b)
        curr_e = torch.where(is_white_3d, e_w, e_b)
        curr_l = torch.where(is_white_3d, l_w, l_b)

        loss, ns, no, ne, nl, s_lat = model(e_in, base_th, curr_s, curr_o, curr_e, curr_l, thinking_steps=thinking_steps)
        acts = torch.argmax(model.get_action_logits(ns) + l_mask, dim=-1)
        actions = torch.where(is_agent, acts, actions)

        accum_losses[is_agent] += loss.item()
        loss_steps[is_agent] += 1

        model.step_muon_dust(e_in, base_th, ns, s_lat, no, ne, nl, l_mask)

        w_upd = is_white_4d & is_agent_4d; b_upd = (~is_white_4d) & is_agent_4d
        w_upd_3d = is_white_3d & is_agent_3d; b_upd_3d = (~is_white_3d) & is_agent_3d

        states[0] = torch.where(w_upd, ns, s_w); states[1] = torch.where(w_upd, no, o_w)
        states[2] = torch.where(w_upd_3d, ne, e_w); states[3] = torch.where(w_upd_3d, nl, l_w)
        states[4] = torch.where(b_upd, ns, s_b); states[5] = torch.where(b_upd, no, o_b)
        states[6] = torch.where(b_upd_3d, ne, e_b); states[7] = torch.where(b_upd_3d, nl, l_b)

    if is_bot.any():
        bot_idx = torch.where(is_bot)[0]
        for isl in range(model.I):
            isl_bot = bot_idx[(bot_idx >= isl * model.sub_B) & (bot_idx < (isl + 1) * model.sub_B)]
            if isl_bot.numel() > 0:
                anti_bait = model.group_configs[isl]["fix_anti_bait_bot"]
                bot_acts = select_cuda_engine_move_repetition_aware(
                    env.boards[isl_bot], l_mask[isl_bot],
                    (env.players[isl_bot] == 0), model.device, env=env, b_idx=isl_bot,
                    enable_anti_bait=anti_bait
                )
                actions[isl_bot] = bot_acts

    env.dones, env.winners = env.step(actions, env.dones, env.winners, legal_mask=l_mask)

# ==========================================
# 4. FULL FACTORIAL SUITE RUNNER
# ==========================================
def run_full_factorial_suite():
    parser = argparse.ArgumentParser(description="Full Factorial 2^3 Diagnostic Suite")
    parser.add_argument("--sub_b", type=int, default=128, help="Batch per island (Total = 32 x sub_b = 4096)")
    parser.add_argument("--batches", type=int, default=20, help="Training batches")
    parser.add_argument("--sf_games", type=int, default=200, help="SF games per island")
    parser.add_argument("--blunder_rate", type=float, default=0.25, help="Opponent blunder probability")
    parser.add_argument("--thinking_steps", type=int, default=32, help="Deliberation steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = importlib.import_module("models.nodechess21")

    # Complete 2^3 Factorial Arms
    arms = [
        #{"name": "0. [000] Baseline (Control)",    "fix_anti_bait_bot": False, "fix_draw_culling_gate": False, "fix_r2_history_loss": False},
        {"name": "1. [100] +anti_bait_bot",        "fix_anti_bait_bot": True,  "fix_draw_culling_gate": False, "fix_r2_history_loss": False},
        #{"name": "2. [010] +draw_culling_gate",    "fix_anti_bait_bot": False, "fix_draw_culling_gate": True,  "fix_r2_history_loss": False},
        #{"name": "3. [001] +r2_history_loss",      "fix_anti_bait_bot": False, "fix_draw_culling_gate": False, "fix_r2_history_loss": True},
        {"name": "4. [110] +anti_bait +gate (-r2)","fix_anti_bait_bot": True,  "fix_draw_culling_gate": True,  "fix_r2_history_loss": False},
        {"name": "5. [101] +anti_bait +r2 (-gate)","fix_anti_bait_bot": True,  "fix_draw_culling_gate": False, "fix_r2_history_loss": True},
        #{"name": "6. [011] +gate +r2 (-anti_bait)","fix_anti_bait_bot": False, "fix_draw_culling_gate": True,  "fix_r2_history_loss": True},
        {"name": "7. [111] +full_stack (All 3)",   "fix_anti_bait_bot": True,  "fix_draw_culling_gate": True,  "fix_r2_history_loss": True},
    ]

    seeds_per_arm = 4
    num_islands = len(arms) * seeds_per_arm  # 8 * 4 = 32 islands
    sub_B = args.sub_b
    total_batch_size = num_islands * sub_B   # 32 * 128 = 4,096

    island_configs = [arms[i // seeds_per_arm] for i in range(num_islands)]
    seeds = [1000 + i * 53 for i in range(num_islands)]

    print("=" * 120)
    print(f" FULL FACTORIAL 2^3 ABLATION ({len(arms)} ARMS x {seeds_per_arm} SEEDS = {num_islands} TOTAL ISLANDS)")
    print(f" Total Parallel Games: {total_batch_size:,} | Thinking Steps: {args.thinking_steps} | Device: {device}")
    for a_idx, arm in enumerate(arms):
        print(f"   [{a_idx}] {arm['name']:<30} -> AntiBait={arm['fix_anti_bait_bot']!s:<5} | Gate10%={arm['fix_draw_culling_gate']!s:<5} | R2Loss={arm['fix_r2_history_loss']!s:<5}")
    print("=" * 120)

    model = FactorialAblationNet(
        num_islands=num_islands, sub_batch_size=sub_B,
        group_configs=island_configs, seeds=seeds, device=device
    )

    env = nc.GuardrailedTorchChess(total_batch_size, device, halfmove_limit=100)
    env.dones = torch.zeros(total_batch_size, dtype=torch.bool, device=device)
    env.winners = torch.zeros(total_batch_size, dtype=torch.int8, device=device)
    env.plies = torch.zeros(total_batch_size, dtype=torch.long, device=device)
    env.match_modes = (torch.arange(total_batch_size, device=device) % 4).to(torch.int8)

    accum_losses = torch.zeros(total_batch_size, device=device)
    accum_variances = torch.zeros(total_batch_size, device=device)
    loss_steps = torch.zeros(total_batch_size, device=device)

    s, o, err, lp = model.init_states(total_batch_size)
    noise = torch.full((total_batch_size,), 0.015, device=device)
    states = [s.clone(), o.clone(), err.clone(), lp.clone(), s.clone(), o.clone(), err.clone(), lp.clone(), noise]

    print("\n---> Training across all 32 Factorial Islands...")
    for b_idx in range(1, args.batches + 1):
        batch_completed = 0
        batch_start = time.time()

        while batch_completed < total_batch_size:
            step_unified_swarm(model, env, states, accum_losses, accum_variances, loss_steps, thinking_steps=args.thinking_steps)
            env.plies[~env.dones] += 1

            if env.dones.any():
                fin_i = torch.where(env.dones)[0]
                batch_completed += len(fin_i)

                resolve_island_agent_states(env, states, accum_losses, accum_variances, loss_steps, model)

                env.match_modes[env.dones] = (env.match_modes[env.dones] + 1) % 4
                env.reset(mask=env.dones)
                env.plies[env.dones] = 0
                accum_losses[env.dones] = 0.0
                accum_variances[env.dones] = 0.0
                loss_steps[env.dones] = 0.0
                env.dones[env.dones] = False
                env.winners[env.dones] = 0

        dt = time.time() - batch_start
        gps = total_batch_size / max(1e-3, dt)
        print(f"  [Batch {b_idx:02d}/{args.batches:02d}] Throughput: {gps:5.1f} games/sec | Completed: {b_idx * total_batch_size:,}")

    # ==========================================
    # 5. BLUNDER-CONVERSION QUALITY GAUNTLET
    # ==========================================
    sf_exec = shutil.which("stockfish")
    if not sf_exec:
        print("[Error] Stockfish not found.")
        return

    print(f"\n---> Running Blunder-Conversion Quality Evaluation ({args.sf_games} games/seed, {args.blunder_rate*100:.0f}% Blunder)...")

    isl_wins = [0] * num_islands
    isl_draws = [0] * num_islands
    isl_losses = [0] * num_islands
    isl_mat_at_draw = []
    isl_mat_at_term = []
    isl_ever_lead_300 = [0] * num_islands
    isl_converted_300 = [0] * num_islands
    isl_threw_300 = [0] * num_islands
    isl_oscillation_moves = [0] * num_islands
    isl_total_moves = [0] * num_islands

    for isl in range(num_islands):
        arm_idx = isl // seeds_per_arm
        arm_name = arms[arm_idx]["name"]
        seed = seeds[isl]

        env_sf = nc.GuardrailedTorchChess(args.sf_games, device=device, halfmove_limit=100)
        sf = BlunderingStockfishEngine(sf_exec, skill_level=0)
        sf_is_black = torch.arange(args.sf_games, device=device) < (args.sf_games // 2)

        s_eval = states[0][isl, :, 0:1, :].expand(-1, args.sf_games, -1).clone()
        o_eval = states[1][isl, :, 0:1, :].expand(-1, args.sf_games, -1).clone()
        e_eval = states[2][isl, :, 0:1].expand(-1, args.sf_games).clone()
        l_eval = states[3][isl, :, 0:1].expand(-1, args.sf_games).clone()

        dones = torch.zeros(args.sf_games, dtype=torch.bool, device=device)
        winners = torch.zeros(args.sf_games, dtype=torch.int8, device=device)
        plies = torch.zeros(args.sf_games, dtype=torch.long, device=device)

        prev_from = torch.full((args.sf_games,), -1, dtype=torch.long, device=device)
        prev_to = torch.full((args.sf_games,), -1, dtype=torch.long, device=device)
        reached_lead = torch.zeros(args.sf_games, dtype=torch.bool, device=device)

        while not dones.all() and (plies < 200).any():
            act_mask = ~dones
            l_mask = env_sf.compute_legal_masks()
            actions = torch.zeros(args.sf_games, dtype=torch.long, device=device)

            is_white = (env_sf.players == 0)
            is_sf = torch.where(is_white, ~sf_is_black, sf_is_black)
            is_ag = act_mask & (~is_sf)

            wm, bm = env_sf.get_material()
            ag_lead = torch.where(sf_is_black, wm - bm, bm - wm)
            reached_lead = reached_lead | (ag_lead >= 300.0)

            if (act_mask & is_sf).any():
                for gi in torch.nonzero(act_mask & is_sf).squeeze(-1).tolist():
                    legals = torch.nonzero(l_mask[gi] > -1e8).squeeze(-1).tolist()
                    if not legals: dones[gi], winners[gi] = True, 3; continue
                    fen = nc.board_to_fen(
                        env_sf.boards[gi], int(env_sf.players[gi].item()),
                        env_sf.castling_rights[gi].tolist(), int(env_sf.en_passant_sq[gi].item()),
                        int(env_sf.halfmove_clock[gi].item()), int(plies[gi].item() // 2 + 1)
                    )
                    actions[gi] = sf.get_move_with_blunder(fen, legals, blunder_prob=args.blunder_rate)

            if is_ag.any():
                e_in, th = env_sf.get_obs(torch.full((args.sf_games,), 0.001, device=device))
                s_eval, o_eval, e_eval, l_eval = model.forward_island(
                    isl, e_in, th, s_eval, o_eval, e_eval, l_eval, thinking_steps=args.thinking_steps
                )
                ag_acts = torch.argmax(model.get_action_logits_island(isl, s_eval) + l_mask, dim=-1)
                actions = torch.where(is_ag, ag_acts, actions)

                ag_idx = torch.nonzero(is_ag).squeeze(-1)
                fr = actions[ag_idx] // 64
                to = actions[ag_idx] % 64
                is_osc = (fr == prev_to[ag_idx]) & (to == prev_from[ag_idx])
                isl_oscillation_moves[isl] += is_osc.sum().item()
                isl_total_moves[isl] += ag_idx.numel()

                prev_from[ag_idx] = fr
                prev_to[ag_idx] = to

            dones, winners = env_sf.step(actions, dones, winners, legal_mask=l_mask)
            plies[act_mask] += 1
            over = (plies >= 200) & (~dones)
            if over.any(): dones = dones | over; winners = torch.where(over, 3, winners)

        sf.close()

        is_ag_white = sf_is_black
        ag_w = (((winners == 1) & is_ag_white) | ((winners == 2) & (~is_ag_white))).sum().item()
        sf_w = (((winners == 2) & is_ag_white) | ((winners == 1) & (~is_ag_white))).sum().item()
        draws = (winners == 3).sum().item()

        wm, bm = env_sf.get_material()
        final_lead = torch.where(sf_is_black, wm - bm, bm - wm)
        d_mask = (winners == 3)
        isl_mat_at_draw.append(final_lead[d_mask].float().mean().item() if d_mask.any() else 0.0)
        isl_mat_at_term.append(final_lead.float().mean().item())

        ag_won_mask = ((winners == 1) & is_ag_white) | ((winners == 2) & (~is_ag_white))
        isl_ever_lead_300[isl] = reached_lead.sum().item()
        isl_converted_300[isl] = (reached_lead & ag_won_mask).sum().item()
        isl_threw_300[isl] = (reached_lead & (~ag_won_mask)).sum().item()

        isl_wins[isl] = ag_w
        isl_draws[isl] = draws
        isl_losses[isl] = sf_w

        osc_pct = (isl_oscillation_moves[isl] / max(1, isl_total_moves[isl])) * 100
        print(f"  Isl {isl:02d} [{arm_name:<28}] Seed: {seed} | WINS: {ag_w:2d}/{args.sf_games} | Draws: {draws:2d} | "
              f"Mat@Draw: {isl_mat_at_draw[-1]:+5.0f} cp | Mat@End: {isl_mat_at_term[-1]:+5.0f} cp | "
              f"Lead+300: {isl_ever_lead_300[isl]:2d} (Conv: {isl_converted_300[isl]:2d}, Threw: {isl_threw_300[isl]:2d}) | Osc: {osc_pct:4.1f}%")

    # ==========================================
    # 6. MASTER FACTORIAL SCORECARD
    # ==========================================
    print("\n" + "=" * 130)
    print("                      FULL FACTORIAL 2^3 BLUNDER-CONVERSION MASTER SCORECARD")
    print("=" * 130)
    print(f"{'Arm':<3} | {'Arm Name':<30} | {'WINS':<8} | {'DRAWS':<8} | {'Mat@Draw':<10} | {'Mat@End':<10} | {'Conv Rate':<10} | {'Throw Rate':<10} | {'Oscillation'}")
    print("-" * 130)

    for a_idx, arm in enumerate(arms):
        idx_slice = slice(a_idx * seeds_per_arm, (a_idx + 1) * seeds_per_arm)
        tot_w = sum(isl_wins[idx_slice])
        tot_d = sum(isl_draws[idx_slice])
        tot_g = seeds_per_arm * args.sf_games

        mean_mat_draw = statistics.mean(isl_mat_at_draw[idx_slice])
        mean_mat_term = statistics.mean(isl_mat_at_term[idx_slice])

        tot_lead = sum(isl_ever_lead_300[idx_slice])
        tot_conv = sum(isl_converted_300[idx_slice])
        tot_throw = sum(isl_threw_300[idx_slice])

        conv_pct = (tot_conv / max(1, tot_lead)) * 100
        throw_pct = (tot_throw / max(1, tot_lead)) * 100

        tot_osc = sum(isl_oscillation_moves[idx_slice])
        tot_moves = sum(isl_total_moves[idx_slice])
        osc_pct = (tot_osc / max(1, tot_moves)) * 100

        print(f"[{a_idx}] | {arm['name']:<30} | {tot_w:2d}/{tot_g}   | {tot_d:2d}/{tot_g}   | {mean_mat_draw:+6.1f} cp  | {mean_mat_term:+6.1f} cp  | "
              f"{conv_pct:5.1f}%     | {throw_pct:5.1f}%     | {osc_pct:5.1f}%")

    print("=" * 130 + "\n")

    os.makedirs("out/diagnostics", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"out/diagnostics/factorial_ablation_{stamp}.pt"
    torch.save({"model_state_dict": model.state_dict(), "arms": arms, "isl_wins": isl_wins, "isl_draws": isl_draws}, out_file)
    print(f"  [Factorial Diagnostic Saved] -> {out_file}\n")

if __name__ == "__main__":
    torch.cuda.init()
    torch.set_float32_matmul_precision('high')
    run_full_factorial_suite()