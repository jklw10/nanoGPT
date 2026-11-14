# ==========================================
# nodechess21.py
# Decentralized Asynchronous Swarm & Harmonic Geometric Motor Latching
# Full Fidelity Lifelong Active Inference
# ==========================================

import argparse
import glob
import math
import os
import re
import shutil
import subprocess
import time

import torch
import torch._inductor.config as inductor_config
import torch.nn as nn
import torch.nn.functional as F

inductor_config.disable_cpp_codegen = True
PLY_STEP_COUNTER = 0

# ==========================================
# 0. SPECTRAL GABOR FILTER & GEOMETRIC HARMONICS
# ==========================================
_CIRC_CACHE = {}

def make_circulant_matrix(dim: int, center_offset=1.0, sigma_val=0.5, device="cuda", dtype=torch.float32):
    k_sz = min(31, dim if dim % 2 == 1 else dim - 1)
    if k_sz < 3: return torch.eye(dim, device=device, dtype=dtype)
    half_k = k_sz // 2
    t = torch.arange(-half_k, half_k + 1, device=device, dtype=dtype)
    kernel = (torch.exp(-0.5 * (t * max(sigma_val * math.pi, 1e-4)) ** 2) * torch.cos(center_offset * math.pi * t))
    kernel = (kernel / (torch.norm(kernel, p=2) + 1e-8)).view(1, 1, k_sz)
    eye = torch.eye(dim, device=device, dtype=dtype).unsqueeze(1)
    return F.conv1d(F.pad(eye, (half_k, half_k), mode='circular'), kernel).squeeze(1)

def apply_spectral_filter(noise: torch.Tensor, center_offset=1.0, sigma_val=0.5) -> torch.Tensor:
    dim = noise.shape[-1]
    if dim < 3: return noise
    key = (dim, round(float(center_offset), 4), round(float(sigma_val), 4), str(noise.device), noise.dtype)
    if key not in _CIRC_CACHE:
        _CIRC_CACHE[key] = make_circulant_matrix(dim, center_offset, sigma_val, noise.device, noise.dtype)
    return torch.matmul(noise, _CIRC_CACHE[key])

def buntanh(x, slope, bound):
    return torch.tanh(x - torch.tanh(x) * slope) * bound

def compute_chessboard_euclidean_affinity(sigma=2.5, device="cuda"):
    r = torch.arange(8, device=device).unsqueeze(1).expand(8, 8).reshape(64)
    c = torch.arange(8, device=device).unsqueeze(0).expand(8, 8).reshape(64)
    coords = torch.stack([r, c], dim=-1).float()
    dist = torch.cdist(coords, coords, p=2)
    affinity = torch.exp(-dist / sigma)
    affinity.fill_diagonal_(0.0)
    return affinity

def compute_geometric_motor_basis(dim=64, device="cuda"):
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
    proj = torch.randn(harmonics.shape[1], dim, generator=g, device=device)
    return F.normalize(torch.matmul(harmonics, proj), p=2, dim=0)

def batched_newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    norm = torch.linalg.matrix_norm(X, keepdim=True) + eps
    X = X / norm

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.transpose(-2, -1)
        transposed = True

    for _ in range(steps):
        A = torch.matmul(X, X.transpose(-2, -1))
        B = b * A + c * torch.matmul(A, A)
        X = a * X + torch.matmul(B, X)

    if transposed:
        X = X.transpose(-2, -1)
    return X.type_as(G)

def project_to_sphere_tangent(G, W, dim=-2):
    dot_prod = (G * W).sum(dim=dim, keepdim=True)
    w_norm_sq = (W * W).sum(dim=dim, keepdim=True) + 1e-8
    return G - (dot_prod / w_norm_sq) * W

# ==========================================
# 1. CUDA FIDE CHESS SIMULATOR
# ==========================================
class GuardrailedTorchChess:
    def __init__(self, batch_size, device="cuda", halfmove_limit=100):
        self.B, self.device, self.halfmove_limit, self.max_history_len = batch_size, device, halfmove_limit, 128
        self.boards = torch.zeros((batch_size, 8, 8), dtype=torch.int8, device=device)
        self.players = torch.zeros(batch_size, dtype=torch.int8, device=device)
        self.piece_vals = torch.tensor([0, 100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000], dtype=torch.float32, device=device)
        self.castling_rights = torch.ones((batch_size, 4), dtype=torch.bool, device=device)
        self.en_passant_sq = torch.full((batch_size,), -1, dtype=torch.int8, device=device)
        self.halfmove_clock = torch.zeros(batch_size, dtype=torch.int16, device=device)
        self.history_len = torch.zeros(batch_size, dtype=torch.int16, device=device)
        self.history_hashes = torch.zeros((batch_size, 128), dtype=torch.int64, device=device)
        self.pseudo_mask = torch.full((batch_size, 4096), -1e9, device=device)
        self.legal_mask = torch.full((batch_size, 4096), -1e9, device=device)
        self.sq_range = torch.arange(64, device=device).unsqueeze(0)

        dirs = torch.tensor([[1,0], [-1,0], [0,1], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1]], dtype=torch.long, device=device)
        self._r_dr, self._r_dc = dirs[:, 0].view(1, 8, 1), dirs[:, 1].view(1, 8, 1)
        self._r_st = torch.arange(1, 8, dtype=torch.long, device=device).view(1, 1, 7)
        self._k_d = torch.tensor([[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]], dtype=torch.long, device=device)
        self._kg_d = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]], dtype=torch.long, device=device)

        g = torch.Generator(device=device).manual_seed(42)
        self.zobrist_pieces = torch.randint(-2**62, 2**62, (64, 13), generator=g, dtype=torch.int64, device=device)
        self.zobrist_turn = torch.randint(-2**62, 2**62, (1,), generator=g, dtype=torch.int64, device=device)
        self.reset()

    def compute_hash(self, boards, players):
        flat = boards.view(boards.shape[0], 64).long().clamp(0, 12)
        return self.zobrist_pieces[self.sq_range.expand(boards.shape[0], -1), flat].sum(dim=-1) + self.zobrist_turn * players.to(torch.int64)

    def reset(self, mask=None):
        idx = torch.nonzero(mask if mask is not None else torch.ones(self.B, dtype=torch.bool, device=self.device)).squeeze(-1)
        if idx.numel() == 0: return
        self.boards[idx] = torch.tensor([
            [4,2,3,5,6,3,2,4], [1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [7,7,7,7,7,7,7,7], [10,8,9,11,12,9,8,10]
        ], dtype=torch.int8, device=self.device)
        self.players[idx], self.castling_rights[idx], self.en_passant_sq[idx], self.halfmove_clock[idx], self.history_len[idx] = 0, True, -1, 0, 1
        self.history_hashes[idx, 0] = self.compute_hash(self.boards[idx], self.players[idx])
        self.history_hashes[idx, 1:] = 0

    def get_material(self):
        flat = self.boards.view(self.B, 64).long().clamp(0, 12)
        vals = self.piece_vals[flat]
        return (vals * ((flat >= 1) & (flat <= 5)).float()).sum(dim=-1), (vals * ((flat >= 7) & (flat <= 11)).float()).sum(dim=-1)

    def is_insufficient_material(self):
        flat = self.boards.view(self.B, 64).long()
        p_maj = ((flat == 1) | (flat == 7) | (flat == 4) | (flat == 10) | (flat == 5) | (flat == 11)).sum(dim=-1)
        wm, bm = ((flat == 2) | (flat == 3)).sum(dim=-1), ((flat == 8) | (flat == 9)).sum(dim=-1)
        return (p_maj == 0) & ((wm == 0) & (bm == 0) | (((wm <= 1) & (bm == 0)) | ((bm <= 1) & (wm == 0))))

    def get_obs(self, agent_noise):
        flat = self.boards.view(self.B, 64).float()
        is_w = (self.players == 0).unsqueeze(1)
        p_val = torch.where((flat >= 1) & (flat <= 6), flat, torch.where((flat >= 7) & (flat <= 12), flat - 6, 0.0))
        is_own = torch.where(is_w, (flat >= 1) & (flat <= 6), (flat >= 7) & (flat <= 12))
        is_opp = torch.where(is_w, (flat >= 7) & (flat <= 12), (flat >= 1) & (flat <= 6))
        
        board_signal = torch.where(is_own, p_val / 6.0, torch.where(is_opp, -p_val / 6.0, 0.0))
        n_sig = agent_noise.unsqueeze(1)
        s_nse = apply_spectral_filter(torch.randn_like(board_signal), 1.0, 0.5)
        obs_d = board_signal * torch.clamp(1.0 - n_sig, min=0.05, max=1.0) + s_nse * n_sig
        return obs_d, agent_noise

    def _is_square_attacked(self, target_sq, attacking_player, boards):
        B = boards.shape[0]
        flat = boards.view(B, 64)
        if not isinstance(target_sq, torch.Tensor): target_sq = torch.full((B,), target_sq, dtype=torch.long, device=self.device)
        elif target_sq.dim() == 0: target_sq = target_sq.expand(B)
        if not isinstance(attacking_player, torch.Tensor): attacking_player = torch.full((B,), attacking_player, dtype=torch.int8, device=self.device)
        elif attacking_player.dim() == 0: attacking_player = attacking_player.expand(B)

        rt, ct = target_sq // 8, target_sq % 8
        is_w = (attacking_player == 0).view(B, 1)
        off = torch.where(is_w, 0, 6).to(torch.int8)

        pr = torch.where(is_w, rt.view(-1, 1) - 1, rt.view(-1, 1) + 1)
        pc = ct.view(-1, 1) + torch.tensor([[-1, 1]], device=self.device)
        p_att = ((pr >= 0) & (pr < 8) & (pc >= 0) & (pc < 8) & (flat.gather(1, (pr.clamp(0, 7) * 8 + pc.clamp(0, 7)).long()) == 1 + off)).any(dim=1)

        def step_att(deltas, p_val):
            dr, dc = rt.view(-1, 1) + deltas[:, 0].view(1, 8), ct.view(-1, 1) + deltas[:, 1].view(1, 8)
            return ((dr >= 0) & (dr < 8) & (dc >= 0) & (dc < 8) & (flat.gather(1, (dr.clamp(0, 7) * 8 + dc.clamp(0, 7)).long()) == p_val)).any(dim=1)

        k_att, kg_att = step_att(self._k_d, 2 + off), step_att(self._kg_d, 6 + off)

        sr, sc = rt.view(-1, 1, 1) + self._r_dr * self._r_st, ct.view(-1, 1, 1) + self._r_dc * self._r_st
        p_sl = flat.gather(1, (sr.clamp(0, 7) * 8 + sc.clamp(0, 7)).long().view(B, 56)).view(B, 8, 7)
        occ = (sr >= 0) & (sr < 8) & (sc >= 0) & (sc < 8) & (p_sl > 0)
        fb = (torch.cumsum(occ.to(torch.int32), dim=2) == 1) & occ
        threat = torch.cat([(p_sl[:, :4, :] == 5 + off.unsqueeze(2)) | (p_sl[:, :4, :] == 4 + off.unsqueeze(2)),
                            (p_sl[:, 4:, :] == 5 + off.unsqueeze(2)) | (p_sl[:, 4:, :] == 3 + off.unsqueeze(2))], dim=1)
        return p_att | k_att | kg_att | (fb & threat).any(dim=2).any(dim=1)

    def is_in_check(self, player, boards=None):
        b = self.boards if boards is None else boards
        k_val = torch.where((player if isinstance(player, torch.Tensor) else torch.full((b.shape[0],), player, device=self.device)) == 0, 6, 12).to(torch.int8).unsqueeze(1)
        return self._is_square_attacked((b.view(b.shape[0], 64) == k_val).max(dim=1).indices, 1 - (player if isinstance(player, torch.Tensor) else player), b)

    def compute_pseudo_legal_masks(self):
        self.pseudo_mask.fill_(-1e9)
        sq_f = torch.arange(64, device=self.device)
        rf, cf = sq_f // 8, sq_f % 8
        pieces = self.boards.view(self.B, 64)
        is_w = (self.players == 0).unsqueeze(1)
        b_idx, f_idx = torch.nonzero(torch.where(is_w, (pieces >= 1) & (pieces <= 6), (pieces >= 7) & (pieces <= 12)), as_tuple=True)
        if b_idx.numel() == 0: return self.pseudo_mask

        pt = torch.where(pieces[b_idx, f_idx] <= 6, pieces[b_idx, f_idx], pieces[b_idx, f_idx] - 6)
        is_pw, r_f, c_f = is_w.squeeze(1)[b_idx], rf[f_idx], cf[f_idx]

        r1 = torch.where(is_pw, r_f + 1, r_f - 1)
        d1 = (r1 * 8 + c_f).clamp(0, 63)
        pf1 = (pt == 1) & (r1 >= 0) & (r1 < 8) & (pieces[b_idx, d1] == 0)
        self.pseudo_mask[b_idx[pf1], f_idx[pf1] * 64 + d1[pf1]] = 0.0

        r2 = torch.where(is_pw, r_f + 2, r_f - 2)
        d2 = (r2 * 8 + c_f).clamp(0, 63)
        pf2 = (pt == 1) & torch.where(is_pw, r_f == 1, r_f == 6) & (pieces[b_idx, d1] == 0) & (pieces[b_idx, d2] == 0)
        self.pseudo_mask[b_idx[pf2], f_idx[pf2] * 64 + d2[pf2]] = 0.0

        for dc in [-1, 1]:
            ct = c_f + dc
            dc_sq = (r1 * 8 + ct).clamp(0, 63)
            d_p = pieces[b_idx, dc_sq]
            enemy = torch.where(is_pw, (d_p >= 7) & (d_p <= 12), (d_p >= 1) & (d_p <= 6))
            pc = (pt == 1) & (r1 >= 0) & (r1 < 8) & (ct >= 0) & (ct < 8) & (enemy | ((self.en_passant_sq[b_idx] == dc_sq) & (self.en_passant_sq[b_idx] >= 0)))
            self.pseudo_mask[b_idx[pc], f_idx[pc] * 64 + dc_sq[pc]] = 0.0

        for deltas, piece_target in [(self._k_d, 2), (self._kg_d, 6)]:
            dr, dc = r_f.view(-1, 1) + deltas[:, 0].view(1, 8), c_f.view(-1, 1) + deltas[:, 1].view(1, 8)
            to_sq = (dr.clamp(0, 7) * 8 + dc.clamp(0, 7)).long()
            b_exp, f_exp = b_idx.view(-1, 1).expand_as(to_sq), f_idx.view(-1, 1).expand_as(to_sq)
            dp = pieces[b_exp, to_sq]
            own = torch.where(is_pw.view(-1, 1), (dp >= 1) & (dp <= 6), (dp >= 7) & (dp <= 12))
            val = (pt.view(-1, 1) == piece_target) & (dr >= 0) & (dr < 8) & (dc >= 0) & (dc < 8) & (~own)
            self.pseudo_mask[b_exp[val], f_exp[val] * 64 + to_sq[val]] = 0.0

        wc, bc = self.is_in_check(0), self.is_in_check(1)
        f1, d1, f8, d8 = [self._is_square_attacked(sq, p, self.boards) for sq, p in [(5,1), (3,1), (61,0), (59,0)]]
        for cnd, act in [
            ((self.players == 0) & self.castling_rights[:, 0] & (pieces[:, 5:7] == 0).all(1) & (~wc) & (~f1), 4 * 64 + 6),
            ((self.players == 0) & self.castling_rights[:, 1] & (pieces[:, 1:4] == 0).all(1) & (~wc) & (~d1), 4 * 64 + 2),
            ((self.players == 1) & self.castling_rights[:, 2] & (pieces[:, 61:63] == 0).all(1) & (~bc) & (~f8), 60 * 64 + 62),
            ((self.players == 1) & self.castling_rights[:, 3] & (pieces[:, 57:60] == 0).all(1) & (~bc) & (~d8), 60 * 64 + 58)
        ]: self.pseudo_mask[cnd, act] = 0.0

        sr, sc = r_f.view(-1, 1, 1) + self._r_dr * self._r_st, c_f.view(-1, 1, 1) + self._r_dc * self._r_st
        in_b = (sr >= 0) & (sr < 8) & (sc >= 0) & (sc < 8)
        to_sq = (sr.clamp(0, 7) * 8 + sc.clamp(0, 7)).long()
        b_exp, f_exp = b_idx.view(-1, 1, 1).expand_as(to_sq), f_idx.view(-1, 1, 1).expand_as(to_sq)
        dp = pieces[b_exp, to_sq]
        d_own = torch.where(is_pw.view(-1, 1, 1), (dp >= 1) & (dp <= 6), (dp >= 7) & (dp <= 12))
        d_val = torch.zeros(b_idx.shape[0], 8, 1, dtype=torch.bool, device=self.device)
        d_val[:, :4, 0], d_val[:, 4:, 0] = ((pt == 4) | (pt == 5)).unsqueeze(1), ((pt == 3) | (pt == 5)).unsqueeze(1)
        can_reach = in_b & (torch.cumsum(F.pad((in_b & (dp > 0))[:, :, :-1], (1, 0), value=0).to(torch.int32), dim=2) == 0) & d_val & (~d_own)
        self.pseudo_mask[b_exp[can_reach], f_exp[can_reach] * 64 + to_sq[can_reach]] = 0.0
        return self.pseudo_mask

    def compute_legal_masks(self):
        p_mask = self.compute_pseudo_legal_masks()
        self.legal_mask.fill_(-1e9)
        b_idx, act_idx = torch.nonzero(p_mask > -1e8, as_tuple=True)
        if b_idx.numel() == 0: return self.legal_mask

        from_sqs, to_sqs, turn = act_idx // 64, act_idx % 64, self.players[b_idx]
        N = b_idx.shape[0]
        sub = self.boards[b_idx].view(N, 64)
        moving = sub[torch.arange(N, device=self.device), from_sqs]

        is_ep = ((moving == 1) | (moving == 7)) & (to_sqs == self.en_passant_sq[b_idx]) & (self.en_passant_sq[b_idx] >= 0)
        if is_ep.any():
            ep_i = torch.nonzero(is_ep).squeeze(-1)
            sub[ep_i, torch.where(turn[ep_i] == 0, to_sqs[ep_i] - 8, to_sqs[ep_i] + 8)] = 0

        sub[torch.arange(N, device=self.device), to_sqs] = moving
        sub[torch.arange(N, device=self.device), from_sqs] = 0

        p_k_sqs = (self.boards.view(self.B, 64) == torch.where(self.players == 0, 6, 12).to(torch.int8).unsqueeze(1)).max(dim=1).indices
        k_sqs = torch.where((moving == 6) | (moving == 12), to_sqs, p_k_sqs[b_idx])
        valid = ~self._is_square_attacked(k_sqs, 1 - turn, sub.view(N, 8, 8))
        self.legal_mask[b_idx[valid], act_idx[valid]] = 0.0
        return self.legal_mask

    def step(self, action_indices, dones, winners, legal_mask=None):
        B_idx = torch.arange(self.B, device=self.device)
        from_sq, to_sq = action_indices // 64, action_indices % 64
        fr_r, fr_c, to_r, to_c = from_sq // 8, from_sq % 8, to_sq // 8, to_sq % 8
        piece, dest_piece, active = self.boards[B_idx, fr_r, fr_c], self.boards[B_idx, to_r, to_c], ~dones

        if legal_mask is not None:
            no_legal = active & (~(legal_mask > -1e8).any(dim=-1))
            if no_legal.any():
                w_loss, b_loss = no_legal & (self.players == 0) & self.is_in_check(0), no_legal & (self.players == 1) & self.is_in_check(1)
                mates, stales = w_loss | b_loss, no_legal & ~(w_loss | b_loss)
                winners = torch.where(w_loss, torch.tensor(2, dtype=torch.int8, device=self.device), winners)
                winners = torch.where(b_loss, torch.tensor(1, dtype=torch.int8, device=self.device), winners)
                winners = torch.where(stales, torch.tensor(3, dtype=torch.int8, device=self.device), winners)
                dones = dones | mates | stales
                active = active & (~dones)

        is_pawn = active & ((piece == 1) | (piece == 7))
        is_irrev = is_pawn | (active & (dest_piece != 0))

        is_ep = is_pawn & (to_sq == self.en_passant_sq) & (self.en_passant_sq >= 0)
        self.boards[B_idx[is_ep], torch.where(self.players == 0, to_r - 1, to_r + 1)[is_ep], to_c[is_ep]] = 0

        for cnd, r, f_c, t_c, p_val in [
            (active & (piece == 6) & (from_sq == 4) & (to_sq == 6), 0, 7, 5, 4),
            (active & (piece == 6) & (from_sq == 4) & (to_sq == 2), 0, 0, 3, 4),
            (active & (piece == 12) & (from_sq == 60) & (to_sq == 62), 7, 7, 5, 10),
            (active & (piece == 12) & (from_sq == 60) & (to_sq == 58), 7, 0, 3, 10)
        ]:
            self.boards[B_idx[cnd], r, f_c], self.boards[B_idx[cnd], r, t_c] = 0, p_val

        self.boards[B_idx[active], to_r[active], to_c[active]] = piece[active]
        self.boards[B_idx[active], fr_r[active], fr_c[active]] = 0

        self.boards[B_idx[active & (piece == 1) & (to_r == 7)], to_r[active & (piece == 1) & (to_r == 7)], to_c[active & (piece == 1) & (to_r == 7)]] = 5
        self.boards[B_idx[active & (piece == 7) & (to_r == 0)], to_r[active & (piece == 7) & (to_r == 0)], to_c[active & (piece == 7) & (to_r == 0)]] = 11

        for sq, rights_slice in [(4, slice(0, 2)), (7, 0), (0, 1), (60, slice(2, 4)), (63, 2), (56, 3)]:
            self.castling_rights[active & ((from_sq == sq) | (to_sq == sq)), rights_slice] = False

        self.en_passant_sq[active] = -1
        self.en_passant_sq[is_pawn & (fr_r == 1) & (to_r == 3)] = (from_sq[is_pawn & (fr_r == 1) & (to_r == 3)] + 8).to(torch.int8)
        self.en_passant_sq[is_pawn & (fr_r == 6) & (to_r == 4)] = (from_sq[is_pawn & (fr_r == 6) & (to_r == 4)] - 8).to(torch.int8)

        self.players[active] = 1 - self.players[active]
        self.halfmove_clock[active] += 1
        self.halfmove_clock[is_irrev] = 0
        self.history_len[is_irrev] = 0

        c_hashes = self.compute_hash(self.boards, self.players)
        act_i = torch.nonzero(active).squeeze(-1)
        if act_i.numel() > 0:
            c_lens = self.history_len[act_i].clamp(max=127).long()
            self.history_hashes[act_i, c_lens] = c_hashes[act_i]
            self.history_len[act_i] = (c_lens + 1).to(torch.int16)

        wm, bm = self.get_material()
        w_res = active & ((bm >= wm + 1200) | ((wm == 0) & (bm > 300)))
        b_res = active & ((wm >= bm + 1200) | ((bm == 0) & (wm > 300)))
        winners = torch.where(b_res, torch.tensor(1, dtype=torch.int8, device=self.device), winners)
        winners = torch.where(w_res, torch.tensor(2, dtype=torch.int8, device=self.device), winners)
        active = active & ~(w_res | b_res)

        rep = ((self.history_hashes == c_hashes.unsqueeze(1)) & (torch.arange(128, device=self.device).unsqueeze(0) < self.history_len.unsqueeze(1))).sum(dim=-1)
        lead = torch.where(self.players == 0, wm - bm, bm - wm)
        draws = active & (((rep >= 3) & (lead <= 150)) | (rep >= 5) | (self.halfmove_clock >= self.halfmove_limit) | self.is_insufficient_material())
        winners = torch.where(draws, torch.tensor(3, dtype=torch.int8, device=self.device), winners)
        return dones | w_res | b_res | draws, winners

# ==========================================
# 2. STOCKFISH INTERFACE & UTILS
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

class StockfishEngine:
    def __init__(self, stockfish_path="stockfish", threads=1, hash_mb=16):
        self.proc = subprocess.Popen([stockfish_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for cmd in ["uci", f"setoption name Threads value {threads}", f"setoption name Hash value {hash_mb}", "setoption name UCI_LimitStrength value false", "isready"]:
            self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        while "readyok" not in self.proc.stdout.readline(): pass

    def get_move(self, fen, nodes=1500):
        self.proc.stdin.write(f"position fen {fen}\ngo nodes {nodes}\n"); self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line: return None
            if line.startswith("bestmove"):
                p = line.split()
                return p[1] if len(p) > 1 and p[1] not in ("(none)", "0000") else None

    def close(self):
        try: self.proc.stdin.write("quit\n"); self.proc.stdin.flush(); self.proc.terminate()
        except Exception: pass

# ==========================================
# 3. CONTINUOUS AGENT ALLOSTASIS & POOL SURVIVAL
# ==========================================
def resolve_fittest_agent_states(finished_mask, winners, match_modes, state_pool, states_w, states_b, agent_noise=None, agent_loss=None):
    if not finished_mask.any(): return
    self_play = (match_modes == 0) | (match_modes == 2)
    sp_draw = finished_mask & self_play & (winners == 3)
    w_surv = finished_mask & ((match_modes == 3) | (self_play & (winners == 1)))
    b_surv = finished_mask & ((match_modes == 1) | (self_play & (winners == 2)))

    for pool, w_buf, b_buf in zip(state_pool, states_w, states_b):
        if w_surv.any():
            if pool.dim() == 3: pool[:, w_surv, :] = w_buf[:, w_surv, :]
            else: pool[:, w_surv] = w_buf[:, w_surv]
        if b_surv.any():
            if pool.dim() == 3: pool[:, b_surv, :] = b_buf[:, b_surv, :]
            else: pool[:, b_surv] = b_buf[:, b_surv]

    if sp_draw.any():
        dead_idx = torch.nonzero(sp_draw).squeeze(-1)
        alive_mask = ~sp_draw
        alive_idx = torch.nonzero(alive_mask).squeeze(-1)

        if alive_idx.numel() > 0:
            loss_metric = agent_loss if agent_loss is not None else state_pool[2].mean(dim=0)
            top_k = max(1, alive_idx.numel() // 4)
            _, best_sub_idx = torch.topk(loss_metric[alive_idx], k=top_k, largest=False)
            donor_pool = alive_idx[best_sub_idx]

            picks = torch.randint(0, donor_pool.numel(), (dead_idx.numel(),), device=finished_mask.device)
            donors = donor_pool[picks]

            for pool in state_pool:
                if pool.dim() == 3: pool[:, dead_idx, :] = pool[:, donors, :].clone()
                else: pool[:, dead_idx] = pool[:, donors].clone()

            if agent_noise is not None:
                agent_noise[dead_idx] = agent_noise[donors].clone()

def update_agent_somatic_noise(agent_noise, finished_mask, winners, match_modes, plies=None, env=None, noise_cap=None):
    if not finished_mask.any(): return
    self_play = (match_modes == 0) | (match_modes == 2)
    valid_mask = finished_mask & ~(self_play & (winners == 3))
    if not valid_mask.any(): return

    idx = torch.nonzero(valid_mask).squeeze(-1)
    sub_B = idx.numel()
    cap = noise_cap if noise_cap is not None else 0.01
    floor = cap * 0.10
    span = max(cap - floor, 1e-6)
    step = cap * 0.08

    modes, wins = match_modes[idx], winners[idx]
    p = plies[idx].float() if plies is not None else torch.full((sub_B,), 40.0, device=agent_noise.device)
    is_bot = ~((modes == 0) | (modes == 2))
    agent_won = ((modes == 3) & (wins == 1)) | ((modes == 1) & (wins == 2)) | ((~is_bot) & ((wins == 1) | (wins == 2)))
    agent_lost = ((modes == 3) & (wins == 2)) | ((modes == 1) & (wins == 1))
    is_draw = (wins == 3) & is_bot

    delta = torch.zeros(sub_B, device=agent_noise.device)
    if agent_won.any():
        w_p = p[agent_won]
        delta[agent_won] = -step * (0.80 + 0.60 * torch.exp(-w_p / 30.0))

    if agent_lost.any():
        l_p = p[agent_lost]
        stamina_factor = torch.clamp((l_p - 42.0) / 22.0, -1.0, 1.0)
        delta[agent_lost] = step * (0.30 - 0.70 * stamina_factor)

    if is_draw.any():
        d_p = p[is_draw]
        delta[is_draw] = torch.where(d_p >= 40.0, -step * 0.80, step * 0.20)

    curr = agent_noise[idx]
    headroom_up = torch.clamp((cap - curr) / span, min=0.02, max=1.0)
    headroom_down = torch.clamp((curr - floor) / span, min=0.02, max=1.0)
    scaled_delta = torch.where(delta > 0, delta * headroom_up, delta * headroom_down)
    agent_noise[idx] = torch.clamp(curr + scaled_delta, min=floor, max=cap)

# ==========================================
# 4. BIDIRECTIONAL CUDA TACTICAL ENGINE
# ==========================================
_ENG_CONST = {}
def select_cuda_engine_move(boards, legal_mask, is_engine_white, device, env=None, b_idx=None):
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

    scores = cap_scores - safety_pen + c['pb'] + pawn_push + dev_bonus + castle_bonus + torch.randn(B, 4096, device=device) * 0.5 + legal_mask
    return torch.argmax(scores, dim=-1)

# ==========================================
# 5. ASYNCHRONOUS STIGMERGIC SWARM ARCHITECTURE
# ==========================================
class NodeParams3(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, device):
        super().__init__()
        # Fix 1: Clean QZ Initialization (diagonal + micro-noise)
        eye_part = torch.eye(in_dim, device=device).unsqueeze(0).expand(num_nodes, -1, -1) * 0.8
        micro_noise = torch.randn(num_nodes, in_dim, in_dim, device=device) * 0.01
        self.weight = nn.Parameter(torch.cat([eye_part, micro_noise], dim=-1), requires_grad=False)
        self.register_buffer('untanh_slope', torch.tensor(0.5, device=device).view(1, 1, 1))
        self.register_buffer('w_scale', torch.tensor(0.8, device=device).view(1, 1, 1))
        self.normalize_weight()

    def normalize_weight(self):
        with torch.no_grad():
            self.weight.data[:, :, :self.weight.shape[1]].copy_(
                F.normalize(self.weight.data[:, :, :self.weight.shape[1]], p=2, dim=-2) * self.w_scale
            )

    def forward(self, x, act_jitter=None):
        raw = torch.bmm(x, self.weight)
        if act_jitter is not None: raw = raw + act_jitter
        return buntanh(raw, self.untanh_slope, math.sqrt(2.0 + math.sqrt(2.0)))

class StigmergicFieldBus(nn.Module):
    def __init__(self, num_nodes: int, dim: int = 32, device="cuda"):
        super().__init__()
        self.N, self.D, self.device = num_nodes, dim, device
        # Fix 2: Coherent Bus (Normalized DFT initialization)
        dft = torch.fft.fft(torch.eye(dim, device=device)).real
        dft_norm = F.normalize(dft, p=2, dim=-1) * 0.8
        self.w_rx = nn.Parameter(dft_norm.unsqueeze(0).expand(num_nodes, -1, -1).clone(), requires_grad=False)
        self.w_tx = nn.Parameter(dft_norm.unsqueeze(0).expand(num_nodes, -1, -1).clone(), requires_grad=False)
        self.normalize_weights()

    def normalize_weights(self):
        with torch.no_grad():
            self.w_rx.data.copy_(F.normalize(self.w_rx.data, p=2, dim=-2) * 0.8)
            self.w_tx.data.copy_(F.normalize(self.w_tx.data, p=2, dim=-2) * 0.8)

    def read(self, bus, rx_jitter=None):
        b_exp = bus.unsqueeze(0).expand(self.N, -1, -1)
        raw_rx = torch.bmm(b_exp, self.w_rx)
        if rx_jitter is not None: raw_rx = raw_rx + rx_jitter
        return buntanh(raw_rx, 0.5, math.sqrt(2.0 + math.sqrt(2.0)))

    def emit(self, state, tx_jitter=None):
        raw_tx = torch.bmm(state, self.w_tx)
        if tx_jitter is not None: raw_tx = raw_tx + tx_jitter
        return buntanh(raw_tx, 0.5, 1.0)

class ChessNodeNet(nn.Module):
    def __init__(self, num_nodes=32, dim=32, input_dim=64, rank_k=16, spatial_sigma=2.5, action_logit_scale=3.0, lp_alpha=0.45, device="cuda"):
        super().__init__()
        self.N, self.D, self.device = num_nodes, dim, device
        self.num_input_nodes = 2
        self.motor_nodes_count = 4
        
        self.sensory_proj = nn.Linear(input_dim, self.num_input_nodes * dim, bias=False, device=device)
        nn.init.orthogonal_(self.sensory_proj.weight, gain=1.0)
        self.sensory_proj.weight.requires_grad = False
        
        self.w1 = NodeParams3(num_nodes, dim, dim * 2, device)
        self.router = StigmergicFieldBus(num_nodes, dim, device=device)

        # Fix 3: Decoupled 2D Harmonic Motor Heads
        geom_motor = compute_geometric_motor_basis(dim=64, device=device)
        self.motor_enc_from = nn.Parameter(geom_motor.clone(), requires_grad=False)
        self.motor_enc_to = nn.Parameter(geom_motor.clone(), requires_grad=False)
        
        self.lp_alpha = lp_alpha
        
        self.register_buffer('a_variance', torch.tensor(0.05, device=device).view(1, 1, 1))
        self.register_buffer('action_logit_scale', torch.tensor(action_logit_scale, device=device))
        self.register_buffer('circ_1_0', make_circulant_matrix(dim, 1.0, 1.0, device=device))
        self.register_buffer('circ_0_25', make_circulant_matrix(dim, 1.0, 0.25, device=device))

        # Fix 4: 2D Euclidean Spatial Affinity
        self.register_buffer('spatial_affinity', compute_chessboard_euclidean_affinity(sigma=spatial_sigma, device=device))
        self.normalize_sensory_weight()
        self.normalize_motor_heads()

    def normalize_sensory_weight(self):
        with torch.no_grad():
            self.sensory_proj.weight.copy_(F.normalize(self.sensory_proj.weight, p=2, dim=0))

    def normalize_motor_heads(self):
        with torch.no_grad():
            self.motor_enc_from.data.copy_(F.normalize(self.motor_enc_from.data, p=2, dim=0))
            self.motor_enc_to.data.copy_(F.normalize(self.motor_enc_to.data, p=2, dim=0))

    def init_states(self, batch_size):
        return (1e-4 * apply_spectral_filter(torch.randn(self.N, batch_size, self.D, device=self.device), 1.0, 1.0),
                1e-4 * apply_spectral_filter(torch.randn(self.N, batch_size, self.D, device=self.device), 1.0, 1.0),
                torch.full((self.N, batch_size), 0.05, device=self.device),
                torch.zeros((self.N, batch_size), device=self.device))

    def get_action_logits(self, state):
        PB = state.shape[1]
        motor = state[-self.motor_nodes_count:, :, :].permute(1, 0, 2).reshape(PB, self.motor_nodes_count * self.D)
        nf, nt = F.normalize(motor[:, :64], p=2, dim=-1), F.normalize(motor[:, 64:128], p=2, dim=-1)
        lf, lt = torch.matmul(nf, self.motor_enc_from), torch.matmul(nt, self.motor_enc_to)
        cohere = torch.sum(nf * nt, dim=-1, keepdim=True).unsqueeze(-1)
        return ((lf.unsqueeze(2) + lt.unsqueeze(1) + cohere * self.spatial_affinity.unsqueeze(0)) * self.action_logit_scale).view(PB, 4096)

    def forward(self, env_in, threat, state, out, err_ema, lp_ema, thinking_steps=1, 
                act_jitter=None, rx_jitter=None, tx_jitter=None, sensory_jitter=None, 
                precomputed_latents=None, gamma_coupling=2.0, use_entropic_restoring=True,
                settled_vel_thresh=0.018, dirac_pulse_sigma=0.35, v_latch=0.025):
        
        PB = env_in.shape[0]
        if precomputed_latents is not None:
            lat = precomputed_latents
        else:
            raw_s = self.sensory_proj(env_in)
            lat = F.softsign((raw_s + (sensory_jitter if sensory_jitter is not None else 0)).view(PB, self.num_input_nodes, self.D).permute(1, 0, 2))

        bus = lat.mean(dim=0)

        if thinking_steps <= 1:
            l, s, o, e, p, v, _, _ = self._step(
                env_in, threat, state, out, err_ema, lp_ema, bus,
                act_j=act_jitter, rx_j=rx_jitter, tx_j=tx_jitter, sens_j=sensory_jitter,
                precomputed=lat, noise_gate=1.0, tick_mask=None, v_latch=1e9,
                gamma_coupling=gamma_coupling, use_entropic_restoring=use_entropic_restoring
            )
            return l, s, o, e, p, v, s

        acc_l, acc_v, acc_s = 0.0, 0.0, torch.zeros_like(state)
        c_s, c_o, c_e, c_p = state, out, err_ema, lp_ema

        latched_s = c_s.clone()
        latched_em = self.router.emit(c_s)
        prev_s = c_s.clone()

        cooldown = torch.zeros(PB, dtype=torch.int16, device=self.device)
        settled_counter = torch.zeros(PB, dtype=torch.int16, device=self.device)

        for step in range(thinking_steps):
            motor_vel = torch.norm(c_s[-self.motor_nodes_count:] - prev_s[-self.motor_nodes_count:], dim=-1).mean(dim=0)
            is_motor_calm = motor_vel < v_latch
            if is_motor_calm.any():
                calm_idx = torch.nonzero(is_motor_calm).squeeze(-1)
                latched_s[:, calm_idx, :] = c_s[:, calm_idx, :].clone()

            overall_vel = torch.norm(c_s - prev_s, dim=-1).mean(dim=0)
            is_settled = (overall_vel < settled_vel_thresh) & (cooldown == 0)
            settled_counter = torch.where(is_settled, settled_counter + 1, torch.zeros_like(settled_counter))

            trigger_pulse = settled_counter >= 2
            if trigger_pulse.any():
                idx_p = torch.nonzero(trigger_pulse).squeeze(-1)
                pulse = torch.randn_like(c_s[:, idx_p, :]) * dirac_pulse_sigma
                c_s[:, idx_p, :] = F.softsign(c_s[:, idx_p, :] + pulse)
                cooldown[idx_p] = 8
                settled_counter[idx_p] = 0

            is_coasting = (cooldown > 0) | (overall_vel > 0.025)
            noise_gate = torch.where(is_coasting, 0.0, 1.0).view(1, PB, 1)
            cooldown = torch.clamp(cooldown - 1, min=0)
            prev_s = c_s.clone()

            n_hidden = self.N - self.num_input_nodes - self.motor_nodes_count
            rand_m = torch.rand(n_hidden, PB, 1, device=self.device)
            hidden_mask = (rand_m < 0.75).float()
            tick_mask = torch.cat([
                torch.ones(self.num_input_nodes, PB, 1, device=self.device),
                hidden_mask,
                torch.ones(self.motor_nodes_count, PB, 1, device=self.device)
            ], dim=0)

            sl, c_s, c_o, c_e, c_p, sv, bus, latched_em = self._step(
                env_in, threat, c_s, c_o, c_e, c_p, bus,
                latched_emissions=latched_em,
                act_j=act_jitter, rx_j=rx_jitter, tx_j=tx_jitter, sens_j=sensory_jitter,
                precomputed=lat, noise_gate=noise_gate, tick_mask=tick_mask, v_latch=v_latch,
                gamma_coupling=gamma_coupling, use_entropic_restoring=use_entropic_restoring
            )
            acc_l, acc_v, acc_s = acc_l + sl, acc_v + sv, acc_s + c_s

        final_motor_vel = torch.norm(c_s[-self.motor_nodes_count:] - prev_s[-self.motor_nodes_count:], dim=-1).mean(dim=0)
        is_motor_calm = final_motor_vel < v_latch
        if is_motor_calm.any():
            calm_idx = torch.nonzero(is_motor_calm).squeeze(-1)
            latched_s[:, calm_idx, :] = c_s[:, calm_idx, :].clone()

        inv = 1.0 / thinking_steps
        return acc_l * inv, latched_s, c_o, c_e, c_p, acc_v * inv, acc_s * inv
    
    def _step(self, env_in, threat, state, out, err_ema, lp_ema, bus, 
              latched_emissions=None, act_j=None, rx_j=None, tx_j=None, sens_j=None, 
              precomputed=None, noise_gate=1.0, tick_mask=None, v_latch=0.025,
              gamma_coupling=2.0, use_entropic_restoring=True):
        PB = env_in.shape[0]
        h = threat.view(1, PB, 1)
        if precomputed is not None and sens_j is None:
            s_lat = precomputed
        else:
            raw_s = self.sensory_proj(env_in)
            s_lat = F.softsign((raw_s + (sens_j if sens_j is not None else 0)).view(PB, self.num_input_nodes, self.D).permute(1, 0, 2))

        perceived = self.router.read(bus, rx_jitter=rx_j)
        raw_boredom = torch.pow(1.0 - torch.clamp(F.relu(lp_ema) / (err_ema + 1e-6), 0.0, 1.0), gamma_coupling / 2.0).unsqueeze(-1)
        boredom = raw_boredom * noise_gate

        tgt = perceived + torch.sqrt(self.a_variance) * torch.matmul(torch.randn_like(out), self.circ_1_0) * boredom
        if use_entropic_restoring: tgt = tgt - (gamma_coupling * 0.01) * state * (1.0 - boredom)
        tgt = F.softsign(tgt)
        tgt = torch.cat([s_lat, tgt[self.num_input_nodes:]], dim=0)

        mean, err_p = torch.chunk(self.w1(state, act_jitter=act_j), 2, dim=-1)
        loss_w1 = (F.mse_loss(mean, tgt, reduction='none') + 0.1 * F.mse_loss(err_p, torch.abs(tgt - mean), reduction='none')) * (1.0 + 2.0 * h)
        cur_err = loss_w1.mean(dim=-1)
        n_err = err_ema * (1.0 - self.lp_alpha) + cur_err * self.lp_alpha
        n_lp = lp_ema * (1.0 - self.lp_alpha) + torch.log((err_ema + 1e-6) / (cur_err + 1e-6)) * self.lp_alpha
        
        raw_omega = h * torch.pow(1.0 - torch.clamp(F.relu(n_lp) / (n_err + 1e-6), 0.0, 1.0), gamma_coupling / 2.0).unsqueeze(-1)
        omega = raw_omega * noise_gate

        settled = F.softsign(state + 0.25 * (tgt - state) + 0.125 * torch.matmul(torch.randn_like(tgt), self.circ_0_25) * omega * 0.40)
        n_s_cand = torch.cat([s_lat, settled[self.num_input_nodes:]], dim=0)

        if tick_mask is not None:
            n_s = torch.where(tick_mask > 0.5, n_s_cand, state)
            n_err = torch.where(tick_mask.squeeze(-1) > 0.5, n_err, err_ema)
            n_lp = torch.where(tick_mask.squeeze(-1) > 0.5, n_lp, lp_ema)
        else:
            n_s = n_s_cand

        raw_emissions = self.router.emit(n_s, tx_jitter=tx_j)
        node_vel = torch.norm(n_s - state, dim=-1, keepdim=True)
        
        if tick_mask is not None:
            update_latch = (node_vel < v_latch) & (tick_mask > 0.5)
        else:
            update_latch = node_vel < v_latch

        if latched_emissions is None:
            new_latched_emissions = raw_emissions
        else:
            new_latched_emissions = torch.where(update_latch, raw_emissions, latched_emissions)

        node_energy = new_latched_emissions.abs().mean(dim=-1, keepdim=True)
        firing_mask = (node_energy > 0.05).float()

        sens_broadcast = s_lat.mean(dim=0)
        swarm_broadcast = (new_latched_emissions[self.num_input_nodes:] * firing_mask[self.num_input_nodes:]).sum(dim=0) / (firing_mask[self.num_input_nodes:].sum(dim=0) + 1.0)
        new_bus = F.softsign((1.0 - 0.35) * bus + 0.35 * (0.60 * sens_broadcast + 0.40 * swarm_broadcast))

        has_j = any(x is not None for x in (act_j, rx_j, tx_j, sens_j))
        return loss_w1.mean(), n_s, mean, n_err, n_lp, (loss_w1.mean(dim=-1, keepdim=True) if has_j else omega.squeeze(-1)), new_bus, new_latched_emissions

# ==========================================
# 6. DUST PLY OPTIMIZER (WITH HARMONIC MOTOR MUON)
# ==========================================
def deliberate_and_train_ply(model, dust_lr, env_input, threat, state_buf, out_buf, err_buf, lp_buf, active_idx, legal_mask, 
                             thinking_steps=32, dust_sigma=0.04, dust_grad_clip=1.0, motor_dust_lr=0.015,
                             gamma_coupling=2.0, use_entropic_restoring=True):
    global PLY_STEP_COUNTER; PLY_STEP_COUNTER += 1
    s_in, th = env_input[active_idx], threat[active_idx]
    s_s, s_o, s_e, s_l = state_buf[:, active_idx, :], out_buf[:, active_idx, :], err_buf[:, active_idx], lp_buf[:, active_idx]
    PB = s_in.shape[0]
    raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model

    with torch.no_grad():
        raw_s = raw_m.sensory_proj(s_in)
        lat = F.softsign(raw_s.view(PB, raw_m.num_input_nodes, raw_m.D).permute(1, 0, 2))

    traj_l, n_s, n_o, n_e, n_l, omgs, s_traj = model(
        s_in, th, s_s, s_o, s_e, s_l, thinking_steps=thinking_steps, precomputed_latents=lat,
        gamma_coupling=gamma_coupling, use_entropic_restoring=use_entropic_restoring
    )
    in2, th2, s2, o2, e2, l2 = (
        s_in.repeat(2, 1), 
        th.repeat(2), 
        s_traj.repeat(1, 2, 1), 
        n_o.repeat(1, 2, 1), 
        n_e.repeat(1, 2), 
        n_l.repeat(1, 2)
    )

    s_magnitude = s_traj.abs().sum(dim=(1, 2), keepdim=True)
    hunger_switch = torch.randn_like(s_traj) / (1.0 + s_magnitude)

    if PLY_STEP_COUNTER % 2 == 0:
        xi = torch.randn(raw_m.N, PB, raw_m.D * 2, device=s_in.device)
        lat2 = lat.repeat(1, 2, 1)
        _, _, _, _, _, l_both, _ = model(
            in2, th2, s2, o2, e2, l2, 
            act_jitter=torch.cat([dust_sigma * xi, -dust_sigma * xi], dim=1), 
            precomputed_latents=lat2, 
            thinking_steps=1,
            gamma_coupling=gamma_coupling, use_entropic_restoring=use_entropic_restoring
        )
        lp, ln = l_both.chunk(2, dim=1)
        gh = torch.clamp((((lp - ln) / (2.0 * dust_sigma)) * xi) / PB, -dust_grad_clip, dust_grad_clip)
        
        gh_demand = gh.std(dim=-1, keepdim=True, correction=0)
        s_hungry_w1 = s_traj + hunger_switch + (torch.randn_like(s_traj) * gh_demand * 0.10)

        raw_m.w1.weight.data.sub_(dust_lr * torch.bmm(s_hungry_w1.transpose(1, 2), gh).clamp_(-1.0, 1.0))
        raw_m.w1.normalize_weight()
    else:
        xi_s = torch.randn(PB, raw_m.num_input_nodes * raw_m.D, device=s_in.device)
        xi_rx = torch.randn(raw_m.N, PB, raw_m.D, device=s_in.device)
        xi_tx = torch.randn(raw_m.N, PB, raw_m.D, device=s_in.device)
        
        _, _, _, _, _, l_both, _ = model(
            in2, th2, s2, o2, e2, l2,
            sensory_jitter=torch.cat([dust_sigma * xi_s, -dust_sigma * xi_s], dim=0),
            rx_jitter=torch.cat([dust_sigma * xi_rx, -dust_sigma * xi_rx], dim=1),
            tx_jitter=torch.cat([dust_sigma * xi_tx, -dust_sigma * xi_tx], dim=1),
            thinking_steps=1,
            gamma_coupling=gamma_coupling, use_entropic_restoring=use_entropic_restoring
        )
        lp, ln = l_both.chunk(2, dim=1)
        d_tgt = lp - ln
        
        g_s = torch.clamp(((d_tgt[:raw_m.num_input_nodes].permute(1, 0, 2).expand(-1, -1, raw_m.D).reshape(PB, -1) / (2.0 * dust_sigma)) * xi_s) / PB, -dust_grad_clip, dust_grad_clip)
        raw_m.sensory_proj.weight.data.sub_(dust_lr * torch.mm(g_s.T, s_in).clamp_(-1.0, 1.0))
        raw_m.normalize_sensory_weight()

        bus_in = lat.mean(dim=0).unsqueeze(0).expand(raw_m.N, -1, -1)
        g_rx = torch.clamp(((d_tgt / (2.0 * dust_sigma)) * xi_rx) / PB, -dust_grad_clip, dust_grad_clip)
        raw_m.router.w_rx.data.sub_(dust_lr * torch.bmm(bus_in.transpose(1, 2), g_rx).clamp_(-1.0, 1.0))

        s_hungry_tx = s_traj + hunger_switch + (torch.randn_like(s_traj) * d_tgt.abs() * 0.10)
        g_tx = torch.clamp(((d_tgt / (2.0 * dust_sigma)) * xi_tx) / PB, -dust_grad_clip, dust_grad_clip)
        raw_m.router.w_tx.data.sub_(dust_lr * torch.bmm(s_hungry_tx.transpose(1, 2), g_tx).clamp_(-1.0, 1.0))
        raw_m.router.normalize_weights()

        # Update decoupled motor heads via Shannon entropy tangent Muon
        xi_mf = torch.randn(64, 64, device=s_in.device)
        xi_mt = torch.randn(64, 64, device=s_in.device)
        motor = n_s[-raw_m.motor_nodes_count:, :, :].permute(1, 0, 2).reshape(PB, raw_m.motor_nodes_count * raw_m.D)
        nf, nt = F.normalize(motor[:, :64], p=2, dim=-1), F.normalize(motor[:, 64:], p=2, dim=-1)

        lf_p = torch.matmul(nf, raw_m.motor_enc_from + dust_sigma * xi_mf)
        lt_p = torch.matmul(nt, raw_m.motor_enc_to + dust_sigma * xi_mt)
        lf_n = torch.matmul(nf, raw_m.motor_enc_from - dust_sigma * xi_mf)
        lt_n = torch.matmul(nt, raw_m.motor_enc_to - dust_sigma * xi_mt)
        cohere = torch.sum(nf * nt, dim=-1, keepdim=True).unsqueeze(-1)

        l_p = ((lf_p.unsqueeze(2) + lt_p.unsqueeze(1) + cohere * raw_m.spatial_affinity.unsqueeze(0)) * raw_m.action_logit_scale).view(PB, 4096)
        l_n = ((lf_n.unsqueeze(2) + lt_n.unsqueeze(1) + cohere * raw_m.spatial_affinity.unsqueeze(0)) * raw_m.action_logit_scale).view(PB, 4096)

        p_log = F.log_softmax(l_p + legal_mask[active_idx], dim=-1)
        n_log = F.log_softmax(l_n + legal_mask[active_idx], dim=-1)
        obj_p = -(p_log.exp() * p_log).sum(dim=-1).mean()
        obj_n = -(n_log.exp() * n_log).sum(dim=-1).mean()

        g_mf = torch.clamp((((obj_p - obj_n) / (2.0 * dust_sigma)) * xi_mf), -dust_grad_clip, dust_grad_clip)
        g_mt = torch.clamp((((obj_p - obj_n) / (2.0 * dust_sigma)) * xi_mt), -dust_grad_clip, dust_grad_clip)

        tan_mf = project_to_sphere_tangent(batched_newtonschulz5(g_mf.unsqueeze(0)).squeeze(0), raw_m.motor_enc_from, dim=0)
        tan_mt = project_to_sphere_tangent(batched_newtonschulz5(g_mt.unsqueeze(0)).squeeze(0), raw_m.motor_enc_to, dim=0)
        raw_m.motor_enc_from.data.sub_(motor_dust_lr * tan_mf)
        raw_m.motor_enc_to.data.sub_(motor_dust_lr * tan_mt)
        raw_m.normalize_motor_heads()

    actions = torch.argmax(model.get_action_logits(n_s) + legal_mask[active_idx], dim=-1)
    return actions, n_s, n_o, n_e, n_l, traj_l.detach(), omgs.mean(dim=0)

# ==========================================
# 7. STOCKFISH TRAINING BATCH
# ==========================================
def train_stockfish_batch(model, dust_lr=0.001, num_games=32, target_nodes=1500, thinking_steps=32, max_plies=250, device="cuda",
                          agent_noise=None, state_b=None, out_b=None, err_b=None, lp_b=None, noise_cap=0.01,
                          dust_sigma=0.04, dust_grad_clip=1.0, motor_dust_lr=0.015,
                          gamma_coupling=2.0, use_entropic_restoring=True):
    sf_exec = shutil.which("stockfish")
    if sf_exec is None: 
        return agent_noise, state_b, out_b, err_b, lp_b

    tier_nodes = torch.empty(num_games, dtype=torch.long, device=device)
    n_easy = int(num_games * 0.50)
    n_med  = int(num_games * 0.35)
    tier_nodes[:n_easy] = 100
    tier_nodes[n_easy:n_easy+n_med] = 350
    tier_nodes[n_easy+n_med:] = 1000

    sf = StockfishEngine(sf_exec)
    env = GuardrailedTorchChess(num_games, device)
    b_noise = agent_noise.clone() if agent_noise is not None else torch.full((num_games,), noise_cap * 0.50, device=device)
    
    ag_is_white = torch.arange(num_games, device=device) < (num_games // 2)
    m_modes = torch.where(ag_is_white, 3, 1).to(torch.int8)

    raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
    if state_b is None: 
        state_b, out_b, err_b, lp_b = raw_m.init_states(num_games)
    else: 
        state_b, out_b, err_b, lp_b = state_b.clone(), out_b.clone(), err_b.clone(), lp_b.clone()

    dones = torch.zeros(num_games, dtype=torch.bool, device=device)
    winners = torch.zeros(num_games, dtype=torch.int8, device=device)
    plies = torch.zeros(num_games, dtype=torch.long, device=device)
    loss_acc, loss_cnt = 0.0, 0

    while not dones.all() and (plies < max_plies).any():
        active = ~dones
        l_masks = env.compute_legal_masks()
        is_ag = active & ((env.players == 0) == ag_is_white)
        actions = torch.zeros(num_games, dtype=torch.long, device=device)

        if is_ag.any():
            ag_idx = torch.nonzero(is_ag).squeeze(-1)
            e_in, th = env.get_obs(b_noise)
            acts, ns, no, ne, nl, ploss, _ = deliberate_and_train_ply(
                model, dust_lr, e_in, th, state_b, out_b, err_b, lp_b, ag_idx, l_masks,
                thinking_steps=thinking_steps, dust_sigma=dust_sigma, dust_grad_clip=dust_grad_clip,
                motor_dust_lr=motor_dust_lr, gamma_coupling=gamma_coupling, use_entropic_restoring=use_entropic_restoring
            )
            actions[ag_idx] = acts
            state_b[:, ag_idx, :], out_b[:, ag_idx, :], err_b[:, ag_idx], lp_b[:, ag_idx] = ns, no, ne, nl
            loss_acc += ploss.item()
            loss_cnt += 1

        is_sf = active & (~is_ag)
        if is_sf.any():
            for i in torch.nonzero(is_sf).squeeze(-1).tolist():
                legals = torch.nonzero(l_masks[i] > -1e8).squeeze(-1).tolist()
                if not legals: 
                    dones[i], winners[i] = True, 3
                    continue
                
                fen = board_to_fen(
                    env.boards[i], int(env.players[i].item()), env.castling_rights[i].tolist(),
                    int(env.en_passant_sq[i].item()), int(env.halfmove_clock[i].item()), int(plies[i].item() // 2 + 1)
                )
                
                target_node_limit = int(tier_nodes[i].item())
                bm = sf.get_move(fen, nodes=target_node_limit)
                act = uci_to_action(bm) if bm else None
                actions[i] = act if act in legals else legals[0]

        dones, winners = env.step(actions, dones, winners, legal_mask=l_masks)
        plies[active] += 1
        over = (plies >= max_plies) & (~dones)
        if over.any(): 
            dones, winners = dones | over, torch.where(over, torch.tensor(3, dtype=torch.int8, device=device), winners)

    sf.close()
    update_agent_somatic_noise(b_noise, dones, winners, m_modes, plies=plies, env=env, noise_cap=noise_cap)
    
    ag_wins = (((winners == 1) & ag_is_white) | ((winners == 2) & (~ag_is_white))).sum().item()
    sf_wins = (((winners == 2) & ag_is_white) | ((winners == 1) & (~ag_is_white))).sum().item()
    draws = (winners == 3).sum().item()
    
    print(f"[Stockfish Multi-Curriculum Batch] Ag: {ag_wins} | SF: {sf_wins} | D: {draws} | "
          f"Loss: {loss_acc / max(1, loss_cnt):.4f} | Stress: {b_noise.mean().item():.4f}")
    return b_noise, state_b, out_b, err_b, lp_b

# ==========================================
# 8. SHARED ROLLOUT HELPER
# ==========================================
def _run_env_rollout(model, env, s_w, o_w, e_w, l_w, s_b, o_b, e_b, l_b, noise, max_plies=300, thinking_steps=32, sf_engine=None, sf_nodes=1000, sf_is_black_mask=None):
    B = env.B
    dones, winners, plies = torch.zeros(B, dtype=torch.bool, device=env.device), torch.zeros(B, dtype=torch.int8, device=env.device), torch.zeros(B, dtype=torch.long, device=env.device)

    while not dones.all() and (plies < max_plies).any():
        active = ~dones
        l_masks = env.compute_legal_masks()
        actions = torch.zeros(B, dtype=torch.long, device=env.device)
        w_turn, b_turn = active & (env.players == 0), active & (env.players == 1)

        for turn_mask, is_white, s_b_curr, o_b_curr, e_b_curr, l_b_curr in [
            (w_turn, True, s_w, o_w, e_w, l_w), (b_turn, False, s_b, o_b, e_b, l_b)
        ]:
            if not turn_mask.any(): continue

            if sf_engine is not None and sf_is_black_mask is not None:
                sf_turn = (~sf_is_black_mask) if is_white else sf_is_black_mask
                active_sf = turn_mask & sf_turn
                active_agent = turn_mask & (~sf_turn)
            else:
                active_sf = torch.zeros(B, dtype=torch.bool, device=env.device)
                active_agent = turn_mask

            if active_sf.any():
                for i in torch.nonzero(active_sf).squeeze(-1).tolist():
                    legals = torch.nonzero(l_masks[i] > -1e8).squeeze(-1).tolist()
                    if not legals: dones[i], winners[i] = True, 3; continue
                    fen = board_to_fen(env.boards[i], 0 if is_white else 1, env.castling_rights[i].tolist(), int(env.en_passant_sq[i].item()), int(env.halfmove_clock[i].item()), int(plies[i].item() // 2 + 1))
                    bm = sf_engine.get_move(fen, nodes=sf_nodes)
                    act = uci_to_action(bm) if bm else None
                    actions[i] = act if act in legals else legals[0]

            if active_agent.any():
                a_idx = torch.nonzero(active_agent).squeeze(-1)
                e_in, th = env.get_obs(noise)
                _, ns, no, ne, nl, _, _ = model(e_in[a_idx], th[a_idx], s_b_curr[:, a_idx, :], o_b_curr[:, a_idx, :], e_b_curr[:, a_idx], l_b_curr[:, a_idx], thinking_steps=thinking_steps)
                s_b_curr[:, a_idx, :], o_b_curr[:, a_idx, :], e_b_curr[:, a_idx], l_b_curr[:, a_idx] = ns, no, ne, nl
                actions[a_idx] = torch.argmax(model.get_action_logits(ns) + l_masks[a_idx], dim=-1)

        dones, winners = env.step(actions, dones, winners, legal_mask=l_masks)
        plies[active] += 1
        over = (plies >= max_plies) & (~dones)
        if over.any(): dones, winners = dones | over, torch.where(over, torch.tensor(3, dtype=torch.int8, device=env.device), winners)

    return winners

# ==========================================
# 9. TOURNAMENT & SF GAUNTLET
# ==========================================
def run_internal_tournament(model, agent_state, agent_output, agent_error, agent_lp, agent_noise, top_k=16, thinking_steps=32, device="cuda"):
    eff_k = min(top_k, agent_noise.shape[0])
    agent_loss = agent_error.mean(dim=0)
    
    _, top_idx = torch.topk(agent_loss, k=eff_k, largest=False)
    cand_list = top_idx.tolist()
    matchups = [(cand_list[i], cand_list[j]) for i in range(eff_k) for j in range(eff_k) if i != j]
    N = len(matchups)
    w_cand = torch.tensor([m[0] for m in matchups], device=device)
    b_cand = torch.tensor([m[1] for m in matchups], device=device)

    print(f"\n---> Running Internal Tournament ({eff_k} Candidates -> {N} Matches @ {thinking_steps} Thinking Steps)...")
    env = GuardrailedTorchChess(N, device=device)
    winners = _run_env_rollout(
        model, env,
        agent_state[:, w_cand, :].clone(), agent_output[:, w_cand, :].clone(), agent_error[:, w_cand].clone(), agent_lp[:, w_cand].clone(),
        agent_state[:, b_cand, :].clone(), agent_output[:, b_cand, :].clone(), agent_error[:, b_cand].clone(), agent_lp[:, b_cand].clone(),
        torch.full((N,), 0.001, device=device), thinking_steps=thinking_steps
    )

    scores = {c: 0.0 for c in cand_list}
    for (wc, bc), win in zip(matchups, winners.tolist()):
        if win == 1: scores[wc] += 1.0
        elif win == 2: scores[bc] += 1.0
        else: scores[wc] += 0.5; scores[bc] += 0.5

    best = max(cand_list, key=lambda c: (scores[c], -agent_loss[c].item()))
    
    print(f"\n[Tournament Complete] ★ Crown Champion: Slot #{best} ★ | "
          f"Score: {scores[best]:.1f}/{(eff_k - 1) * 2} | "
          f"Loss: {agent_loss[best].item():.4f} | "
          f"Noise: {agent_noise[best].item():.4f}\n")
          
    return {
        'slot_id': best,
        'score': scores[best],
        'loss': agent_loss[best].item(),
        'state': agent_state[:, best:best+1, :].clone(),
        'output': agent_output[:, best:best+1, :].clone(),
        'error_ema': agent_error[:, best:best+1].clone(),
        'lp_ema': agent_lp[:, best:best+1].clone(),
        'noise': agent_noise[best:best+1].clone()
    }

def run_stockfish_gauntlet(model, champ, thinking_steps=32, node_levels=[100, 500, 1000, 1500, 3000], games_per_tier=20, device="cuda"):
    sf_exec = shutil.which("stockfish")
    if sf_exec is None: 
        print("[Gauntlet] Stockfish binary not found. Skipping.")
        return
    print(f"\n---> Stockfish Multi-Tier Gauntlet ({games_per_tier} g/tier @ {thinking_steps} Thinking Steps)")
    for nodes in node_levels:
        sf = StockfishEngine(sf_exec)
        env = GuardrailedTorchChess(games_per_tier, device)
        sf_is_black = torch.arange(games_per_tier, device=device) < (games_per_tier // 2)

        winners = _run_env_rollout(
            model, env,
            champ['state'].expand(-1, games_per_tier, -1).clone(), champ['output'].expand(-1, games_per_tier, -1).clone(),
            champ['error_ema'].expand(-1, games_per_tier).clone(), champ['lp_ema'].expand(-1, games_per_tier).clone(),
            champ['state'].expand(-1, games_per_tier, -1).clone(), champ['output'].expand(-1, games_per_tier, -1).clone(),
            champ['error_ema'].expand(-1, games_per_tier).clone(), champ['lp_ema'].expand(-1, games_per_tier).clone(),
            torch.full((games_per_tier,), 0.001, device=device),
            thinking_steps=thinking_steps, sf_engine=sf, sf_nodes=nodes, sf_is_black_mask=sf_is_black
        )
        sf.close()
        ag_w = (((winners == 1) & sf_is_black) | ((winners == 2) & (~sf_is_black))).sum().item()
        sf_w = (((winners == 2) & sf_is_black) | ((winners == 1) & (~sf_is_black))).sum().item()
        draws = (winners == 3).sum().item()
        print(f"  SF @ {nodes:5d} nodes => Ag WINS: {ag_w:2d} | Draws: {draws:2d} | Ag Losses: {sf_w:2d} | Score: {((ag_w + 0.5 * draws) / games_per_tier) * 100.0:4.1f}%")

# ==========================================
# 10. LIFELONG TRAINING & MIGRATION LOADER
# ==========================================
def resolve_checkpoint_path(resume_arg: str) -> str:
    if not resume_arg: return None
    if resume_arg.lower() == "latest":
        ckpt_files = glob.glob(os.path.join("out", "ckpt(*).pt"))
        if ckpt_files:
            def extract_batch(p):
                m = re.search(r"ckpt\((\d+)\)\.pt", p)
                return int(m.group(1)) if m else -1
            chosen = max(ckpt_files, key=extract_batch)
            print(f"[Checkpoint Resolution] '--resume latest' matched highest numbered checkpoint: {chosen}")
            return chosen
        
        all_pts = glob.glob(os.path.join("out", "*.pt"))
        if all_pts:
            chosen = max(all_pts, key=os.path.getmtime)
            print(f"[Checkpoint Resolution] '--resume latest' fell back to newest file: {chosen}")
            return chosen
        return None

    if resume_arg.lower() == "champion":
        champ_path = os.path.join("out", "champion_agent.pt")
        return champ_path if os.path.exists(champ_path) else None

    return resume_arg

def load_checkpoint(model, resume_path, agent_noise, agent_state, agent_output, agent_error, agent_lp, device):
    resolved_path = resolve_checkpoint_path(resume_path)
    if not resolved_path or not os.path.exists(resolved_path):
        print(f"\n[Resume Error] Target checkpoint file could not be found: '{resume_path}'. Starting fresh.\n")
        return 1, 0

    print("\n" + "=" * 76)
    print(f"[Checkpoint Loader] Loading: {resolved_path}")
    ckpt = torch.load(resolved_path, map_location=device, weights_only=False)

    raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
    if 'model_state_dict' in ckpt:
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt['model_state_dict'].items()}
        res = raw_m.load_state_dict(sd, strict=False)
        print(f"[Model Weights]     State dict restored. Missing: {len(res.missing_keys)} | Unexpected: {len(res.unexpected_keys)}")
        raw_m.w1.normalize_weight()
        raw_m.router.normalize_weights()
        raw_m.normalize_motor_heads()

    batch_size = agent_noise.shape[0]
    has_population_pool = ('agent_state' in ckpt and ckpt['agent_state'] is not None)

    if 'champion' in ckpt and not has_population_pool:
        champ = ckpt['champion']
        print(f"[Population Pool]   Broadcasting Champion across batch size {batch_size}...")
        agent_noise.fill_(float(champ['noise'].squeeze().item()))
        agent_state.copy_(champ['state'].expand(-1, batch_size, -1))
        agent_output.copy_(champ['output'].expand(-1, batch_size, -1))
        agent_error.copy_(champ['error_ema'].expand(-1, batch_size))
        agent_lp.copy_(champ['lp_ema'].expand(-1, batch_size))
    else:
        for key, target in [('agent_noise', agent_noise), ('agent_state', agent_state),
                            ('agent_output', agent_output), ('agent_error', agent_error), ('agent_lp', agent_lp)]:
            if key in ckpt and ckpt[key] is not None:
                src = ckpt[key].to(device)
                if src.shape == target.shape:
                    target.copy_(src)
                elif src.dim() == target.dim() and src.shape[0] == target.shape[0]:
                    reps = math.ceil(target.shape[1] / src.shape[1])
                    if target.dim() == 3: tiled = src.repeat(1, reps, 1)[:, :target.shape[1], :]
                    else: tiled = src.repeat(1, reps)[:, :target.shape[1]]
                    target.copy_(tiled)

    completed_games = ckpt.get('completed_games', 0)
    batch_idx = ckpt.get('batch_idx', 1)
    print(f"[Resumed Counters]  Completed Games: {completed_games:,} | Resuming at Batch: {batch_idx}")
    print("=" * 76 + "\n")
    return batch_idx, completed_games

def run_agent_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    model = ChessNodeNet(
        num_nodes=args.num_nodes, dim=args.dim, input_dim=args.input_dim, rank_k=args.rank_k,
        spatial_sigma=args.spatial_sigma, action_logit_scale=args.action_logit_scale,
        lp_alpha=args.lp_alpha, device=device
    ).to(device)

    try:
        compiled_model = torch.compile(model, dynamic=True)
    except Exception:
        compiled_model = model

    dust_lr = args.dust_lr
    env = GuardrailedTorchChess(args.batch_size, device, halfmove_limit=args.halfmove_limit)
    dones = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
    winners = torch.zeros(args.batch_size, dtype=torch.int8, device=device)
    match_modes = (torch.arange(args.batch_size, device=device) % 4).to(torch.int8)

    agent_state, agent_output, agent_error, agent_lp = model.init_states(args.batch_size)
    agent_noise = torch.full((args.batch_size,), args.noise_cap * 0.50, device=device)
    completed_games = 0
    batch_idx = 1

    if args.resume:
        batch_idx, completed_games = load_checkpoint(
            model, args.resume, agent_noise, agent_state, agent_output, agent_error, agent_lp, device
        )

    state_w, output_w, error_w, lp_w = agent_state.clone(), agent_output.clone(), agent_error.clone(), agent_lp.clone()
    state_b, output_b, error_b, lp_b = agent_state.clone(), agent_output.clone(), agent_error.clone(), agent_lp.clone()
    plies = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    accum_losses = torch.zeros(args.batch_size, device=device)
    loss_steps = torch.zeros(args.batch_size, device=device)

    sp_games = sp_draws = bot_games = bot_agent_wins = bot_bot_wins = bot_draws = 0
    b_plies, b_losses = [], []
    batch_start = time.time()

    print("=" * 90)
    print(" NODECHESS 21: FULLY PARAMETERIZED LIFELONG ACTIVE INFERENCE")
    print(f" Target Games: {args.total_games:,} | Batch: {args.batch_size} | Thinking Steps: {args.thinking_steps} | Device: {device}")
    print(f" Spatial Sigma: {args.spatial_sigma} | DUST LR: {args.dust_lr} | Motor DUST LR: {args.motor_dust_lr}")
    print("=" * 90)

    if not args.eval_only:
        while completed_games < args.total_games:
            active = ~dones
            l_mask = env.compute_legal_masks()
            actions = torch.zeros(args.batch_size, dtype=torch.long, device=device)

            w_turn, b_turn = active & (env.players == 0), active & (env.players == 1)
            w_agent, w_bot = w_turn & (match_modes != 1), w_turn & (match_modes == 1)
            b_agent, b_bot = b_turn & (match_modes != 3), b_turn & (match_modes == 3)

            for mask, is_agent, is_white, sb, ob, eb, lb in [
                (w_agent, True, True, state_w, output_w, error_w, lp_w),
                (w_bot, False, True, None, None, None, None),
                (b_agent, True, False, state_b, output_b, error_b, lp_b),
                (b_bot, False, False, None, None, None, None)
            ]:
                if not mask.any(): continue
                idx = torch.where(mask)[0]
                if is_agent:
                    e_in, th = env.get_obs(agent_noise)
                    acts, ns, no, ne, nl, loss, _ = deliberate_and_train_ply(
                        compiled_model, dust_lr, e_in, th, sb, ob, eb, lb, idx, l_mask, 
                        thinking_steps=args.thinking_steps, dust_sigma=args.dust_sigma, dust_grad_clip=args.dust_grad_clip,
                        motor_dust_lr=args.motor_dust_lr, gamma_coupling=args.gamma_coupling,
                        use_entropic_restoring=args.use_entropic_restoring
                    )
                    actions[idx] = acts
                    sb[:, idx, :], ob[:, idx, :], eb[:, idx], lb[:, idx] = ns, no, ne, nl
                    accum_losses[idx] += loss
                    loss_steps[idx] += 1
                else:
                    actions[idx] = select_cuda_engine_move(
                        env.boards[idx], l_mask[idx], 
                        torch.full((len(idx),), is_white, dtype=torch.bool, device=device), 
                        device, env=env, b_idx=idx
                    )

            dones, winners = env.step(actions, dones, winners, legal_mask=l_mask)
            plies[active] += 1
            over = (plies >= args.max_plies) & (~dones)
            if over.any():
                dones = dones | over
                winners = torch.where(over, torch.tensor(3, dtype=torch.int8, device=device), winners)

            if dones.any():
                fin_i = torch.where(dones)[0]

                live_loss = accum_losses / torch.clamp(loss_steps, min=1.0)
                agent_loss = torch.where(loss_steps > 0, live_loss, agent_error.mean(dim=0))

                resolve_fittest_agent_states(
                    dones, winners, match_modes,
                    [agent_state, agent_output, agent_error, agent_lp],
                    [state_w, output_w, error_w, lp_w],
                    [state_b, output_b, error_b, lp_b],
                    agent_noise=agent_noise,
                    agent_loss=agent_loss
                )
                
                update_agent_somatic_noise(
                    agent_noise, dones, winners, match_modes, 
                    plies=plies, env=env, noise_cap=args.noise_cap
                )

                for i in fin_i.tolist():
                    completed_games += 1
                    b_plies.append(plies[i].item())
                    b_losses.append((accum_losses[i] / max(1.0, loss_steps[i].item())).item())

                    is_sp = (match_modes[i] == 0 or match_modes[i] == 2)
                    wc = winners[i].item()
                    ag_won = ((match_modes[i] == 3 and wc == 1) or (match_modes[i] == 1 and wc == 2) or (is_sp and wc in (1, 2)))
                    b_won = ((match_modes[i] == 3 and wc == 2) or (match_modes[i] == 1 and wc == 1))

                    if is_sp:
                        sp_games += 1
                        if wc == 3: sp_draws += 1
                    else:
                        bot_games += 1
                        if ag_won: bot_agent_wins += 1
                        elif b_won: bot_bot_wins += 1
                        elif wc == 3: bot_draws += 1

                    if completed_games % args.batch_size == 0 or completed_games >= args.total_games:
                        dt = time.time() - batch_start
                        print(f"[BATCH {batch_idx:03d} | Games: {completed_games:06d}/{args.total_games:,}] "
                              f"{len(b_plies) / max(1e-3, dt):4.1f} g/s | "
                              f"Bot: [W: {(bot_agent_wins / max(1, bot_games)) * 100:4.1f}% | "
                              f"L: {(bot_bot_wins / max(1, bot_games)) * 100:4.1f}% | "
                              f"D: {(bot_draws / max(1, bot_games)) * 100:4.1f}%] | "
                              f"SP Rep-D: {(sp_draws / max(1, sp_games)) * 100:4.1f}% | "
                              f"AvgPlies: {sum(b_plies) / max(1, len(b_plies)):4.1f} | "
                              f"Loss: {sum(b_losses) / max(1, len(b_losses)):.4f} | "
                              f"Stress: {agent_noise.mean().item():.4f}")

                        sp_games = sp_draws = bot_games = bot_agent_wins = bot_bot_wins = bot_draws = 0

                        if batch_idx % args.sf_interval == 0:
                            k_eff = min(args.sf_train_games, args.batch_size)
                            pop_loss = agent_error.mean(dim=0)
                            _, sf_idx = torch.topk(pop_loss, k=k_eff, largest=False)
                            
                            sfn, sfs, sfo, sfe, sfl = train_stockfish_batch(
                                compiled_model, dust_lr=dust_lr, num_games=k_eff, 
                                thinking_steps=args.thinking_steps, device=device,
                                agent_noise=agent_noise[sf_idx], state_b=agent_state[:, sf_idx, :], 
                                out_b=agent_output[:, sf_idx, :], err_b=agent_error[:, sf_idx], lp_b=agent_lp[:, sf_idx],
                                noise_cap=args.noise_cap, dust_sigma=args.dust_sigma, dust_grad_clip=args.dust_grad_clip,
                                motor_dust_lr=args.motor_dust_lr, gamma_coupling=args.gamma_coupling,
                                use_entropic_restoring=args.use_entropic_restoring
                            )
                            agent_noise[sf_idx], agent_state[:, sf_idx, :], agent_output[:, sf_idx, :] = sfn, sfs, sfo
                            agent_error[:, sf_idx], agent_lp[:, sf_idx] = sfe, sfl
                            for sb, ob, eb, lb in [(state_w, output_w, error_w, lp_w), (state_b, output_b, error_b, lp_b)]:
                                sb[:, sf_idx, :], ob[:, sf_idx, :], eb[:, sf_idx], lb[:, sf_idx] = sfs, sfo, sfe, sfl

                            os.makedirs("out", exist_ok=True)
                            save_file = os.path.join("out", f"ckpt({batch_idx}).pt")
                            raw_save_m = model._orig_mod if hasattr(model, "_orig_mod") else model
                            torch.save({
                                'batch_idx': batch_idx + 1,
                                'completed_games': completed_games,
                                'model_state_dict': raw_save_m.state_dict(),
                                'agent_noise': agent_noise.cpu(),
                                'agent_state': agent_state.cpu(),
                                'agent_output': agent_output.cpu(),
                                'agent_error': agent_error.cpu(),
                                'agent_lp': agent_lp.cpu()
                            }, save_file)
                            print(f"[Checkpoint Saved] -> {save_file} (Next Batch: {batch_idx + 1}, Games: {completed_games:,})")

                        batch_idx += 1
                        b_plies, b_losses = [], []
                        batch_start = time.time()
                    
                    if completed_games >= args.total_games:
                        break

                match_modes[dones] = (match_modes[dones] + 1) % 4
                env.reset(mask=dones)
                for sb, ob, eb, lb in [(state_w, output_w, error_w, lp_w), (state_b, output_b, error_b, lp_b)]:
                    sb[:, fin_i, :], ob[:, fin_i, :] = agent_state[:, fin_i, :], agent_output[:, fin_i, :]
                    eb[:, fin_i], lb[:, fin_i] = agent_error[:, fin_i], agent_lp[:, fin_i]
                plies[dones], accum_losses[dones], loss_steps[dones], dones[dones], winners[dones] = 0, 0.0, 0.0, False, 0

    champion = run_internal_tournament(
        compiled_model, agent_state, agent_output, agent_error, agent_lp, agent_noise, 
        top_k=args.top_k_tourney, thinking_steps=args.thinking_steps, device=device
    )
    
    os.makedirs("out", exist_ok=True)
    champ_file = os.path.join("out", "champion_agent.pt")
    raw_save_m = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        'batch_idx': batch_idx,
        'completed_games': completed_games,
        'model_state_dict': raw_save_m.state_dict(),
        'champion': champion,
        'agent_noise': agent_noise.cpu(),
        'agent_state': agent_state.cpu(),
        'agent_output': agent_output.cpu(),
        'agent_error': agent_error.cpu(),
        'agent_lp': agent_lp.cpu()
    }, champ_file)
    print(f"[Champion Checkpoint Saved] -> {champ_file} (Completed Games: {completed_games:,})")

    node_levels = [int(x) for x in args.sf_gauntlet_nodes.split(",")]
    run_stockfish_gauntlet(
        compiled_model, champion, thinking_steps=args.thinking_steps, 
        node_levels=node_levels, games_per_tier=args.sf_gauntlet_games, device=device
    )

# ==========================================
# 11. CENTRALIZED ARGPARSE CONFIGURATION
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Active Inference NodeChess Lifelong Training")
    
    # Runtime & Experiment Budget
    parser.add_argument("--total_games", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_plies", type=int, default=400)
    parser.add_argument("--thinking_steps", type=int, default=32)
    parser.add_argument("--top_k_tourney", type=int, default=16)
    parser.add_argument("--sf_interval", type=int, default=10)
    parser.add_argument("--sf_train_games", type=int, default=256)
    parser.add_argument("--noise_cap", type=float, default=0.03, help="Max somatic noise level")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")

    # Swarm Architecture & Geometry
    parser.add_argument("--num_nodes", type=int, default=32)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--input_dim", type=int, default=64)
    parser.add_argument("--rank_k", type=int, default=16)
    parser.add_argument("--spatial_sigma", type=float, default=2.5, help="Euclidean affinity Gaussian sigma")
    parser.add_argument("--action_logit_scale", type=float, default=3.0, help="Softmax temperature scale")
    parser.add_argument("--lp_alpha", type=float, default=0.45)
    parser.add_argument("--gamma_coupling", type=float, default=2.0)
    parser.add_argument("--use_entropic_restoring", action="store_true", default=True)

    # Optimization Hyperparameters
    parser.add_argument("--dust_lr", type=float, default=0.001)
    parser.add_argument("--motor_dust_lr", type=float, default=0.015)
    parser.add_argument("--dust_sigma", type=float, default=0.04)
    parser.add_argument("--dust_grad_clip", type=float, default=1.0)

    # Chess Simulation & Gauntlet
    parser.add_argument("--halfmove_limit", type=int, default=100)
    parser.add_argument("--sf_gauntlet_nodes", type=str, default="100,500,1000,1500,3000")
    parser.add_argument("--sf_gauntlet_games", type=int, default=20)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.init()
    torch.set_float32_matmul_precision('high')
    run_agent_experiment(args)