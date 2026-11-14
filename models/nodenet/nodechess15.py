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

noise_cap = 0.10

# ==========================================
# OPTIMIZATION MODE CONFIGURATION
# ==========================================
TRAINING_MODE = "dust"    # Pure Activation-Space Zeroth-Order (Zero Autograd)
DUST_SIGMA = 0.04         # Antithetic activation perturbation radius
DUST_GRAD_CLIP = 1.0      # Gradient proxy clipping threshold

# ==========================================
# 0. SPECTRAL GABOR FILTER HELPERS
# ==========================================
def make_spectral_kernel(dim: int, center_offset: float = 1.0, sigma_val: float = 0.5, device="cuda", dtype=torch.float32):
    kernel_size = min(31, dim if dim % 2 == 1 else dim - 1)
    if kernel_size < 3:
        return torch.ones(1, 1, 1, device=device, dtype=dtype)
    half_k = kernel_size // 2
    omega_0 = center_offset * math.pi
    sigma_t = max(sigma_val * math.pi, 1e-4)

    t = torch.arange(-half_k, half_k + 1, device=device, dtype=dtype)
    envelope = torch.exp(-0.5 * (t * sigma_t) ** 2)
    carrier = torch.cos(omega_0 * t)
    kernel = envelope * carrier
    kernel = kernel / (torch.norm(kernel, p=2) + 1e-8)
    return kernel.view(1, 1, kernel_size)

def make_circulant_matrix(dim: int, center_offset: float = 1.0, sigma_val: float = 0.5, device="cuda", dtype=torch.float32):
    kernel = make_spectral_kernel(dim, center_offset, sigma_val, device, dtype)
    kernel_size = kernel.shape[-1]
    if kernel_size < 3:
        return torch.eye(dim, device=device, dtype=dtype)
    half_k = kernel_size // 2
    eye = torch.eye(dim, device=device, dtype=dtype).unsqueeze(1)
    eye_pad = F.pad(eye, (half_k, half_k), mode='circular')
    # Row i of M is conv(e_i), so for vector x: x @ M == conv(x)
    return F.conv1d(eye_pad, kernel).squeeze(1)

_CIRCULANT_CACHE = {}

def apply_spectral_filter(noise: torch.Tensor, center_offset: float = 1.0, sigma_val: float = 0.5) -> torch.Tensor:
    dim = noise.shape[-1]
    if dim < 3:
        return noise
    device = noise.device
    dtype = noise.dtype

    cache_key = (dim, round(float(center_offset), 4), round(float(sigma_val), 4), str(device), dtype)
    if cache_key in _CIRCULANT_CACHE:
        circ_mat = _CIRCULANT_CACHE[cache_key]
    else:
        circ_mat = make_circulant_matrix(dim, center_offset, sigma_val, device, dtype)
        _CIRCULANT_CACHE[cache_key] = circ_mat

    return torch.matmul(noise, circ_mat)

class FallbackUtils:
    @staticmethod
    def softsign(x): 
        return F.softsign(x)

    @staticmethod
    def buntanh(x, slope, bound):
        return torch.tanh(x - torch.tanh(x) * slope) * bound

    @staticmethod
    def apply_spectral_filter(noise, center_offset=1.0, sigma_val=0.5):
        return apply_spectral_filter(noise, center_offset, sigma_val)

utils = FallbackUtils()
activation = utils.buntanh

# ==========================================
# 1. PURE CUDA FIDE CHESS SIMULATOR
# ==========================================
class GuardrailedTorchChess:
    def __init__(self, batch_size, device="cuda", halfmove_limit=100):
        self.B = batch_size
        self.device = device
        self.halfmove_limit = halfmove_limit
        self.max_history_len = 128
        
        self.boards = torch.zeros((batch_size, 8, 8), dtype=torch.int8, device=device)
        self.players = torch.zeros(batch_size, dtype=torch.int8, device=device)
        self.piece_vals = torch.tensor([0, 100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000], dtype=torch.float32, device=device)

        self.castling_rights = torch.ones((batch_size, 4), dtype=torch.bool, device=device)
        self.en_passant_sq = torch.full((batch_size,), -1, dtype=torch.int8, device=device)

        self.halfmove_clock = torch.zeros(batch_size, dtype=torch.int16, device=device)
        self.history_len = torch.zeros(batch_size, dtype=torch.int16, device=device)
        self.history_hashes = torch.zeros((batch_size, self.max_history_len), dtype=torch.int64, device=device)

        # Preallocated buffers
        self.pseudo_mask = torch.full((batch_size, 4096), -1e9, device=device)
        self.legal_mask = torch.full((batch_size, 4096), -1e9, device=device)
        self.sq_range = torch.arange(64, device=device).unsqueeze(0)

        self._ray_dirs = torch.tensor([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [1, -1], [-1, 1], [-1, -1]
        ], dtype=torch.long, device=device)
        self._ray_dr = self._ray_dirs[:, 0].view(1, 8, 1)
        self._ray_dc = self._ray_dirs[:, 1].view(1, 8, 1)
        self._ray_steps = torch.arange(1, 8, dtype=torch.long, device=device).view(1, 1, 7)

        self._k_deltas = torch.tensor([
            [-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]
        ], dtype=torch.long, device=device)

        self._kg_deltas = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]
        ], dtype=torch.long, device=device)

        # ----------------------------------------------------
        # PRECOMPUTED STATIC GEOMETRY FOR ALL 64 SQUARES
        # ----------------------------------------------------
        sqs = torch.arange(64, device=device)
        r = sqs // 8
        c = sqs % 8

        # Slider rays: (64, 8, 7) and (64, 56)
        sr = r.view(64, 1, 1) + self._ray_dr * self._ray_steps
        sc = c.view(64, 1, 1) + self._ray_dc * self._ray_steps
        self.slider_valid = ((sr >= 0) & (sr < 8) & (sc >= 0) & (sc < 8)).view(64, 8, 7)
        self.slider_sqs = (sr.clamp(0, 7) * 8 + sc.clamp(0, 7)).long().view(64, 56)

        # Knight hops: (64, 8)
        kr = r.view(64, 1) + self._k_deltas[:, 0].view(1, 8)
        kc = c.view(64, 1) + self._k_deltas[:, 1].view(1, 8)
        self.knight_valid = (kr >= 0) & (kr < 8) & (kc >= 0) & (kc < 8)
        self.knight_sqs = (kr.clamp(0, 7) * 8 + kc.clamp(0, 7)).long()

        # King steps: (64, 8)
        kgr = r.view(64, 1) + self._kg_deltas[:, 0].view(1, 8)
        kgc = c.view(64, 1) + self._kg_deltas[:, 1].view(1, 8)
        self.king_valid = (kgr >= 0) & (kgr < 8) & (kgc >= 0) & (kgc < 8)
        self.king_sqs = (kgr.clamp(0, 7) * 8 + kgc.clamp(0, 7)).long()

        # Pawn attack sources: (2, 64, 2) [0: White attacking from r-1, 1: Black attacking from r+1]
        p_dc = torch.tensor([-1, 1], device=device).view(1, 2)
        pr_w = r.view(64, 1) - 1
        pc_w = c.view(64, 1) + p_dc
        valid_pw = (pr_w >= 0) & (pr_w < 8) & (pc_w >= 0) & (pc_w < 8)
        sqs_pw = (pr_w.clamp(0, 7) * 8 + pc_w.clamp(0, 7)).long()

        pr_b = r.view(64, 1) + 1
        pc_b = c.view(64, 1) + p_dc
        valid_pb = (pr_b >= 0) & (pr_b < 8) & (pc_b >= 0) & (pc_b < 8)
        sqs_pb = (pr_b.clamp(0, 7) * 8 + pc_b.clamp(0, 7)).long()

        self.pawn_valid = torch.stack([valid_pw, valid_pb], dim=0)
        self.pawn_sqs = torch.stack([sqs_pw, sqs_pb], dim=0)

        g = torch.Generator(device=device)
        g.manual_seed(42)
        self.zobrist_pieces = torch.randint(-2**62, 2**62, (64, 13), generator=g, dtype=torch.int64, device=device)
        self.zobrist_turn = torch.randint(-2**62, 2**62, (1,), generator=g, dtype=torch.int64, device=device)

        self.reset()

    def compute_hash(self, boards, players):
        b_size = boards.shape[0]
        flat = boards.view(b_size, 64).long().clamp(0, 12)
        sq = self.sq_range.expand(b_size, -1)
        piece_hashes = self.zobrist_pieces[sq, flat]
        return piece_hashes.sum(dim=-1) + self.zobrist_turn * players.to(torch.int64)

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        idx = torch.nonzero(mask).squeeze(-1)
        if idx.numel() == 0:
            return

        initial = torch.tensor([
            [4, 2, 3, 5, 6, 3, 2, 4],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [10, 8, 9, 11, 12, 9, 8, 10]
        ], dtype=torch.int8, device=self.device)

        self.boards[idx] = initial
        self.players[idx] = 0
        self.castling_rights[idx] = True
        self.en_passant_sq[idx] = -1
        self.halfmove_clock[idx] = 0
        self.history_len[idx] = 1

        init_hashes = self.compute_hash(self.boards[idx], self.players[idx])
        self.history_hashes[idx, 0] = init_hashes
        self.history_hashes[idx, 1:] = 0

    def get_material(self):
        flat = self.boards.view(self.B, 64).long().clamp(0, 12)
        vals = self.piece_vals[flat]
        white_mat = (vals * ((flat >= 1) & (flat <= 5)).float()).sum(dim=-1)
        black_mat = (vals * ((flat >= 7) & (flat <= 11)).float()).sum(dim=-1)
        return white_mat, black_mat

    def is_insufficient_material(self):
        flat = self.boards.view(self.B, 64).long()
        pawns_majors = ((flat == 1) | (flat == 7) | (flat == 4) | (flat == 10) | (flat == 5) | (flat == 11)).sum(dim=-1)
        w_minors = ((flat == 2) | (flat == 3)).sum(dim=-1)
        b_minors = ((flat == 8) | (flat == 9)).sum(dim=-1)
        
        bare_kings = (pawns_majors == 0) & (w_minors == 0) & (b_minors == 0)
        kn_or_kb = (pawns_majors == 0) & (((w_minors <= 1) & (b_minors == 0)) | ((b_minors <= 1) & (w_minors == 0)))
        return bare_kings | kn_or_kb

    def get_obs(self, agent_noise):
        flat = self.boards.view(self.B, 64).float() / 12.0
        noise_sig = agent_noise.unsqueeze(1)
        raw_noise = torch.randn_like(flat)
        spectral_noise = utils.apply_spectral_filter(raw_noise, center_offset=1.0, sigma_val=0.5)

        signal_clarity = torch.clamp(1.0 - noise_sig, min=0.05, max=1.0)
        obs_distorted = flat * signal_clarity + spectral_noise * noise_sig

        somatic_raw = torch.randn(self.B, 34, device=self.device)
        somatic_noise = utils.apply_spectral_filter(somatic_raw, center_offset=1.0, sigma_val=0.5) * noise_sig

        meta = torch.cat([self.players.float().unsqueeze(1), noise_sig, somatic_noise], dim=1)
        return torch.cat([obs_distorted, meta], dim=1), agent_noise
    def _is_square_attacked(self, target_sq, attacking_player, boards):
        B_size = boards.shape[0]
        flat = boards.view(B_size, 64)

        if not isinstance(target_sq, torch.Tensor):
            target_sq = torch.full((B_size,), target_sq, dtype=torch.long, device=self.device)
        elif target_sq.dim() == 0:
            target_sq = target_sq.expand(B_size)

        if not isinstance(attacking_player, torch.Tensor):
            attacking_player = torch.full((B_size,), attacking_player, dtype=torch.int8, device=self.device)
        elif attacking_player.dim() == 0:
            attacking_player = attacking_player.expand(B_size)

        r_t, c_t = target_sq // 8, target_sq % 8
        is_att_w = (attacking_player == 0).view(B_size, 1)
        offset = torch.where(is_att_w, 0, 6).to(torch.int8)

        pawn_val   = 1 + offset
        knight_val = 2 + offset
        bishop_val = 3 + offset
        rook_val   = 4 + offset
        queen_val  = 5 + offset
        king_val   = 6 + offset

        p_src_r = torch.where(is_att_w, r_t.view(-1, 1) - 1, r_t.view(-1, 1) + 1)
        p_src_c = c_t.view(-1, 1) + torch.tensor([[-1, 1]], device=self.device)
        valid_p = (p_src_r >= 0) & (p_src_r < 8) & (p_src_c >= 0) & (p_src_c < 8)
        sqs_p = (p_src_r.clamp(0, 7) * 8 + p_src_c.clamp(0, 7)).long()
        p_p = flat.gather(1, sqs_p)
        pawn_att = (valid_p & (p_p == pawn_val)).any(dim=1)

        kr = r_t.view(-1, 1) + self._k_deltas[:, 0].view(1, 8)
        kc = c_t.view(-1, 1) + self._k_deltas[:, 1].view(1, 8)
        valid_k = (kr >= 0) & (kr < 8) & (kc >= 0) & (kc < 8)
        sqs_k = (kr.clamp(0, 7) * 8 + kc.clamp(0, 7)).long()
        p_k = flat.gather(1, sqs_k)
        knight_att = (valid_k & (p_k == knight_val)).any(dim=1)

        kgr = r_t.view(-1, 1) + self._kg_deltas[:, 0].view(1, 8)
        kgc = c_t.view(-1, 1) + self._kg_deltas[:, 1].view(1, 8)
        valid_kg = (kgr >= 0) & (kgr < 8) & (kgc >= 0) & (kgc < 8)
        sqs_kg = (kgr.clamp(0, 7) * 8 + kgc.clamp(0, 7)).long()
        p_kg = flat.gather(1, sqs_kg)
        king_att = (valid_kg & (p_kg == king_val)).any(dim=1)

        sr = r_t.view(-1, 1, 1) + self._ray_dr * self._ray_steps
        sc = c_t.view(-1, 1, 1) + self._ray_dc * self._ray_steps
        valid_step = (sr >= 0) & (sr < 8) & (sc >= 0) & (sc < 8)
        sqs_slider = (sr.clamp(0, 7) * 8 + sc.clamp(0, 7)).long()
        p_slider = flat.gather(1, sqs_slider.view(B_size, 56)).view(B_size, 8, 7)

        is_occupied = valid_step & (p_slider > 0)
        first_blocker = (torch.cumsum(is_occupied.to(torch.int32), dim=2) == 1) & is_occupied

        straight_threat = (p_slider[:, :4, :] == queen_val.unsqueeze(2)) | (p_slider[:, :4, :] == rook_val.unsqueeze(2))
        diag_threat     = (p_slider[:, 4:, :] == queen_val.unsqueeze(2)) | (p_slider[:, 4:, :] == bishop_val.unsqueeze(2))
        slider_threat   = torch.cat([straight_threat, diag_threat], dim=1)

        slider_att = (first_blocker & slider_threat).any(dim=2).any(dim=1)
        return pawn_att | knight_att | king_att | slider_att

    def is_in_check(self, player, boards=None):
        if boards is None:
            boards = self.boards
        B_size = boards.shape[0]
        flat = boards.view(B_size, 64)

        if isinstance(player, int):
            k_val = 6 if player == 0 else 12
            opp = 1 - player
            k_val_tensor = torch.full((B_size, 1), k_val, dtype=torch.int8, device=self.device)
        else:
            k_val_tensor = torch.where(player == 0, 6, 12).to(torch.int8).unsqueeze(1)
            opp = 1 - player

        king_sqs = (flat == k_val_tensor).max(dim=1).indices
        return self._is_square_attacked(king_sqs, opp, boards)

    def compute_pseudo_legal_masks(self):
        self.pseudo_mask.fill_(-1e9)
        pseudo_mask = self.pseudo_mask

        sq_from = torch.arange(64, device=self.device)
        r_f, c_f = sq_from // 8, sq_from % 8

        pieces = self.boards.view(self.B, 64)
        is_white = (self.players == 0).unsqueeze(1)
        owns = torch.where(is_white, (pieces >= 1) & (pieces <= 6), (pieces >= 7) & (pieces <= 12))

        b_idx, f_idx = torch.nonzero(owns, as_tuple=True)
        if b_idx.numel() == 0:
            return pseudo_mask

        M = b_idx.shape[0]
        p_types = pieces[b_idx, f_idx]
        is_p_white = is_white.squeeze(1)[b_idx]
        pt = torch.where(p_types <= 6, p_types, p_types - 6)
        rf, cf = r_f[f_idx], c_f[f_idx]

        r_t1 = torch.where(is_p_white, rf + 1, rf - 1)
        in_b1 = (r_t1 >= 0) & (r_t1 < 8)
        dest_sq1 = (r_t1 * 8 + cf).clamp(0, 63)
        dest_p1 = pieces[b_idx, dest_sq1]
        pawn_fwd1 = (pt == 1) & in_b1 & (dest_p1 == 0)

        r_t2 = torch.where(is_p_white, rf + 2, rf - 2)
        on_start = torch.where(is_p_white, rf == 1, rf == 6)
        dest_sq2 = (r_t2 * 8 + cf).clamp(0, 63)
        dest_p2 = pieces[b_idx, dest_sq2]
        pawn_fwd2 = (pt == 1) & on_start & (dest_p1 == 0) & (dest_p2 == 0)

        pseudo_mask[b_idx[pawn_fwd1], f_idx[pawn_fwd1] * 64 + dest_sq1[pawn_fwd1]] = 0.0
        pseudo_mask[b_idx[pawn_fwd2], f_idx[pawn_fwd2] * 64 + dest_sq2[pawn_fwd2]] = 0.0

        for dc in [-1, 1]:
            ct = cf + dc
            in_c = (ct >= 0) & (ct < 8)
            dest_sq_cap = (r_t1 * 8 + ct).clamp(0, 63)
            dest_cap = pieces[b_idx, dest_sq_cap]
            has_enemy = torch.where(is_p_white, (dest_cap >= 7) & (dest_cap <= 12), (dest_cap >= 1) & (dest_cap <= 6))
            is_ep = (self.en_passant_sq[b_idx] == dest_sq_cap) & (self.en_passant_sq[b_idx] >= 0)
            pawn_cap = (pt == 1) & in_b1 & in_c & (has_enemy | is_ep)
            pseudo_mask[b_idx[pawn_cap], f_idx[pawn_cap] * 64 + dest_sq_cap[pawn_cap]] = 0.0

        kr = rf.view(-1, 1) + self._k_deltas[:, 0].view(1, 8)
        kc = cf.view(-1, 1) + self._k_deltas[:, 1].view(1, 8)
        valid_k = (pt.view(-1, 1) == 2) & (kr >= 0) & (kr < 8) & (kc >= 0) & (kc < 8)
        to_sq_k = (kr.clamp(0, 7) * 8 + kc.clamp(0, 7)).long()
        b_exp_k = b_idx.view(-1, 1).expand_as(to_sq_k)
        d_p_k = pieces[b_exp_k, to_sq_k]
        d_own_k = torch.where(is_p_white.view(-1, 1), (d_p_k >= 1) & (d_p_k <= 6), (d_p_k >= 7) & (d_p_k <= 12))
        valid_k = valid_k & (~d_own_k)
        f_exp_k = f_idx.view(-1, 1).expand_as(to_sq_k)
        pseudo_mask[b_exp_k[valid_k], f_exp_k[valid_k] * 64 + to_sq_k[valid_k]] = 0.0

        kgr = rf.view(-1, 1) + self._kg_deltas[:, 0].view(1, 8)
        kgc = cf.view(-1, 1) + self._kg_deltas[:, 1].view(1, 8)
        valid_kg = (pt.view(-1, 1) == 6) & (kgr >= 0) & (kgr < 8) & (kgc >= 0) & (kgc < 8)
        to_sq_kg = (kgr.clamp(0, 7) * 8 + kgc.clamp(0, 7)).long()
        b_exp_kg = b_idx.view(-1, 1).expand_as(to_sq_kg)
        d_p_kg = pieces[b_exp_kg, to_sq_kg]
        d_own_kg = torch.where(is_p_white.view(-1, 1), (d_p_kg >= 1) & (d_p_kg <= 6), (d_p_kg >= 7) & (d_p_kg <= 12))
        valid_kg = valid_kg & (~d_own_kg)
        f_exp_kg = f_idx.view(-1, 1).expand_as(to_sq_kg)
        pseudo_mask[b_exp_kg[valid_kg], f_exp_kg[valid_kg] * 64 + to_sq_kg[valid_kg]] = 0.0

        w_in_check = self.is_in_check(0)
        b_in_check = self.is_in_check(1)

        f1_att = self._is_square_attacked(5, 1, self.boards)
        d1_att = self._is_square_attacked(3, 1, self.boards)
        f8_att = self._is_square_attacked(61, 0, self.boards)
        d8_att = self._is_square_attacked(59, 0, self.boards)

        w_k_castle = (self.players == 0) & self.castling_rights[:, 0] & (pieces[:, 5] == 0) & (pieces[:, 6] == 0) & (~w_in_check) & (~f1_att)
        w_q_castle = (self.players == 0) & self.castling_rights[:, 1] & (pieces[:, 1] == 0) & (pieces[:, 2] == 0) & (pieces[:, 3] == 0) & (~w_in_check) & (~d1_att)
        b_k_castle = (self.players == 1) & self.castling_rights[:, 2] & (pieces[:, 61] == 0) & (pieces[:, 62] == 0) & (~b_in_check) & (~f8_att)
        b_q_castle = (self.players == 1) & self.castling_rights[:, 3] & (pieces[:, 57] == 0) & (pieces[:, 58] == 0) & (pieces[:, 59] == 0) & (~b_in_check) & (~d8_att)

        pseudo_mask[w_k_castle, 4 * 64 + 6] = 0.0
        pseudo_mask[w_q_castle, 4 * 64 + 2] = 0.0
        pseudo_mask[b_k_castle, 60 * 64 + 62] = 0.0
        pseudo_mask[b_q_castle, 60 * 64 + 58] = 0.0

        sr = rf.view(-1, 1, 1) + self._ray_dr * self._ray_steps
        sc = cf.view(-1, 1, 1) + self._ray_dc * self._ray_steps
        in_b = (sr >= 0) & (sr < 8) & (sc >= 0) & (sc < 8)
        to_sq = (sr.clamp(0, 7) * 8 + sc.clamp(0, 7)).long()

        b_expanded = b_idx.view(-1, 1, 1).expand_as(to_sq)
        d_piece = pieces[b_expanded, to_sq]
        d_own = torch.where(is_p_white.view(-1, 1, 1), (d_piece >= 1) & (d_piece <= 6), (d_piece >= 7) & (d_piece <= 12))

        dir_valid = torch.zeros(M, 8, 1, dtype=torch.bool, device=self.device)
        dir_valid[:, :4, 0] = ((pt == 4) | (pt == 5)).unsqueeze(1)
        dir_valid[:, 4:, 0] = ((pt == 3) | (pt == 5)).unsqueeze(1)

        has_piece = in_b & (d_piece > 0)
        prev_pieces = F.pad(has_piece[:, :, :-1], (1, 0), value=0)
        cum_prev = torch.cumsum(prev_pieces.to(torch.int32), dim=2)
        can_reach = in_b & (cum_prev == 0)

        can_land = can_reach & dir_valid & (~d_own)
        f_expanded = f_idx.view(-1, 1, 1).expand_as(to_sq)
        pseudo_mask[b_expanded[can_land], f_expanded[can_land] * 64 + to_sq[can_land]] = 0.0
        return pseudo_mask

    def compute_legal_masks(self):
        pseudo_mask = self.compute_pseudo_legal_masks()
        self.legal_mask.fill_(-1e9)
        legal_mask = self.legal_mask

        b_idx, act_idx = torch.nonzero(pseudo_mask > -1e8, as_tuple=True)
        if b_idx.numel() == 0:
            return legal_mask

        from_sqs = act_idx // 64
        to_sqs = act_idx % 64
        turn = self.players[b_idx]
        N_moves = b_idx.shape[0]

        sub_boards = self.boards[b_idx].view(N_moves, 64)
        idx_range = torch.arange(N_moves, device=self.device)
        moving_pieces = sub_boards[idx_range, from_sqs]

        is_pawn = (moving_pieces == 1) | (moving_pieces == 7)
        is_ep = is_pawn & (to_sqs == self.en_passant_sq[b_idx]) & (self.en_passant_sq[b_idx] >= 0)
        if is_ep.any():
            ep_indices = torch.nonzero(is_ep).squeeze(-1)
            ep_cap_sq = torch.where(turn[ep_indices] == 0, to_sqs[ep_indices] - 8, to_sqs[ep_indices] + 8)
            sub_boards[ep_indices, ep_cap_sq] = 0

        sub_boards[idx_range, to_sqs] = moving_pieces
        sub_boards[idx_range, from_sqs] = 0

        # Fast king location: check parent boards (256 items) instead of 10,000 sub-boards
        parent_flat = self.boards.view(self.B, 64)
        parent_k_val = torch.where(self.players == 0, 6, 12).to(torch.int8).unsqueeze(1)
        parent_k_sqs = (parent_flat == parent_k_val).max(dim=1).indices

        is_king_move = (moving_pieces == 6) | (moving_pieces == 12)
        k_sqs = torch.where(is_king_move, to_sqs, parent_k_sqs[b_idx])

        opp = 1 - turn
        in_check = self._is_square_attacked(k_sqs, opp, sub_boards.view(N_moves, 8, 8))
        valid_moves = ~in_check

        legal_mask[b_idx[valid_moves], act_idx[valid_moves]] = 0.0
        return legal_mask

    def step(self, action_indices, dones, winners, legal_mask=None):
        B_idx = torch.arange(self.B, device=self.device)
        from_sq = action_indices // 64
        to_sq = action_indices % 64
        from_r, from_c = from_sq // 8, from_sq % 8
        to_r, to_c = to_sq // 8, to_sq % 8

        piece = self.boards[B_idx, from_r, from_c]
        dest_piece = self.boards[B_idx, to_r, to_c]
        active = ~dones

        if legal_mask is not None:
            has_legal = (legal_mask > -1e8).any(dim=-1)
            no_legal = active & (~has_legal)
            if no_legal.any():
                in_chk_w = self.is_in_check(0)
                in_chk_b = self.is_in_check(1)

                checkmate_w_loses = no_legal & (self.players == 0) & in_chk_w
                checkmate_b_loses = no_legal & (self.players == 1) & in_chk_b
                checkmates = checkmate_w_loses | checkmate_b_loses
                stalemates = no_legal & (~checkmates)

                winners = torch.where(checkmate_w_loses, torch.tensor(2, dtype=torch.int8, device=self.device), winners)
                winners = torch.where(checkmate_b_loses, torch.tensor(1, dtype=torch.int8, device=self.device), winners)
                winners = torch.where(stalemates, torch.tensor(3, dtype=torch.int8, device=self.device), winners)

                dones = dones | checkmates | stalemates
                active = active & (~dones)

        is_pawn_move = active & ((piece == 1) | (piece == 7))
        is_capture = active & (dest_piece != 0)
        is_irreversible = is_pawn_move | is_capture

        is_ep_move = is_pawn_move & (to_sq == self.en_passant_sq) & (self.en_passant_sq >= 0)
        ep_victim_r = torch.where(self.players == 0, to_r - 1, to_r + 1)
        self.boards[B_idx[is_ep_move], ep_victim_r[is_ep_move], to_c[is_ep_move]] = 0

        w_k_cast = active & (piece == 6) & (from_sq == 4) & (to_sq == 6)
        w_q_cast = active & (piece == 6) & (from_sq == 4) & (to_sq == 2)
        b_k_cast = active & (piece == 12) & (from_sq == 60) & (to_sq == 62)
        b_q_cast = active & (piece == 12) & (from_sq == 60) & (to_sq == 58)

        self.boards[B_idx[w_k_cast], 0, 5] = 4; self.boards[B_idx[w_k_cast], 0, 7] = 0
        self.boards[B_idx[w_q_cast], 0, 3] = 4; self.boards[B_idx[w_q_cast], 0, 0] = 0
        self.boards[B_idx[b_k_cast], 7, 5] = 10; self.boards[B_idx[b_k_cast], 7, 7] = 0
        self.boards[B_idx[b_q_cast], 7, 3] = 10; self.boards[B_idx[b_q_cast], 7, 0] = 0

        self.boards[B_idx[active], to_r[active], to_c[active]] = piece[active]
        self.boards[B_idx[active], from_r[active], from_c[active]] = 0

        promo_w = active & (piece == 1) & (to_r == 7)
        promo_b = active & (piece == 7) & (to_r == 0)
        self.boards[B_idx[promo_w], to_r[promo_w], to_c[promo_w]] = 5
        self.boards[B_idx[promo_b], to_r[promo_b], to_c[promo_b]] = 11

        self.castling_rights[active & (from_sq == 4), 0:2] = False
        self.castling_rights[active & (from_sq == 7), 0] = False
        self.castling_rights[active & (from_sq == 0), 1] = False
        self.castling_rights[active & (from_sq == 60), 2:4] = False
        self.castling_rights[active & (from_sq == 63), 2] = False
        self.castling_rights[active & (from_sq == 56), 3] = False

        self.castling_rights[active & (to_sq == 7), 0] = False
        self.castling_rights[active & (to_sq == 0), 1] = False
        self.castling_rights[active & (to_sq == 63), 2] = False
        self.castling_rights[active & (to_sq == 56), 3] = False

        pawn_double_w = is_pawn_move & (from_r == 1) & (to_r == 3)
        pawn_double_b = is_pawn_move & (from_r == 6) & (to_r == 4)
        self.en_passant_sq[active] = -1
        self.en_passant_sq[pawn_double_w] = (from_sq[pawn_double_w] + 8).to(torch.int8)
        self.en_passant_sq[pawn_double_b] = (from_sq[pawn_double_b] - 8).to(torch.int8)

        self.players[active] = 1 - self.players[active]
        self.halfmove_clock[active] += 1
        self.halfmove_clock[is_irreversible] = 0

        self.history_len[is_irreversible] = 0
        curr_hashes = self.compute_hash(self.boards, self.players)
        act_idx = torch.nonzero(active).squeeze(-1)
        if act_idx.numel() > 0:
            curr_lens = self.history_len[act_idx].clamp(max=self.max_history_len - 1).long()
            self.history_hashes[act_idx, curr_lens] = curr_hashes[act_idx]
            self.history_len[act_idx] = (curr_lens + 1).to(torch.int16)

        w_mat, b_mat = self.get_material()
        w_resigns = active & ((b_mat >= w_mat + 1200) | ((w_mat == 0) & (b_mat > 300)))
        b_resigns = active & ((w_mat >= b_mat + 1200) | ((b_mat == 0) & (w_mat > 300)))

        winners = torch.where(b_resigns, torch.tensor(1, dtype=torch.int8, device=self.device), winners)
        winners = torch.where(w_resigns, torch.tensor(2, dtype=torch.int8, device=self.device), winners)
        decisive = w_resigns | b_resigns
        active = active & (~decisive)

        step_idx = torch.arange(self.max_history_len, device=self.device).unsqueeze(0)
        valid_history = step_idx < self.history_len.unsqueeze(1)
        matches = (self.history_hashes == curr_hashes.unsqueeze(1)) & valid_history
        rep_count = matches.sum(dim=-1)

        turn = self.players
        side_lead = torch.where(turn == 0, w_mat - b_mat, b_mat - w_mat)
        claim_rep_3 = (rep_count >= 3) & (side_lead <= 150)
        auto_rep_5  = (rep_count >= 5)
        draw_rep = active & (claim_rep_3 | auto_rep_5)
        draw_50 = active & (self.halfmove_clock >= self.halfmove_limit)
        draw_insufficient = active & self.is_insufficient_material()

        draws = draw_rep | draw_50 | draw_insufficient
        winners = torch.where(draws, torch.tensor(3, dtype=torch.int8, device=self.device), winners)
        dones = dones | decisive | draws
        return dones, winners
    
def resolve_fittest_agent_states(
    finished_mask, winners, match_modes,
    agent_state, agent_output, agent_error, agent_lp,
    state_w, output_w, error_w, lp_w,
    state_b, output_b, error_b, lp_b,
    p_loser_survives=0.00
):
    if not finished_mask.any():
        return

    w_won = winners == 1
    b_won = winners == 2
    draw  = winners == 3

    B = finished_mask.shape[0]
    device = finished_mask.device

    self_play = (match_modes == 0) | (match_modes == 2)
    draw_coin = torch.rand(B, device=device) < 0.5
    keep_loser = torch.rand(B, device=device) < p_loser_survives

    white_survives = finished_mask & (
        (match_modes == 3) | 
        (self_play & ((w_won & ~keep_loser) | (b_won & keep_loser) | (draw & draw_coin)))
    )

    black_survives = finished_mask & (
        (match_modes == 1) | 
        (self_play & ((b_won & ~keep_loser) | (w_won & keep_loser) | (draw & ~draw_coin)))
    )

    if white_survives.any():
        agent_state[:, white_survives, :]  = state_w[:, white_survives, :]
        agent_output[:, white_survives, :] = output_w[:, white_survives, :]
        agent_error[:, white_survives]     = error_w[:, white_survives]
        agent_lp[:, white_survives]        = lp_w[:, white_survives]

    if black_survives.any():
        agent_state[:, black_survives, :]  = state_b[:, black_survives, :]
        agent_output[:, black_survives, :] = output_b[:, black_survives, :]
        agent_error[:, black_survives]     = error_b[:, black_survives]
        agent_lp[:, black_survives]        = lp_b[:, black_survives]

# ==========================================
# 2. CONTINUOUS AGENT ALLOSTASIS
# ==========================================
def update_agent_somatic_noise(agent_noise, finished_mask, winners, match_modes):
    if not finished_mask.any():
        return

    w_won = winners == 1
    b_won = winners == 2
    draw  = winners == 3

    agent_won = (
        ((match_modes == 3) & w_won) | 
        ((match_modes == 1) & b_won) | 
        (((match_modes == 0) | (match_modes == 2)) & (w_won | b_won))
    )

    agent_failed = (
        ((match_modes == 3) & b_won) | 
        ((match_modes == 1) & w_won) | 
        draw                           
    )

    win_mask = finished_mask & agent_won
    fail_mask = finished_mask & agent_failed

    agent_noise[win_mask] = torch.clamp(
        agent_noise[win_mask] * 0.80 - 0.003, 
        min=0.01, 
        max=noise_cap
    )

    agent_noise[fail_mask] = torch.clamp(
        agent_noise[fail_mask] * 1.10 + 0.003, 
        min=0.01, 
        max=noise_cap
    )

# ==========================================
# 3. BIDIRECTIONAL CUDA TACTICAL ENGINE
# ==========================================
_ENGINE_CONSTS = {}

def get_engine_constants(device):
    if device in _ENGINE_CONSTS:
        return _ENGINE_CONSTS[device]
    to_sq = torch.arange(4096, device=device) % 64
    from_sq = torch.arange(4096, device=device) // 64
    dest_r, dest_c = to_sq // 8, to_sq % 8
    from_r, from_c = from_sq // 8, from_sq % 8
    center_dist = ((dest_r.float() - 3.5).abs() + (dest_c.float() - 3.5).abs())
    pos_bonus = (7.0 - center_dist) * 2.5
    is_sweet_center = ((dest_r == 3) | (dest_r == 4)) & ((dest_c == 3) | (dest_c == 4))
    pos_bonus = pos_bonus + torch.where(is_sweet_center, torch.tensor(15.0, device=device), torch.zeros_like(pos_bonus))

    consts = {
        'to_sq': to_sq, 'from_sq': from_sq,
        'dest_r': dest_r, 'dest_c': dest_c,
        'from_r': from_r, 'from_c': from_c,
        'pos_bonus': pos_bonus
    }
    _ENGINE_CONSTS[device] = consts
    return consts

def select_cuda_engine_move(boards, legal_mask, is_engine_white, device, env=None, b_idx=None):
    B = boards.shape[0]
    c = get_engine_constants(device)
    to_sq, from_sq = c['to_sq'], c['from_sq']
    dest_r, dest_c = c['dest_r'], c['dest_c']
    from_r, from_c = c['from_r'], c['from_c']

    from_sq_b = from_sq.unsqueeze(0).expand(B, -1)
    to_sq_b   = to_sq.unsqueeze(0).expand(B, -1)

    flat_boards = boards.view(B, 64)
    target_pieces = flat_boards.gather(1, to_sq_b).long().clamp(0, 12)
    moving_pieces = flat_boards.gather(1, from_sq_b).long().clamp(0, 12)

    piece_vals = torch.tensor([0, 100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000], dtype=torch.float32, device=device)

    is_w = is_engine_white.unsqueeze(1)
    is_enemy = torch.where(is_w, (target_pieces >= 7) & (target_pieces <= 12), (target_pieces >= 1) & (target_pieces <= 6))

    target_val = piece_vals[target_pieces]
    moving_val = piece_vals[moving_pieces]
    capture_scores = torch.where(is_enemy, target_val * 2.0 - moving_val * 0.1 + 100.0, torch.zeros_like(target_val))

    dest_rf = dest_r.unsqueeze(0).expand(B, -1)
    dest_cf = dest_c.unsqueeze(0).expand(B, -1)
    attacker_r = torch.where(is_w, (dest_rf + 1).clamp(0, 7), (dest_rf - 1).clamp(0, 7))
    enemy_pawn = torch.where(is_w, torch.tensor(7, device=device), torch.tensor(1, device=device))

    left_c = (dest_cf - 1).clamp(0, 7)
    right_c = (dest_cf + 1).clamp(0, 7)
    batch_sub_idx = torch.arange(B, device=device).unsqueeze(1)

    pawn_threat_left = (dest_cf > 0) & (boards[batch_sub_idx, attacker_r, left_c] == enemy_pawn)
    pawn_threat_right = (dest_cf < 7) & (boards[batch_sub_idx, attacker_r, right_c] == enemy_pawn)
    r_valid = torch.where(is_w, dest_rf < 7, dest_rf > 0)
    under_pawn_threat = (pawn_threat_left | pawn_threat_right) & r_valid

    is_own = torch.where(is_w, (moving_pieces >= 1) & (moving_pieces <= 6), (moving_pieces >= 7) & (moving_pieces <= 12))
    piece_risk = torch.where(is_own, moving_val, torch.zeros_like(moving_val))
    safety_penalty = torch.where(under_pawn_threat, piece_risk * 0.9, torch.zeros_like(piece_risk))

    is_pawn = (moving_pieces == 1) | (moving_pieces == 7)
    pawn_dir = torch.where(is_w, dest_r.float() - from_r.float(), from_r.float() - dest_r.float())
    pawn_rank = torch.where(is_w, dest_r.float(), 7.0 - dest_r.float())
    pawn_push_bonus = torch.where(is_pawn & (pawn_dir > 0), pawn_dir * 12.0 + pawn_rank * 6.0, torch.zeros_like(pawn_dir))
    is_promo = is_pawn & torch.where(is_w, dest_r == 7, dest_r == 0)
    promo_bonus = torch.where(is_promo, torch.tensor(800.0, device=device), torch.zeros_like(pawn_dir))

    is_minor = (moving_pieces == 2) | (moving_pieces == 3) | (moving_pieces == 8) | (moving_pieces == 9)
    developed = torch.where(is_w, (from_r == 0) & (dest_r > 0), (from_r == 7) & (dest_r < 7))
    develop_bonus = torch.where(is_minor & developed, torch.tensor(30.0, device=device), torch.zeros_like(pawn_dir))

    is_castling = (
        (is_w & (moving_pieces == 6) & (from_sq == 4) & ((to_sq == 6) | (to_sq == 2))) |
        ((~is_w) & (moving_pieces == 12) & (from_sq == 60) & ((to_sq == 62) | (to_sq == 58)))
    )
    castle_bonus = torch.where(is_castling, torch.tensor(80.0, device=device), torch.zeros_like(pawn_dir))

    is_king = (moving_pieces == 6) | (moving_pieces == 12)
    king_wander_penalty = torch.where(is_king & (~is_castling), torch.tensor(25.0, device=device), torch.zeros_like(pawn_dir))
    pos_bonus = c['pos_bonus']

    if env is not None and b_idx is not None:
        curr_lens = env.history_len[b_idx]
        curr_hash = env.history_hashes[b_idx, (curr_lens - 1).clamp(min=0).long()].unsqueeze(1)

        p_after = torch.where(
            is_w & (dest_r == 7) & (moving_pieces == 1),
            torch.tensor(5, dtype=torch.long, device=device),
            torch.where((~is_w) & (dest_r == 0) & (moving_pieces == 7), torch.tensor(11, dtype=torch.long, device=device), moving_pieces)
        )

        z_from_sub = env.zobrist_pieces[from_sq_b, moving_pieces]
        z_from_add = env.zobrist_pieces[from_sq_b, 0]
        z_to_sub   = env.zobrist_pieces[to_sq_b, target_pieces]
        z_to_add   = env.zobrist_pieces[to_sq_b, p_after]
        turn_flip  = torch.where(env.players[b_idx] == 0, env.zobrist_turn, -env.zobrist_turn).unsqueeze(1)

        delta = - z_from_sub + z_from_add - z_to_sub + z_to_add + turn_flip
        cand_hash = curr_hash + delta

        lookback = 16
        step_idx = torch.arange(env.max_history_len, device=device).unsqueeze(0)
        valid = step_idx < curr_lens.unsqueeze(1)

        idx_start = (curr_lens - lookback).clamp(min=0)
        gather_indices = (idx_start.unsqueeze(1) + torch.arange(lookback, device=device).unsqueeze(0)).clamp(max=env.max_history_len - 1)
        recent_hist = torch.gather(env.history_hashes[b_idx], 1, gather_indices)
        recent_valid = torch.gather(valid, 1, gather_indices)

        dummy_hash = torch.tensor(-9223372036854775807, device=device, dtype=torch.int64)
        recent_hist = torch.where(recent_valid, recent_hist, dummy_hash)

        is_repeat = (cand_hash.unsqueeze(-1) == recent_hist.unsqueeze(1)).any(dim=-1)
        repetition_penalty = torch.where(is_repeat, torch.tensor(400.0, device=device), torch.zeros(1, device=device))

        half_clock = env.halfmove_clock[b_idx].float().unsqueeze(1)
        clock_urgency = F.relu(half_clock - 50.0) * 3.5
        clock_bonus = torch.where(is_pawn | is_enemy, clock_urgency, torch.zeros(1, device=device))
    else:
        repetition_penalty = 0.0
        clock_bonus = 0.0

    move_scores = (
        capture_scores - safety_penalty + pos_bonus + pawn_push_bonus +
        promo_bonus + develop_bonus + castle_bonus - king_wander_penalty -
        repetition_penalty + clock_bonus +
        torch.randn(B, 4096, device=device) * 0.5 + legal_mask
    )
    return torch.argmax(move_scores, dim=-1)

# ==========================================
# 4. STOCKFISH INTERFACE & BENCHMARK
# ==========================================
PIECE_TO_FEN = {
    0: '', 1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
    7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'
}

def board_to_fen(board_tensor, player_turn, castling_rights, ep_sq=-1, halfmove_clock=0, fullmove_num=1):
    fen_ranks = []
    for r in range(7, -1, -1):
        empty_count = 0
        r_str = ""
        for c in range(8):
            p = int(board_tensor[r, c].item())
            if p == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    r_str += str(empty_count)
                    empty_count = 0
                r_str += PIECE_TO_FEN.get(p, '')
        if empty_count > 0:
            r_str += str(empty_count)
        fen_ranks.append(r_str)
    turn_str = 'w' if player_turn == 0 else 'b'

    c_str = ""
    if castling_rights[0]: c_str += "K"
    if castling_rights[1]: c_str += "Q"
    if castling_rights[2]: c_str += "k"
    if castling_rights[3]: c_str += "q"
    if not c_str: c_str = "-"

    if ep_sq >= 0:
        ep_c = chr(ord('a') + (ep_sq % 8))
        ep_r = str((ep_sq // 8) + 1)
        ep_str = f"{ep_c}{ep_r}"
    else:
        ep_str = "-"

    return f"{'/'.join(fen_ranks)} {turn_str} {c_str} {ep_str} {halfmove_clock} {fullmove_num}"

def uci_to_action(uci_str):
    if not uci_str or len(uci_str) < 4:
        return None
    try:
        cf = ord(uci_str[0]) - ord('a')
        rf = int(uci_str[1]) - 1
        ct = ord(uci_str[2]) - ord('a')
        rt = int(uci_str[3]) - 1
        return (rf * 8 + cf) * 64 + (rt * 8 + ct)
    except Exception:
        return None

class StockfishEngine:
    def __init__(self, stockfish_path="stockfish", threads=1, hash_mb=16):
        self.proc = subprocess.Popen(
            [stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        self._send("uci")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("setoption name UCI_LimitStrength value false")
        self._send("isready")
        while True:
            line = self.proc.stdout.readline()
            if "readyok" in line:
                break

    def _send(self, cmd):
        if self.proc.poll() is None:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()

    def get_move(self, fen, nodes=1500):
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) > 1 and parts[1] not in ("(none)", "0000"):
                    return parts[1]
                return None

    def close(self):
        try:
            self._send("quit")
            self.proc.terminate()
        except Exception:
            pass

# ==========================================
# 5. LIVE STOCKFISH TRAINING BATCH
# ==========================================
# ==========================================
# 5. LIVE STOCKFISH TRAINING BATCH (PERSISTENT STRESS FIXED)
# ==========================================
def train_stockfish_batch(
    model,
    dust_lr=0.001,
    num_games=32,
    target_nodes=1500,
    thinking_steps=8,
    max_plies=250,
    device="cuda",
    agent_noise_slice=None,
    agent_state_slice=None,
    agent_output_slice=None,
    agent_error_slice=None,
    agent_lp_slice=None
):
    sf_exec = shutil.which("stockfish")
    if sf_exec is None:
        print("[Stockfish Train Batch] 'stockfish' executable not found in PATH. Skipping.")
        return agent_noise_slice, agent_state_slice, agent_output_slice, agent_error_slice, agent_lp_slice

    print(f"\n---> [LIVE TRAINING ({TRAINING_MODE.upper()})] Executing Stockfish Training Batch ({num_games} matches | SF Nodes: {target_nodes})...")
    sf = StockfishEngine(stockfish_path=sf_exec)
    sf_env = GuardrailedTorchChess(batch_size=num_games, device=device)

    # 1. Inherit real somatic noise from the persistent population (or fallback if standalone)
    if agent_noise_slice is not None:
        batch_noise = agent_noise_slice.clone()
    else:
        batch_noise = torch.full((num_games,), 0.05, dtype=torch.float32, device=device)

    agent_is_white = torch.zeros(num_games, dtype=torch.bool, device=device)
    agent_is_white[:num_games // 2] = True
    match_modes = torch.where(agent_is_white, torch.tensor(3, device=device), torch.tensor(1, device=device)).to(torch.int8)

    # 2. Inherit persistent brain states from the selected population
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if agent_state_slice is not None:
        state_buf = agent_state_slice.clone()
        output_buf = agent_output_slice.clone() if agent_output_slice is not None else torch.zeros_like(state_buf)
        error_buf = agent_error_slice.clone() if agent_error_slice is not None else torch.ones(raw_model.N, num_games, device=device) * 0.05
        lp_buf = agent_lp_slice.clone() if agent_lp_slice is not None else torch.zeros(raw_model.N, num_games, device=device)
    else:
        state_buf, output_buf, error_buf, lp_buf = raw_model.init_states(batch_size=num_games)

    dones = torch.zeros(num_games, dtype=torch.bool, device=device)
    winners = torch.zeros(num_games, dtype=torch.int8, device=device)
    plies = torch.zeros(num_games, dtype=torch.long, device=device)

    total_loss_accum = 0.0
    loss_steps_count = 0

    while not dones.all() and (plies < max_plies).any():
        active = ~dones
        legal_masks = sf_env.compute_legal_masks()
        turns = sf_env.players
        is_agent_turn = active & ((turns == 0) == agent_is_white)
        is_sf_turn = active & ~is_agent_turn

        actions = torch.zeros(num_games, dtype=torch.long, device=device)

        if is_agent_turn.any():
            ag_idx = torch.nonzero(is_agent_turn).squeeze(-1)
            env_input, threat = sf_env.get_obs(agent_noise=batch_noise)

            acts, next_s, next_o, next_e, next_l, ply_loss, _ = deliberate_and_train_ply(
                model=model,
                dust_lr=dust_lr,
                env_input=env_input,
                threat=threat,
                state_buf=state_buf,
                output_buf=output_buf,
                error_ema_buf=error_buf,
                lp_ema_buf=lp_buf,
                active_idx=ag_idx,
                legal_mask=legal_masks,
                thinking_steps=thinking_steps
            )

            actions[ag_idx] = acts
            state_buf[:, ag_idx, :] = next_s
            output_buf[:, ag_idx, :] = next_o
            error_buf[:, ag_idx] = next_e
            lp_buf[:, ag_idx] = next_l

            total_loss_accum += ply_loss.item()
            loss_steps_count += 1

        if is_sf_turn.any():
            sf_indices = torch.nonzero(is_sf_turn).squeeze(-1).tolist()
            for i in sf_indices:
                board_i = sf_env.boards[i]
                turn_i = int(turns[i].item())
                half_c = int(sf_env.halfmove_clock[i].item())
                full_n = int(plies[i].item() // 2 + 1)
                c_rights = sf_env.castling_rights[i].tolist()
                ep = int(sf_env.en_passant_sq[i].item())
                fen = board_to_fen(board_i, turn_i, c_rights, ep_sq=ep, halfmove_clock=half_c, fullmove_num=full_n)

                legal_acts = torch.nonzero(legal_masks[i] > -1e8).squeeze(-1).tolist()
                if len(legal_acts) == 0:
                    dones[i] = True
                    winners[i] = 3
                    continue

                best_m = sf.get_move(fen, nodes=target_nodes)
                act = uci_to_action(best_m) if best_m else None
                if act is None or act not in legal_acts:
                    act = legal_acts[0]
                actions[i] = act

        dones, winners = sf_env.step(actions, dones, winners, legal_mask=legal_masks)
        plies[active] += 1

        overlength = (plies >= max_plies) & (~dones)
        if overlength.any():
            dones = dones | overlength
            winners = torch.where(overlength, torch.tensor(3, dtype=torch.int8, device=device), winners)

    sf.close()
    
    # Update somatic noise based on win/loss against Stockfish
    update_agent_somatic_noise(batch_noise, dones, winners, match_modes)

    agent_won = (((winners == 1) & agent_is_white) | ((winners == 2) & (~agent_is_white))).sum().item()
    sf_won = (((winners == 2) & agent_is_white) | ((winners == 1) & (~agent_is_white))).sum().item()
    draws_count = (winners == 3).sum().item()
    avg_loss = total_loss_accum / max(1, loss_steps_count)

    print(f"[Stockfish Train Batch Complete] "
          f"Agent Wins: {agent_won} | SF Wins: {sf_won} | Draws: {draws_count} | "
          f"Mean Loss: {avg_loss:.4f} | AvgPlies: {plies.float().mean().item():4.1f} | Stress: {batch_noise.mean().item():.3f}\n")

    # Return updated somatic noise and state tensors to update caller's persistent memory
    return batch_noise, state_buf, output_buf, error_buf, lp_buf
# ==========================================
# 6. NODE NETWORK (NATIVE N, B, D ARCHITECTURE)
# ==========================================
USE_DIMENSIONLESS_BOREDOM = True
USE_ENTROPIC_RESTORING_FORCE = True
GAMMA_COUPLING = 2.0

DEFAULT_HYPERPARAMS = {
    'UNTANH': 1.725268,
    'w_scale': 0.548835,
    'EMA_SPEED': 0.071671,
    'REW_SENS': 1.330760,
    'speed': 1.355162,
    'width': 0.001000,
    'sprs': 0.580194,
    'c_moment': 0.905514,
    'a_variance': 0.05,
    'opt_gd_lr': 0.001,
}

class HyperParamsConfig:
    def __init__(self, device):
        self.params = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in DEFAULT_HYPERPARAMS.items()}
    def get(self, key): 
        return self.params[key]

def color_noise(size, center, width, device):
    noise = torch.randn(size, device=device)
    return utils.apply_spectral_filter(noise, center_offset=center, sigma_val=width)

class NodeParams3(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, device):
        super().__init__()
        self.N = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        base_weight = color_noise((num_nodes, in_dim, out_dim), 0.5, 0.2, device=device) / math.sqrt(in_dim)
        self.weight = nn.Parameter(base_weight, requires_grad=False)
        self.register_buffer('untanh_slope', torch.tensor(0.5, device=device).view(1, 1, 1))
        self.register_buffer('w_scale', torch.tensor(0.8, device=device).view(1, 1, 1))
        self.normalize_weight()

    def normalize_weight(self):
        with torch.no_grad():
            self.weight.data.copy_(F.normalize(self.weight.data, p=2, dim=-2) * self.w_scale)

    def forward(self, x, act_jitter=None):
        # x is natively (N, B, in_dim)
        # weight is (N, in_dim, out_dim) - already normalized and scaled
        raw_pred = torch.bmm(x, self.weight)
        if act_jitter is not None:
            raw_pred = raw_pred + act_jitter
        return activation(raw_pred, self.untanh_slope, math.sqrt(2.0 + math.sqrt(2.0)))

    def apply_hyperparams(self, config):
        self.untanh_slope.copy_(config.get('UNTANH').view(1, 1, 1))
        self.w_scale.copy_(config.get('w_scale').view(1, 1, 1))
        self.normalize_weight()

class LinearBusRouting(nn.Module):
    def __init__(self, num_nodes: int, dim: int = 32, rank_k: int = 16, device="cuda"):
        super().__init__()
        self.N = num_nodes
        self.D = dim
        self.K = rank_k
        self.w_qk = nn.Parameter(
            torch.randn(num_nodes, dim, 2 * rank_k, device=device) * 0.50, 
            requires_grad=False
        )

    def forward(self, state: torch.Tensor, output: torch.Tensor, 
                q_jitter: torch.Tensor = None, k_jitter: torch.Tensor = None, 
                sprs: torch.Tensor = None):
        global_mean = output.mean(dim=0, keepdim=True) # (1, B, D)

        raw_qk = torch.bmm(state, self.w_qk)
        raw_q, raw_k = raw_qk.chunk(2, dim=-1)

        if q_jitter is not None:
            raw_q = raw_q + q_jitter
        Q = F.elu(raw_q) + 1.0 # (N, B, K)

        if k_jitter is not None:
            raw_k = raw_k + k_jitter
        K = F.elu(raw_k) + 1.0 # (N, B, K)

        K_b = K.permute(1, 2, 0)
        Out_b = output.permute(1, 0, 2)
        blackboard = torch.bmm(K_b, Out_b)
        norm_k = K_b.sum(dim=-1, keepdim=True) # (B, K, 1)

        Q_b = Q.permute(1, 0, 2)
        numerator = torch.bmm(Q_b, blackboard)
        denominator = torch.bmm(Q_b, norm_k) + 1e-6
        bus_diffusion = (numerator / denominator).permute(1, 0, 2) # (N, B, D)

        return torch.lerp(bus_diffusion, global_mean, sprs)

class ChessNodeNet(nn.Module):
    def __init__(self, num_nodes=32, dim=32, input_dim=100, rank_k=16, device="cuda"):
        super().__init__()
        self.N, self.D, self.device = num_nodes, dim, device
        self.num_input_nodes = math.ceil(input_dim / dim)  # 4 nodes (128 dims)
        self.motor_nodes_count = 4                        # 4 nodes (128 dims)

        # 1. Unified Sensory & Reflex Projection (WEIGHT TIED)
        # Both sensory boundary (nodes 0..3) and motor reflex (nodes 28..31) share the exact same encoder
        self.sensory_proj = nn.Linear(input_dim, self.num_input_nodes * dim, bias=False, device=device)
        nn.init.orthogonal_(self.sensory_proj.weight, gain=1.0)
        self.sensory_proj.weight.requires_grad = False
        
        # Tied reference: reflex_proj points directly to sensory_proj
        self.reflex_proj = self.sensory_proj

        self.w1 = NodeParams3(num_nodes, dim, dim * 2, device)
        self.router = LinearBusRouting(num_nodes, dim, rank_k=rank_k, device=device)

        self.lp_alpha = 0.25
        self.register_buffer('sprs', torch.tensor(0.5, device=device).view(1, 1, 1))
        self.register_buffer('a_variance', torch.tensor(0.05, device=device).view(1, 1, 1))

        # Action Temperature Scale (Sharpens cosine similarities into [-9.0, +9.0] logit spread)
        self.register_buffer('action_logit_scale', torch.tensor(6.0, device=device))

        # Precompute circulant matrix operators
        self.register_buffer('circ_filter_1_0', make_circulant_matrix(dim, 1.0, 1.0, device=device))
        self.register_buffer('circ_filter_0_25', make_circulant_matrix(dim, 1.0, 0.25, device=device))

        self.normalize_sensory_weight()

    def normalize_sensory_weight(self):
        """Prevents sensory channels from drifting and keeps square embeddings on the unit sphere."""
        with torch.no_grad():
            # Normalize square projection channels (cols 0..63)
            sq_w = self.sensory_proj.weight[:, :64]
            normed_sq = F.normalize(sq_w, p=2, dim=0)
            self.sensory_proj.weight[:, :64].copy_(normed_sq)
            self.update_cached_spatial()

    def update_cached_spatial(self):
        """Ties from_enc and to_enc to a single unified square embedding and computes normalized affinity."""
        with torch.no_grad():
            # 2. Unified Square Embedding (WEIGHT TIED: From-Square <-> To-Square)
            # We take the 64-dim representation from sensory features for all 64 board squares
            self.square_enc = self.sensory_proj.weight[:64, :64]  # (64, 64)
            
            # Unit-normalize columns so no square gets an unfair baseline logit bias
            norm_sq = F.normalize(self.square_enc, p=2, dim=0)
            self.normalized_square_enc = norm_sq

            # 3. Normalized Spatial Affinity (Symmetric Gram Matrix)
            # Diagonal is 1.0; off-diagonals are true cosine similarities in [-1.0, 1.0]
            self.spatial_affinity = torch.matmul(norm_sq.T, norm_sq)

    def init_states(self, batch_size):
        state = 1e-4 * color_noise((self.N, batch_size, self.D), 1.0, 1.0, device=self.device)
        output = 1e-4 * color_noise((self.N, batch_size, self.D), 1.0, 1.0, device=self.device)
        error_ema = torch.ones(self.N, batch_size, device=self.device) * 0.05
        lp_ema = torch.zeros(self.N, batch_size, device=self.device)
        return state, output, error_ema, lp_ema

    def apply_hyperparams(self, config):
        self.sprs.copy_(config.get('sprs').view(1, 1, 1))
        self.a_variance.copy_(config.get('a_variance').view(1, 1, 1))
        self.lp_alpha = float(config.get('EMA_SPEED').cpu())
        self.w1.apply_hyperparams(config)

    def get_action_logits(self, state):
        PB = state.shape[1]
        motor_vector = state[-self.motor_nodes_count:, :, :].permute(1, 0, 2).reshape(PB, self.motor_nodes_count * self.D)
        from_intent, to_intent = torch.chunk(motor_vector, 2, dim=-1)  # (PB, 64) each

        # -------------------------------------------------------------
        # COSINE-NORMALIZED ACTION HEAD (Weight-Tied & Drift-Proof)
        # -------------------------------------------------------------
        # L2-normalize intent vectors: prevents logit collapse/explosion
        norm_from = F.normalize(from_intent, p=2, dim=-1)  # (PB, 64)
        norm_to   = F.normalize(to_intent,   p=2, dim=-1)  # (PB, 64)

        # Both intents query the EXACT SAME unified square representation:
        from_logits = torch.matmul(norm_from, self.normalized_square_enc)  # (PB, 64) in [-1, 1]
        to_logits   = torch.matmul(norm_to,   self.normalized_square_enc)  # (PB, 64) in [-1, 1]

        # Normalized Coherence between origin and destination intents
        coherence_scalar = torch.sum(norm_from * norm_to, dim=-1, keepdim=True).unsqueeze(-1)  # in [-1, 1]
        
        # Joint scores are strictly bounded before temperature scaling
        joint_scores = from_logits.unsqueeze(2) + to_logits.unsqueeze(1) + coherence_scalar * self.spatial_affinity.unsqueeze(0)
        
        # Scale by action temperature factor
        return (joint_scores * self.action_logit_scale).view(PB, 4096)

    def forward(self, env_input, threat, state, output, error_ema, lp_ema,
                thinking_steps: int = 1,
                act_jitter=None, q_jitter=None, k_jitter=None,
                sensory_jitter=None, reflex_jitter=None,
                precomputed_latents=None):
        
        if thinking_steps <= 1:
            clean_loss, n_state, n_out, n_err, n_lp, last_val = self._forward(
                env_input, threat, state, output, error_ema, lp_ema,
                act_jitter=act_jitter, q_jitter=q_jitter, k_jitter=k_jitter,
                sensory_jitter=sensory_jitter, reflex_jitter=reflex_jitter,
                precomputed_latents=precomputed_latents
            )
            return clean_loss, n_state, n_out, n_err, n_lp, last_val, n_state

        accum_loss = 0.0
        accum_val = 0.0
        accum_state = torch.zeros_like(state)

        curr_state = state
        curr_out = output
        curr_err = error_ema
        curr_lp = lp_ema

        for _ in range(thinking_steps):
            step_loss, curr_state, curr_out, curr_err, curr_lp, step_val = self._forward(
                env_input, threat, curr_state, curr_out, curr_err, curr_lp,
                act_jitter=act_jitter, q_jitter=q_jitter, k_jitter=k_jitter,
                sensory_jitter=sensory_jitter, reflex_jitter=reflex_jitter,
                precomputed_latents=precomputed_latents
            )
            accum_loss = accum_loss + step_loss
            accum_val = accum_val + step_val
            accum_state = accum_state + curr_state

        inv_k = 1.0 / thinking_steps
        trajectory_loss = accum_loss * inv_k
        trajectory_val  = accum_val * inv_k
        mean_trajectory_state = accum_state * inv_k

        return trajectory_loss, curr_state, curr_out, curr_err, curr_lp, trajectory_val, mean_trajectory_state

    def _forward(self, env_input, threat, state, output, error_ema, lp_ema, 
                act_jitter=None, q_jitter=None, k_jitter=None,
                sensory_jitter=None, reflex_jitter=None,
                precomputed_latents=None):
        PB = env_input.shape[0]
        h = threat.view(1, PB, 1)

        # -------------------------------------------------------------
        # TIED SENSORY & REFLEX BOUNDARY COMPUTATION
        # -------------------------------------------------------------
        if precomputed_latents is not None and sensory_jitter is None and reflex_jitter is None:
            sensory_latent, motor_reflex = precomputed_latents
        else:
            raw_s = self.sensory_proj(env_input)
            if sensory_jitter is not None:
                raw_s = raw_s + sensory_jitter
            
            # Unit-normalized projection through softsign
            sensory_latent = utils.softsign(raw_s.view(PB, self.num_input_nodes, self.D).permute(1, 0, 2))
            
            # Reflex shares the exact same sensory feature mapping
            if reflex_jitter is not None:
                raw_r = raw_s + reflex_jitter
                motor_reflex = utils.softsign(raw_r.view(PB, self.motor_nodes_count, self.D).permute(1, 0, 2))
            else:
                motor_reflex = sensory_latent

        white_noise = torch.randn_like(output)
        filtered_noise = torch.matmul(white_noise, self.circ_filter_1_0)
        base_noise = torch.sqrt(self.a_variance) * filtered_noise

        relative_lp = torch.clamp(F.relu(lp_ema) / (error_ema + 1e-6), min=0.0, max=1.0)
        boredom_scale = torch.pow(1.0 - relative_lp, GAMMA_COUPLING / 2.0).unsqueeze(-1)

        target = self.router(state, output, q_jitter=q_jitter, k_jitter=k_jitter, sprs=self.sprs) + base_noise * boredom_scale
        
        if USE_ENTROPIC_RESTORING_FORCE:
            target = target - (GAMMA_COUPLING * 0.01) * state * (1.0 - boredom_scale)

        target = utils.softsign(target)
        target = torch.cat([
            sensory_latent, 
            target[self.num_input_nodes:-self.motor_nodes_count, :, :],
            utils.softsign(target[-self.motor_nodes_count:, :, :] + 1.5 * motor_reflex)
        ], dim=0)

        prediction = self.w1(state, act_jitter=act_jitter)
        mean, error_pred = torch.chunk(prediction, 2, dim=-1)
        loss_pred = F.mse_loss(mean, target, reduction='none')
        loss_err_pred = F.mse_loss(error_pred, torch.abs(target - mean), reduction='none')
        loss_w1 = (loss_pred + 0.1 * loss_err_pred) * (1.0 + 2.0 * h)

        current_error = loss_w1.mean(dim=-1)
        ratio_lp = torch.log((error_ema + 1e-6) / (current_error + 1e-6))
        next_error_ema = error_ema * (1.0 - self.lp_alpha) + current_error * self.lp_alpha
        next_lp_ema = lp_ema * (1.0 - self.lp_alpha) + ratio_lp * self.lp_alpha

        total_loss = loss_w1.mean()

        next_relative_lp = torch.clamp(F.relu(next_lp_ema) / (next_error_ema + 1e-6), min=0.0, max=1.0)
        next_boredom_scale = torch.pow(1.0 - next_relative_lp, GAMMA_COUPLING / 2.0).unsqueeze(-1)

        omega = h * next_boredom_scale
        v_pragmatic = target - state

        target_noise = torch.randn_like(target)
        filtered_target_noise = torch.matmul(target_noise, self.circ_filter_0_25)

        v_epistemic = filtered_target_noise * omega * 0.40
        alpha = 0.25 
        settled_state = utils.softsign(state + alpha * v_pragmatic + (alpha * 0.5) * v_epistemic)
        
        next_state = torch.cat([sensory_latent, settled_state[self.num_input_nodes:, :, :]], dim=0)
        next_output = mean

        any_jitter = (act_jitter is not None) or (q_jitter is not None) or (k_jitter is not None) or (sensory_jitter is not None) or (reflex_jitter is not None)
        if any_jitter:
            return total_loss, next_state, next_output, next_error_ema, next_lp_ema, loss_w1.mean(dim=-1, keepdim=True)
            
        return total_loss, next_state, next_output, next_error_ema, next_lp_ema, omega.squeeze(-1)
      
def select_agent_actions(model, state, legal_mask):
    raw_logits = model.get_action_logits(state)
    masked_logits = raw_logits + legal_mask
    return torch.argmax(masked_logits, dim=-1)

PLY_STEP_COUNTER = 0

def deliberate_and_train_ply(
    model,
    dust_lr,
    env_input,
    threat,
    state_buf,
    output_buf,
    error_ema_buf,
    lp_ema_buf,
    active_idx,
    legal_mask,
    thinking_steps=8
):
    global PLY_STEP_COUNTER
    PLY_STEP_COUNTER += 1

    sub_in  = env_input[active_idx]
    sub_th  = threat[active_idx]
    sub_s   = state_buf[:, active_idx, :]
    sub_o   = output_buf[:, active_idx, :]
    sub_e   = error_ema_buf[:, active_idx]
    sub_l   = lp_ema_buf[:, active_idx]
    PB      = sub_in.shape[0]

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Capture fixed sensory boundary snapshot once for all recurrence steps
    with torch.no_grad():
        raw_s = raw_model.sensory_proj(sub_in)
        raw_r = raw_model.reflex_proj(sub_in)
        sensory_latent = utils.softsign(raw_s.view(PB, raw_model.num_input_nodes, raw_model.D).permute(1, 0, 2))
        motor_reflex   = utils.softsign(raw_r.view(PB, raw_model.motor_nodes_count, raw_model.D).permute(1, 0, 2))
        latents = (sensory_latent, motor_reflex)

    # 1. Deliberation Phase: integrates trajectory loss and trajectory mean state
    trajectory_loss, next_s, next_o, next_e, next_l, omegas, s_trajectory = model(
        sub_in, sub_th, sub_s, sub_o, sub_e, sub_l,
        thinking_steps=thinking_steps,
        precomputed_latents=latents
    )

    # 2. Vectorized 2x batch repetition using the ACTIVE TRAJECTORY state (not the dead terminal state)
    in_2x = sub_in.repeat(2, 1)
    th_2x = sub_th.repeat(2)
    s_2x  = s_trajectory.repeat(1, 2, 1)  # <--- Includes all middle thinking steps!
    o_2x  = next_o.repeat(1, 2, 1)
    e_2x  = next_e.repeat(1, 2)
    l_2x  = next_l.repeat(1, 2)

    # 3. Direct Analytical DUST Parameter Update across Deliberation
    if PLY_STEP_COUNTER % 2 == 0:
        # PHASE A: Predictor Update (w1)
        xi_w1 = torch.randn(raw_model.N, PB, raw_model.D * 2, device=sub_in.device)
        jitter_w1 = torch.cat([+DUST_SIGMA * xi_w1, -DUST_SIGMA * xi_w1], dim=1)

        s_lat_2x = sensory_latent.repeat(1, 2, 1)
        m_ref_2x = motor_reflex.repeat(1, 2, 1)
        latents_2x = (s_lat_2x, m_ref_2x)

        _, _, _, _, _, l_both, _ = model(
            in_2x, th_2x, s_2x, o_2x, e_2x, l_2x,
            act_jitter=jitter_w1,
            precomputed_latents=latents_2x,
            thinking_steps=1
        )

        l_pos_w1, l_neg_w1 = l_both.chunk(2, dim=1)
        delta_l_w1 = (l_pos_w1 - l_neg_w1)
        g_hat_w1 = torch.clamp(((delta_l_w1 / (2.0 * DUST_SIGMA)) * xi_w1) / PB, -DUST_GRAD_CLIP, DUST_GRAD_CLIP)

        # Uses s_trajectory transpose so thinking nodes receive their full gradient
        grad_w1 = torch.bmm(s_trajectory.transpose(1, 2), g_hat_w1)
        grad_w1.clamp_(-1.0, 1.0)
        raw_model.w1.weight.data.sub_(dust_lr * grad_w1)
        raw_model.w1.normalize_weight()

    else:
        # PHASE B: Target, Encoders & Fused Routing Update
        sensory_dim = raw_model.num_input_nodes * raw_model.D
        reflex_dim  = raw_model.motor_nodes_count * raw_model.D

        xi_s = torch.randn(PB, sensory_dim, device=sub_in.device)
        xi_r = torch.randn(PB, reflex_dim,  device=sub_in.device)
        xi_q = torch.randn(raw_model.N, PB, raw_model.router.K, device=sub_in.device)
        xi_k = torch.randn(raw_model.N, PB, raw_model.router.K, device=sub_in.device)

        j_s = torch.cat([+DUST_SIGMA * xi_s, -DUST_SIGMA * xi_s], dim=0)
        j_r = torch.cat([+DUST_SIGMA * xi_r, -DUST_SIGMA * xi_r], dim=0)
        j_q = torch.cat([+DUST_SIGMA * xi_q, -DUST_SIGMA * xi_q], dim=1)
        j_k = torch.cat([+DUST_SIGMA * xi_k, -DUST_SIGMA * xi_k], dim=1)

        _, _, _, _, _, l_both, _ = model(
            in_2x, th_2x, s_2x, o_2x, e_2x, l_2x,
            sensory_jitter=j_s, reflex_jitter=j_r,
            q_jitter=j_q, k_jitter=j_k,
            thinking_steps=1
        )

        l_pos_tgt, l_neg_tgt = l_both.chunk(2, dim=1)
        delta_l_tgt = (l_pos_tgt - l_neg_tgt)

        delta_l_s = delta_l_tgt[:raw_model.num_input_nodes, :, :].permute(1, 0, 2).expand(-1, -1, raw_model.D).reshape(PB, -1)
        delta_l_r = delta_l_tgt[-raw_model.motor_nodes_count:, :, :].permute(1, 0, 2).expand(-1, -1, raw_model.D).reshape(PB, -1)

        g_hat_s = torch.clamp(((delta_l_s / (2.0 * DUST_SIGMA)) * xi_s) / PB, -DUST_GRAD_CLIP, DUST_GRAD_CLIP)
        g_hat_r = torch.clamp(((delta_l_r / (2.0 * DUST_SIGMA)) * xi_r) / PB, -DUST_GRAD_CLIP, DUST_GRAD_CLIP)
        g_hat_q = torch.clamp(((delta_l_tgt / (2.0 * DUST_SIGMA)) * xi_q) / PB, -DUST_GRAD_CLIP, DUST_GRAD_CLIP)
        g_hat_k = torch.clamp(((delta_l_tgt / (2.0 * DUST_SIGMA)) * xi_k) / PB, -DUST_GRAD_CLIP, DUST_GRAD_CLIP)

        g_hat_qk = torch.cat([g_hat_q, g_hat_k], dim=-1)
        grad_qk = torch.bmm(s_trajectory.transpose(1, 2), g_hat_qk)
        grad_qk.clamp_(-1.0, 1.0)
        raw_model.router.w_qk.data.sub_(dust_lr * grad_qk)


       #grad_sensory = torch.mm(g_hat_s.T, sub_in)
       #grad_reflex  = torch.mm(g_hat_r.T, sub_in)
       #grad_sensory.clamp_(-1.0, 1.0)
       #grad_reflex.clamp_(-1.0, 1.0)
       #raw_model.sensory_proj.weight.data.sub_(dust_lr * grad_sensory)
       #raw_model.reflex_proj.weight.data.sub_(dust_lr * grad_reflex)

        grad_sensory = torch.mm((g_hat_s + g_hat_r).T, sub_in)
        grad_sensory.clamp_(-1.0, 1.0)
        raw_model.sensory_proj.weight.data.sub_(dust_lr * grad_sensory)
        #raw_model.update_cached_spatial()
        raw_model.normalize_sensory_weight()

    actions = select_agent_actions(model, next_s, legal_mask[active_idx])
    ply_loss_val = trajectory_loss.detach()
    latest_omegas = omegas.mean(dim=0)

    return (
        actions,
        next_s,
        next_o,
        next_e,
        next_l,
        ply_loss_val,
        latest_omegas
    )
# ==========================================
# 7. CHECKPOINT RESOLVER
# ==========================================
def find_latest_checkpoint(ckpt_dir="out"):
    if not os.path.exists(ckpt_dir):
        return None
    patterns = [
        re.compile(r"ckpt\((\d+)\)\.pt"),
        re.compile(r"ckpt_(\d+)\.pt")
    ]
    candidates = []
    for f in os.listdir(ckpt_dir):
        full_path = os.path.join(ckpt_dir, f)
        if not os.path.isfile(full_path):
            continue
        for p in patterns:
            m = p.match(f)
            if m:
                candidates.append((int(m.group(1)), full_path))
                break
    if not candidates:
        all_pts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if all_pts:
            all_pts.sort(key=os.path.getmtime, reverse=True)
            return all_pts[0]
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# ==========================================
# 8. UNIFIED LIFELONG TRAINING LOOP
# ==========================================
def run_agent_experiment(
    total_target_games=50000,
    batch_size=64,
    num_nodes=1024,
    sf_train_interval_batches=10,
    sf_train_games=32,
    thinking_steps=8,
    max_plies=400,
    top_k_tourney=16,
    resume_checkpoint=None,
    eval_only=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    print(f"=== Active Inference Continuous Lifelong Agent on {device} ===")
    print(f"Layout Architecture:    Native (N, B, D) | Zero-Transpose Batched GEMM")
    print(f"Training Engine:        {TRAINING_MODE.upper()} Optimizer (Autograd Off)")
    print(f"Deliberation Ticks:     {thinking_steps} Thinking Steps per Ply")
    print(f"Persistent Agents:      {batch_size} | Fittest Survival Mechanics: Active")
    print(f"Stockfish Live Train:   Every {sf_train_interval_batches} batches ({sf_train_games} games)\n")

    cfg = HyperParamsConfig(device)
    model = ChessNodeNet(num_nodes=num_nodes, dim=32, input_dim=100, rank_k=16, device=device).to(device)
    model.apply_hyperparams(cfg)
    
    try:
        compiled_model = torch.compile(model, dynamic=True)
    except Exception as e:
        print(f"[Notice] torch.compile bypassed ({e}), using eager execution mode.")
        compiled_model = model

    dust_lr = float(cfg.get('opt_gd_lr').cpu())

    env = GuardrailedTorchChess(batch_size=batch_size, device=device, halfmove_limit=100)
    dones = torch.zeros(batch_size, dtype=torch.bool, device=device)
    winners = torch.zeros(batch_size, dtype=torch.int8, device=device)

    match_modes = (torch.arange(batch_size, device=device) % 4).to(torch.int8)

    agent_state, agent_output, agent_error, agent_lp = model.init_states(batch_size=batch_size)
    agent_noise = torch.full((batch_size,), 0.05, dtype=torch.float32, device=device)

    completed_games = 0
    batch_idx = 1

    if resume_checkpoint:
        if resume_checkpoint.lower() == "latest":
            ckpt_path = find_latest_checkpoint("out")
            if ckpt_path is None:
                print("[Resume] No checkpoint found in 'out/'. Starting fresh.")
            else:
                resume_checkpoint = ckpt_path

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"[Resume] Loading checkpoint from: {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location=device)

            raw_state_dict = ckpt.get('model_state_dict', {})
            cleaned_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()}
            model.load_state_dict(cleaned_sd, strict=False)
            model.w1.normalize_weight()
            model.update_cached_spatial()

            if 'agent_noise' in ckpt:
                loaded_noise = ckpt['agent_noise'].to(device)
                if loaded_noise.numel() == batch_size:
                    agent_noise.copy_(loaded_noise)

            if 'agent_state' in ckpt and ckpt['agent_state'].shape == agent_state.shape:
                agent_state.copy_(ckpt['agent_state'].to(device))
                agent_output.copy_(ckpt['agent_output'].to(device))
                agent_error.copy_(ckpt['agent_error'].to(device))
                agent_lp.copy_(ckpt['agent_lp'].to(device))

            batch_idx = ckpt.get('batch_idx', 1)
            completed_games = ckpt.get('completed_games', 0)
            print(f"[Resume] Successfully restored at Game {completed_games}, Batch {batch_idx}\n")

    state_w, output_w, error_w, lp_w = agent_state.clone(), agent_output.clone(), agent_error.clone(), agent_lp.clone()
    state_b, output_b, error_b, lp_b = agent_state.clone(), agent_output.clone(), agent_error.clone(), agent_lp.clone()

    plies = torch.zeros(batch_size, dtype=torch.long, device=device)
    accum_losses = torch.zeros(batch_size, device=device)
    loss_steps = torch.zeros(batch_size, device=device)

    white_wins, black_wins, draws = 0, 0, 0
    b_w_wins, b_b_wins, b_draws = 0, 0, 0
    b_plies, b_losses, b_omegas = [], [], []

    batch_start_time = time.time()
    total_start_time = time.time()
    latest_omegas = torch.zeros(batch_size, device=device)

    if not eval_only and completed_games < total_target_games:
        while completed_games < total_target_games:
            active = ~dones
            legal_mask = env.compute_legal_masks()

            white_turn = active & (env.players == 0)
            black_turn = active & (env.players == 1)

            w_agent_mask = white_turn & (match_modes != 1)
            w_engine_mask = white_turn & (match_modes == 1)

            b_agent_mask = black_turn & (match_modes != 3)
            b_engine_mask = black_turn & (match_modes == 3)

            actions = torch.zeros(batch_size, dtype=torch.long, device=device)

            if w_agent_mask.any():
                w_idx = torch.where(w_agent_mask)[0]
                env_input, threat = env.get_obs(agent_noise=agent_noise)

                acts, next_s, next_o, next_e, next_l, m_loss, omgs = deliberate_and_train_ply(
                    model=compiled_model, dust_lr=dust_lr,
                    env_input=env_input, threat=threat,
                    state_buf=state_w, output_buf=output_w,
                    error_ema_buf=error_w, lp_ema_buf=lp_w,
                    active_idx=w_idx, legal_mask=legal_mask,
                    thinking_steps=thinking_steps
                )

                actions[w_idx] = acts
                state_w[:, w_idx, :] = next_s
                output_w[:, w_idx, :] = next_o
                error_w[:, w_idx] = next_e
                lp_w[:, w_idx] = next_l
                latest_omegas[w_idx] = omgs
                accum_losses[w_idx] += m_loss
                loss_steps[w_idx] += 1

            if w_engine_mask.any():
                w_eng_idx = torch.where(w_engine_mask)[0]
                is_w = torch.ones(len(w_eng_idx), dtype=torch.bool, device=device)
                actions[w_eng_idx] = select_cuda_engine_move(
                    env.boards[w_eng_idx], legal_mask[w_eng_idx], is_w, device,
                    env=env, b_idx=w_eng_idx
                )

            if b_agent_mask.any():
                b_idx = torch.where(b_agent_mask)[0]
                env_input, threat = env.get_obs(agent_noise=agent_noise)

                acts, next_s, next_o, next_e, next_l, m_loss, omgs = deliberate_and_train_ply(
                    model=compiled_model, dust_lr=dust_lr,
                    env_input=env_input, threat=threat,
                    state_buf=state_b, output_buf=output_b,
                    error_ema_buf=error_b, lp_ema_buf=lp_b,
                    active_idx=b_idx, legal_mask=legal_mask,
                    thinking_steps=thinking_steps
                )

                actions[b_idx] = acts
                state_b[:, b_idx, :] = next_s
                output_b[:, b_idx, :] = next_o
                error_b[:, b_idx] = next_e
                lp_b[:, b_idx] = next_l
                latest_omegas[b_idx] = omgs
                accum_losses[b_idx] += m_loss
                loss_steps[b_idx] += 1

            if b_engine_mask.any():
                b_eng_idx = torch.where(b_engine_mask)[0]
                is_w = torch.zeros(len(b_eng_idx), dtype=torch.bool, device=device)
                actions[b_eng_idx] = select_cuda_engine_move(
                    env.boards[b_eng_idx], legal_mask[b_eng_idx], is_w, device,
                    env=env, b_idx=b_eng_idx
                )

            dones, winners = env.step(actions, dones, winners, legal_mask=legal_mask)
            plies[active] += 1

            max_plies_draw = (plies >= max_plies) & (~dones)
            if max_plies_draw.any():
                dones = dones | max_plies_draw
                winners = torch.where(max_plies_draw, torch.tensor(3, dtype=torch.int8, device=device), winners)

            finished_mask = dones.clone()
            num_finished = finished_mask.sum().item()

            if num_finished > 0:
                finished_indices = torch.where(finished_mask)[0]

                resolve_fittest_agent_states(
                    finished_mask, winners, match_modes,
                    agent_state, agent_output, agent_error, agent_lp,
                    state_w, output_w, error_w, lp_w,
                    state_b, output_b, error_b, lp_b
                )

                update_agent_somatic_noise(agent_noise, finished_mask, winners, match_modes)

                for idx in finished_indices:
                    completed_games += 1
                    p = plies[idx].item()
                    w_code = winners[idx].item()
                    m_loss = (accum_losses[idx] / max(1.0, loss_steps[idx].item())).item()
                    omg = latest_omegas[idx].item()

                    b_plies.append(p)
                    b_losses.append(m_loss)
                    b_omegas.append(omg)

                    if w_code == 1:
                        white_wins += 1
                        b_w_wins += 1
                    elif w_code == 2:
                        black_wins += 1
                        b_b_wins += 1
                    else:
                        draws += 1
                        b_draws += 1

                    if completed_games % batch_size == 0 or completed_games >= total_target_games:
                        batch_time = time.time() - batch_start_time
                        b_count = len(b_plies)
                        b_w_pct = (b_w_wins / max(1, b_count)) * 100.0
                        b_b_pct = (b_b_wins / max(1, b_count)) * 100.0
                        b_d_pct = (b_draws / max(1, b_count)) * 100.0
                        b_avg_p = sum(b_plies) / max(1, b_count)
                        b_avg_l = sum(b_losses) / max(1, b_count)
                        b_avg_o = sum(b_omegas) / max(1, b_count)
                        cum_w_pct = (white_wins / completed_games) * 100.0
                        avg_noise = agent_noise.mean().item()

                        print(f"[FITTEST | BATCH {batch_idx:02d} | Games {completed_games:05d}/{total_target_games}] "
                              f"Time: {batch_time:5.2f}s ({b_count / max(1e-3, batch_time):5.1f} g/s) | "
                              f"Batch: [W: {b_w_pct:4.2f}% | B: {b_b_pct:4.2f}% | D: {b_d_pct:4.2f}%] | "
                              f"Cum W: {cum_w_pct:4.2f}% | "
                              f"AvgPlies: {b_avg_p:4.1f} | Loss: {b_avg_l:.4f} | Ω: {b_avg_o:.4f} | AvgStress: {avg_noise:.3f}")

                        # -------------------------------------------------------------------------
                        # LIVE STOCKFISH TRAINING BATCH (INTERRUPT & TRAIN WITH REAL POPULATION)
                        # -------------------------------------------------------------------------
                        if batch_idx % sf_train_interval_batches == 0:
                            # Select top-fittest agents (lowest noise) to represent the population
                            effective_sf_games = min(sf_train_games, batch_size)
                            _, sf_agent_idx = torch.topk(agent_noise, k=effective_sf_games, largest=False)

                            sf_noise, sf_s, sf_o, sf_e, sf_l = train_stockfish_batch(
                                model=compiled_model,
                                dust_lr=dust_lr,
                                num_games=effective_sf_games,
                                target_nodes=1500,
                                thinking_steps=thinking_steps,
                                device=device,
                                agent_noise_slice=agent_noise[sf_agent_idx],
                                agent_state_slice=agent_state[:, sf_agent_idx, :],
                                agent_output_slice=agent_output[:, sf_agent_idx, :],
                                agent_error_slice=agent_error[:, sf_agent_idx],
                                agent_lp_slice=agent_lp[:, sf_agent_idx]
                            )

                            # Persist elevated allostatic noise and settled states back into the main population
                            if sf_noise is not None:
                                # 1. Update persistent bank
                                agent_noise[sf_agent_idx] = sf_noise
                                agent_state[:, sf_agent_idx, :] = sf_s
                                agent_output[:, sf_agent_idx, :] = sf_o
                                agent_error[:, sf_agent_idx] = sf_e
                                agent_lp[:, sf_agent_idx] = sf_l

                                # 2. Sync immediately into active working buffers (White & Black)
                                state_w[:, sf_agent_idx, :] = sf_s
                                output_w[:, sf_agent_idx, :] = sf_o
                                error_w[:, sf_agent_idx] = sf_e
                                lp_w[:, sf_agent_idx] = sf_l

                                state_b[:, sf_agent_idx, :] = sf_s
                                output_b[:, sf_agent_idx, :] = sf_o
                                error_b[:, sf_agent_idx] = sf_e
                                lp_b[:, sf_agent_idx] = sf_l

                            os.makedirs("out", exist_ok=True)
                            ckpt_path = f"out/ckpt({batch_idx}).pt"
                            torch.save({
                                'batch_idx': batch_idx,
                                'completed_games': completed_games,
                                'model_state_dict': model.state_dict(),
                                'agent_noise': agent_noise.clone(),
                                'agent_state': agent_state.clone(),
                                'agent_output': agent_output.clone(),
                                'agent_error': agent_error.clone(),
                                'agent_lp': agent_lp.clone(),
                                'hyperparams': {k: v.data for k, v in cfg.params.items()}
                            }, ckpt_path)
                            print(f"[Checkpoint] Saved to {ckpt_path}\n")

                        batch_idx += 1
                        b_w_wins, b_b_wins, b_draws = 0, 0, 0
                        b_plies, b_losses, b_omegas = [], [], []
                        batch_start_time = time.time()
                        torch.cuda.empty_cache()

                    if completed_games >= total_target_games:
                        break

                match_modes[finished_mask] = (match_modes[finished_mask] + 1) % 4
                env.reset(mask=finished_mask)

                state_w[:, finished_indices, :] = agent_state[:, finished_indices, :].clone()
                output_w[:, finished_indices, :] = agent_output[:, finished_indices, :].clone()
                error_w[:, finished_indices] = agent_error[:, finished_indices].clone()
                lp_w[:, finished_indices] = agent_lp[:, finished_indices].clone()

                state_b[:, finished_indices, :] = agent_state[:, finished_indices, :].clone()
                output_b[:, finished_indices, :] = agent_output[:, finished_indices, :].clone()
                error_b[:, finished_indices] = agent_error[:, finished_indices].clone()
                lp_b[:, finished_indices] = agent_lp[:, finished_indices].clone()

                plies[finished_mask] = 0
                accum_losses[finished_mask] = 0.0
                loss_steps[finished_mask] = 0.0
                dones[finished_mask] = False
                winners[finished_mask] = 0

        total_time = time.time() - total_start_time
        print(f"\nCompleted {completed_games} games in {total_time:.2f}s ({completed_games / max(1e-3, total_time):.1f} games/sec)")

    os.makedirs("out", exist_ok=True)
    
    champion = run_internal_tournament(
        model=compiled_model,
        agent_state=agent_state,
        agent_output=agent_output,
        agent_error=agent_error,
        agent_lp=agent_lp,
        agent_noise=agent_noise,
        top_k=top_k_tourney,
        thinking_steps=thinking_steps,
        device=device
    )

    champion_save_path = "out/champion_agent.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparams': {k: v.data for k, v in cfg.params.items()},
        'champion_slot_id': champion['slot_id'],
        'champion_state': champion['state'].cpu(),
        'champion_output': champion['output'].cpu(),
        'champion_error_ema': champion['error_ema'].cpu(),
        'champion_lp_ema': champion['lp_ema'].cpu(),
        'champion_noise': champion['noise'].cpu(),
    }, champion_save_path)
    print(f"\n[Artifact Saved] Playable Champion checkpoint written to: {champion_save_path}")

    run_stockfish_gauntlet(
        model=compiled_model,
        champion_package=champion,
        node_levels=[100, 500, 1000, 1500, 3000, 10000],
        games_per_tier=20,
        thinking_steps=thinking_steps,
        device=device
    )
# ==========================================
# 9. CHAMPION TOURNAMENT & SF GAUNTLET
# ==========================================
def run_internal_tournament(model, agent_state, agent_output, agent_error, agent_lp, agent_noise, top_k=16, thinking_steps=32,device="cuda"):
    available_agents = agent_noise.shape[0]
    effective_top_k = min(top_k, available_agents)
    
    print(f"\n========================================================")
    print(f"---> STAGE 1: Running Internal Championship Tournament (Top {effective_top_k} Candidates)")
    print(f"========================================================")

    _, top_indices = torch.topk(agent_noise, k=effective_top_k, largest=False)
    top_indices = top_indices.tolist()

    matchups = []
    for i in range(effective_top_k):
        for j in range(effective_top_k):
            if i != j:
                matchups.append((top_indices[i], top_indices[j]))

    num_matches = len(matchups)
    print(f"Running {num_matches} simultaneous tournament matches on {device}...")

    tourney_env = GuardrailedTorchChess(batch_size=num_matches, device=device)
    scores = {cand_idx: 0.0 for cand_idx in top_indices}

    w_cand = torch.tensor([m[0] for m in matchups], device=device)
    b_cand = torch.tensor([m[1] for m in matchups], device=device)

    s_w = agent_state[:, w_cand, :].clone()
    o_w = agent_output[:, w_cand, :].clone()
    e_w = agent_error[:, w_cand].clone()
    l_w = agent_lp[:, w_cand].clone()

    s_b = agent_state[:, b_cand, :].clone()
    o_b = agent_output[:, b_cand, :].clone()
    e_b = agent_error[:, b_cand].clone()
    l_b = agent_lp[:, b_cand].clone()

    tourney_noise = torch.full((num_matches,), 0.01, device=device)

    dones = torch.zeros(num_matches, dtype=torch.bool, device=device)
    winners = torch.zeros(num_matches, dtype=torch.int8, device=device)
    plies = torch.zeros(num_matches, dtype=torch.long, device=device)

    max_plies = 300

    while not dones.all() and (plies < max_plies).any():
        active = ~dones
        legal_mask = tourney_env.compute_legal_masks()
        actions = torch.zeros(num_matches, dtype=torch.long, device=device)

        w_turn = active & (tourney_env.players == 0)
        b_turn = active & (tourney_env.players == 1)

        if w_turn.any():
            w_idx = torch.where(w_turn)[0]
            env_input, threat = tourney_env.get_obs(agent_noise=tourney_noise)
            _, next_s, next_o, next_e, next_l, _, _ = model(
                env_input[w_idx], threat[w_idx],
                s_w[:, w_idx, :], o_w[:, w_idx, :], e_w[:, w_idx], l_w[:, w_idx]
                , thinking_steps=thinking_steps
            )
            s_w[:, w_idx, :] = next_s
            o_w[:, w_idx, :] = next_o
            e_w[:, w_idx] = next_e
            l_w[:, w_idx] = next_l
            actions[w_idx] = select_agent_actions(model, next_s, legal_mask[w_idx])

        if b_turn.any():
            b_idx = torch.where(b_turn)[0]
            env_input, threat = tourney_env.get_obs(agent_noise=tourney_noise)
            _, next_s, next_o, next_e, next_l, _, _ = model(
                env_input[b_idx], threat[b_idx],
                s_b[:, b_idx, :], o_b[:, b_idx, :], e_b[:, b_idx], l_b[:, b_idx]
                , thinking_steps=thinking_steps
            )
            s_b[:, b_idx, :] = next_s
            o_b[:, b_idx, :] = next_o
            e_b[:, b_idx] = next_e
            l_b[:, b_idx] = next_l
            actions[b_idx] = select_agent_actions(model, next_s, legal_mask[b_idx])

        dones, winners = tourney_env.step(actions, dones, winners, legal_mask=legal_mask)
        plies[active] += 1

        overlength = (plies >= max_plies) & (~dones)
        if overlength.any():
            dones = dones | overlength
            winners = torch.where(overlength, torch.tensor(3, dtype=torch.int8, device=device), winners)

    for m_idx, (w_c, b_c) in enumerate(matchups):
        w_code = winners[m_idx].item()
        if w_code == 1:
            scores[w_c] += 1.0
        elif w_code == 2:
            scores[b_c] += 1.0
        else:
            scores[w_c] += 0.5
            scores[b_c] += 0.5

    best_cand_idx = max(scores, key=scores.get)
    max_pts = scores[best_cand_idx]
    total_cand_games = (effective_top_k - 1) * 2
    winrate = (max_pts / max(1, total_cand_games)) * 100.0

    print(f"\n---> Tournament Complete!")
    print(f"  Crown Champion: Agent Slot #{best_cand_idx}")
    print(f"  Score: {max_pts:.1f} / {total_cand_games} ({winrate:.1f}% Win/Draw Rate)")
    print(f"  Allostatic Noise Level: {agent_noise[best_cand_idx].item():.4f}")

    champion_package = {
        'slot_id': best_cand_idx,
        'score': max_pts,
        'state': agent_state[:, best_cand_idx:best_cand_idx+1, :].clone(),
        'output': agent_output[:, best_cand_idx:best_cand_idx+1, :].clone(),
        'error_ema': agent_error[:, best_cand_idx:best_cand_idx+1].clone(),
        'lp_ema': agent_lp[:, best_cand_idx:best_cand_idx+1].clone(),
        'noise': agent_noise[best_cand_idx:best_cand_idx+1].clone(),
    }
    return champion_package

def run_stockfish_gauntlet(model, champion_package, thinking_steps=32, node_levels=[100, 500, 1000, 1500], games_per_tier=20, device="cuda"):
    sf_exec = shutil.which("stockfish")
    if sf_exec is None:
        print("[Stockfish Gauntlet] 'stockfish' executable not found in PATH. Skipping.")
        return

    print(f"\n========================================================")
    print(f"---> STAGE 2: Stockfish Multi-Tier Gauntlet ({games_per_tier} games per tier)")
    print(f"========================================================")

    champ_s = champion_package['state']
    champ_o = champion_package['output']
    champ_e = champion_package['error_ema']
    champ_l = champion_package['lp_ema']

    results = {}

    for nodes in node_levels:
        print(f"\n--- Testing vs Stockfish @ {nodes:5d} nodes ---")
        sf = StockfishEngine(stockfish_path=sf_exec)
        eval_env = GuardrailedTorchChess(batch_size=games_per_tier, device=device)
        eval_noise = torch.full((games_per_tier,), 0.01, dtype=torch.float32, device=device)

        agent_is_white = torch.zeros(games_per_tier, dtype=torch.bool, device=device)
        agent_is_white[:games_per_tier // 2] = True

        dones = torch.zeros(games_per_tier, dtype=torch.bool, device=device)
        winners = torch.zeros(games_per_tier, dtype=torch.int8, device=device)
        plies = torch.zeros(games_per_tier, dtype=torch.long, device=device)

        s_eval = champ_s.expand(-1, games_per_tier, -1).clone()
        o_eval = champ_o.expand(-1, games_per_tier, -1).clone()
        e_eval = champ_e.expand(-1, games_per_tier).clone()
        l_eval = champ_l.expand(-1, games_per_tier).clone()

        max_eval_plies = 300

        while not dones.all() and (plies < max_eval_plies).any():
            active = ~dones
            legal_masks = eval_env.compute_legal_masks()
            turns = eval_env.players
            is_agent_turn = active & ((turns == 0) == agent_is_white)
            is_sf_turn = active & ~is_agent_turn
            actions = torch.zeros(games_per_tier, dtype=torch.long, device=device)

            if is_agent_turn.any():
                ag_idx = torch.nonzero(is_agent_turn).squeeze(-1)
                env_input, threat = eval_env.get_obs(agent_noise=eval_noise)

                _, next_s, next_o, next_e, next_l, _, _ = model(
                    env_input[ag_idx], threat[ag_idx],
                    s_eval[:, ag_idx, :], o_eval[:, ag_idx, :], e_eval[:, ag_idx], l_eval[:, ag_idx]
                    , thinking_steps=thinking_steps
                )
                s_eval[:, ag_idx, :] = next_s
                o_eval[:, ag_idx, :] = next_o
                e_eval[:, ag_idx] = next_e
                l_eval[:, ag_idx] = next_l

                actions[ag_idx] = select_agent_actions(model, next_s, legal_masks[ag_idx])

            if is_sf_turn.any():
                sf_indices = torch.nonzero(is_sf_turn).squeeze(-1).tolist()
                for i in sf_indices:
                    board_i = eval_env.boards[i]
                    turn_i = int(turns[i].item())
                    half_c = int(eval_env.halfmove_clock[i].item())
                    full_n = int(plies[i].item() // 2 + 1)
                    c_rights = eval_env.castling_rights[i].tolist()
                    ep = int(eval_env.en_passant_sq[i].item())
                    fen = board_to_fen(board_i, turn_i, c_rights, ep_sq=ep, halfmove_clock=half_c, fullmove_num=full_n)

                    legal_acts = torch.nonzero(legal_masks[i] > -1e8).squeeze(-1).tolist()
                    if len(legal_acts) == 0:
                        dones[i] = True
                        winners[i] = 3
                        continue

                    best_m = sf.get_move(fen, nodes=nodes)
                    act = uci_to_action(best_m) if best_m else None
                    if act is None or act not in legal_acts:
                        act = legal_acts[0]
                    actions[i] = act

            dones, winners = eval_env.step(actions, dones, winners, legal_mask=legal_masks)
            plies[active] += 1

            overlength = (plies >= max_eval_plies) & (~dones)
            if overlength.any():
                dones = dones | overlength
                winners = torch.where(overlength, torch.tensor(3, dtype=torch.int8, device=device), winners)

        sf.close()

        agent_won = (((winners == 1) & agent_is_white) | ((winners == 2) & (~agent_is_white))).sum().item()
        sf_won = (((winners == 2) & agent_is_white) | ((winners == 1) & (~agent_is_white))).sum().item()
        draws = (winners == 3).sum().item()
        score_pct = ((agent_won + 0.5 * draws) / games_per_tier) * 100.0

        results[nodes] = {'W': agent_won, 'L': sf_won, 'D': draws, 'Score': score_pct}
        print(f"  Result @ {nodes:5d} nodes => Champ Wins: {agent_won:02d} | SF Wins: {sf_won:02d} | Draws: {draws:02d} | Score: {score_pct:4.1f}%")

    print(f"\n========================================================")
    print(f"STOCKFISH GAUNTLET FINAL SUMMARY:")
    for nodes, res in results.items():
        print(f"  SF {nodes:5d} nodes: Score {res['Score']:5.1f}% (W: {res['W']} | L: {res['L']} | D: {res['D']})")
    print(f"========================================================")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Inference NodeChess Lifelong Training")
    parser.add_argument("--total_games", type=int, default=50000, help="Total target games")
    parser.add_argument("--batch_size", type=int, default=4096, help="Number of concurrent environments")
    parser.add_argument("--num_nodes", type=int, default=32, help="Latent nodes in network")
    parser.add_argument("--thinking_steps", type=int, default=32, help="Internal recurrence steps per ply")
    parser.add_argument("--top_k_tourney", type=int, default=16, help="Top candidates to battle in tournament")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint or 'latest' to load newest ckpt")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run tournament & gauntlet directly")
    args = parser.parse_args()

    torch.cuda.init()
    torch.set_float32_matmul_precision('high')

    run_agent_experiment(
        total_target_games=args.total_games,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        sf_train_interval_batches=10,
        sf_train_games=128,
        thinking_steps=args.thinking_steps,
        max_plies=400,
        top_k_tourney=args.top_k_tourney,
        resume_checkpoint=args.resume,
        eval_only=args.eval_only
    )