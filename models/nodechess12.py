import math
import random
import time
import os
import sys
import shutil
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import chess
    HAS_CHESS_PKG = True
except ImportError:
    HAS_CHESS_PKG = False

noise_cap = 0.10

# ==========================================
# 0. LOCAL MODULE FALLBACKS
# ==========================================
try:
    from parts import quantizer
    import parts.utils as utils
    from unit import bench
except ImportError:
    class FallbackUtils:
        @staticmethod
        def softsign(x): 
            return F.softsign(x)

        @staticmethod
        def buntanh(x, slope, bound):
            return torch.tanh(x - torch.tanh(x) * slope) * bound

        @staticmethod
        def apply_spectral_filter(noise, center_offset=1.0, sigma_val=0.5):
            dim = noise.shape[-1]
            fft = torch.fft.rfft(noise, dim=-1)
            freqs = torch.fft.rfftfreq(dim, device=noise.device)
            filter_curve = torch.exp(-0.5 * ((freqs - center_offset) / (sigma_val + 1e-6)) ** 2)
            return torch.fft.irfft(fft * filter_curve, n=dim, dim=-1)

    class FallbackQuantizer:
        @staticmethod
        def topkhotSTE(logits, k):
            topk_indices = torch.topk(logits, k=k, dim=-1).indices
            mask = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
            return (mask - logits).detach() + logits

    class FallbackOptimizer(torch.optim.AdamW):
        def __init__(self, params, lr=1e-3, **kwargs):
            super().__init__(params, lr=lr, weight_decay=kwargs.get('weight_decay', 1e-4))

    utils = FallbackUtils()
    quantizer = FallbackQuantizer()
    class bench: 
        UnifiedKreinEntropicOptimizer = FallbackOptimizer

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

        g = torch.Generator(device=device)
        g.manual_seed(42)
        self.zobrist_pieces = torch.randint(-2**62, 2**62, (64, 13), generator=g, dtype=torch.int64, device=device)
        self.zobrist_turn = torch.randint(-2**62, 2**62, (1,), generator=g, dtype=torch.int64, device=device)

        self.reset()

    def compute_hash(self, boards, players):
        b_size = boards.shape[0]
        flat = boards.view(b_size, 64).long().clamp(0, 12)
        sq = torch.arange(64, device=self.device).unsqueeze(0).expand(b_size, -1)
        piece_hashes = self.zobrist_pieces[sq, flat]
        return piece_hashes.sum(dim=-1) + self.zobrist_turn * players.to(torch.int64)

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        idx = torch.nonzero(mask).squeeze(-1)
        if len(idx) == 0:
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
        """
        Pure outcome-driven noise allostasis.
        Observation signal clarity and noise injection are exclusively controlled
        by the win/loss/draw outcome noise signal.
        """
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
        is_att_w = (attacking_player == 0)
        offset = torch.where(is_att_w, 0, 6).to(torch.int8)

        pawn_val   = 1 + offset
        knight_val = 2 + offset
        bishop_val = 3 + offset
        rook_val   = 4 + offset
        queen_val  = 5 + offset
        king_val   = 6 + offset

        p_src_r = torch.where(is_att_w, r_t - 1, r_t + 1)
        valid_r = (p_src_r >= 0) & (p_src_r < 8)

        pawn_att = torch.zeros(B_size, dtype=torch.bool, device=self.device)
        for dc in [-1, 1]:
            p_src_c = c_t + dc
            in_c = (p_src_c >= 0) & (p_src_c < 8) & valid_r
            sq = (p_src_r * 8 + p_src_c).clamp(0, 63).long()
            p_at = flat.gather(1, sq.unsqueeze(1)).squeeze(1)
            pawn_att |= in_c & (p_at == pawn_val)

        knight_att = torch.zeros(B_size, dtype=torch.bool, device=self.device)
        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
            kr, kc = r_t + dr, c_t + dc
            valid_k = (kr >= 0) & (kr < 8) & (kc >= 0) & (kc < 8)
            sq = (kr * 8 + kc).clamp(0, 63).long()
            p_at = flat.gather(1, sq.unsqueeze(1)).squeeze(1)
            knight_att |= valid_k & (p_at == knight_val)

        king_att = torch.zeros(B_size, dtype=torch.bool, device=self.device)
        for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            kr, kc = r_t + dr, c_t + dc
            valid_kg = (kr >= 0) & (kr < 8) & (kc >= 0) & (kc < 8)
            sq = (kr * 8 + kc).clamp(0, 63).long()
            p_at = flat.gather(1, sq.unsqueeze(1)).squeeze(1)
            king_att |= valid_kg & (p_at == king_val)

        slider_att = torch.zeros(B_size, dtype=torch.bool, device=self.device)
        for is_diag, dirs in [(False, [(1,0), (-1,0), (0,1), (0,-1)]), (True, [(1,1), (1,-1), (-1,1), (-1,-1)])]:
            slider_target = bishop_val if is_diag else rook_val
            for dr, dc in dirs:
                blocked = torch.zeros(B_size, dtype=torch.bool, device=self.device)
                for step in range(1, 8):
                    sr, sc = r_t + dr * step, c_t + dc * step
                    valid_step = (sr >= 0) & (sr < 8) & (sc >= 0) & (sc < 8)
                    sq = (sr * 8 + sc).clamp(0, 63).long()
                    p_at = flat.gather(1, sq.unsqueeze(1)).squeeze(1)

                    is_slider = (p_at == queen_val) | (p_at == slider_target)
                    slider_att |= valid_step & (~blocked) & is_slider
                    blocked |= valid_step & (p_at > 0)

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
        pseudo_mask = torch.full((self.B, 4096), -1e9, device=self.device)
        sq_from = torch.arange(64, device=self.device)
        r_f, c_f = sq_from // 8, sq_from % 8

        pieces = self.boards.view(self.B, 64)
        is_white = (self.players == 0).unsqueeze(1)
        owns = torch.where(is_white, (pieces >= 1) & (pieces <= 6), (pieces >= 7) & (pieces <= 12))

        b_idx, f_idx = torch.nonzero(owns, as_tuple=True)
        if len(b_idx) == 0:
            return pseudo_mask

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

        for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
            rt, ct = rf + dr, cf + dc
            valid_k = (pt == 2) & (rt >= 0) & (rt < 8) & (ct >= 0) & (ct < 8)
            to_sq = (rt * 8 + ct).clamp(0, 63)
            d_p = pieces[b_idx, to_sq]
            d_own = torch.where(is_p_white, (d_p >= 1) & (d_p <= 6), (d_p >= 7) & (d_p <= 12))
            valid_k = valid_k & (~d_own)
            pseudo_mask[b_idx[valid_k], f_idx[valid_k] * 64 + to_sq[valid_k]] = 0.0

        for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            rt, ct = rf + dr, cf + dc
            valid_kg = (pt == 6) & (rt >= 0) & (rt < 8) & (ct >= 0) & (ct < 8)
            to_sq = (rt * 8 + ct).clamp(0, 63)
            d_p = pieces[b_idx, to_sq]
            d_own = torch.where(is_p_white, (d_p >= 1) & (d_p <= 6), (d_p >= 7) & (d_p <= 12))
            valid_kg = valid_kg & (~d_own)
            pseudo_mask[b_idx[valid_kg], f_idx[valid_kg] * 64 + to_sq[valid_kg]] = 0.0

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

        for is_slider, dirs in [((pt == 4) | (pt == 5), [(1,0), (-1,0), (0,1), (0,-1)]),
                                ((pt == 3) | (pt == 5), [(1,1), (1,-1), (-1,1), (-1,-1)])]:
            for dr, dc in dirs:
                blocked = torch.zeros_like(b_idx, dtype=torch.bool)
                for step in range(1, 8):
                    rt, ct = rf + dr * step, cf + dc * step
                    in_b = is_slider & (rt >= 0) & (rt < 8) & (ct >= 0) & (ct < 8) & (~blocked)
                    to_sq = (rt * 8 + ct).clamp(0, 63)
                    d_piece = pieces[b_idx, to_sq]
                    d_own = torch.where(is_p_white, (d_piece >= 1) & (d_piece <= 6), (d_piece >= 7) & (d_piece <= 12))
                    can_land = in_b & (~d_own)
                    pseudo_mask[b_idx[can_land], f_idx[can_land] * 64 + to_sq[can_land]] = 0.0
                    blocked = blocked | (in_b & (d_piece > 0))

        return pseudo_mask

    def compute_legal_masks(self):
        pseudo_mask = self.compute_pseudo_legal_masks()
        legal_mask = torch.full_like(pseudo_mask, -1e9)

        b_idx, act_idx = torch.nonzero(pseudo_mask > -1e8, as_tuple=True)
        if len(b_idx) == 0:
            return legal_mask

        from_sqs = act_idx // 64
        to_sqs = act_idx % 64
        turn = self.players[b_idx]
        N_moves = len(b_idx)

        sub_boards = self.boards[b_idx].clone().view(N_moves, 64)
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

        k_val = torch.where(turn == 0, 6, 12).to(torch.int8)
        k_sqs = (sub_boards == k_val.unsqueeze(1)).max(dim=1).indices

        opp = 1 - turn
        in_check = self._is_square_attacked(k_sqs, opp, sub_boards.view(N_moves, 8, 8))
        valid_moves = ~in_check

        legal_mask[b_idx[valid_moves], act_idx[valid_moves]] = 0.0
        return legal_mask

    def step(self, action_indices, dones, winners):
        B_idx = torch.arange(self.B, device=self.device)
        from_sq = action_indices // 64
        to_sq = action_indices % 64
        from_r, from_c = from_sq // 8, from_sq % 8
        to_r, to_c = to_sq // 8, to_sq % 8

        piece = self.boards[B_idx, from_r, from_c]
        dest_piece = self.boards[B_idx, to_r, to_c]
        active = ~dones

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
        if len(act_idx) > 0:
            curr_lens = self.history_len[act_idx].clamp(max=self.max_history_len - 1).long()
            self.history_hashes[act_idx, curr_lens] = curr_hashes[act_idx]
            self.history_len[act_idx] = (curr_lens + 1).to(torch.int16)

        legal_mask = self.compute_legal_masks()
        has_legal = (legal_mask > -1e8).any(dim=-1)
        no_legal = active & (~has_legal)

        in_chk_w = self.is_in_check(0)
        in_chk_b = self.is_in_check(1)

        checkmate_w_loses = no_legal & (self.players == 0) & in_chk_w
        checkmate_b_loses = no_legal & (self.players == 1) & in_chk_b
        checkmates = checkmate_w_loses | checkmate_b_loses
        stalemates = no_legal & (~checkmates)

        w_mat, b_mat = self.get_material()
        w_resigns = active & ((b_mat >= w_mat + 1200) | ((w_mat == 0) & (b_mat > 300)))
        b_resigns = active & ((w_mat >= b_mat + 1200) | ((b_mat == 0) & (w_mat > 300)))

        w_wins = checkmate_b_loses | b_resigns
        b_wins = checkmate_w_loses | w_resigns

        winners = torch.where(w_wins, torch.tensor(1, dtype=torch.int8, device=self.device), winners)
        winners = torch.where(b_wins, torch.tensor(2, dtype=torch.int8, device=self.device), winners)
        decisive = w_wins | b_wins
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

        draws = stalemates | draw_rep | draw_50 | draw_insufficient
        winners = torch.where(draws, torch.tensor(3, dtype=torch.int8, device=self.device), winners)
        dones = dones | decisive | draws

        return dones, winners
    
def resolve_fittest_agent_states(
    finished_mask, winners, match_modes,
    agent_state, agent_output, agent_error, agent_lp,
    state_w, output_w, error_w, lp_w,
    state_b, output_b, error_b, lp_b,
    p_loser_survives=0.20
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

    sp_white_wins_and_kept = (w_won & ~keep_loser)
    sp_black_wins_but_white_kept = (b_won & keep_loser)
    sp_white_draw_kept = (draw & draw_coin)

    white_survives = finished_mask & (
        (match_modes == 3) | 
        (self_play & (sp_white_wins_and_kept | sp_black_wins_but_white_kept | sp_white_draw_kept))
    )

    sp_black_wins_and_kept = (b_won & ~keep_loser)
    sp_white_wins_but_black_kept = (w_won & keep_loser)
    sp_black_draw_kept = (draw & ~draw_coin)

    black_survives = finished_mask & (
        (match_modes == 1) | 
        (self_play & (sp_black_wins_and_kept | sp_white_wins_but_black_kept | sp_black_draw_kept))
    )

    if white_survives.any():
        agent_state[white_survives]  = state_w[white_survives]
        agent_output[white_survives] = output_w[white_survives]
        agent_error[white_survives]  = error_w[white_survives]
        agent_lp[white_survives]     = lp_w[white_survives]

    if black_survives.any():
        agent_state[black_survives]  = state_b[black_survives]
        agent_output[black_survives] = output_b[black_survives]
        agent_error[black_survives]  = error_b[black_survives]
        agent_lp[black_survives]     = lp_b[black_survives]

# ==========================================
# 2. CONTINUOUS AGENT ALLOSTASIS
# ==========================================
def update_agent_somatic_noise(agent_noise, finished_mask, winners, match_modes):
    """
    Pure outcome noise signal:
    Decisive wins relieve noise stress; losses and draws elevate stress.
    """
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
def select_cuda_engine_move(boards, legal_mask, is_engine_white, device, env=None, b_idx=None):
    B = boards.shape[0]
    to_sq = torch.arange(4096, device=device) % 64
    from_sq = torch.arange(4096, device=device) // 64
    dest_r, dest_c = to_sq // 8, to_sq % 8
    from_r, from_c = from_sq // 8, from_sq % 8

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

    center_dist = ((dest_r.float() - 3.5).abs() + (dest_c.float() - 3.5).abs())
    pos_bonus = (7.0 - center_dist) * 2.5
    is_sweet_center = ((dest_r == 3) | (dest_r == 4)) & ((dest_c == 3) | (dest_c == 4))
    pos_bonus = pos_bonus + torch.where(is_sweet_center, torch.tensor(15.0, device=device), torch.zeros_like(pos_bonus))

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
def train_stockfish_batch(
    model,
    optimizer,
    num_games=32,
    target_nodes=1500,
    thinking_steps=16,
    max_plies=250,
    device="cuda"
):
    """
    Executes a full live gradient-training batch against Stockfish.
    Updates the model weights, backpropagating on every ply played by the agent.
    """
    sf_exec = shutil.which("stockfish")
    if sf_exec is None:
        print("[Stockfish Train Batch] 'stockfish' executable not found in PATH. Skipping.")
        return

    print(f"\n---> [LIVE TRAINING] Executing Stockfish Training Batch ({num_games} matches | SF Nodes: {target_nodes})...")
    sf = StockfishEngine(stockfish_path=sf_exec)
    sf_env = GuardrailedTorchChess(batch_size=num_games, device=device)

    # Allostatic stress noise for this batch
    batch_noise = torch.full((num_games,), 0.05, dtype=torch.float32, device=device)

    # Half games agent is White (mode 3), half agent is Black (mode 1)
    agent_is_white = torch.zeros(num_games, dtype=torch.bool, device=device)
    agent_is_white[:num_games // 2] = True
    match_modes = torch.where(agent_is_white, torch.tensor(3, device=device), torch.tensor(1, device=device)).to(torch.int8)

    state_buf, output_buf, error_buf, lp_buf = model.init_states(batch_size=num_games)
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

        # 1. Active Agent Move with deliberate_and_train_ply (GRADIENT STEP)
        if is_agent_turn.any():
            ag_idx = torch.nonzero(is_agent_turn).squeeze(-1)
            env_input, threat = sf_env.get_obs(agent_noise=batch_noise)

            acts, next_s, next_o, next_e, next_l, ply_loss, _ = deliberate_and_train_ply(
                model=model,
                optimizer=optimizer,
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
            state_buf[ag_idx] = next_s
            output_buf[ag_idx] = next_o
            error_buf[ag_idx] = next_e
            lp_buf[ag_idx] = next_l

            total_loss_accum += ply_loss
            loss_steps_count += 1

        # 2. Stockfish Move
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

        dones, winners = sf_env.step(actions, dones, winners)
        plies[active] += 1

        overlength = (plies >= max_plies) & (~dones)
        if overlength.any():
            dones = dones | overlength
            winners = torch.where(overlength, torch.tensor(3, dtype=torch.int8, device=device), winners)

    sf.close()

    # Update outcome-driven stress noise
    update_agent_somatic_noise(batch_noise, dones, winners, match_modes)

    agent_won = (((winners == 1) & agent_is_white) | ((winners == 2) & (~agent_is_white))).sum().item()
    sf_won = (((winners == 2) & agent_is_white) | ((winners == 1) & (~agent_is_white))).sum().item()
    draws_count = (winners == 3).sum().item()
    avg_loss = total_loss_accum / max(1, loss_steps_count)

    print(f"[Stockfish Train Batch Complete] "
          f"Agent Wins: {agent_won} | SF Wins: {sf_won} | Draws: {draws_count} | "
          f"Mean Loss: {avg_loss:.4f} | AvgPlies: {plies.float().mean().item():4.1f} | Stress: {batch_noise.mean().item():.3f}\n")


def run_stockfish_benchmark(model, num_games=50, target_nodes=1500, eval_noise=0.05, device="cuda"):
    sf_exec = shutil.which("stockfish")
    if sf_exec is None:
        print("[Stockfish Benchmark] 'stockfish' executable not found in PATH. Skipping.")
        return

    print(f"\n---> Running Fast Batched Stockfish Benchmark ({num_games} matches | SF Nodes: {target_nodes} | Noise: {eval_noise:.3f})...")
    sf = StockfishEngine(stockfish_path=sf_exec)
    eval_env = GuardrailedTorchChess(batch_size=num_games, device=device)
    eval_agent_noise = torch.full((num_games,), eval_noise, dtype=torch.float32, device=device)

    agent_is_white = torch.zeros(num_games, dtype=torch.bool, device=device)
    agent_is_white[:num_games // 2] = True

    dones = torch.zeros(num_games, dtype=torch.bool, device=device)
    winners = torch.zeros(num_games, dtype=torch.int8, device=device)
    finish_plies = torch.zeros(num_games, dtype=torch.long, device=device)
    eval_state, eval_out, eval_err, eval_lp = model.init_states(batch_size=num_games)
    plies = torch.zeros(num_games, dtype=torch.long, device=device)

    model.eval()
    ply_step = 0
    max_eval_plies = 400
    first_loss_ply = None

    while not dones.all() and (plies < max_eval_plies).any():
        ply_step += 1
        active = ~dones
        actions = torch.zeros(num_games, dtype=torch.long, device=device)

        legal_masks = eval_env.compute_legal_masks()
        turns = eval_env.players
        is_agent_turn = active & ((turns == 0) == agent_is_white)
        is_sf_turn = active & ~is_agent_turn

        if is_agent_turn.any():
            agent_indices = torch.nonzero(is_agent_turn).squeeze(-1)
            env_input, threat = eval_env.get_obs(agent_noise=eval_agent_noise)

            with torch.no_grad():
                _, next_s, next_o, next_e, next_l, _ = model(
                    env_input[agent_indices],
                    threat[agent_indices],
                    eval_state[agent_indices],
                    eval_out[agent_indices],
                    eval_err[agent_indices],
                    eval_lp[agent_indices]
                )
                eval_state[agent_indices] = next_s
                eval_out[agent_indices] = next_o
                eval_err[agent_indices] = next_e
                eval_lp[agent_indices] = next_l

                raw_logits = model.get_action_logits(next_s)
                masked_logits = raw_logits + legal_masks[agent_indices]
                actions[agent_indices] = torch.argmax(masked_logits, dim=-1)

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
                    finish_plies[i] = plies[i]
                    continue

                best_m = sf.get_move(fen, nodes=target_nodes)
                act = uci_to_action(best_m) if best_m else None
                if act is None or act not in legal_acts:
                    act = legal_acts[0]
                actions[i] = act

        prev_dones = dones.clone()
        dones, winners = eval_env.step(actions, dones, winners)
        plies[active] += 1

        just_finished = (~prev_dones) & dones
        if just_finished.any():
            finish_plies[just_finished] = plies[just_finished]
            for j in torch.where(just_finished)[0]:
                is_loss = ((winners[j] == 2) and agent_is_white[j]) or ((winners[j] == 1) and not agent_is_white[j])
                if is_loss and first_loss_ply is None:
                    first_loss_ply = finish_plies[j].item()

        overlength = (plies >= max_eval_plies) & (~dones)
        if overlength.any():
            dones = dones | overlength
            winners = torch.where(overlength, torch.tensor(3, dtype=torch.int8, device=device), winners)
            finish_plies[overlength] = plies[overlength]

        done_cnt = dones.sum().item()
        w_wins = ((winners == 1) & agent_is_white).sum().item() + ((winners == 2) & (~agent_is_white)).sum().item()
        sf_wins = ((winners == 2) & agent_is_white).sum().item() + ((winners == 1) & (~agent_is_white)).sum().item()
        draw_cnt = (winners == 3).sum().item()
        first_str = f"{first_loss_ply:02d}" if first_loss_ply is not None else "--"
        print(f"\r[SF Eval] Ply: {ply_step:03d} | Games Done: {done_cnt:02d}/{num_games} | Agent W: {w_wins} | SF W: {sf_wins} (1st: {first_str}) | D: {draw_cnt}", end="", flush=True)

    sf.close()
    print()

    agent_won = (((winners == 1) & agent_is_white) | ((winners == 2) & (~agent_is_white))).sum().item()
    sf_won = (((winners == 2) & agent_is_white) | ((winners == 1) & (~agent_is_white))).sum().item()
    draws_count = (winners == 3).sum().item()
    score = (agent_won + 0.5 * draws_count) / max(1, num_games)

    loss_mask = ((winners == 2) & agent_is_white) | ((winners == 1) & (~agent_is_white))
    if loss_mask.any():
        loss_plies = finish_plies[loss_mask].float()
        ply_analytics = f"Min: {int(loss_plies.min().item()):02d} | Max: {int(loss_plies.max().item()):02d} | Avg: {loss_plies.mean().item():4.1f}"
    else:
        ply_analytics = "N/A (No losses)"

    print(f"\n[Strict Stockfish Benchmark Report]")
    print(f"  Matches: {num_games} | Agent Wins: {agent_won} | SF Wins: {sf_won} | Draws: {draws_count}")
    print(f"  Agent Score: {score * 100:.1f}%")
    print(f"  Loss Ply Analytics => {ply_analytics}\n")


# ==========================================
# 6. NODE NETWORK (ACTIVE INFERENCE MODEL)
# ==========================================
USE_DIMENSIONLESS_BOREDOM = True
USE_ENTROPIC_RESTORING_FORCE = True
GAMMA_COUPLING = 2.0

DEFAULT_HYPERPARAMS = {
    #'UNTANH': 0.725268,
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
    'opt_beta_cov': 0.99,
    'opt_damping': 1e-5,
    'opt_decay': 1e-5,
    'opt_gamma': 1e-5
}

class HyperParamsConfig:
    def __init__(self, device):
        self.params = {k: nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device)) for k, v in DEFAULT_HYPERPARAMS.items()}
    def get(self, key): 
        return self.params[key]

def color_noise(size, center, width, device):
    noise = torch.randn(size, device=device)
    return utils.apply_spectral_filter(noise, center_offset=center, sigma_val=width)

class NodeParams3(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, device):
        super().__init__()
        base_weight = color_noise((num_nodes, in_dim, out_dim), 0.5, 0.2, device=device) / math.sqrt(in_dim)
        self.weight = nn.Parameter(base_weight)
        self.untanh_slope = torch.tensor(0.5, device=device)
        self.w_scale = torch.tensor(0.8, device=device)

    def forward(self, x):
        norm_weight = F.normalize(self.weight, p=2, dim=-2) * self.w_scale
        raw_pred = torch.einsum('bni,nio->bno', x, norm_weight)
        return activation(raw_pred, self.untanh_slope, math.sqrt(2.0 + math.sqrt(2.0)))

    def apply_hyperparams(self, config):
        self.untanh_slope = config.get('UNTANH').view(-1, 1, 1)
        self.w_scale = config.get('w_scale').view(-1, 1, 1)

def sparse_route_batched(router, node_count):
    probs = torch.sigmoid(router)
    mask = quantizer.topkhotSTE(router, max(1, int(math.sqrt(node_count))))
    sparse_probs = mask * probs
    sparse_probs = sparse_probs / (sparse_probs.sum(dim=-1, keepdim=True) + 1e-8)
    return sparse_probs, 0.0, mask

class ChessNodeNet(nn.Module):
    def __init__(self, num_nodes=64, dim=32, input_dim=100, device="cuda"):
        super().__init__()
        self.N, self.D, self.device = num_nodes, dim, device
        self.num_input_nodes = math.ceil(input_dim / dim)

        self.sensory_proj = nn.Linear(input_dim, self.num_input_nodes * dim, bias=False, device=device)
        nn.init.orthogonal_(self.sensory_proj.weight, gain=0.8)

        self.motor_nodes_count = 4
        self.reflex_proj = nn.Linear(input_dim, self.motor_nodes_count * dim, bias=False, device=device)
        nn.init.normal_(self.reflex_proj.weight, std=0.02)

        self.w1 = NodeParams3(num_nodes, dim, dim * 2, device)
        self.router = NodeParams3(num_nodes, dim, num_nodes, device)

        self.lp_alpha = 0.25
        self.sprs = torch.tensor(0.5, device=device)
        self.a_variance = torch.tensor(0.05, device=device)

    def init_states(self, batch_size):
        state = 1e-4 * color_noise((batch_size, self.N, self.D), 1.0, 1.0, device=self.device)
        output = 1e-4 * color_noise((batch_size, self.N, self.D), 1.0, 1.0, device=self.device)
        error_ema = torch.ones(batch_size, self.N, device=self.device) * 0.05
        lp_ema = torch.zeros(batch_size, self.N, device=self.device)
        return state, output, error_ema, lp_ema

    def apply_hyperparams(self, config):
        self.sprs = config.get('sprs').view(-1, 1, 1)
        self.a_variance = config.get('a_variance').view(-1, 1, 1)
        self.lp_alpha = float(config.get('EMA_SPEED').item())
        self.w1.apply_hyperparams(config)
        self.router.apply_hyperparams(config)

    def get_action_logits(self, state):
        PB = state.shape[0]
        motor_vector = state[:, -self.motor_nodes_count:, :].reshape(PB, self.motor_nodes_count * self.D)
        from_intent, to_intent = torch.chunk(motor_vector, 2, dim=-1)

        sensory_square_enc = self.sensory_proj.weight[:, :64]
        sensory_from_enc, sensory_to_enc = torch.chunk(sensory_square_enc, 2, dim=0)

        from_logits = torch.matmul(from_intent, sensory_from_enc)
        to_logits   = torch.matmul(to_intent,   sensory_to_enc)

        spatial_affinity = torch.matmul(sensory_from_enc.T, sensory_to_enc)
        coherence_scalar = torch.sum(from_intent * to_intent, dim=-1, keepdim=True).unsqueeze(-1)

        joint_scores = from_logits.unsqueeze(2) + to_logits.unsqueeze(1) + coherence_scalar * spatial_affinity.unsqueeze(0)
        return joint_scores.view(PB, 4096)
    
    def forward(self, env_input, threat, state, output, error_ema, lp_ema):
        PB = env_input.shape[0]
        h = threat.view(PB, 1, 1)

        sensory_latent = utils.softsign(self.sensory_proj(env_input).view(PB, self.num_input_nodes, self.D))
        motor_reflex = utils.softsign(self.reflex_proj(env_input).view(PB, self.motor_nodes_count, self.D))

        router_logits = self.router(state)
        cognitive_lp = lp_ema.clone()
        cognitive_lp[:, :self.num_input_nodes] = 0.0
        route_logits = router_logits + F.relu(cognitive_lp).unsqueeze(1) * 3.0
        sparse_probs, aux, _ = sparse_route_batched(route_logits, self.N)

        current_var = ((1.0 - self.sprs) ** 2) * sparse_probs.var(dim=-1, keepdim=True)
        vd = F.relu(self.a_variance - current_var)

        white_noise = torch.randn_like(output)
        filtered_noise = utils.apply_spectral_filter(white_noise, center_offset=1.0, sigma_val=1.0)
        base_noise = torch.sqrt(vd) * filtered_noise

        relative_lp = torch.clamp(F.relu(lp_ema) / (error_ema + 1e-6), min=0.0, max=1.0)
        boredom_scale = torch.pow(1.0 - relative_lp, GAMMA_COUPLING / 2.0).unsqueeze(-1)

        local_diffusion = torch.matmul(sparse_probs, output)
        global_mean = torch.mean(output, dim=1, keepdim=True)
        target = (1.0 - self.sprs) * local_diffusion + self.sprs * global_mean + base_noise * boredom_scale

        if USE_ENTROPIC_RESTORING_FORCE:
            target = target - (GAMMA_COUPLING * 0.01) * state * (1.0 - boredom_scale)

        target = utils.softsign(target)
        target = torch.cat([
            sensory_latent, 
            target[:, self.num_input_nodes:-self.motor_nodes_count, :],
            utils.softsign(target[:, -self.motor_nodes_count:, :] + 0.3 * motor_reflex)
        ], dim=1)

        prediction = self.w1(state)
        mean, error_pred = torch.chunk(prediction, 2, dim=-1)
        loss_pred = F.mse_loss(mean, target, reduction='none')
        loss_err_pred = F.mse_loss(error_pred, torch.abs(target - mean).detach(), reduction='none')
        loss_w1 = (loss_pred + 0.1 * loss_err_pred) * (1.0 + 2.0 * h)

        with torch.no_grad():
            current_error = loss_w1.mean(dim=-1).detach()
            ratio_lp = torch.log((error_ema + 1e-6) / (current_error + 1e-6))
            next_error_ema = error_ema * (1.0 - self.lp_alpha) + current_error * self.lp_alpha
            next_lp_ema = lp_ema * (1.0 - self.lp_alpha) + ratio_lp * self.lp_alpha

        total_loss = (loss_w1.mean()) + (aux / PB)

        next_relative_lp = torch.clamp(F.relu(next_lp_ema) / (next_error_ema + 1e-6), min=0.0, max=1.0)
        next_boredom_scale = torch.pow(1.0 - next_relative_lp, GAMMA_COUPLING / 2.0).unsqueeze(-1)

        omega = h * next_boredom_scale
        v_pragmatic = target - state
        target_noise = torch.randn_like(target)
        filtered_target_noise = utils.apply_spectral_filter(target_noise, center_offset=1.0, sigma_val=0.25)
        v_epistemic = filtered_target_noise * omega * 0.40

        alpha = 0.25 
        settled_state = utils.softsign(state + alpha * v_pragmatic + (alpha * 0.5) * v_epistemic)
        
        next_state = torch.cat([sensory_latent, settled_state[:, self.num_input_nodes:, :]], dim=1)
        next_output = mean

        return total_loss, next_state, next_output, next_error_ema, next_lp_ema, omega.squeeze(-1)

def select_agent_actions(model, state, legal_mask):
    model.eval()
    with torch.no_grad():
        raw_logits = model.get_action_logits(state)
        masked_logits = raw_logits + legal_mask
        return torch.argmax(masked_logits, dim=-1)

def deliberate_and_train_ply(
    model,
    optimizer,
    env_input,
    threat,
    state_buf,
    output_buf,
    error_ema_buf,
    lp_ema_buf,
    active_idx,
    legal_mask,
    thinking_steps=4
):
    model.train()
    optimizer.zero_grad()

    sub_in  = env_input[active_idx]
    sub_th  = threat[active_idx]
    sub_s   = state_buf[active_idx].detach()
    sub_o   = output_buf[active_idx].detach()
    sub_e   = error_ema_buf[active_idx].detach()
    sub_l   = lp_ema_buf[active_idx].detach()

    trajectory_loss = 0.0

    for _ in range(thinking_steps):
        step_loss, sub_s, sub_o, sub_e, sub_l, omegas = model(
            sub_in, sub_th, sub_s, sub_o, sub_e, sub_l
        )
        trajectory_loss = trajectory_loss + step_loss

    ply_loss = trajectory_loss / thinking_steps
    ply_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        actions = select_agent_actions(model, sub_s, legal_mask[active_idx])

    latest_omegas = omegas.mean(dim=-1).detach()
    return (
        actions,
        sub_s.detach(),
        sub_o.detach(),
        sub_e.detach(),
        sub_l.detach(),
        ply_loss.item(),
        latest_omegas
    )


# ==========================================
# 7. UNIFIED LIFELONG TRAINING LOOP
# ==========================================
def run_agent_experiment(
    total_target_games=50000,
    batch_size=512,
    num_nodes=128,
    sf_train_interval_batches=10,
    sf_train_games=32,
    thinking_steps=64,
    max_plies=400
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Active Inference Continuous Lifelong Agent on {device} ===")
    print(f"Persistent Agents: {batch_size} | Fittest Survival Mechanics: Active")
    print(f"Noise Signal: PURE Outcome Allostasis (Wins, Losses, Draws only)")
    print(f"Stockfish Live Training: Every {sf_train_interval_batches} batches ({sf_train_games} games)\n")

    cfg = HyperParamsConfig(device)
    model = ChessNodeNet(num_nodes=num_nodes, dim=32, input_dim=100, device=device).to(device)
    model.apply_hyperparams(cfg)
    model = torch.compile(model)
    optimizer = bench.UnifiedKreinEntropicOptimizer(
        model.parameters(),
        lr=float(cfg.get('opt_gd_lr').item())
    )

    env = GuardrailedTorchChess(batch_size=batch_size, device=device, halfmove_limit=100)
    dones = torch.zeros(batch_size, dtype=torch.bool, device=device)
    winners = torch.zeros(batch_size, dtype=torch.int8, device=device)

    match_modes = (torch.arange(batch_size, device=device) % 4).to(torch.int8)

    agent_state, agent_output, agent_error, agent_lp = model.init_states(batch_size=batch_size)
    agent_noise = torch.full((batch_size,), 0.05, dtype=torch.float32, device=device)

    state_w, output_w, error_w, lp_w = agent_state.clone(), agent_output.clone(), agent_error.clone(), agent_lp.clone()
    state_b, output_b, error_b, lp_b = agent_state.clone(), agent_output.clone(), agent_error.clone(), agent_lp.clone()

    plies = torch.zeros(batch_size, dtype=torch.long, device=device)
    accum_losses = torch.zeros(batch_size, device=device)
    loss_steps = torch.zeros(batch_size, device=device)

    white_wins, black_wins, draws = 0, 0, 0
    completed_games = 0

    batch_idx = 1
    b_w_wins, b_b_wins, b_draws = 0, 0, 0
    b_plies, b_losses, b_omegas = [], [], []

    batch_start_time = time.time()
    total_start_time = time.time()
    latest_omegas = torch.zeros(batch_size, device=device)

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

        # 1. White Agent Move
        if w_agent_mask.any():
            w_idx = torch.where(w_agent_mask)[0]
            env_input, threat = env.get_obs(agent_noise=agent_noise)

            acts, next_s, next_o, next_e, next_l, m_loss, omgs = deliberate_and_train_ply(
                model=model, optimizer=optimizer,
                env_input=env_input, threat=threat,
                state_buf=state_w, output_buf=output_w,
                error_ema_buf=error_w, lp_ema_buf=lp_w,
                active_idx=w_idx, legal_mask=legal_mask,
                thinking_steps=thinking_steps
            )

            actions[w_idx] = acts
            state_w[w_idx] = next_s
            output_w[w_idx] = next_o
            error_w[w_idx] = next_e
            lp_w[w_idx] = next_l
            latest_omegas[w_idx] = omgs
            accum_losses[w_idx] += m_loss
            loss_steps[w_idx] += 1

        # 2. White Engine Move
        if w_engine_mask.any():
            w_eng_idx = torch.where(w_engine_mask)[0]
            with torch.no_grad():
                is_w = torch.ones(len(w_eng_idx), dtype=torch.bool, device=device)
                actions[w_eng_idx] = select_cuda_engine_move(
                    env.boards[w_eng_idx], legal_mask[w_eng_idx], is_w, device,
                    env=env, b_idx=w_eng_idx
                )

        # 3. Black Agent Move
        if b_agent_mask.any():
            b_idx = torch.where(b_agent_mask)[0]
            env_input, threat = env.get_obs(agent_noise=agent_noise)

            acts, next_s, next_o, next_e, next_l, m_loss, omgs = deliberate_and_train_ply(
                model=model, optimizer=optimizer,
                env_input=env_input, threat=threat,
                state_buf=state_b, output_buf=output_b,
                error_ema_buf=error_b, lp_ema_buf=lp_b,
                active_idx=b_idx, legal_mask=legal_mask,
                thinking_steps=thinking_steps
            )

            actions[b_idx] = acts
            state_b[b_idx] = next_s
            output_b[b_idx] = next_o
            error_b[b_idx] = next_e
            lp_b[b_idx] = next_l
            latest_omegas[b_idx] = omgs
            accum_losses[b_idx] += m_loss
            loss_steps[b_idx] += 1

        # 4. Black Engine Move
        if b_engine_mask.any():
            b_eng_idx = torch.where(b_engine_mask)[0]
            with torch.no_grad():
                is_w = torch.zeros(len(b_eng_idx), dtype=torch.bool, device=device)
                actions[b_eng_idx] = select_cuda_engine_move(
                    env.boards[b_eng_idx], legal_mask[b_eng_idx], is_w, device,
                    env=env, b_idx=b_eng_idx
                )

        dones, winners = env.step(actions, dones, winners)
        plies[active] += 1

        max_plies_draw = (plies >= max_plies) & (~dones)
        if max_plies_draw.any():
            dones = dones | max_plies_draw
            winners = torch.where(max_plies_draw, torch.tensor(3, dtype=torch.int8, device=device), winners)

        finished_mask = dones.clone()
        num_finished = finished_mask.sum().item()

        if num_finished > 0:
            finished_indices = torch.where(finished_mask)[0]

            # 1. Survival selection
            resolve_fittest_agent_states(
                finished_mask, winners, match_modes,
                agent_state, agent_output, agent_error, agent_lp,
                state_w, output_w, error_w, lp_w,
                state_b, output_b, error_b, lp_b
            )

            # 2. Update pure outcome noise
            update_agent_somatic_noise(agent_noise, finished_mask, winners, match_modes)

            # 3. Log metrics
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

                    # TRIGGER STOCKFISH TRAINING BATCH EVERY 10 BATCHES
                    if batch_idx % sf_train_interval_batches == 0:
                        train_stockfish_batch(
                            model=model,
                            optimizer=optimizer,
                            num_games=sf_train_games,
                            target_nodes=1500,
                            thinking_steps=thinking_steps,
                            device=device
                        )

                        os.makedirs("out", exist_ok=True)
                        ckpt_path = f"out/ckpt({batch_idx}).pt"
                        torch.save({
                            'batch_idx': batch_idx,
                            'completed_games': completed_games,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'agent_noise': agent_noise,
                            'hyperparams': {k: v.data for k, v in cfg.params.items()}
                        }, ckpt_path)
                        print(f"[Checkpoint] Saved to {ckpt_path}\n")

                    batch_idx += 1
                    b_w_wins, b_b_wins, b_draws = 0, 0, 0
                    b_plies, b_losses, b_omegas = [], [], []
                    batch_start_time = time.time()

                if completed_games >= total_target_games:
                    break

            match_modes[finished_mask] = (match_modes[finished_mask] + 1) % 4
            env.reset(mask=finished_mask)

            state_w[finished_indices] = agent_state[finished_indices].clone()
            output_w[finished_indices] = agent_output[finished_indices].clone()
            error_w[finished_indices] = agent_error[finished_indices].clone()
            lp_w[finished_indices] = agent_lp[finished_indices].clone()

            state_b[finished_indices] = agent_state[finished_indices].clone()
            output_b[finished_indices] = agent_output[finished_indices].clone()
            error_b[finished_indices] = agent_error[finished_indices].clone()
            lp_b[finished_indices] = agent_lp[finished_indices].clone()

            plies[finished_mask] = 0
            accum_losses[finished_mask] = 0.0
            loss_steps[finished_mask] = 0.0
            dones[finished_mask] = False
            winners[finished_mask] = 0

    total_time = time.time() - total_start_time
    print(f"\nCompleted {completed_games} games in {total_time:.2f}s ({completed_games / total_time:.1f} games/sec)")

    # ========================================================
    # POST-TRAINING: CROWN THE CHAMPION & RUN GAUNTLET
    # ========================================================
    os.makedirs("out", exist_ok=True)
    
    champion = run_internal_tournament(
        model=model,
        agent_state=agent_state,
        agent_output=agent_output,
        agent_error=agent_error,
        agent_lp=agent_lp,
        agent_noise=agent_noise,
        top_k=128,
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
        model=model,
        champion_package=champion,
        node_levels=[100, 500, 1000, 1500, 3000, 10000],
        games_per_tier=20,
        device=device
    )


# ==========================================
# 8. CHAMPION TOURNAMENT & SF GAUNTLET
# ==========================================
def run_internal_tournament(model, agent_state, agent_output, agent_error, agent_lp, agent_noise, top_k=16, device="cuda"):
    print(f"\n========================================================")
    print(f"---> STAGE 1: Running Internal Championship Tournament (Top {top_k} Candidates)")
    print(f"========================================================")

    _, top_indices = torch.topk(agent_noise, k=top_k, largest=False)
    top_indices = top_indices.tolist()

    matchups = []
    for i in range(top_k):
        for j in range(top_k):
            if i != j:
                matchups.append((top_indices[i], top_indices[j]))

    num_matches = len(matchups)
    print(f"Running {num_matches} simultaneous tournament matches on {device}...")

    tourney_env = GuardrailedTorchChess(batch_size=num_matches, device=device)
    scores = {cand_idx: 0.0 for cand_idx in top_indices}

    w_cand = torch.tensor([m[0] for m in matchups], device=device)
    b_cand = torch.tensor([m[1] for m in matchups], device=device)

    s_w = agent_state[w_cand].clone()
    o_w = agent_output[w_cand].clone()
    e_w = agent_error[w_cand].clone()
    l_w = agent_lp[w_cand].clone()

    s_b = agent_state[b_cand].clone()
    o_b = agent_output[b_cand].clone()
    e_b = agent_error[b_cand].clone()
    l_b = agent_lp[b_cand].clone()

    tourney_noise = torch.full((num_matches,), 0.01, device=device)

    dones = torch.zeros(num_matches, dtype=torch.bool, device=device)
    winners = torch.zeros(num_matches, dtype=torch.int8, device=device)
    plies = torch.zeros(num_matches, dtype=torch.long, device=device)

    model.eval()
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
            with torch.no_grad():
                _, next_s, next_o, next_e, next_l, _ = model(
                    env_input[w_idx], threat[w_idx],
                    s_w[w_idx], o_w[w_idx], e_w[w_idx], l_w[w_idx]
                )
                s_w[w_idx], o_w[w_idx], e_w[w_idx], l_w[w_idx] = next_s, next_o, next_e, next_l
                raw_logits = model.get_action_logits(next_s)
                actions[w_idx] = torch.argmax(raw_logits + legal_mask[w_idx], dim=-1)

        if b_turn.any():
            b_idx = torch.where(b_turn)[0]
            env_input, threat = tourney_env.get_obs(agent_noise=tourney_noise)
            with torch.no_grad():
                _, next_s, next_o, next_e, next_l, _ = model(
                    env_input[b_idx], threat[b_idx],
                    s_b[b_idx], o_b[b_idx], e_b[b_idx], l_b[b_idx]
                )
                s_b[b_idx], o_b[b_idx], e_b[b_idx], l_b[b_idx] = next_s, next_o, next_e, next_l
                raw_logits = model.get_action_logits(next_s)
                actions[b_idx] = torch.argmax(raw_logits + legal_mask[b_idx], dim=-1)

        dones, winners = tourney_env.step(actions, dones, winners)
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
    total_cand_games = (top_k - 1) * 2
    winrate = (max_pts / total_cand_games) * 100.0

    print(f"\n---> Tournament Complete!")
    print(f"  Crown Champion: Agent Slot #{best_cand_idx}")
    print(f"  Score: {max_pts:.1f} / {total_cand_games} ({winrate:.1f}% Win/Draw Rate)")
    print(f"  Allostatic Noise Level: {agent_noise[best_cand_idx].item():.4f}")

    champion_package = {
        'slot_id': best_cand_idx,
        'score': max_pts,
        'state': agent_state[best_cand_idx:best_cand_idx+1].clone(),
        'output': agent_output[best_cand_idx:best_cand_idx+1].clone(),
        'error_ema': agent_error[best_cand_idx:best_cand_idx+1].clone(),
        'lp_ema': agent_lp[best_cand_idx:best_cand_idx+1].clone(),
        'noise': agent_noise[best_cand_idx:best_cand_idx+1].clone(),
    }
    return champion_package


def run_stockfish_gauntlet(model, champion_package, node_levels=[100, 500, 1000, 1500], games_per_tier=20, device="cuda"):
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

        s_eval = champ_s.expand(games_per_tier, -1, -1).clone()
        o_eval = champ_o.expand(games_per_tier, -1, -1).clone()
        e_eval = champ_e.expand(games_per_tier, -1).clone()
        l_eval = champ_l.expand(games_per_tier, -1).clone()

        model.eval()
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

                with torch.no_grad():
                    _, next_s, next_o, next_e, next_l, _ = model(
                        env_input[ag_idx], threat[ag_idx],
                        s_eval[ag_idx], o_eval[ag_idx], e_eval[ag_idx], l_eval[ag_idx]
                    )
                    s_eval[ag_idx], o_eval[ag_idx] = next_s, next_o
                    e_eval[ag_idx], l_eval[ag_idx] = next_e, next_l

                    raw_logits = model.get_action_logits(next_s)
                    actions[ag_idx] = torch.argmax(raw_logits + legal_masks[ag_idx], dim=-1)

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

            dones, winners = eval_env.step(actions, dones, winners)
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
    torch.set_float32_matmul_precision('high')
    run_agent_experiment(
        total_target_games=50000,
        batch_size=64,
        num_nodes=1024,
        sf_train_interval_batches=10,
        sf_train_games=32,
        thinking_steps=4,
        max_plies=400
    )