# =====================================================================
# chess_bot.py
# High-Throughput Dialectic PATT Chess Bot (Champion Architecture)
# =====================================================================

import argparse
import math
import os
import queue
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. CHESS ENVIRONMENT (Fast Tensor Engine)
# ==========================================
PIECE_CHARS = ["", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]

def fast_board_to_fen(board_np, is_w, rights_np, ep, halfc, fulln):
    rows = []
    for r in range(7, -1, -1):
        empty = 0
        r_chars = []
        for c in range(8):
            p = board_np[r, c]
            if p == 0:
                empty += 1
            else:
                if empty:
                    r_chars.append(str(empty))
                    empty = 0
                r_chars.append(PIECE_CHARS[p])
        if empty:
            r_chars.append(str(empty))
        rows.append("".join(r_chars))
    c_s = "".join([c for c, has in zip("KQkq", rights_np) if has]) or "-"
    ep_s = f"{chr(ord('a') + (ep % 8))}{(ep // 8) + 1}" if ep >= 0 else "-"
    return f"{'/'.join(rows)} {'w' if is_w else 'b'} {c_s} {ep_s} {halfc} {fulln}"

def uci_to_action(uci_str):
    if not uci_str or len(uci_str) < 4:
        return None
    try:
        fr = (int(uci_str[1]) - 1) * 8 + (ord(uci_str[0]) - ord('a'))
        to = (int(uci_str[3]) - 1) * 8 + (ord(uci_str[2]) - ord('a'))
        return fr * 64 + to
    except Exception:
        return None

class GuardrailedTorchChess:
    def __init__(self, batch_size, device="cuda", halfmove_limit=100):
        self.B, self.device, self.halfmove_limit = batch_size, device, halfmove_limit
        self.boards = torch.zeros((batch_size, 8, 8), dtype=torch.int8, device=device)
        self.players = torch.zeros(batch_size, dtype=torch.int8, device=device)
        self.piece_vals = torch.tensor([0, 100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000], dtype=torch.float32, device=device)
        self.castling_rights = torch.ones((batch_size, 4), dtype=torch.bool, device=device)
        self.en_passant_sq = torch.full((batch_size,), -1, dtype=torch.int8, device=device)
        self.halfmove_clock = torch.zeros(batch_size, dtype=torch.int16, device=device)
        self.pseudo_mask = torch.full((batch_size, 4096), -1e9, device=device)
        self.legal_mask = torch.full((batch_size, 4096), -1e9, device=device)

        dirs = torch.tensor([[1,0], [-1,0], [0,1], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1]], dtype=torch.long, device=device)
        self._r_dr, self._r_dc = dirs[:, 0].view(1, 8, 1), dirs[:, 1].view(1, 8, 1)
        self._r_st = torch.arange(1, 8, dtype=torch.long, device=device).view(1, 1, 7)
        self._k_d = torch.tensor([[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]], dtype=torch.long, device=device)
        self._kg_d = torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]], dtype=torch.long, device=device)
        self.reset()

    def reset(self, mask=None):
        idx = torch.nonzero(mask if mask is not None else torch.ones(self.B, dtype=torch.bool, device=self.device)).squeeze(-1)
        if idx.numel() == 0:
            return
        self.boards[idx] = torch.tensor([
            [4,2,3,5,6,3,2,4], [1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [7,7,7,7,7,7,7,7], [10,8,9,11,12,9,8,10]
        ], dtype=torch.int8, device=self.device)
        self.players[idx] = 0
        self.castling_rights[idx] = True
        self.en_passant_sq[idx] = -1
        self.halfmove_clock[idx] = 0

    def get_material(self):
        flat = self.boards.view(self.B, 64).long().clamp(0, 12)
        vals = self.piece_vals[flat]
        return (vals * ((flat >= 1) & (flat <= 5)).float()).sum(dim=-1), (vals * ((flat >= 7) & (flat <= 11)).float()).sum(dim=-1)

    def _is_square_attacked(self, target_sq, attacking_player, boards):
        B = boards.shape[0]
        flat = boards.view(B, 64)
        if not isinstance(target_sq, torch.Tensor):
            target_sq = torch.full((B,), target_sq, dtype=torch.long, device=self.device)
        elif target_sq.dim() == 0:
            target_sq = target_sq.expand(B)

        if not isinstance(attacking_player, torch.Tensor):
            attacking_player = torch.full((B,), attacking_player, dtype=torch.int8, device=self.device)
        elif attacking_player.dim() == 0:
            attacking_player = attacking_player.expand(B)

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
        B = b.shape[0]
        p_val = player if isinstance(player, torch.Tensor) else torch.full((B,), player, device=self.device)
        k_val = torch.where(p_val == 0, 6, 12).to(torch.int8).unsqueeze(1)
        k_sqs = (b.view(B, 64) == k_val).max(dim=1).indices
        return self._is_square_attacked(k_sqs, 1 - p_val, b)

    def compute_legal_masks(self):
        self.pseudo_mask.fill_(-1e9)
        sq_f = torch.arange(64, device=self.device)
        rf, cf = sq_f // 8, sq_f % 8
        pieces = self.boards.view(self.B, 64)
        is_w = (self.players == 0).unsqueeze(1)
        b_idx, f_idx = torch.nonzero(torch.where(is_w, (pieces >= 1) & (pieces <= 6), (pieces >= 7) & (pieces <= 12)), as_tuple=True)
        if b_idx.numel() == 0:
            return self.pseudo_mask

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
        f1, d1_sq, f8, d8 = [self._is_square_attacked(sq, p, self.boards) for sq, p in [(5,1), (3,1), (61,0), (59,0)]]
        for cnd, act in [
            ((self.players == 0) & self.castling_rights[:, 0] & (pieces[:, 5:7] == 0).all(1) & (~wc) & (~f1), 4 * 64 + 6),
            ((self.players == 0) & self.castling_rights[:, 1] & (pieces[:, 1:4] == 0).all(1) & (~wc) & (~d1_sq), 4 * 64 + 2),
            ((self.players == 1) & self.castling_rights[:, 2] & (pieces[:, 61:63] == 0).all(1) & (~bc) & (~f8), 60 * 64 + 62),
            ((self.players == 1) & self.castling_rights[:, 3] & (pieces[:, 57:60] == 0).all(1) & (~bc) & (~d8), 60 * 64 + 58)
        ]: self.pseudo_mask[cnd, act] = 0.0

        sr, sc = r_f.view(-1, 1, 1) + self._r_dr * self._r_st, c_f.view(-1, 1, 1) + self._r_dc * self._r_st
        in_b = (sr >= 0) & (sr < 8) & (sc >= 0) & (sc < 8)
        to_sq = (sr.clamp(0, 7) * 8 + sc.clamp(0, 7)).long()

        b_exp = b_idx.view(-1, 1, 1).expand_as(to_sq)
        f_exp = f_idx.view(-1, 1, 1).expand_as(to_sq)

        dp = pieces[b_exp, to_sq]
        d_own = torch.where(is_pw.view(-1, 1, 1), (dp >= 1) & (dp <= 6), (dp >= 7) & (dp <= 12))
        d_val = torch.zeros(b_idx.shape[0], 8, 1, dtype=torch.bool, device=self.device)
        d_val[:, :4, 0], d_val[:, 4:, 0] = ((pt == 4) | (pt == 5)).unsqueeze(1), ((pt == 3) | (pt == 5)).unsqueeze(1)
        can_reach = in_b & (torch.cumsum(F.pad((in_b & (dp > 0))[:, :, :-1], (1, 0), value=0).to(torch.int32), dim=2) == 0) & d_val & (~d_own)
        self.pseudo_mask[b_exp[can_reach], f_exp[can_reach] * 64 + to_sq[can_reach]] = 0.0

        self.legal_mask.fill_(-1e9)
        val_b, val_act = torch.nonzero(self.pseudo_mask > -1e8, as_tuple=True)
        if val_b.numel() == 0:
            return self.legal_mask

        fr_sqs, to_sqs, turn = val_act // 64, val_act % 64, self.players[val_b]
        N = val_b.shape[0]
        sub = self.boards[val_b].clone().view(N, 64)
        moving = sub[torch.arange(N, device=self.device), fr_sqs]

        is_ep = ((moving == 1) | (moving == 7)) & (to_sqs == self.en_passant_sq[val_b]) & (self.en_passant_sq[val_b] >= 0)
        if is_ep.any():
            ep_i = torch.nonzero(is_ep).squeeze(-1)
            sub[ep_i, torch.where(turn[ep_i] == 0, to_sqs[ep_i] - 8, to_sqs[ep_i] + 8)] = 0

        sub[torch.arange(N, device=self.device), to_sqs] = moving
        sub[torch.arange(N, device=self.device), fr_sqs] = 0

        p_k_sqs = (self.boards.view(self.B, 64) == torch.where(self.players == 0, 6, 12).to(torch.int8).unsqueeze(1)).max(dim=1).indices
        k_sqs = torch.where((moving == 6) | (moving == 12), to_sqs, p_k_sqs[val_b])
        valid = ~self._is_square_attacked(k_sqs, 1 - turn, sub.view(N, 8, 8))
        self.legal_mask[val_b[valid], val_act[valid]] = 0.0
        return self.legal_mask

    def step(self, action_indices, dones, winners, legal_mask=None):
        B_idx = torch.arange(self.B, device=self.device)
        from_sq, to_sq = action_indices // 64, action_indices % 64
        fr_r, fr_c, to_r, to_c = from_sq // 8, from_sq % 8, to_sq // 8, to_sq % 8
        piece, dest_piece, active = self.boards[B_idx, fr_r, fr_c], self.boards[B_idx, to_r, to_c], ~dones

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

        wm, bm = self.get_material()
        w_res = active & ((wm >= bm + 1200) | ((bm == 0) & (wm > 300)))
        b_res = active & ((bm >= wm + 1200) | ((wm == 0) & (bm > 300)))
        winners = torch.where(w_res, 1, winners)
        winners = torch.where(b_res, 2, winners)
        active = active & ~(w_res | b_res)

        draws = active & (self.halfmove_clock >= self.halfmove_limit)
        winners = torch.where(draws, 3, winners)
        return dones | w_res | b_res | draws, winners

# ==========================================
# 1. CHAMPION PATT ARCHITECTURE
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        out = (x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
        return out * self.weight

class FlexPattLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int, cosine_patt: bool = True):
        super().__init__()
        self.cosine_patt = cosine_patt
        self.key_tokens = nn.Parameter(torch.empty(num_tokens, input_dim))
        self.val_tokens = nn.Parameter(torch.empty(num_tokens, output_dim))
        nn.init.kaiming_uniform_(self.key_tokens, a=math.sqrt(5))
        if cosine_patt:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            nn.init.normal_(self.val_tokens, std=1.0 / math.sqrt(output_dim))
        else:
            nn.init.kaiming_uniform_(self.val_tokens, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cosine_patt:
            x_n = F.normalize(x.float(), dim=-1).to(x.dtype)
            k_n = F.normalize(self.key_tokens.float(), dim=-1).to(x.dtype)
            scale = self.logit_scale.exp().clamp(max=50.0).to(x.dtype)
            scores = (x_n @ k_n.T) * scale
        else:
            scores = (x @ self.key_tokens.T) / math.sqrt(self.key_tokens.shape[0])
        attn = F.softmax(scores.float(), dim=-1).to(self.val_tokens.dtype)
        return attn @ self.val_tokens

class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = max(32, ((int(dim * 8 / 3) + 15) // 16) * 16)
        self.w13 = nn.Linear(dim, hidden * 2, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * val)

class ModernPattAttentionLayer(nn.Module):
    """Post-LN + RMSNorm + Wo + QKNorm + SwiGLU + CosinePatt Engine."""
    def __init__(self, dim: int = 128, num_patt_tokens: int = 256, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.patt_q = FlexPattLayer(dim, dim, num_patt_tokens, cosine_patt=True)
        self.patt_k = FlexPattLayer(dim, dim, num_patt_tokens, cosine_patt=True)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_o = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.mlp = SwiGLUFFN(dim)

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def _attn(self, h: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        q = self.patt_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.patt_k(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        ctx = F.scaled_dot_product_attention(q, k, v)
        ctx = ctx.transpose(1, 2).reshape(B, N, D)
        return self.proj_o(ctx)

    def forward(self, x: torch.Tensor, depth_token: torch.Tensor = None) -> torch.Tensor:
        h = (x + depth_token) if depth_token is not None else x
        x = self.norm1(x + self._attn(h))
        x = self.norm2(x + self.mlp(x))
        return x

class DialecticPattChessBot(nn.Module):
    def __init__(self, dim: int = 128, num_patt_tokens: int = 256, rollout_steps: int = 6, num_heads: int = 8, head_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rollout_steps = rollout_steps

        self.piece_embed = nn.Embedding(14, dim)
        self.rank_embed = nn.Embedding(8, dim)
        self.file_embed = nn.Embedding(8, dim)
        self.turn_embed = nn.Embedding(2, dim)
        self.castling_proj = nn.Linear(4, dim, bias=False)
        self.ply_embed = nn.Embedding(16, dim)
        self.register_buffer("ply_indices", torch.arange(16), persistent=False)

        # Weight-Tied Thesis & Antithesis engines
        self.thesis_engine = ModernPattAttentionLayer(dim=dim, num_patt_tokens=num_patt_tokens, num_heads=num_heads)
        self.antithesis_engine = ModernPattAttentionLayer(dim=dim, num_patt_tokens=num_patt_tokens, num_heads=num_heads)

        self.final_norm = RMSNorm(dim)
        self.proj_from = nn.Linear(dim, head_dim, bias=False)
        self.proj_to = nn.Linear(dim, head_dim, bias=False)
        self.bias_from = nn.Linear(dim, 1, bias=False)
        self.bias_to = nn.Linear(dim, 1, bias=False)

        sqs = torch.arange(64)
        self.register_buffer("sq_ranks", sqs // 8, persistent=False)
        self.register_buffer("sq_files", sqs % 8, persistent=False)

    def forward(self, boards: torch.Tensor, is_white: torch.Tensor, castling_rights: torch.Tensor = None, legal_mask: torch.Tensor = None):
        B = boards.shape[0]
        flat = boards.view(B, 64).long().clamp(0, 12)
        turn_idx = torch.where(is_white, 0, 1).long()

        pos_embed = self.rank_embed(self.sq_ranks) + self.file_embed(self.sq_files)
        tokens = self.piece_embed(flat) + pos_embed.unsqueeze(0) + self.turn_embed(turn_idx).unsqueeze(1)

        if castling_rights is not None:
            tokens = tokens + self.castling_proj(castling_rights.float()).unsqueeze(1)

        x = tokens
        for ply in range(self.rollout_steps):
            depth_token = self.ply_embed(self.ply_indices[ply]).view(1, 1, self.dim)
            engine = self.thesis_engine if (ply % 2 == 0) else self.antithesis_engine
            x = engine(x, depth_token=depth_token)

        x = self.final_norm(x)

        q_fr = self.proj_from(x)
        k_to = self.proj_to(x)
        pairwise = torch.bmm(q_fr, k_to.transpose(1, 2)) * (1.0 / math.sqrt(self.head_dim))

        b_fr = self.bias_from(x)
        b_to = self.bias_to(x).transpose(1, 2)

        logits = (pairwise + b_fr + b_to).view(B, 4096)
        if legal_mask is not None:
            logits = logits + legal_mask
        return logits

    @torch.no_grad()
    def select_action(self, boards, is_white, legal_mask, castling_rights=None, temperature=0.0):
        logits = self.forward(boards, is_white, castling_rights=castling_rights, legal_mask=legal_mask)
        if temperature <= 1e-4:
            return torch.argmax(logits, dim=-1)
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

# ==========================================
# 2. FAST STOCKFISH ENGINE & WORKER POOL
# ==========================================
#hard error on failure, this is used everywhere. silent stupid is worthless.
class StockfishEngine:
    def __init__(self, stockfish_path="stockfish", threads=4, hash_mb=128):
        self.path = stockfish_path
        self.threads = threads
        self.hash_mb = hash_mb
        self.proc = None
        self.current_mode = None
        self._spawn()

    def _wait_ready(self, context="isready"):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                rc = self.proc.poll()
                raise RuntimeError(f"Stockfish EOF during {context} (exit_code={rc})")
            if "readyok" in line:
                return

    def _spawn(self):
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send([
            "uci",
            f"setoption name Threads value {self.threads}",
            f"setoption name Hash value {self.hash_mb}",
            "isready",
        ])
        self._wait_ready("_spawn")

    def _send(self, commands):
        if self.proc is None or self.proc.poll() is not None:
            rc = None if self.proc is None else self.proc.poll()
            raise RuntimeError(f"Cannot send to dead Stockfish process (exit_code={rc})")
        try:
            self.proc.stdin.write("\n".join(commands) + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"Stockfish stdin write failed ({commands}): {e}") from e

    def _read_bestmove(self, fen=""):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                rc = self.proc.poll()
                raise RuntimeError(f"Stockfish EOF in _read_bestmove (exit_code={rc}, fen={fen!r})")
            line = line.strip()
            if line.startswith("bestmove"):
                parts = line.split()
                bm = parts[1] if len(parts) > 1 else None
                if not bm or bm in ("(none)", "0000"):
                    raise RuntimeError(f"Stockfish returned empty bestmove={bm!r} for FEN={fen!r}")
                return bm

    def _validate_fen_kings(self, fen: str):
        board_part = fen.split()[0]
        if "K" not in board_part or "k" not in board_part:
            raise RuntimeError(f"Corrupted FEN (missing King): {fen!r}")

    def set_elo(self, elo: int):
        elo = int(elo)
        mode_key = f"elo_{elo}"
        if self.current_mode == mode_key:
            return
        self._send([
            "setoption name UCI_LimitStrength value true",
            f"setoption name UCI_Elo value {elo}",
            "isready",
        ])
        self._wait_ready(f"set_elo({elo})")
        self.current_mode = mode_key

    
    def get_elo_move(self, fen: str, elo: int = 1350):
        self._validate_fen_kings(fen)
        elo = int(elo)
        if self.current_mode != f"elo_{elo}":
            self.set_elo(elo)
            if elo > 2000:
                print(
                    f"[Warning] Elo {elo} > 2000: Skill::pick_best may require depth > 8 "
                    f"(playing at capped depth 8 instead)."
                )
        self._send([f"position fen {fen}", "go depth 8"])
        return self._read_bestmove(fen=fen)

    def get_teacher_move(self, fen: str, nodes: int = 10000):
        self._validate_fen_kings(fen)
        if self.current_mode != "teacher":
            self._send([
                "setoption name UCI_LimitStrength value false",
                "isready",
            ])
            self._wait_ready("get_teacher_move")
            self.current_mode = "teacher"
        self._send([f"position fen {fen}", f"go nodes {nodes}"])
        return self._read_bestmove(fen=fen)

    def get_move(self, fen: str, nodes: int = None, depth: int = 8, movetime_ms: int = None, elo: int = None, **kwargs):
        if nodes is not None:
            return self.get_teacher_move(fen, nodes=nodes)
        return self.get_elo_move(fen, elo=elo or 1350, depth=depth, movetime_ms=movetime_ms)

    def close(self):
        if self.proc:
            try:
                if self.proc.poll() is None:
                    self.proc.stdin.write("quit\n")
                    self.proc.stdin.flush()
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None

class FastStockfishEvalPool:
    def __init__(self, sf_path, num_workers=8, hash_mb=32):
        self.num_workers = num_workers
        self.engine_queue = queue.Queue()
        self.engines = []
        for _ in range(num_workers):
            eng = StockfishEngine(sf_path, threads=1, hash_mb=hash_mb)
            self.engines.append(eng)
            self.engine_queue.put(eng)
        self.pool = ThreadPoolExecutor(max_workers=num_workers)
        self.current_elo = None

    def set_elo(self, elo: int):
        elo = int(elo)
        if self.current_elo == elo:
            return
        for eng in self.engines:
            eng.set_elo(elo)
        self.current_elo = elo

    def play_batch(self, tasks, elo: int, movetime_ms: int = 20):
        if not tasks:
            return {}
        self.set_elo(elo)

        def query(task):
            idx, fen, legals = task
            eng = self.engine_queue.get()
            try:
                bm = eng.get_elo_move(fen, elo=elo, movetime_ms=movetime_ms)
                act = uci_to_action(bm) if bm else None
                return idx, (act if (act is not None and act in legals) else legals[0])
            finally:
                self.engine_queue.put(eng)

        return dict(self.pool.map(query, tasks))

    def close(self):
        self.pool.shutdown(wait=True)
        for eng in self.engines:
            eng.close()

# ==========================================
# 3. FAST EVALUATION GAUNTLET
# ==========================================
@torch.no_grad()
def evaluate_model_elo(model, sf_pool: FastStockfishEvalPool, elo=1350, games=64, max_plies=400, movetime_ms=20, device="cuda"):
    model.eval()
    sf_pool.set_elo(elo)

    env = GuardrailedTorchChess(games, device=device, halfmove_limit=min(100, max_plies))
    bot_is_white = torch.arange(games, device=device) < (games // 2)

    dones = torch.zeros(games, dtype=torch.bool, device=device)
    winners = torch.zeros(games, dtype=torch.int8, device=device)
    plies = torch.zeros(games, dtype=torch.long, device=device)

    while not dones.all() and (plies < max_plies).any():
        active = ~dones
        l_mask = env.compute_legal_masks()
        actions = torch.zeros(games, dtype=torch.long, device=device)

        is_w = (env.players == 0)
        is_bot = active & (is_w == bot_is_white)
        is_sf = active & (~is_bot)
        in_chk_all = env.is_in_check(env.players, env.boards)

        if is_bot.any():
            has_legals = (l_mask > -1e8).any(dim=-1)
            bot_terminal = is_bot & (~has_legals)

            if bot_terminal.any():
                dones = dones | bot_terminal
                opp_wins = torch.where(is_w, 2, 1)
                winners = torch.where(bot_terminal, torch.where(in_chk_all, opp_wins, 3), winners)
                is_bot = is_bot & has_legals

            if is_bot.any():
                bot_acts = model.select_action(env.boards, is_w, l_mask, castling_rights=env.castling_rights, temperature=0.0)
                actions[is_bot] = bot_acts[is_bot]

        if is_sf.any():
            sf_indices = torch.nonzero(is_sf).squeeze(-1).tolist()
            boards_np = env.boards.cpu().numpy()
            is_w_list = is_w.cpu().tolist()
            rights_np = env.castling_rights.cpu().numpy()
            ep_list = env.en_passant_sq.cpu().tolist()
            half_list = env.halfmove_clock.cpu().tolist()
            plies_list = plies.cpu().tolist()
            chk_list = in_chk_all.cpu().tolist()

            nz_cpu = torch.nonzero(l_mask > -1e8).cpu()
            legals_map = {i: [] for i in sf_indices}
            sf_set = set(sf_indices)
            for b_i, act in nz_cpu.tolist():
                if b_i in sf_set:
                    legals_map[b_i].append(act)

            sf_tasks = []
            for i in sf_indices:
                legals = legals_map[i]
                if not legals:
                    dones[i] = True
                    winners[i] = (2 if is_w_list[i] else 1) if chk_list[i] else 3
                else:
                    fen = fast_board_to_fen(boards_np[i], is_w_list[i], rights_np[i], ep_list[i], half_list[i], (plies_list[i] // 2) + 1)
                    sf_tasks.append((i, fen, legals))

            if sf_tasks:
                sf_moves = sf_pool.play_batch(sf_tasks, elo=elo, movetime_ms=movetime_ms)
                for idx, act in sf_moves.items():
                    actions[idx] = act

        dones, winners = env.step(actions, dones, winners, legal_mask=l_mask)
        plies[active] += 1

        over = (plies >= max_plies) & (~dones)
        if over.any():
            dones = dones | over
            wm, bm = env.get_material()
            diff = wm - bm
            winners = torch.where(over & (diff >= 200), 1, winners)
            winners = torch.where(over & (diff <= -200), 2, winners)
            winners = torch.where(over & (diff.abs() < 200), 3, winners)

    bot_wins = (((winners == 1) & bot_is_white) | ((winners == 2) & (~bot_is_white))).sum().item()
    sf_wins = (((winners == 2) & bot_is_white) | ((winners == 1) & (~bot_is_white))).sum().item()
    draws = (winners == 3).sum().item()

    wm, bm = env.get_material()
    mat_lead = torch.where(bot_is_white, wm - bm, bm - wm).float().mean().item()

    return {
        "wins": bot_wins, "draws": draws, "losses": sf_wins,
        "score_str": f"{bot_wins}W/{draws}D/{sf_wins}L",
        "mat_str": f"{mat_lead:+5.0f} cp"
    }

# ==========================================
# 4. COMPILED TRAINING PIPELINE
# ==========================================
def train_model(model, dataset, epochs=25, batch_size=2048, lr=1.0e-3, device="cuda"):
    print("\n" + "=" * 90)
    print(f" TRAINING CHAMPION DIALECTIC PATT BOT ({epochs} Epochs | Batch: {batch_size})")
    print("=" * 90)

    boards, is_white, targets = dataset["boards"], dataset["is_white"], dataset["targets"]
    castling = dataset.get("castling_rights", None)
    dataset_size = targets.shape[0]

    effective_size = dataset_size - (dataset_size % batch_size)
    steps_per_epoch = effective_size // batch_size
    total_steps = steps_per_epoch * epochs

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim < 2 or any(k in name for k in ("embed", "castling_proj")):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "lr": lr, "weight_decay": 1e-4},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0}
    ], fused=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.03)
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    compiled_model = torch.compile(model, mode="reduce-overhead")
    compiled_model.train()

    start_t = time.time()
    global_step = 0

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(effective_size, device=device)
        epoch_start_t = time.time()
        last_loss = 0.0

        for i in range(0, effective_size, batch_size):
            global_step += 1
            idx = perm[i:i + batch_size]
            b, w, y = boards[idx], is_white[idx], targets[idx]
            c = castling[idx] if castling is not None else None

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits = compiled_model(b, w, castling_rights=c)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            last_loss = loss.item()

        dt = time.time() - epoch_start_t
        speed = effective_size / max(1e-4, dt)
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] Step [{global_step:05d}/{total_steps:05d}] | "
              f"Loss: {last_loss:.4f} | Speed: {speed:6.0f} pos/s | Elapsed: {dt:.1f}s")

    total_time = time.time() - start_t
    print(f"\n[Training Finished] Processed {effective_size * epochs:,} positions in {total_time:.2f}s")
# ==========================================
# 5. RL SPARRING PARTNER LOADER
# ==========================================
class PattBotSparringPartner:
    """
    Self-play / RL sparring partner loader for DialecticPattChessBot checkpoints.
    Supports fast batched tensor inference, temperature sampling, and greedy play.
    """
    def __init__(self, ckpt_path="out/dialectic_patt_bot.pt", device="cuda"):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})

        dim = cfg.get("dim", 128)
        num_heads = cfg.get("num_heads", 8 if dim >= 128 else (6 if dim == 96 else 4))
        head_dim = cfg.get("head_dim", dim // num_heads)

        self.model = DialecticPattChessBot(
            dim=dim,
            num_patt_tokens=cfg.get("num_patt_tokens", 256),
            rollout_steps=cfg.get("rollout_steps", 6),
            num_heads=num_heads,
            head_dim=head_dim,
        ).to(device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def get_action(self, boards, is_white, legal_mask, castling_rights=None, temperature=0.0):
        return self.model.select_action(
            boards,
            is_white,
            legal_mask=legal_mask,
            castling_rights=castling_rights,
            temperature=temperature,
        )
    
# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Champion Dialectic Patt Chess Bot Trainer")
    parser.add_argument("--dataset", type=str, default="data/chess/sf_dataset.pt", help="Path to cleaned SF dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Learning rate")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_tokens", type=int, default=256, help="PATT concept tokens")
    parser.add_argument("--rollout_steps", type=int, default=6, help="Tied dialectic rollout plies")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--out", type=str, default="out/dialectic_patt_bot.pt", help="Checkpoint output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    os.makedirs("out", exist_ok=True)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found at '{args.dataset}'")

    print(f"Loading dataset from '{args.dataset}' into {device}...")
    dataset = torch.load(args.dataset, map_location=device, weights_only=False)
    if "legal_masks" in dataset:
        del dataset["legal_masks"]
        torch.cuda.empty_cache()

    model_cfg = {
        "dim": args.dim,
        "num_patt_tokens": args.num_tokens,
        "rollout_steps": args.rollout_steps,
        "num_heads": args.num_heads,
        "head_dim": args.dim // args.num_heads
    }
    model = DialecticPattChessBot(**model_cfg).to(device)

    train_model(model, dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)

    torch.save({"model_state_dict": model.state_dict(), "config": model_cfg}, args.out)
    print(f"[Artifact Saved] -> {args.out}")

    sf_exec = shutil.which("stockfish") or ("./stockfish.exe" if os.path.exists("stockfish.exe") else None)
    if sf_exec:
        pool = FastStockfishEvalPool(sf_exec, num_workers=min(12, os.cpu_count() or 4), hash_mb=32)
        for elo in [1350, 1600]:
            print(f"\nEvaluating vs Stockfish Elo {elo} (64 Games)...")
            res = evaluate_model_elo(model, pool, elo=elo, games=64, max_plies=400, device=device)
            print(f"  Result vs {elo}: {res['score_str']} | Mean Material Delta: {res['mat_str']}")
        pool.close()

if __name__ == "__main__":
    main()