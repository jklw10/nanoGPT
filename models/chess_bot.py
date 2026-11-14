# =====================================================================
# chess_bot.py
# High-Throughput Dialectic PATT Chess Bot & Parallel Environment
# =====================================================================

import argparse
import math
import os
import shutil
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. CHESS ENVIRONMENT (For Eval & Sparring)
# ==========================================
PIECE_TO_FEN = {
    0: '', 1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
    7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'
}

def board_to_fen(board, turn, rights, ep=-1, halfc=0, fulln=1):
    rows = []
    for r in range(7, -1, -1):
        empty, r_s = 0, ""
        for c in range(8):
            p = int(board[r, c].item())
            if p == 0:
                empty += 1
            else:
                if empty:
                    r_s += str(empty)
                    empty = 0
                r_s += PIECE_TO_FEN.get(p, '')
        if empty:
            r_s += str(empty)
        rows.append(r_s)
    c_s = "".join([s for s, has in zip("KQkq", rights) if has]) or "-"
    ep_s = f"{chr(ord('a') + (ep % 8))}{(ep // 8) + 1}" if ep >= 0 else "-"
    return f"{'/'.join(rows)} {'w' if turn == 0 else 'b'} {c_s} {ep_s} {halfc} {fulln}"

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
# 1. OPTIMIZED DIALECTIC PATT ARCHITECTURE
# ==========================================
def tokenformer_theta(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    scale = 1.0 / math.sqrt(scores.shape[-1])
    return F.softmax(scores * scale, dim=dim)

class PattLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int):
        super().__init__()
        self.key_tokens = nn.Parameter(torch.empty(num_tokens, input_dim))
        self.val_tokens = nn.Parameter(torch.empty(num_tokens, output_dim))
        nn.init.kaiming_uniform_(self.key_tokens, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.val_tokens, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = x @ self.key_tokens.T
        attn = tokenformer_theta(scores, dim=-1)
        return attn @ self.val_tokens

class PattAttentionLayer(nn.Module):
    def __init__(self, dim: int = 64, num_patt_tokens: int = 128, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.patt_q = PattLayer(input_dim=dim, output_dim=dim, num_tokens=num_patt_tokens)
        self.patt_k = PattLayer(input_dim=dim, output_dim=dim, num_tokens=num_patt_tokens)
        self.proj_v = nn.Linear(dim, dim, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_norm = self.norm1(x)

        q = self.patt_q(x_norm)
        k = self.patt_k(x_norm)
        v = self.proj_v(x_norm)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        context = F.scaled_dot_product_attention(q, k, v)
        context = context.transpose(1, 2).reshape(B, N, D)

        x = x + context
        x = x + self.mlp(self.norm2(x))
        return x

class DialecticPattChessBot(nn.Module):
    def __init__(self, dim: int = 64, num_patt_tokens: int = 128, rollout_steps: int = 4, head_dim: int = 32):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rollout_steps = rollout_steps

        self.piece_embed = nn.Embedding(14, dim)
        self.rank_embed = nn.Embedding(8, dim)
        self.file_embed = nn.Embedding(8, dim)
        self.turn_embed = nn.Embedding(2, dim)
        self.castling_proj = nn.Linear(4, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

        self.rollout_layers = nn.ModuleList([
            PattAttentionLayer(dim=dim, num_patt_tokens=num_patt_tokens, num_heads=4)
            for _ in range(rollout_steps)
        ])

        self.final_norm = nn.LayerNorm(dim)
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

        x = self.norm(tokens)
        for layer in self.rollout_layers:
            x = layer(x)
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
# 2. STOCKFISH ENGINE INTERFACE
# ==========================================
class StockfishEngine:
    def __init__(self, stockfish_path="stockfish", threads=4, hash_mb=128):
        self.path = stockfish_path
        self.threads = threads
        self.hash_mb = hash_mb
        self.proc = None
        self.current_mode = None
        self._spawn()

    def _spawn(self):
        if self.proc:
            try: self.proc.terminate()
            except Exception: pass
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        self._send([
            "uci",
            f"setoption name Threads value {self.threads}",
            f"setoption name Hash value {self.hash_mb}",
            "isready"
        ])
        while "readyok" not in self.proc.stdout.readline():
            pass

    def _send(self, commands):
        try:
            self.proc.stdin.write("\n".join(commands) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass

    def _read_bestmove(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 and parts[1] not in ("(none)", "0000") else None

    def set_elo(self, elo: int):
        """
        Configures Stockfish to limit its strength to the requested Elo rating.
        Synchronizes with the engine via 'isready' -> 'readyok'.
        """
        elo = int(elo)
        mode_key = f"elo_{elo}"
        if self.current_mode == mode_key:
            return

        self._send([
            "setoption name UCI_LimitStrength value true",
            f"setoption name UCI_Elo value {elo}",
            "isready"
        ])
        while True:
            line = self.proc.stdout.readline()
            if not line or "readyok" in line:
                break

        self.current_mode = mode_key

    # USE CASE 1: MATCH PLAY (Elo Calibrated)
    def get_elo_move(self, fen: str, elo: int = 1350, movetime_ms: int = 50):
        """
        Stockfish Elo mode: Uses Stockfish's calibrated strength limiter.
        Does NOT constrain nodes, allowing the Elo curve to function properly.
        """
        if "K" not in fen.split()[0] or "k" not in fen.split()[0]:
            return None

        if self.current_mode != f"elo_{elo}":
            self.set_elo(elo)

        self._send([f"position fen {fen}", f"go movetime {movetime_ms}"])
        return self._read_bestmove()

    # USE CASE 2: DATASET TEACHER (Full Strength, Pure Search)
    def get_teacher_move(self, fen: str, nodes: int = 10000):
        """
        Pure Teacher mode: UCI_LimitStrength is completely disabled.
        Search is dictated purely by the exact node budget.
        """
        if "K" not in fen.split()[0] or "k" not in fen.split()[0]:
            return None

        if self.current_mode != "teacher":
            self._send([
                "setoption name UCI_LimitStrength value false",
                "isready"
            ])
            while "readyok" not in self.proc.stdout.readline():
                pass
            self.current_mode = "teacher"

        self._send([f"position fen {fen}", f"go nodes {nodes}"])
        return self._read_bestmove()

    def get_move(self, fen: str, nodes: int = None, movetime_ms: int = None, elo: int = None, **kwargs):
        if nodes is not None:
            return self.get_teacher_move(fen, nodes=nodes)
        return self.get_elo_move(fen, elo=elo or 1350, movetime_ms=movetime_ms or 20)

    def close(self):
        if self.proc:
            try:
                self._send(["quit"])
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None

# ==========================================
# 3. HIGH-SPEED COMPILED TRAINER
# ==========================================
def train_offline_stockfish(model, dataset, epochs=10, batch_size=2048, lr=2e-3, device="cuda"):
    print("\n" + "=" * 90)
    print(f" ULTRA-FAST OFFLINE TRAINING (Batch: {batch_size} | Pure Unmasked CE)")
    print("=" * 90)

    boards = dataset["boards"]
    is_white = dataset["is_white"]
    targets = dataset["targets"]
    castling = dataset.get("castling_rights", None)
    dataset_size = targets.shape[0]

    effective_size = dataset_size - (dataset_size % batch_size)
    steps_per_epoch = effective_size // batch_size
    total_steps = steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.05)

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"  [Setup] CUDA Device: {torch.cuda.get_device_name(0)} | AMP Type: {amp_dtype}")
    print(f"  [Volume] {effective_size:,} pos/epoch * {epochs} epochs = {effective_size * epochs:,} total pos")

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

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits = compiled_model(b, w, castling_rights=c)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            last_loss = loss.item()

            if global_step % max(1, steps_per_epoch // 2) == 0:
                elapsed = time.time() - start_t
                pos_processed = global_step * batch_size
                rate = pos_processed / max(1e-4, elapsed)
                print(f"   Epoch [{epoch:02d}/{epochs:02d}] Step [{global_step:05d}/{total_steps:05d}] | "
                      f"Loss: {last_loss:.4f} | Throughput: {rate:8.0f} pos/sec")

        dt = time.time() - epoch_start_t
        print(f"  --> Epoch {epoch:02d} Completed in {dt:.2f}s ({(effective_size / max(1e-4, dt)):.0f} pos/s) | Final Loss: {last_loss:.4f}")

    total_time = time.time() - start_t
    print(f"\n[Training Finished] Processed {effective_size * epochs:,} positions in {total_time:.2f}s "
          f"({(effective_size * epochs / max(1e-4, total_time)):.0f} avg pos/sec)")

# ==========================================
# 4. EVALUATION & BENCHMARKING
# ==========================================
@torch.no_grad()
def evaluate_model_elo(model, sf_engine, elo=1350, games=32, max_plies=400, device="cuda"):
    model.eval()
    sf_engine.set_elo(elo)

    env = GuardrailedTorchChess(games, device=device, halfmove_limit=max_plies)
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

        # 1. Batched GPU inference for bot
        if is_bot.any():
            has_legals = (l_mask > -1e8).any(dim=-1)
            bot_terminal = is_bot & (~has_legals)

            if bot_terminal.any():
                dones = dones | bot_terminal
                in_chk = env.is_in_check(env.players, env.boards)
                opp_wins = torch.where(is_w, 2, 1)  # White mated -> Black (2) wins; Black mated -> White (1) wins
                winners = torch.where(bot_terminal, torch.where(in_chk, opp_wins, 3), winners)
                is_bot = is_bot & has_legals

            if is_bot.any():
                bot_acts = model.select_action(
                    env.boards, is_w, l_mask, castling_rights=env.castling_rights, temperature=0.0
                )
                actions[is_bot] = bot_acts[is_bot]

        # 2. Stockfish inference (Elo-calibrated)
        if is_sf.any():
            for i in torch.nonzero(is_sf).squeeze(-1).tolist():
                legals = torch.nonzero(l_mask[i] > -1e8).squeeze(-1).tolist()
                if not legals:
                    dones[i] = True
                    in_chk = env.is_in_check(env.players[i:i+1], env.boards[i:i+1]).item()
                    winners[i] = (2 if is_w[i] else 1) if in_chk else 3
                    continue

                fen = board_to_fen(
                    env.boards[i], 0 if is_w[i] else 1, env.castling_rights[i].tolist(),
                    int(env.en_passant_sq[i].item()), int(env.halfmove_clock[i].item()),
                    int(plies[i].item() // 2 + 1)
                )
                bm = sf_engine.get_elo_move(fen, elo=elo, movetime_ms=20)
                act = uci_to_action(bm) if bm else None
                actions[i] = act if (act is not None and act in legals) else legals[0]

        dones, winners = env.step(actions, dones, winners, legal_mask=l_mask)
        plies[active] += 1

        # Timeout adjudication (+/- 200 cp)
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
# 5. RL SPARRING PARTNER LOADER
# ==========================================
class PattBotSparringPartner:
    def __init__(self, ckpt_path="out/dialectic_patt_bot.pt", device="cuda"):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {"dim": 64, "num_patt_tokens": 128, "rollout_steps": 4})
        self.model = DialecticPattChessBot(
            dim=cfg["dim"],
            num_patt_tokens=cfg["num_patt_tokens"],
            rollout_steps=cfg["rollout_steps"]
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def get_action(self, boards, is_white, legal_mask, castling_rights=None, temperature=0.0):
        return self.model.select_action(
            boards, is_white, legal_mask, castling_rights=castling_rights, temperature=temperature
        )

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Fast Supervised Training for DialecticPattChessBot")
    parser.add_argument("--dataset", type=str, default="data/chess/sf_dataset.pt", help="Path to offline SF dataset")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size (2048+ recommended for GPUs)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Optimizer learning rate")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_tokens", type=int, default=128, help="Patt concept tokens")
    parser.add_argument("--rollout_steps", type=int, default=4, help="Rollout steps")
    parser.add_argument("--out", type=str, default="out/dialectic_patt_bot.pt", help="Output checkpoint path")
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

    model_cfg = {"dim": args.dim, "num_patt_tokens": args.num_tokens, "rollout_steps": args.rollout_steps}
    model = DialecticPattChessBot(**model_cfg).to(device)

    train_offline_stockfish(model, dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_cfg
    }, args.out)
    print(f"[Artifact Saved] -> {args.out}")

    sf_exec = shutil.which("stockfish") or ("./stockfish.exe" if os.path.exists("stockfish.exe") else None)
    if sf_exec:
        sf_engine = StockfishEngine(sf_exec, threads=4, hash_mb=128)
        for elo in [1350, 1600]:
            print(f"\nEvaluating vs Stockfish Elo {elo} (32 Games)...")
            res = evaluate_model_elo(model, sf_engine, elo=elo, games=32, device=device)
            print(f"  Result vs {elo}: {res['score_str']} | Mean Material Delta: {res['mat_str']}")
        sf_engine.close()
    else:
        print("\n[Notice] Stockfish binary not detected. Evaluation skipped.")

if __name__ == "__main__":
    main()