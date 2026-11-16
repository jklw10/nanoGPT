# =====================================================================
# chess_sf_sweep2.py
# Transformer Learnings Ablation Sweep for 2-Engine Tied Dialectic Bot
# + Parallel Stockfish Pool Evaluation
# =====================================================================

import argparse
import math
import os
import queue
import shutil
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.chess_bot import (
    DialecticPattChessBot,
    GuardrailedTorchChess,
    StockfishEngine,
    uci_to_action,
)
from parts.sweep import (
    Leaderboard,
    ThroughputMeter,
    cleanup_compiled,
    compile_and_warmup,
    cosine_warmup_scheduler,
    cudagraph_step,
    seed_everything,
    select_arms,
    setup_torch,
    split_decay_params,
)

PIECE_CHARS = ["", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


# =====================================================================
# 1. FAST PARALLEL STOCKFISH EVALUATION (Adapted from generate_sf_dataset)
# =====================================================================
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

    def play_batch(self, tasks, elo: int):
        """tasks: list of (game_idx, fen_str, legals_set_or_list)"""
        if not tasks:
            return {}
        self.set_elo(elo)

        def query(task):
            idx, fen, legals = task
            eng = self.engine_queue.get()
            try:
                bm = eng.get_elo_move(fen, elo=elo)
                act = uci_to_action(bm) if bm else None
                return idx, (act if (act is not None and act in legals) else legals[0])
            finally:
                self.engine_queue.put(eng)

        return dict(self.pool.map(query, tasks))

    def close(self):
        self.pool.shutdown(wait=True)
        for eng in self.engines:
            eng.close()


@torch.no_grad()
def fast_evaluate_model_elo(
    model,
    sf_pool: FastStockfishEvalPool,
    elo=1350,
    games=100,
    max_plies=400,
    stagger_plies=0,
    has_castling=True,
    device="cuda",
):
    model.eval()
    sf_pool.set_elo(elo)

    env = GuardrailedTorchChess(games, device=device, halfmove_limit=max_plies)
    bot_is_white = torch.arange(games, device=device) < (games // 2)

    dones = torch.zeros(games, dtype=torch.bool, device=device)
    winners = torch.zeros(games, dtype=torch.int8, device=device)
    plies = torch.zeros(games, dtype=torch.long, device=device)

    # Optional seeded opening diversity so all games don't start identically at T=0.0
    if stagger_plies > 0:
        gen = torch.Generator(device=device).manual_seed(42)
        target_stagger = torch.randint(0, stagger_plies + 1, (games,), generator=gen, device=device)
        for s_ply in range(stagger_plies):
            need = (~dones) & (target_stagger > s_ply)
            if not need.any():
                break
            lm = env.compute_legal_masks()
            acts = torch.zeros(games, dtype=torch.long, device=device)
            alive = torch.zeros(games, dtype=torch.bool, device=device)
            nz = torch.nonzero(lm > -1e8).cpu()
            leg_map = [[] for _ in range(games)]
            for b_i, a in nz.tolist():
                leg_map[b_i].append(a)
            for i in torch.nonzero(need).squeeze(-1).tolist():
                if leg_map[i]:
                    pick = int(torch.randint(0, len(leg_map[i]), (1,), generator=gen, device=device).item())
                    acts[i] = leg_map[i][pick]
                    alive[i] = True
            if alive.any():
                dones, winners = env.step(acts, ~alive, winners, legal_mask=lm)
                plies[alive] += 1

    while not dones.all() and (plies < max_plies).any():
        active = ~dones
        l_mask = env.compute_legal_masks()
        actions = torch.zeros(games, dtype=torch.long, device=device)

        is_w = (env.players == 0)
        is_bot = active & (is_w == bot_is_white)
        is_sf = active & (~is_bot)

        # Compute check status once per ply on GPU
        in_chk_all = env.is_in_check(env.players, env.boards)

        # 1. Batched GPU inference for bot
        if is_bot.any():
            has_legals = (l_mask > -1e8).any(dim=-1)
            bot_terminal = is_bot & (~has_legals)

            if bot_terminal.any():
                dones = dones | bot_terminal
                opp_wins = torch.where(is_w, 2, 1)
                winners = torch.where(bot_terminal, torch.where(in_chk_all, opp_wins, 3), winners)
                is_bot = is_bot & has_legals

            if is_bot.any():
                c_arg = env.castling_rights if has_castling else None
                bot_acts = model.select_action(
                    env.boards, is_w, l_mask, castling_rights=c_arg, temperature=0.0
                )
                actions[is_bot] = bot_acts[is_bot]

        # 2. Parallel Stockfish inference via worker pool + bulk CPU transfer
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
                    fen = fast_board_to_fen(
                        boards_np[i], is_w_list[i], rights_np[i],
                        ep_list[i], half_list[i], (plies_list[i] // 2) + 1
                    )
                    sf_tasks.append((i, fen, legals))

            if sf_tasks:
                sf_moves = sf_pool.play_batch(sf_tasks, elo=elo)
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
        "wins": bot_wins,
        "draws": draws,
        "losses": sf_wins,
        "mat_lead": mat_lead,
        "score_str": f"{bot_wins}W/{draws}D/{sf_wins}L",
        "mat_str": f"{mat_lead:+5.0f} cp",
    }


# =====================================================================
# 2. TRANSFORMER / PATT ARCHITECTURAL VARIANTS
# =====================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        out = (x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
        return out * self.weight if self.weight is not None else out


def make_norm(norm_type: str, dim: int, affine: bool = True) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim, elementwise_affine=affine)
    return nn.LayerNorm(dim, elementwise_affine=affine)


class FlexPattLayer(nn.Module):
    """PattLayer with optional Tokenformer-style cosine-normalized keys."""
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int, cosine_patt: bool = False):
        super().__init__()
        self.cosine_patt = cosine_patt
        self.key_tokens = nn.Parameter(torch.empty(num_tokens, input_dim))
        self.val_tokens = nn.Parameter(torch.empty(num_tokens, output_dim))
        nn.init.kaiming_uniform_(self.key_tokens, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.val_tokens, a=math.sqrt(5))
        if cosine_patt:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            nn.init.normal_(self.val_tokens, std=1.0 / math.sqrt(output_dim))

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


class ExperimentPattAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        num_patt_tokens: int = 128,
        num_heads: int = 4,
        norm_order: str = "pre_ln",     # pre_ln | post_ln | peri_ln | layerscale | adaln_zero
        norm_type: str = "layernorm",   # layernorm | rmsnorm
        use_wo: bool = False,
        use_qk_norm: bool = False,
        use_swiglu: bool = False,
        cosine_patt: bool = False,
        layerscale_init: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_order = norm_order

        self.patt_q = FlexPattLayer(dim, dim, num_patt_tokens, cosine_patt=cosine_patt)
        self.patt_k = FlexPattLayer(dim, dim, num_patt_tokens, cosine_patt=cosine_patt)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_o = nn.Linear(dim, dim, bias=False) if use_wo else nn.Identity()

        self.q_norm = make_norm(norm_type, self.head_dim) if use_qk_norm else nn.Identity()
        self.k_norm = make_norm(norm_type, self.head_dim) if use_qk_norm else nn.Identity()

        if use_swiglu:
            self.mlp = SwiGLUFFN(dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Linear(dim * 4, dim),
            )

        use_affine = (norm_order != "adaln_zero")
        self.norm1 = make_norm(norm_type, dim, affine=use_affine)
        self.norm2 = make_norm(norm_type, dim, affine=use_affine)

        if norm_order == "peri_ln":
            self.post_norm1 = make_norm(norm_type, dim)
            self.post_norm2 = make_norm(norm_type, dim)
        elif norm_order == "layerscale":
            self.ls1 = nn.Parameter(torch.full((dim,), layerscale_init))
            self.ls2 = nn.Parameter(torch.full((dim,), layerscale_init))
        elif norm_order == "adaln_zero":
            self.ada_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
            nn.init.zeros_(self.ada_mlp[1].weight)
            nn.init.zeros_(self.ada_mlp[1].bias)

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
        if self.norm_order == "pre_ln":
            h = self.norm1(x)
            if depth_token is not None:
                h = h + depth_token
            x = x + self._attn(h)
            x = x + self.mlp(self.norm2(x))
            return x

        if self.norm_order == "post_ln":
            h = (x + depth_token) if depth_token is not None else x
            x = self.norm1(x + self._attn(h))
            x = self.norm2(x + self.mlp(x))
            return x

        if self.norm_order == "peri_ln":
            h = self.norm1(x)
            if depth_token is not None:
                h = h + depth_token
            x = x + self.post_norm1(self._attn(h))
            x = x + self.post_norm2(self.mlp(self.norm2(x)))
            return x

        if self.norm_order == "layerscale":
            h = self.norm1(x)
            if depth_token is not None:
                h = h + depth_token
            x = x + self.ls1 * self._attn(h)
            x = x + self.ls2 * self.mlp(self.norm2(x))
            return x

        if self.norm_order == "adaln_zero":
            s1, g1_s, g1, s2, g2_s, g2 = self.ada_mlp(depth_token).chunk(6, dim=-1)
            h1 = self.norm1(x) * (1.0 + g1_s) + s1
            x = x + g1 * self._attn(h1)
            h2 = self.norm2(x) * (1.0 + g2_s) + s2
            x = x + g2 * self.mlp(h2)
            return x

        raise ValueError(f"Unknown norm_order: {self.norm_order}")


class ExperimentTiedBot(DialecticPattChessBot):
    def __init__(
        self,
        dim: int = 64,
        num_patt_tokens: int = 128,
        rollout_steps: int = 6,
        head_dim: int = 32,
        num_heads: int = 4,
        input_norm: bool = False,
        pollute_residual: bool = False,
        norm_order: str = "pre_ln",
        norm_type: str = "layernorm",
        use_wo: bool = False,
        use_qk_norm: bool = False,
        use_swiglu: bool = False,
        cosine_patt: bool = False,
    ):
        super().__init__(dim=dim, num_patt_tokens=num_patt_tokens, rollout_steps=rollout_steps, head_dim=head_dim)
        del self.rollout_layers

        self.input_norm = input_norm
        self.pollute_residual = pollute_residual
        self.ply_embed = nn.Embedding(16, dim)
        self.register_buffer("ply_indices", torch.arange(16), persistent=False)

        if not input_norm:
            self.norm = nn.Identity()
        if norm_type == "rmsnorm":
            self.final_norm = RMSNorm(dim)

        layer_kwargs = dict(
            dim=dim,
            num_patt_tokens=num_patt_tokens,
            num_heads=num_heads,
            norm_order=norm_order,
            norm_type=norm_type,
            use_wo=use_wo,
            use_qk_norm=use_qk_norm,
            use_swiglu=use_swiglu,
            cosine_patt=cosine_patt,
        )
        self.thesis_engine = ExperimentPattAttentionLayer(**layer_kwargs)
        self.antithesis_engine = ExperimentPattAttentionLayer(**layer_kwargs)

    def forward(
        self,
        boards: torch.Tensor,
        is_white: torch.Tensor,
        castling_rights: torch.Tensor = None,
        legal_mask: torch.Tensor = None,
    ):
        B = boards.shape[0]
        flat = boards.view(B, 64).long().clamp(0, 12)
        turn_idx = torch.where(is_white, 0, 1).long()

        pos_embed = self.rank_embed(self.sq_ranks) + self.file_embed(self.sq_files)
        tokens = self.piece_embed(flat) + pos_embed.unsqueeze(0) + self.turn_embed(turn_idx).unsqueeze(1)

        if castling_rights is not None and getattr(self, "has_castling", True):
            tokens = tokens + self.castling_proj(castling_rights.float()).unsqueeze(1)

        x = self.norm(tokens)

        for ply in range(self.rollout_steps):
            depth_token = self.ply_embed(self.ply_indices[ply]).view(1, 1, self.dim)
            engine = self.thesis_engine if (ply % 2 == 0) else self.antithesis_engine
            if self.pollute_residual:
                x = engine(x + depth_token, depth_token=None)
            else:
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


# =====================================================================
# 3. TRAINING + GAUNTLET PER ARM
# =====================================================================
def train_and_eval_arm(
    arm_cfg,
    dataset,
    sf_pool: FastStockfishEvalPool,
    epochs=10,
    default_batch=2048,
    elos=(1350, 1600),
    eval_temps=(0.0, 0.7),
    games_per_elo=100,
    max_plies=60,
    eval_stagger=0,
    use_masked_ce=False,
    amp_dtype=torch.bfloat16,
    device=torch.device("cuda"),
):
    dim = arm_cfg["dim"]
    num_tokens = arm_cfg["num_tokens"]
    rollout_steps = arm_cfg["rollout_steps"]
    num_heads = arm_cfg.get("num_heads", 4)
    lr = arm_cfg.get("lr", 1.5e-3)
    wd = arm_cfg.get("weight_decay", 1e-4)
    batch_size = arm_cfg.get("batch_size", default_batch)
    clip_norm = arm_cfg.get("clip_norm", 1.0)

    model = ExperimentTiedBot(
        dim=dim,
        num_patt_tokens=num_tokens,
        rollout_steps=rollout_steps,
        num_heads=num_heads,
        input_norm=arm_cfg.get("input_norm", False),
        pollute_residual=arm_cfg.get("pollute_residual", False),
        norm_order=arm_cfg.get("norm_order", "pre_ln"),
        norm_type=arm_cfg.get("norm_type", "layernorm"),
        use_wo=arm_cfg.get("use_wo", False),
        use_qk_norm=arm_cfg.get("use_qk_norm", False),
        use_swiglu=arm_cfg.get("use_swiglu", False),
        cosine_patt=arm_cfg.get("cosine_patt", False),
    ).to(device)

    param_groups = split_decay_params(model, lr=lr, weight_decay=wd)
    optimizer = torch.optim.AdamW(param_groups, fused=(device.type == "cuda"))

    boards = dataset["boards"]
    is_white = dataset["is_white"]
    targets = dataset["targets"]
    castling = dataset.get("castling_rights", None)
    legal_bool = dataset.get("legal_bool", None) if use_masked_ce else None

    has_castling = (castling is not None)
    model.has_castling = has_castling

    raw_dataset_size = targets.shape[0]
    effective_size = raw_dataset_size - (raw_dataset_size % batch_size)
    steps_per_epoch = effective_size // batch_size
    total_steps = steps_per_epoch * epochs
    total_positions = effective_size * epochs
    scheduler = cosine_warmup_scheduler(optimizer, total_steps, warmup_frac=0.03, min_lr_ratio=0.03)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"   [Arch] Dim: {dim} | Tokens: {num_tokens} | Plies: {rollout_steps} | "
        f"Norm: {arm_cfg.get('norm_order')} ({arm_cfg.get('norm_type', 'layernorm')}) | "
        f"Params: {param_count:,} | LR: {lr:.1e}"
    )
    print(f"   [Volume] {effective_size:,} pos/epoch * {epochs} epochs = {total_positions:,} total pos ({total_steps:,} steps)")

    wb, ww, wy = boards[:batch_size], is_white[:batch_size], targets[:batch_size]
    wc = castling[:batch_size] if has_castling else None

    def warmup_step(m):
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            w_logits = m(wb, ww, castling_rights=wc)
            if legal_bool is not None:
                w_lm = legal_bool[:batch_size].clone()
                w_lm.scatter_(1, wy.unsqueeze(1), True)
                w_logits = w_logits.masked_fill(~w_lm, -1e4)
            w_loss = F.cross_entropy(w_logits, wy)
        w_loss.backward()

    print("   [Warmup] Compiling model CUDA graphs...", end="", flush=True)
    compiled_model = compile_and_warmup(
        model,
        warmup_step,
        mode="reduce-overhead",
        warmup_iters=2,
        optimizer=optimizer,
        restore_state=True,
        clear_opt_state=False,
        device=str(device),
    )
    print(" Ready.")

    meter = ThroughputMeter(total_positions)
    last_epoch_loss, last_epoch_correct, last_epoch_samples = 0.0, 0, 0
    global_step = 0

    compiled_model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(effective_size, device=device)
        is_last_epoch = (epoch == epochs)

        for i in range(0, effective_size, batch_size):
            global_step += 1
            idx = perm[i:i + batch_size]
            b, w, y = boards[idx], is_white[idx], targets[idx]
            c = castling[idx] if has_castling else None

            cudagraph_step()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits = compiled_model(b, w, castling_rights=c)
                if legal_bool is not None:
                    lm = legal_bool[idx].clone()
                    lm.scatter_(1, y.unsqueeze(1), True)
                    logits = logits.masked_fill(~lm, -1e4)
                loss = F.cross_entropy(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            scheduler.step()
            meter.update(batch_size)

            if is_last_epoch:
                with torch.no_grad():
                    correct = (torch.argmax(logits.detach(), dim=-1) == y).sum().item()
                    last_epoch_loss += loss.item() * batch_size
                    last_epoch_correct += correct
                    last_epoch_samples += batch_size

        rate, elapsed_str, eta_str = meter.stats()
        print(
            f"     Epoch [{epoch:02d}/{epochs:02d}] Step [{global_step:04d}/{total_steps:04d}] | "
            f"Loss: {loss.item():.4f} | Speed: {rate:7.0f} pos/s | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str}"
        )

    train_rate, total_elapsed_str, _ = meter.stats()
    final_loss = last_epoch_loss / max(1, last_epoch_samples)
    final_top1 = (last_epoch_correct / max(1, last_epoch_samples)) * 100.0

    print(
        f"   [Arm Trained] {total_positions:,} pos in {total_elapsed_str} ({train_rate:7.0f} pos/s) | "
        f"Top-1: {final_top1:4.1f}% | Loss: {final_loss:.4f}"
    )
    print(f"   [Gauntlet] Running parallel SF pool ({sf_pool.num_workers} workers, {games_per_elo} games/temp/elo, max_plies={max_plies})...")

    orig_select = model.select_action
    elo_data = {}

    for temp in eval_temps:
        print(f"     -> Evaluating [T={temp:.1f}] ({'Deterministic Greedy' if temp == 0.0 else 'Stochastic Sampling'})...")

        def make_eval_wrapper(t_val):
            def eval_select_wrapper(boards, is_white, legal_mask, castling_rights=None, temperature=0.0, current_ply=0):
                c_rights = castling_rights if has_castling else None
                return orig_select(boards, is_white, legal_mask, castling_rights=c_rights, temperature=t_val)
            return eval_select_wrapper

        model.select_action = make_eval_wrapper(temp)
        elo_data[temp] = {}

        for elo in elos:
            res_elo = fast_evaluate_model_elo(
                model,
                sf_pool,
                elo=elo,
                games=games_per_elo,
                max_plies=max_plies,
                stagger_plies=eval_stagger,
                has_castling=has_castling,
                device=device,
            )
            elo_data[temp][elo] = res_elo
            print(f"        Elo {elo:<4d} | Score: {res_elo['score_str']:<14} | Mat: {res_elo['mat_str']}")

    model.select_action = orig_select
    cleanup_compiled()

    return {
        "loss": final_loss,
        "top1": final_top1,
        "throughput": train_rate,
        "params": param_count,
        "elo": elo_data,
        "model": model,
        "cfg": arm_cfg,
    }


# =====================================================================
# 4. MAIN SWEEP RUNNER
# =====================================================================
def main():
    cpu_cores = os.cpu_count() or 8
    default_workers = max(2, min(cpu_cores - 2, 16))

    parser = argparse.ArgumentParser(description="Transformer Learnings Sweep for Tied Dialectic Patt Bot")
    parser.add_argument("--dataset", type=str, default="data/chess/sf_dataset.pt", help="Path to cleaned SF dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs per arm")
    parser.add_argument("--batch", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Base learning rate")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_tokens", type=int, default=128, help="PATT concept tokens")
    parser.add_argument("--rollout_steps", type=int, default=6, help="Tied dialectic rollout plies")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--games_per_elo", type=int, default=500, help="Eval games per Elo and Temperature")
    parser.add_argument("--max_plies", type=int, default=400, help="Max plies per eval game")
    parser.add_argument("--eval_stagger", type=int, default=0, help="Seeded random opening plies during eval (0=standard)")
    parser.add_argument("--eval_workers", type=int, default=default_workers, help="Parallel 1-thread SF workers for eval")
    parser.add_argument("--eval_temps", type=float, nargs="+", default=[0.0, 0.7], help="Evaluation temperatures")
    parser.add_argument("--masked_ce", action="store_true", help="Use legal_masks in CE loss if present in dataset")
    parser.add_argument("--only", type=str, nargs="+", default=None, help="Run only arms matching substring(s)")
    parser.add_argument("--skip", type=str, nargs="+", default=None, help="Skip arms matching substring(s)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    device, amp_dtype = setup_torch()
    seed_everything(args.seed)
    os.makedirs("out", exist_ok=True)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Stockfish dataset not found at '{args.dataset}'")

    print(f"\nLoading Stockfish dataset from '{args.dataset}' into {device}...")
    sf_data = torch.load(args.dataset, map_location=device, weights_only=False)

    if "legal_masks" in sf_data:
        if args.masked_ce:
            sf_data["legal_bool"] = (sf_data.pop("legal_masks") > -1e8).contiguous()
            print("  -> Converted legal_masks to compact bool tensor for Masked CE.")
        else:
            del sf_data["legal_masks"]
            if device.type == "cuda":
                torch.cuda.empty_cache()
    elif args.masked_ce:
        print("  -> [Notice] --masked_ce requested, but 'legal_masks' not in dataset. Using unmasked CE.")

    sf_data["boards"] = sf_data["boards"].contiguous()
    sf_data["is_white"] = sf_data["is_white"].contiguous()
    sf_data["targets"] = sf_data["targets"].contiguous()
    if "castling_rights" in sf_data:
        sf_data["castling_rights"] = sf_data["castling_rights"].contiguous()
        print("  -> Castling rights detected in dataset.")
    else:
        print("  -> [Notice] No castling rights in dataset. Forward will bypass castling projections.")

    print(f"Loaded {sf_data['targets'].shape[0]:,} positions successfully (Cached in VRAM).")

    sf_exec = shutil.which("stockfish") or ("./stockfish.exe" if os.path.exists("stockfish.exe") else None)
    if not sf_exec:
        raise RuntimeError("Stockfish binary required for evaluation gauntlet.")

    test_elos = [1350, 1600]
    arch_8 = {
        "norm_order": "post_ln",
        "norm_type": "rmsnorm",
        "use_wo": True,
        "use_qk_norm": True,
        "use_swiglu": True,
        "cosine_patt": True,
        "input_norm": False,
        "pollute_residual": False,
        "weight_decay": 1e-4,
    }

    all_arms = [
        {
            **arch_8,
            "name": "1. Dim96 (25 Epochs, tok=256, ply=6, h=6)",
            "dim": 96, "num_tokens": 256, "rollout_steps": 6, "num_heads": 6,
            "lr": 1.2e-3, "batch_size": args.batch,
        },

        # Scaling to Dim128 (head_dim=16, 8 heads)
        {
            **arch_8,
            "name": "2. Dim128 (25 Epochs, tok=256, ply=6, h=8)",
            "dim": 128, "num_tokens": 256, "rollout_steps": 6, "num_heads": 8,
            "lr": 1.0e-3, "batch_size": args.batch,
        },

        # Dim128 with 512 concept tokens
        {
            **arch_8,
            "name": "3. Dim128 + Tok512 (25 Epochs, ply=6, h=8)",
            "dim": 128, "num_tokens": 512, "rollout_steps": 6, "num_heads": 8,
            "lr": 1.0e-3, "batch_size": args.batch,
        },
        # 1. Direct Control: Arm 8 base (157k params, 6 plies, 128 tokens)
        #{
        #    **arch_8,
        #    "name": "1. Arm8 Control (dim=64, tok=128, ply=6)",
        #    "dim": 64, "num_tokens": 128, "rollout_steps": 6, "num_heads": 4,
        #    "lr": 1.5e-3, "batch_size": args.batch,
        #},

        ## 2. Concept Token Scaling: 256 tokens (173k params)
        #{
        #    **arch_8,
        #    "name": "2. Token Scale (dim=64, tok=256, ply=6)",
        #    "dim": 64, "num_tokens": 256, "rollout_steps": 6, "num_heads": 4,
        #    "lr": 1.5e-3, "batch_size": args.batch,
        #},
#
        ## 3. Concept Token Scaling: 512 tokens (206k params)
        #{
        #    **arch_8,
        #    "name": "3. Token Scale (dim=64, tok=512, ply=6)",
        #    "dim": 64, "num_tokens": 512, "rollout_steps": 6, "num_heads": 4,
        #    "lr": 1.5e-3, "batch_size": args.batch,
        #},
#
        ## 4. Deep Recurrence: 8 plies (0 extra params: 157k params)
        #{
        #    **arch_8,
        #    "name": "4. Deep Recurrence (dim=64, tok=128, ply=8)",
        #    "dim": 64, "num_tokens": 128, "rollout_steps": 8, "num_heads": 4,
        #    "lr": 1.5e-3, "batch_size": args.batch,
        #},
#
        ## 5. Width Scale: dim=96, heads=6 (head_dim=16), 256 tokens (~350k params)
        #{
        #    **arch_8,
        #    "name": "5. Width Scale (dim=96, tok=256, ply=6, h=6)",
        #    "dim": 96, "num_tokens": 256, "rollout_steps": 6, "num_heads": 6,
        #    "lr": 1.2e-3, "batch_size": args.batch,
        #},
#
        ## 6. Combo Scale: dim=96, heads=6, 256 tokens, 8 plies (~350k params)
        #{
        #    **arch_8,
        #    "name": "6. Combo Scale (dim=96, tok=256, ply=8, h=6)",
        #    "dim": 96, "num_tokens": 256, "rollout_steps": 8, "num_heads": 6,
        #    "lr": 1.2e-3, "batch_size": args.batch,
        #},

        
    ]

    sweep_arms = select_arms(all_arms, only=args.only, skip=args.skip)
    if not sweep_arms:
        print("[Notice] No sweep arms matched filters.")
        return

    print("\n" + "=" * 145)
    print("             TRANSFORMER ARCHITECTURE ABLATION SWEEP (RESIDUALS, NORMS, ATTENTION, COSINE-PATT)")
    print(
        f" {len(sweep_arms)} Configs | {args.epochs} Epochs/Arm | Eval Pool: {args.eval_workers} SF workers | "
        f"Gauntlet: {args.games_per_elo} Games @ Elos {test_elos} across Temps {args.eval_temps}"
    )
    print("=" * 145)

    def fmt_cell(t_dict, elo):
        if not t_dict or elo not in t_dict:
            return "N/A"
        cell = t_dict[elo]
        return f"{cell['score_str']} ({cell['mat_str'].strip()})"

    def compute_composite_score(res):
        score = res["top1"] * 0.05
        weights = {0.0: 1.0, 0.7: 0.8}
        for temp, t_data in res["elo"].items():
            tw = weights.get(temp, 1.0)
            e13 = t_data.get(1350, {})
            e16 = t_data.get(1600, {})
            tot13 = max(1, e13.get("wins", 0) + e13.get("draws", 0) + e13.get("losses", 0))
            tot16 = max(1, e16.get("wins", 0) + e16.get("draws", 0) + e16.get("losses", 0))

            w13 = e13.get("wins", 0) / tot13
            w16 = e16.get("wins", 0) / tot16
            d13 = e13.get("draws", 0) / tot13
            d16 = e16.get("draws", 0) / tot16
            m13 = float(e13.get("mat_lead", 0.0))
            m16 = float(e16.get("mat_lead", 0.0))
            score += tw * ((w16 * 4.0) + (w13 * 2.0) + ((d13 + d16) * 0.75) + ((m13 + m16) / 1000.0 * 1.5))
        return score

    lb_cols = [
        ("Params", "params", 8, ",d"),
        ("Loss", "loss", 7, ".4f"),
        ("Top-1%", "top1", 6, ".1f"),
    ]
    for t in args.eval_temps:
        for e in test_elos:
            lb_cols.append((
                f"{e} (T={t:.1f})",
                lambda m, temp=t, elo=e: fmt_cell(m["elo"].get(temp, {}), elo),
                20,
                "<20",
            ))

    lb = Leaderboard(
        title="TRANSFORMER LEARNINGS SWEEP LEADERBOARD",
        columns=lb_cols,
        name_width=40,
    )

    sf_pool = FastStockfishEvalPool(sf_exec, num_workers=args.eval_workers, hash_mb=32)
    try:
        for idx, arm in enumerate(sweep_arms):
            print(f"\n[Arm {idx+1}/{len(sweep_arms)}] Running: {arm['name']} ...")
            res = train_and_eval_arm(
                arm_cfg=arm,
                dataset=sf_data,
                sf_pool=sf_pool,
                epochs=args.epochs,
                default_batch=args.batch,
                elos=test_elos,
                eval_temps=args.eval_temps,
                games_per_elo=args.games_per_elo,
                max_plies=args.max_plies,
                eval_stagger=args.eval_stagger,
                use_masked_ce=args.masked_ce,
                amp_dtype=amp_dtype,
                device=device,
            )

            arm_ckpt_path = f"out/sweep_arm_{idx}.pt"
            torch.save(
                {
                    "model_state_dict": res["model"].state_dict(),
                    "config": arm,
                    "arm_name": arm["name"],
                    "results": {"loss": res["loss"], "top1": res["top1"], "elo": res["elo"]},
                },
                arm_ckpt_path,
            )

            lb.add(
                name=arm["name"],
                metrics=res,
                score=compute_composite_score(res),
                payload={"arm": arm, "res": res},
            )
    finally:
        sf_pool.close()

    if not lb.rows:
        print("[Notice] No arms completed.")
        return

    ranked = lb.render()
    best = ranked[0]["payload"]
    champion_path = "out/dialectic_patt_bot.pt"
    torch.save(
        {
            "model_state_dict": best["res"]["model"].state_dict(),
            "config": best["arm"],
            "arm_name": best["arm"]["name"],
        },
        champion_path,
    )
    print(f"[Champion Artifact Saved] -> {champion_path} (from {best['arm']['name']})\n")


if __name__ == "__main__":
    main()