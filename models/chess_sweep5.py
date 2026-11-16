# =====================================================================
# chess_sweep_dagger.py
# DAgger / Policy-Recovery Sweep with LR Ramps & Warm-Start Controls
# =====================================================================

import argparse
import math
import os
import queue
import shutil
import time
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
# 1. FAST STOCKFISH PARALLEL GAUNTLET
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


@torch.no_grad()
def evaluate_model_elo(
    model,
    sf_pool: FastStockfishEvalPool,
    elo=1350,
    games=500,
    max_plies=400,
    movetime_ms=15,
    device="cuda",
):
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
                bot_acts = model.select_action(
                    env.boards, is_w, l_mask, castling_rights=env.castling_rights, temperature=0.0
                )
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
                    fen = fast_board_to_fen(
                        boards_np[i], is_w_list[i], rights_np[i],
                        ep_list[i], half_list[i], (plies_list[i] // 2) + 1
                    )
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
        "wins": bot_wins,
        "draws": draws,
        "losses": sf_wins,
        "mat_lead": mat_lead,
        "score_str": f"{bot_wins}W/{draws}D/{sf_wins}L",
        "mat_str": f"{mat_lead:+5.0f} cp",
    }


# =====================================================================
# 2. TRAINING & EVALUATION PER ARM
# =====================================================================
def train_and_eval_arm(
    arm_cfg,
    dataset,
    sf_pool: FastStockfishEvalPool,
    warm_start_ckpt="out/dialectic_patt_bot.pt",
    epochs=15,
    default_batch=2048,
    elos=(1350, 1600),
    eval_temps=(0.0, 0.7),
    games_per_elo=500,
    max_plies=400,
    amp_dtype=torch.bfloat16,
    device=torch.device("cuda"),
):
    dim = arm_cfg.get("dim", 128)
    num_tokens = arm_cfg.get("num_tokens", 256)
    rollout_steps = arm_cfg.get("rollout_steps", 6)
    num_heads = arm_cfg.get("num_heads", 8)
    head_dim = dim // num_heads

    lr = arm_cfg.get("lr", 3.0e-4)
    wd = arm_cfg.get("weight_decay", 1e-4)
    batch_size = arm_cfg.get("batch_size", default_batch)
    warmup_frac = arm_cfg.get("warmup_frac", 0.05)
    min_lr_ratio = arm_cfg.get("min_lr_ratio", 0.1)
    use_warm_start = arm_cfg.get("warm_start", True)
    use_masked_ce = arm_cfg.get("masked_ce", False)

    model = DialecticPattChessBot(
        dim=dim,
        num_patt_tokens=num_tokens,
        rollout_steps=rollout_steps,
        num_heads=num_heads,
        head_dim=head_dim,
    ).to(device)

    # Warm-start from previous champion if specified
    if use_warm_start and warm_start_ckpt and os.path.exists(warm_start_ckpt):
        ckpt = torch.load(warm_start_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"   [Checkpoint] Warm-started weights from {warm_start_ckpt}")
    else:
        print("   [Checkpoint] Initialized from scratch (no warm-start)")

    param_groups = split_decay_params(model, lr=lr, weight_decay=wd)
    optimizer = torch.optim.AdamW(param_groups, fused=(device.type == "cuda"))

    boards = dataset["boards"]
    is_white = dataset["is_white"]
    targets = dataset["targets"]
    castling = dataset.get("castling_rights", None)
    legal_bool = dataset.get("legal_bool", None) if use_masked_ce else None

    raw_dataset_size = targets.shape[0]
    effective_size = raw_dataset_size - (raw_dataset_size % batch_size)
    steps_per_epoch = effective_size // batch_size
    total_steps = steps_per_epoch * epochs
    total_positions = effective_size * epochs

    scheduler = cosine_warmup_scheduler(
        optimizer, total_steps, warmup_frac=warmup_frac, min_lr_ratio=min_lr_ratio
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"   [Arch] Dim: {dim} | Tokens: {num_tokens} | Plies: {rollout_steps} | "
        f"Params: {param_count:,} | Peak LR: {lr:.1e} | Warmup: {warmup_frac*100:.1f}% | Masked-CE: {use_masked_ce}"
    )
    print(f"   [Volume] {effective_size:,} pos/epoch * {epochs} epochs = {total_positions:,} total pos ({total_steps:,} steps)")

    wb, ww, wy = boards[:batch_size], is_white[:batch_size], targets[:batch_size]
    wc = castling[:batch_size] if castling is not None else None

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
            c = castling[idx] if castling is not None else None

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            f"     Epoch [{epoch:02d}/{epochs:02d}] Step [{global_step:04d}/{total_steps:04d}] | "
            f"Loss: {loss.item():.4f} | LR: {curr_lr:.2e} | Speed: {rate:7.0f} pos/s | "
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
            def eval_select_wrapper(boards, is_white, legal_mask, castling_rights=None, temperature=0.0):
                return orig_select(boards, is_white, legal_mask, castling_rights=castling_rights, temperature=t_val)
            return eval_select_wrapper

        model.select_action = make_eval_wrapper(temp)
        elo_data[temp] = {}

        for elo in elos:
            res_elo = evaluate_model_elo(
                model,
                sf_pool,
                elo=elo,
                games=games_per_elo,
                max_plies=max_plies,
                movetime_ms=15,
                device=device,
            )
            elo_data[temp][elo] = res_elo
            print(f"        Elo {elo:<4d} | Score: {res_elo['score_str']:<16} | Mat: {res_elo['mat_str']}")

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
# 3. MAIN RUNNER
# =====================================================================
def main():
    cpu_cores = os.cpu_count() or 8
    default_workers = max(2, min(cpu_cores - 2, 16))

    parser = argparse.ArgumentParser(description="DAgger / Policy-Recovery Tuning Sweep")
    parser.add_argument("--dataset", type=str, default="data/chess/sf_dagger_dataset.pt", help="Path to DAgger dataset")
    parser.add_argument("--warm_start_ckpt", type=str, default="out/dialectic_patt_bot.pt", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per arm")
    parser.add_argument("--batch", type=int, default=2048, help="Batch size")
    parser.add_argument("--games_per_elo", type=int, default=500, help="Eval games per Elo and Temperature")
    parser.add_argument("--max_plies", type=int, default=400, help="Max plies per eval game")
    parser.add_argument("--eval_workers", type=int, default=default_workers, help="Parallel SF workers")
    parser.add_argument("--eval_temps", type=float, nargs="+", default=[0.0, 0.7], help="Evaluation temperatures")
    parser.add_argument("--masked_ce", action="store_true", help="Enable masked CE where specified")
    parser.add_argument("--only", type=str, nargs="+", default=None, help="Run only matching arms")
    parser.add_argument("--skip", type=str, nargs="+", default=None, help="Skip matching arms")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--champion_out", type=str, default="out/dialectic_patt_bot_dagger.pt", help="Target path for winning DAgger model")
    args = parser.parse_args()

    device, amp_dtype = setup_torch()
    seed_everything(args.seed)
    os.makedirs("out", exist_ok=True)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"DAgger dataset not found at '{args.dataset}'")

    print(f"\nLoading DAgger dataset from '{args.dataset}' into {device}...")
    sf_data = torch.load(args.dataset, map_location=device, weights_only=False)

    if "legal_masks" in sf_data:
        if args.masked_ce:
            sf_data["legal_bool"] = (sf_data.pop("legal_masks") > -1e8).contiguous()
        else:
            del sf_data["legal_masks"]
            if device.type == "cuda":
                torch.cuda.empty_cache()

    sf_data["boards"] = sf_data["boards"].contiguous()
    sf_data["is_white"] = sf_data["is_white"].contiguous()
    sf_data["targets"] = sf_data["targets"].contiguous()
    if "castling_rights" in sf_data:
        sf_data["castling_rights"] = sf_data["castling_rights"].contiguous()

    print(f"Loaded {sf_data['targets'].shape[0]:,} DAgger positions successfully (Cached in VRAM).")

    sf_exec = shutil.which("stockfish") or ("./stockfish.exe" if os.path.exists("stockfish.exe") else None)
    if not sf_exec:
        raise RuntimeError("Stockfish binary required for evaluation gauntlet.")

    test_elos = [1350, 1600]

    base_winner = {
        "dim": 128,
        "num_tokens": 256,
        "rollout_steps": 6,
        "num_heads": 8,
        "batch_size": args.batch,
    }

    all_arms = [
        # Arm 0: Control: Train from scratch on the DAgger blended data
        {
            **base_winner,
            "name": "0. Scratch Baseline (LR 1e-3, Warmup 3%)",
            "warm_start": False,
            "lr": 1.0e-3,
            "warmup_frac": 0.03,
            "min_lr_ratio": 0.03,
            "weight_decay": 1e-4,
        },

        # Arm 1: Conservative Fine-Tune (Low LR, warm-started)
        {
            **base_winner,
            "name": "1. Fine-Tune Low-LR (LR 2.5e-4, Warmup 2%)",
            "warm_start": True,
            "lr": 2.5e-4,
            "warmup_frac": 0.02,
            "min_lr_ratio": 0.10,
            "weight_decay": 1e-4,
        },

        # Arm 2: Moderate Fine-Tune with longer LR ramp
        {
            **base_winner,
            "name": "2. Fine-Tune Ramp (LR 4.0e-4, Warmup 6%)",
            "warm_start": True,
            "lr": 4.0e-4,
            "warmup_frac": 0.06,
            "min_lr_ratio": 0.08,
            "weight_decay": 1e-4,
        },

        # Arm 3: Fine-Tune with higher Weight Decay (retains sharp boundaries)
        {
            **base_winner,
            "name": "3. Fine-Tune High-WD (LR 3.5e-4, WD 1e-3)",
            "warm_start": True,
            "lr": 3.5e-4,
            "warmup_frac": 0.05,
            "min_lr_ratio": 0.10,
            "weight_decay": 1e-3,
        },

        # Arm 4: Fine-Tune with Masked Cross-Entropy (restricts logit mass to legal moves)
        {
            **base_winner,
            "name": "4. Fine-Tune Masked-CE (LR 3.5e-4, Warmup 5%)",
            "warm_start": True,
            "lr": 3.5e-4,
            "warmup_frac": 0.05,
            "min_lr_ratio": 0.10,
            "weight_decay": 1e-4,
            "masked_ce": True,
        },
    ]

    sweep_arms = select_arms(all_arms, only=args.only, skip=args.skip)
    if not sweep_arms:
        print("[Notice] No sweep arms matched filters.")
        return

    print("\n" + "=" * 145)
    print("             DAGGER POLICY-DISTRIBUTION FINE-TUNING & LR RAMP SWEEP")
    print(
        f" {len(sweep_arms)} Arms | {args.epochs} Epochs/Arm | Eval Pool: {args.eval_workers} SF workers | "
        f"Gauntlet: {args.games_per_elo} Games @ Elos {test_elos} across Temps {args.eval_temps} (Max Plies: {args.max_plies})"
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
                22,
                "<22",
            ))

    lb = Leaderboard(
        title="DAGGER FINE-TUNING LEADERBOARD",
        columns=lb_cols,
        name_width=44,
    )

    sf_pool = FastStockfishEvalPool(sf_exec, num_workers=args.eval_workers, hash_mb=32)
    try:
        for idx, arm in enumerate(sweep_arms):
            print(f"\n[Arm {idx+1}/{len(sweep_arms)}] Running: {arm['name']} ...")
            res = train_and_eval_arm(
                arm_cfg=arm,
                dataset=sf_data,
                sf_pool=sf_pool,
                warm_start_ckpt=args.warm_start_ckpt,
                epochs=args.epochs,
                default_batch=args.batch,
                elos=test_elos,
                eval_temps=args.eval_temps,
                games_per_elo=args.games_per_elo,
                max_plies=args.max_plies,
                amp_dtype=amp_dtype,
                device=device,
            )

            arm_ckpt_path = f"out/dagger_sweep_arm_{idx}.pt"
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
    
    # Save to a distinct DAgger checkpoint to preserve out/dialectic_patt_bot.pt
    dagger_champion_path = getattr(args, "champion_out", "out/dialectic_patt_bot_dagger.pt")
    torch.save(
        {
            "model_state_dict": best["res"]["model"].state_dict(),
            "config": best["arm"],
            "arm_name": best["arm"]["name"],
            "base_checkpoint": args.warm_start_ckpt,
        },
        dagger_champion_path,
    )
    print(f"[New DAgger Champion Saved] -> {dagger_champion_path} (from {best['arm']['name']})")
    print(f"[Pretrained Preserved]       -> {args.warm_start_ckpt} was untouched.\n")


if __name__ == "__main__":
    main()