# =====================================================================
# chess_sf_sweep2.py
# High-Throughput Architectural Sweep (Dim, P-Tokens, Latent Plies)
# Multi-Temperature Evaluation Gauntlet (T=0.0 vs T=0.7 @ 100 Games)
# =====================================================================

import argparse
import math
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.chess_bot import (
    DialecticPattChessBot,
    PattAttentionLayer,
    StockfishEngine,
    evaluate_model_elo,
)

# =====================================================================
# 1. TIED DIALECTIC ARCHITECTURE (Flexible Scaling Engine)
# =====================================================================
class TiedDialecticPattChessBot(DialecticPattChessBot):
    def __init__(self, dim: int = 64, num_patt_tokens: int = 128, rollout_steps: int = 4,
                 head_dim: int = 32, num_heads: int = 4):
        super().__init__(dim=dim, num_patt_tokens=num_patt_tokens, rollout_steps=rollout_steps, head_dim=head_dim)
        self.ply_embed = nn.Embedding(16, dim)
        self.register_buffer("ply_indices", torch.arange(16), persistent=False)

        # Replace untied layers with 2-cycle bipartite tied engines
        del self.rollout_layers
        self.thesis_engine = PattAttentionLayer(dim=dim, num_patt_tokens=num_patt_tokens, num_heads=num_heads)
        self.antithesis_engine = PattAttentionLayer(dim=dim, num_patt_tokens=num_patt_tokens, num_heads=num_heads)

    def forward(self, boards: torch.Tensor, is_white: torch.Tensor,
                castling_rights: torch.Tensor = None, legal_mask: torch.Tensor = None):
        B = boards.shape[0]
        flat = boards.view(B, 64).long().clamp(0, 12)
        turn_idx = torch.where(is_white, 0, 1).long()

        pos_embed = self.rank_embed(self.sq_ranks) + self.file_embed(self.sq_files)
        tokens = self.piece_embed(flat) + pos_embed.unsqueeze(0) + self.turn_embed(turn_idx).unsqueeze(1)

        if castling_rights is not None and getattr(self, "has_castling", True):
            tokens = tokens + self.castling_proj(castling_rights.float()).unsqueeze(1)

        x = self.norm(tokens)

        # 2-Cycle Bipartite Latent Rollout
        for ply in range(self.rollout_steps):
            depth_token = self.ply_embed(self.ply_indices[ply]).view(1, 1, self.dim)
            x_in = x + depth_token
            x = self.thesis_engine(x_in) if (ply % 2 == 0) else self.antithesis_engine(x_in)

        x = self.final_norm(x)

        # Bilinear dot-product readout
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
# 2. UTILITIES & TRAINING LOOP
# =====================================================================
def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.02):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_and_eval_arm(arm_cfg, dataset, sf_engine,
                       epochs=10, default_batch=2048, elos=(1350, 1600),
                       eval_temps=(0.0, 0.7), games_per_elo=100, device="cuda"):
    dim = arm_cfg["dim"]
    num_tokens = arm_cfg["num_tokens"]
    rollout_steps = arm_cfg["rollout_steps"]
    num_heads = arm_cfg.get("num_heads", 4)
    lr = arm_cfg.get("lr", 2.8e-3)
    wd = arm_cfg.get("weight_decay", 1e-4)
    batch_size = arm_cfg.get("batch_size", default_batch)
    clip_norm = arm_cfg.get("clip_norm", 1.0)

    model = TiedDialecticPattChessBot(
        dim=dim,
        num_patt_tokens=num_tokens,
        rollout_steps=rollout_steps,
        num_heads=num_heads
    ).to(device)

    # Parameter grouping: 0 weight decay on embeddings
    raw_model = getattr(model, "_orig_mod", model)
    embed_params = [
        raw_model.piece_embed.weight,
        raw_model.rank_embed.weight,
        raw_model.file_embed.weight,
        raw_model.turn_embed.weight,
        raw_model.castling_proj.weight,
        raw_model.ply_embed.weight
    ]
    embed_param_set = set(embed_params)
    backbone_params = [p for p in raw_model.parameters() if p not in embed_param_set]

    param_groups = [
        {"params": embed_params, "lr": lr, "weight_decay": 0.0},
        {"params": backbone_params, "lr": lr, "weight_decay": wd}
    ]
    optimizer = torch.optim.AdamW(param_groups, fused=True)

    boards = dataset["boards"]
    is_white = dataset["is_white"]
    targets = dataset["targets"]
    castling = dataset.get("castling_rights", None)
    has_castling = (castling is not None)
    model.has_castling = has_castling

    raw_dataset_size = targets.shape[0]
    effective_size = raw_dataset_size - (raw_dataset_size % batch_size)
    steps_per_epoch = effective_size // batch_size
    total_steps = steps_per_epoch * epochs
    total_positions = effective_size * epochs
    warmup_steps = int(total_steps * 0.02)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.03)

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   [Arch] Dim: {dim} | Tokens: {num_tokens} | Plies: {rollout_steps} | Params: {param_count:,} | LR: {lr:.1e}")
    print(f"   [Volume] {effective_size:,} pos/epoch * {epochs} epochs = {total_positions:,} total pos ({total_steps:,} steps)")

    print("   [Warmup] Compiling model CUDA graphs...", end="", flush=True)
    compiled_model = torch.compile(model, mode="reduce-overhead")

    wb, ww, wy = boards[:batch_size], is_white[:batch_size], targets[:batch_size]
    wc = castling[:batch_size] if has_castling else None

    # Clean warmup: compile forward & backward without advancing optimizer step
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            w_logits = compiled_model(wb, ww, castling_rights=wc)
            w_loss = F.cross_entropy(w_logits, wy)
        w_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    model.load_state_dict(init_state)
    torch.cuda.synchronize()
    print(" Ready.")

    arm_start_t = time.time()
    last_epoch_loss, last_epoch_correct = 0.0, 0
    last_epoch_samples = 0
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

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits = compiled_model(b, w, castling_rights=c)
                loss = F.cross_entropy(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            scheduler.step()

            if is_last_epoch:
                with torch.no_grad():
                    correct = (torch.argmax(logits, dim=-1) == y).sum().item()
                    last_epoch_loss += loss.item() * batch_size
                    last_epoch_correct += correct
                    last_epoch_samples += batch_size
        elapsed = time.time() - arm_start_t
        processed_positions = global_step * batch_size
        rate = processed_positions / max(1e-4, elapsed)
        remaining_positions = max(0, total_positions - processed_positions)
        eta = remaining_positions / max(1e-4, rate)
        print(f"     Epoch [{epoch:02d}/{epochs:02d}] Step [{global_step:04d}/{total_steps:04d}] | "
              f"Loss: {loss.item():.4f} | Speed: {rate:7.0f} pos/s | "
              f"Elapsed: {format_eta(elapsed)} | ETA: {format_eta(eta)}")

    torch.cuda.synchronize()
    total_time = time.time() - arm_start_t
    train_rate = total_positions / max(1e-4, total_time)
    final_loss = last_epoch_loss / max(1, last_epoch_samples)
    final_top1 = (last_epoch_correct / max(1, last_epoch_samples)) * 100.0

    print(f"   [Arm Trained] {total_positions:,} pos in {format_eta(total_time)} ({train_rate:7.0f} pos/s) | Top-1: {final_top1:4.1f}% | Loss: {final_loss:.4f}")
    print(f"   [Gauntlet] Running Stockfish match play validation ({games_per_elo} games per temp/elo)...")

    # Evaluate across specified temperatures (e.g. 0.0 pure greedy vs 0.7 stochastic recovery)
    orig_select = model.select_action
    elo_data = {}

    for temp in eval_temps:
        temp_tag = f"T={temp:.1f}"
        print(f"     -> Evaluating condition [{temp_tag}] ({'Deterministic Greedy' if temp == 0.0 else 'Stochastic Sampling'})...")

        def make_eval_wrapper(t_val):
            def eval_select_wrapper(boards, is_white, legal_mask, castling_rights=None, temperature=0.0, current_ply=0):
                c_rights = castling_rights if has_castling else None
                return orig_select(boards, is_white, legal_mask, castling_rights=c_rights, temperature=t_val)
            return eval_select_wrapper

        model.select_action = make_eval_wrapper(temp)
        elo_data[temp] = {}

        for elo in elos:
            res_elo = evaluate_model_elo(
                model, sf_engine, elo=elo, games=games_per_elo, max_plies=60, device=device
            )
            elo_data[temp][elo] = res_elo
            print(f"        Elo {elo:<4d} | Score: {res_elo['score_str']:<14} | Mat: {res_elo['mat_str']}")

    model.select_action = orig_select

    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        torch.compiler.reset()
    torch.cuda.empty_cache()

    return {
        "final_loss": final_loss,
        "final_top1": final_top1,
        "throughput": train_rate,
        "params": param_count,
        "elo_data": elo_data,
        "model": model,
        "cfg": arm_cfg
    }


# =====================================================================
# 3. ARCHITECTURE SWEEP DEFINITION
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Architectural Scaling Sweep for Dialectic Patt Bot")
    parser.add_argument("--dataset", type=str, default="data/chess/sf_dataset.pt", help="Path to offline SF dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per arm")
    parser.add_argument("--batch", type=int, default=2048, help="Default batch size")
    parser.add_argument("--games_per_elo", type=int, default=100, help="Evaluation games per Elo and Temperature")
    parser.add_argument("--eval_temps", type=float, nargs="+", default=[0.0, 0.7], help="Evaluation temperatures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    os.makedirs("out", exist_ok=True)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Stockfish dataset not found at '{args.dataset}'")

    print(f"\nLoading Stockfish dataset from '{args.dataset}' into {device}...")
    sf_data = torch.load(args.dataset, map_location=device, weights_only=False)

    if "legal_masks" in sf_data:
        del sf_data["legal_masks"]
        torch.cuda.empty_cache()

    sf_data["boards"] = sf_data["boards"].contiguous()
    sf_data["is_white"] = sf_data["is_white"].contiguous()
    sf_data["targets"] = sf_data["targets"].contiguous()
    if "castling_rights" in sf_data:
        sf_data["castling_rights"] = sf_data["castling_rights"].contiguous()
        print("  -> Castling rights detected in dataset.")
    else:
        print("  -> [Notice] No castling rights in dataset. Forward will bypass castling projections.")

    print(f"Loaded {sf_data['targets'].shape[0]:,} positions successfully (Dataset cached in VRAM).")

    sf_exec = shutil.which("stockfish") or ("./stockfish.exe" if os.path.exists("stockfish.exe") else None)
    if not sf_exec:
        raise RuntimeError("Stockfish binary required for evaluation gauntlet.")

    sf_engine = StockfishEngine(sf_exec, threads=4, hash_mb=128)
    test_elos = [1350, 1600]

    sweep_arms = [
        {
            "name": "Baseline",
            "dim": 64, "num_tokens": 128, "rollout_steps": 4, "num_heads": 4,
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        },
        #{
        #    "name": "1. Deep Latent Horizon (6 Plies / 3 Cycles)",
        #    "dim": 64, "num_tokens": 128, "rollout_steps": 6, "num_heads": 4,
        #    "lr": 2.6e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "2. Expanded Motif Bank (256 Tokens)",
        #    "dim": 64, "num_tokens": 256, "rollout_steps": 4, "num_heads": 4,
        #    "lr": 2.8e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "3. Compact Motif Bank (64 Tokens)",
        #    "dim": 64, "num_tokens": 64, "rollout_steps": 4, "num_heads": 4,
        #    "lr": 3.0e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "4. High-Dim Latent Space (Dim 96, 4 Plies)",
        #    "dim": 96, "num_tokens": 128, "rollout_steps": 4, "num_heads": 4,
        #    "lr": 2.4e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "lowlr ",
        #    "dim": 64, "num_tokens": 128, "rollout_steps": 4, "num_heads": 4,
        #    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "low lr, hightok",
        #    "dim": 64, "num_tokens": 256, "rollout_steps": 4, "num_heads": 4,
        #    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "low lr, high step",
        #    "dim": 64, "num_tokens": 128, "rollout_steps": 6, "num_heads": 4,
        #    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "low lr, supertok",
        #    "dim": 64, "num_tokens": 1024, "rollout_steps": 4, "num_heads": 4,
        #    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "low lr, high step hightok",
        #    "dim": 64, "num_tokens": 256, "rollout_steps": 6, "num_heads": 4,
        #    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "low dim",
        #    "dim": 32, "num_tokens": 128, "rollout_steps": 4, "num_heads": 4,
        #    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "low dim, low head",
        #    "dim": 32, "num_tokens": 128, "rollout_steps": 4, "num_heads": 2,
        #    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "low head",
        #    "dim": 64, "num_tokens": 128, "rollout_steps": 4, "num_heads": 2,
        #    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        #{
        #    "name": "high dim",
        #    "dim": 128, "num_tokens": 128, "rollout_steps": 4, "num_heads": 4,
        #    "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        #},
        {
            "name": "super ply",
            "dim": 32, "num_tokens": 64, "rollout_steps": 16, "num_heads": 4,
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        },
        {
            "name": "high head",
            "dim": 64, "num_tokens": 128, "rollout_steps": 4, "num_heads": 8,
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        },
        {
            "name": "high head, highdim",
            "dim": 128, "num_tokens": 128, "rollout_steps": 4, "num_heads": 8,
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        },
        {
            "name": "high head dim tok",
            "dim": 128, "num_tokens": 256, "rollout_steps": 4, "num_heads": 8,
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        },
        {
            "name": "super ply tok",
            "dim": 64, "num_tokens": 512, "rollout_steps": 8, "num_heads": 8,
            "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch
        },
    ]

    print("\n" + "=" * 140)
    print("                    ARCHITECTURE SCALING SWEEP (DIM, TOKENS, PLIES)")
    print(f" {len(sweep_arms)} Configs | {args.epochs} Epochs/Arm | Gauntlet: {args.games_per_elo} Games @ Elos {test_elos} across Temps {args.eval_temps}")
    print("=" * 140)

    results = []
    try:
        for idx, arm in enumerate(sweep_arms):
            print(f"\n[Arm {idx+1}/{len(sweep_arms)}] Running: {arm['name']} ...")
            res = train_and_eval_arm(
                arm_cfg=arm,
                dataset=sf_data,
                sf_engine=sf_engine,
                epochs=args.epochs,
                default_batch=args.batch,
                elos=test_elos,
                eval_temps=args.eval_temps,
                games_per_elo=args.games_per_elo,
                device=device
            )
            results.append((arm, res))

            arm_ckpt_path = f"out/sweep_arm_{idx}.pt"
            torch.save({
                "model_state_dict": res["model"].state_dict(),
                "config": {"dim": arm["dim"], "num_patt_tokens": arm["num_tokens"], "rollout_steps": arm["rollout_steps"]},
                "arm_name": arm["name"],
                "results": {"loss": res["final_loss"], "top1": res["final_top1"], "elo": res["elo_data"]}
            }, arm_ckpt_path)

    finally:
        sf_engine.close()

    if not results:
        print("[Notice] No arms completed.")
        return

    # =========================================================================
    # ENHANCED ARCHITECTURAL LEADERBOARD (T=0.0 vs T=0.7)
    # =========================================================================
    print("\n\n" + "=" * 170)
    print("                                            ARCHITECTURE SWEEP LEADERBOARD (T=0.0 vs T=0.7)")
    print("=" * 170)
    header = (
        f"{'Model / Arm':<20} | {'Dim':<4} | {'Tok':<4} | {'Ply':<3} | {'Params':<8} | "
        f"{'Top-1':<6} | {'1350 (T=0.0)':<18} | {'1350 (T=0.7)':<18} | {'1600 (T=0.0)':<18} | {'1600 (T=0.7)':<18}"
    )
    print(header)
    print("-" * 170)

    def sort_key(item):
        a, r = item
        # Multi-objective composite scoring:
        # 1. Peak skill at T=0.0 (high weight)
        # 2. Recovery / generalization under noise at T=0.7 (resilience weight)
        score = r["final_top1"] * 0.05
        weights = {0.0: 1.0, 0.7: 0.8}

        for temp, t_data in r["elo_data"].items():
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

    def fmt_cell(t_dict, elo):
        if not t_dict or elo not in t_dict:
            return "N/A"
        cell = t_dict[elo]
        return f"{cell['score_str']} ({cell['mat_str'].strip()})"

    ranked_arms = sorted(results, key=sort_key, reverse=True)
    for rank, (arm, res) in enumerate(ranked_arms, 1):
        cfg = arm
        ed = res["elo_data"]
        d0 = ed.get(0.0, {})
        d7 = ed.get(0.7, {})

        c13_0 = fmt_cell(d0, 1350)
        c13_7 = fmt_cell(d7, 1350)
        c16_0 = fmt_cell(d0, 1600)
        c16_7 = fmt_cell(d7, 1600)

        print(
            f"#{rank:<2} {arm['name']:<20} | {cfg['dim']:<4} | {cfg['num_tokens']:<4} | {cfg['rollout_steps']:<3} | "
            f"{res['params']:<8,d} | {res['final_top1']:4.1f}% | "
            f"{c13_0:<18} | {c13_7:<18} | {c16_0:<18} | {c16_7:<18}"
        )
    print("=" * 170 + "\n")

    best_arm, best_res = ranked_arms[0]
    champion_path = "out/dialectic_patt_bot.pt"
    torch.save({
        "model_state_dict": best_res["model"].state_dict(),
        "config": {
            "dim": best_arm["dim"],
            "num_patt_tokens": best_arm["num_tokens"],
            "rollout_steps": best_arm["rollout_steps"]
        },
        "arm_name": best_arm["name"]
    }, champion_path)
    print(f"[Champion Artifact Saved] -> {champion_path} (from {best_arm['name']})\n")

if __name__ == "__main__":
    main()