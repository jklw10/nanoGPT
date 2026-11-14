# =====================================================================
# chess_sf_sweep2.py
# High-Throughput Pure Stockfish Dataset Polish Sweep
# =====================================================================

import argparse
import math
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_bot import (
    DialecticPattChessBot,
    StockfishEngine,
    evaluate_model_elo,
)

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

def train_and_eval_arm(arm_cfg, model_cfg, dataset, sf_engine,
                       epochs=10, default_batch=2048, elos=(1350, 1600), games_per_elo=24, device="cuda"):
    model = DialecticPattChessBot(
        dim=model_cfg["dim"],
        num_patt_tokens=model_cfg["num_tokens"],
        rollout_steps=model_cfg["rollout_steps"]
    ).to(device)

    # 1. Parameter grouping with separable spatial & context embeddings
    raw_model = getattr(model, "_orig_mod", model)
    embed_param_set = {
        raw_model.piece_embed.weight,
        raw_model.rank_embed.weight,
        raw_model.file_embed.weight,
        raw_model.turn_embed.weight,
        raw_model.castling_proj.weight,
    }
    embed_params = [p for p in raw_model.parameters() if p in embed_param_set]
    backbone_params = [p for p in raw_model.parameters() if p not in embed_param_set]

    freeze_embeds = arm_cfg.get("freeze_embeddings", False)
    embed_lr = arm_cfg.get("embed_lr", None)
    lr = arm_cfg.get("lr", 2e-3)
    wd = arm_cfg.get("weight_decay", 1e-4)
    batch_size = arm_cfg.get("batch_size", default_batch)
    label_smoothing = arm_cfg.get("label_smoothing", 0.0)
    min_lr_ratio = arm_cfg.get("min_lr_ratio", 0.05)
    clip_norm = arm_cfg.get("clip_norm", 1.0)

    if freeze_embeds:
        for p in embed_params:
            p.requires_grad = False
        param_groups = [{"params": backbone_params, "lr": lr, "weight_decay": wd}]
    elif embed_lr is not None:
        param_groups = [
            {"params": embed_params, "lr": embed_lr, "weight_decay": 0.0},
            {"params": backbone_params, "lr": lr, "weight_decay": wd}
        ]
    else:
        param_groups = [{"params": list(raw_model.parameters()), "lr": lr, "weight_decay": wd}]

    optimizer = torch.optim.AdamW(param_groups, fused=True)

    boards = dataset["boards"]
    is_white = dataset["is_white"]
    targets = dataset["targets"]
    castling = dataset.get("castling_rights", None)
    raw_dataset_size = targets.shape[0]

    effective_size = raw_dataset_size - (raw_dataset_size % batch_size)
    steps_per_epoch = effective_size // batch_size
    total_steps = steps_per_epoch * epochs
    total_positions = effective_size * epochs
    warmup_steps = int(total_steps * 0.02)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=min_lr_ratio)

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print(f"   [Config] Batch: {batch_size} | LR: {lr:.1e} | Embed LR: {embed_lr if embed_lr else ('FROZEN' if freeze_embeds else lr)} | Smoothing: {label_smoothing} | WD: {wd:.1e}")
    print(f"   [Volume] {effective_size:,} pos/epoch * {epochs} epochs = {total_positions:,} total pos ({total_steps:,} steps)")

    print("   [Warmup] Compiling model CUDA graphs...", end="", flush=True)
    compiled_model = torch.compile(model, mode="reduce-overhead")

    # Warmup passes to prime static allocations
    wb, ww, wy = boards[:batch_size], is_white[:batch_size], targets[:batch_size]
    wc = castling[:batch_size] if castling is not None else None
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            w_logits = compiled_model(wb, ww, castling_rights=wc)
            w_loss = F.cross_entropy(w_logits, wy, label_smoothing=label_smoothing)
        w_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(" Ready.")

    arm_start_t = time.time()
    last_epoch_loss, last_epoch_correct = 0.0, 0
    last_epoch_samples = 0
    global_step = 0
    report_interval = max(20, steps_per_epoch // 4)

    compiled_model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(effective_size, device=device)
        is_last_epoch = (epoch == epochs)

        for i in range(0, effective_size, batch_size):
            global_step += 1
            idx = perm[i:i + batch_size]
            b, w, y = boards[idx], is_white[idx], targets[idx]
            c = castling[idx] if castling is not None else None

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits = compiled_model(b, w, castling_rights=c)
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

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

            if global_step % report_interval == 0:
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
    print("   [Gauntlet] Running Stockfish match play validation...")

    elo_data = {}
    for elo in elos:
        elo_data[elo] = evaluate_model_elo(
            model, sf_engine, elo=elo, games=games_per_elo, max_plies=400, device=device
        )

    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        torch.compiler.reset()
    torch.cuda.empty_cache()

    return {
        "final_loss": final_loss,
        "final_top1": final_top1,
        "throughput": train_rate,
        "elo_data": elo_data,
        "model": model
    }

def main():
    parser = argparse.ArgumentParser(description="High-Throughput Pure Stockfish Dataset Polish Sweep")
    parser.add_argument("--dataset", type=str, default="data/chess/sf_dataset.pt", help="Path to offline SF dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per arm")
    parser.add_argument("--batch", type=int, default=2048, help="Default batch size (2048+ recommended)")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_tokens", type=int, default=128, help="Patt concept tokens")
    parser.add_argument("--rollout_steps", type=int, default=4, help="Rollout steps")
    parser.add_argument("--games_per_elo", type=int, default=24, help="Evaluation games per Elo")
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
    print(f"Loaded {sf_data['targets'].shape[0]:,} positions successfully (Dataset cached in VRAM).")

    sf_exec = shutil.which("stockfish")
    if not sf_exec and os.path.exists("stockfish.exe"):
        sf_exec = "./stockfish.exe"
    if not sf_exec:
        raise RuntimeError("Stockfish binary required for evaluation gauntlet.")

    sf_engine = StockfishEngine(sf_exec, threads=2, hash_mb=64)
    test_elos = [1350, 1600]

    model_cfg = {
        "dim": args.dim,
        "num_tokens": args.num_tokens,
        "rollout_steps": args.rollout_steps
    }

    # ==========================================
    # EXPERIMENTAL SWEEP ARMS
    # ==========================================
    sweep_arms = [
        {
            "name": "0. Control Baseline (Batch 2048, lr 2.0e-3, wd 1e-4)",
            "lr": 2.0e-3, "weight_decay": 1e-4, "label_smoothing": 0.0,
            "freeze_embeddings": False, "batch_size": args.batch, "min_lr_ratio": 0.05
        },
        {
            "name": "1. Differential LR (Embeds 2e-4, Backbone 2.5e-3)",
            "lr": 2.5e-3, "embed_lr": 2e-4, "weight_decay": 1e-4, "label_smoothing": 0.0,
            "freeze_embeddings": False, "batch_size": args.batch, "min_lr_ratio": 0.05
        },
        {
            "name": "2. Regularized (lr 2.5e-3, smooth 0.02, wd 5e-4)",
            "lr": 2.5e-3, "weight_decay": 5e-4, "label_smoothing": 0.02,
            "freeze_embeddings": False, "batch_size": args.batch, "min_lr_ratio": 0.05
        },
        {
            "name": "3. Ultra-Batch Saturator (Batch 4096, lr 3.5e-3, wd 5e-4)",
            "lr": 3.5e-3, "weight_decay": 5e-4, "label_smoothing": 0.01,
            "freeze_embeddings": False, "batch_size": 4096, "min_lr_ratio": 0.05
        },
        {
            "name": "4. Conservative SFT (lr 8e-4, wd 1e-3, smooth 0.01)",
            "lr": 8.0e-4, "weight_decay": 1e-3, "label_smoothing": 0.01,
            "freeze_embeddings": False, "batch_size": args.batch, "min_lr_ratio": 0.10
        },
        {
            "name": "5. Deep Anneal (lr 3.0e-3 -> min_ratio 0.01, clip 0.5)",
            "lr": 3.0e-3, "weight_decay": 1e-4, "label_smoothing": 0.01,
            "freeze_embeddings": False, "batch_size": args.batch, "min_lr_ratio": 0.01, "clip_norm": 0.5
        },
    ]

    print("\n" + "=" * 115)
    print("                      OFFLINE STOCKFISH DATASET POLISH SWEEP")
    print(f" {len(sweep_arms)} Arms | {args.epochs} Epochs/Arm | Eval: {args.games_per_elo} Games vs Stockfish {test_elos}")
    print("=" * 115)

    results = []
    try:
        for idx, arm in enumerate(sweep_arms):
            print(f"\n[Arm {idx+1}/{len(sweep_arms)}] Running: {arm['name']} ...")
            res = train_and_eval_arm(
                arm_cfg=arm,
                model_cfg=model_cfg,
                dataset=sf_data,
                sf_engine=sf_engine,
                epochs=args.epochs,
                default_batch=args.batch,
                elos=test_elos,
                games_per_elo=args.games_per_elo,
                device=device
            )
            results.append((arm, res))

            e13 = res["elo_data"][1350]
            e16 = res["elo_data"][1600]
            print(f"   --> CE Loss: {res['final_loss']:.4f} | Top-1: {res['final_top1']:4.1f}% | Speed: {res['throughput']:.0f} pos/s")
            print(f"   --> Match Play: Elo 1350: {e13['score_str']} ({e13['mat_str']}) | Elo 1600: {e16['score_str']} ({e16['mat_str']})")

            arm_ckpt_path = f"out/sweep_arm_{idx}.pt"
            torch.save({
                "model_state_dict": res["model"].state_dict(),
                "config": model_cfg,
                "arm_name": arm["name"],
                "results": {"loss": res["final_loss"], "top1": res["final_top1"], "elo": res["elo_data"]}
            }, arm_ckpt_path)

    finally:
        sf_engine.close()

    if not results:
        print("[Notice] No arms completed.")
        return

    # Leaderboard
    print("\n\n" + "=" * 130)
    print("                                   FINAL SWEEP LEADERBOARD")
    print("=" * 130)
    header = (
        f"{'Model / Arm':<44} | {'Loss':<8} | {'Top-1':<8} | {'Speed (pos/s)':<14} | "
        f"{'Elo 1350 (W/D/L)':<16} | {'Elo 1600 (W/D/L)':<16}"
    )
    print(header)
    print("-" * 130)

    def sort_key(item):
        a, r = item
        e13 = r["elo_data"][1350]
        e16 = r["elo_data"][1600]
        w13 = e13["wins"] / max(1, e13["wins"] + e13["draws"] + e13["losses"])
        w16 = e16["wins"] / max(1, e16["wins"] + e16["draws"] + e16["losses"])
        d13 = e13["draws"] / max(1, e13["wins"] + e13["draws"] + e13["losses"])
        d16 = e16["draws"] / max(1, e16["wins"] + e16["draws"] + e16["losses"])
        return (w16 * 3.0) + (w13 * 2.0) + ((d13 + d16) * 0.5) + (r["final_top1"] * 0.05)

    ranked_arms = sorted(results, key=sort_key, reverse=True)
    for rank, (arm, res) in enumerate(ranked_arms, 1):
        e13 = res["elo_data"][1350]
        e16 = res["elo_data"][1600]
        print(
            f"#{rank:<2} {arm['name']:<40} | {res['final_loss']:<8.4f} | {res['final_top1']:5.1f}%   | "
            f"{res['throughput']:<14.0f} | {e13['score_str']:<16} | {e16['score_str']:<16}"
        )
    print("=" * 130 + "\n")

    best_arm, best_res = ranked_arms[0]
    champion_path = "out/dialectic_patt_bot.pt"
    torch.save({
        "model_state_dict": best_res["model"].state_dict(),
        "config": model_cfg,
        "arm_name": best_arm["name"]
    }, champion_path)
    print(f"[Champion Artifact Saved] -> {champion_path} (from {best_arm['name']})\n")

if __name__ == "__main__":
    main()