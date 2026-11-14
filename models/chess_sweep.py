# =====================================================================
# models/chess_sweep.py
# Comprehensive Architecture & Hyperparameter Sweep
# Combines: Loss, Top-K, Throughput, and Multi-Elo Stockfish Match Play
#
# Run via: python -m models.chess_sweep
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
    GuardrailedTorchChess,
    compute_cuda_heuristic_scores,
    board_to_fen,
    evaluate_model_elo,
    uci_to_action,
    StockfishEngine,
)


# ==========================================
# 1. LEARNING RATE SCHEDULE (WARMUP + COSINE)
# ==========================================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.02):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==========================================
# 3. ARM TRAINER & BENCHMARK
# ==========================================
def train_and_eval_arm(cfg, steps=300, batch_size=2048, sf_engine=None, elos=(1350, 1600), games_per_elo=16, device="cuda"):
    model = DialecticPattChessBot(
        dim=cfg["dim"],
        num_patt_tokens=cfg["num_tokens"],
        rollout_steps=cfg["rollout_steps"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["peak_lr"], weight_decay=1e-4)
    warmup_steps = int(steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, steps, min_lr_ratio=0.02)
    env = GuardrailedTorchChess(batch_size, device=device, halfmove_limit=80)

    tau = cfg["tau"]
    start_t = time.time()
    accum_top1, accum_top3, accum_loss = 0.0, 0.0, 0.0
    eval_window = min(60, steps // 4)

    # Phase 1: Train with Warmup + Cosine Annealing
    model.train()
    for step in range(1, steps + 1):
        l_mask = env.compute_legal_masks()
        is_w = (env.players == 0)

        with torch.no_grad():
            teacher_scores = compute_cuda_heuristic_scores(env.boards, l_mask, is_w, device)
            teacher_probs = F.softmax(teacher_scores / tau, dim=-1)

        student_logits = model(env.boards, is_w, legal_mask=l_mask)
        student_log_probs = F.log_softmax(student_logits / tau, dim=-1)

        # Stable masked KL Divergence
        valid = teacher_probs > 1e-7
        kl = teacher_probs * (torch.log(teacher_probs.clamp(min=1e-7)) - student_log_probs)
        loss = torch.where(valid, kl, torch.zeros_like(kl)).sum(dim=-1).mean() * (tau ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Telemetry
        with torch.no_grad():
            t_best = torch.argmax(teacher_scores, dim=-1)
            s_best = torch.argmax(student_logits, dim=-1)
            top1 = (s_best == t_best).float().mean().item()
            top3 = (torch.topk(student_logits, 3, dim=-1).indices == t_best.unsqueeze(-1)).any(dim=-1).float().mean().item()

            if step > (steps - eval_window):
                accum_loss += loss.item()
                accum_top1 += top1
                accum_top3 += top3

            # Step games with exploration
            actions = torch.multinomial(F.softmax(student_logits / 1.0, dim=-1), 1).squeeze(-1)
            dones = torch.zeros(batch_size, dtype=torch.bool, device=device)
            winners = torch.zeros(batch_size, dtype=torch.int8, device=device)
            dones, winners = env.step(actions, dones, winners, legal_mask=l_mask)
            if dones.any():
                env.reset(mask=dones)

    train_dt = time.time() - start_t
    train_throughput = (steps * batch_size) / max(1e-4, train_dt)
    final_loss = accum_loss / max(1, eval_window)
    final_top1 = (accum_top1 / max(1, eval_window)) * 100.0
    final_top3 = (accum_top3 / max(1, eval_window)) * 100.0

    # Phase 2: Elo Gauntlet Match Play
    elo_data = {}
    if sf_engine is not None and games_per_elo > 0:
        for elo in elos:
            elo_data[elo] = evaluate_model_elo(
                model, sf_engine, elo=elo, games=games_per_elo, max_plies=200, device=device
            )
    else:
        for elo in elos:
            elo_data[elo] = {"score_str": "N/A", "mat_str": "N/A", "win_pct": 0.0, "mat_delta": 0.0}

    return {
        "final_loss": final_loss,
        "final_top1": final_top1,
        "final_top3": final_top3,
        "throughput": train_throughput,
        "elo_data": elo_data,
        "model": model
    }


# ==========================================
# 4. MAIN SWEEP RUNNER
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Elo Architecture Sweep for DialecticPattChessBot")
    parser.add_argument("--steps", type=int, default=350, help="Training steps per arm")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for Stage 1 flash-bake")
    parser.add_argument("--games_per_elo", type=int, default=16, help="Evaluation games per Stockfish Elo")
    parser.add_argument("--save_best", action="store_true", default=True, help="Save the best arm checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    os.makedirs("out", exist_ok=True)

    # Locate Stockfish
    sf_exec = shutil.which("stockfish")
    if not sf_exec and os.path.exists("stockfish.exe"):
        sf_exec = "./stockfish.exe"

    sf_engine = None
    if sf_exec:
        print(f"[Stockfish Engine Located] -> {sf_exec}")
        sf_engine = StockfishEngine(sf_exec, threads=1, hash_mb=32)
    else:
        print("[Notice] Stockfish binary not found in PATH or working directory. Elo match play will be skipped.")

    test_elos = [1350, 1600]

    # Orthogonal candidate configurations
    candidate_configs = [
        #{"name": "0. Baseline (tau 1.3, K=4, 128 tok, lr 2e-3)      ", "tau": 1.3, "rollout_steps": 4, "peak_lr": 2e-3, "num_tokens": 128, "dim": 64},
        #{"name": "1. Sharp Temp (tau 1.0, K=4, 128 tok, lr 2e-3)    ", "tau": 1.0, "rollout_steps": 4, "peak_lr": 2e-3, "num_tokens": 128, "dim": 64},
        {"name": "2. Deep Ponder (K=6, tau 1.0, 128 tok, lr 4e-3)   ", "tau": 1.0, "rollout_steps": 6, "peak_lr": 4e-3, "num_tokens": 128, "dim": 64},
        #{"name": "3. Shallow Ponder (K=2, tau 1.0, 128 tok, lr 2e-3)", "tau": 1.0, "rollout_steps": 2, "peak_lr": 2e-3, "num_tokens": 128, "dim": 64},
        #{"name": "4. High Memory (256 tok, K=4, tau 1.0, lr 2e-3)   ", "tau": 1.0, "rollout_steps": 4, "peak_lr": 2e-3, "num_tokens": 256, "dim": 64},
        #{"name": "5. Aggressive LR (tau 1.0, K=4, 128 tok, lr 4e-3) ", "tau": 1.0, "rollout_steps": 4, "peak_lr": 4e-3, "num_tokens": 128, "dim": 64},
        #{"name": "5. low T High LR (tau 0.5, K=4, 128 tok, lr 4e-3) ", "tau": 0.5, "rollout_steps": 4, "peak_lr": 2e-3, "num_tokens": 128, "dim": 64},
    ]

    print("\n" + "=" * 115)
    print("                     DIALECTIC PATT CHESS BOT: MULTI-ELO HYPERPARAMETER SWEEP")
    print(f" Train: {args.steps} Steps @ {args.batch_size:,} Games ({args.steps * args.batch_size:,} pos/arm) | Eval: {args.games_per_elo} Games vs SF {test_elos}")
    print("=" * 115)

    results = []
    best_arm_score = -999999.0
    best_model_ckpt = None

    for idx, cfg in enumerate(candidate_configs):
        print(f"\n[Arm {idx+1}/{len(candidate_configs)}] Running: {cfg['name']} ...")
        res = train_and_eval_arm(
            cfg,
            steps=args.steps,
            batch_size=args.batch_size,
            sf_engine=sf_engine,
            elos=test_elos,
            games_per_elo=args.games_per_elo,
            device=device
        )
        results.append((cfg, res))

        # Real-time telemetry print
        e1350 = res["elo_data"].get(1350, {})
        e1600 = res["elo_data"].get(1600, {})
        print(f"   --> Train Metrics : Loss: {res['final_loss']:.4f} | Top-1: {res['final_top1']:4.1f}% | Top-3: {res['final_top3']:4.1f}% | Speed: {res['throughput']:7.0f} pos/s")
        if sf_engine:
            print(f"   --> SF Match Play : Elo 1350: {e1350.get('score_str')} ({e1350.get('mat_str')}) | "
                  f"Elo 1600: {e1600.get('score_str')} ({e1600.get('mat_str')})")

        # Composite score: rewards Elo win rate and material advantage, breaks ties with Top-1 accuracy
        arm_score = (
            e1350.get("win_pct", 0) * 1.5 +
            e1600.get("win_pct", 0) * 2.5 +
            (e1350.get("mat_delta", 0) + e1600.get("mat_delta", 0)) * 0.05 +
            res["final_top1"] * 0.5
        )

        if arm_score > best_arm_score:
            best_arm_score = arm_score
            best_model_ckpt = {
                "model_state_dict": res["model"].state_dict(),
                "config": {
                    "dim": cfg["dim"],
                    "num_patt_tokens": cfg["num_tokens"],
                    "rollout_steps": cfg["rollout_steps"]
                },
                "arm_name": cfg["name"]
            }

    if sf_engine:
        sf_engine.close()

    # ==========================================
    # 5. CONSOLIDATED LEADERBOARD
    # ==========================================
    print("\n\n" + "=" * 135)
    print("                                            FINAL SWEEP LEADERBOARD")
    print("=" * 135)
    header = (
        f"{'Rank':<4} | {'Arm Name':<42} | {'Loss':<7} | {'Top-1':<6} | {'Top-3':<6} | "
        f"{'Speed (pos/s)':<13} | {'Elo 1350 (W/D/L)':<16} | {'ΔMat 1350':<10} | {'Elo 1600 (W/D/L)':<16} | {'ΔMat 1600':<10}"
    )
    print(header)
    print("-" * 135)

    # Sort leaderboard primarily by SF playing performance (or Top-1 if SF is unavailable)
    def sort_key(item):
        c, r = item
        if sf_exec:
            return (
                r["elo_data"].get(1600, {}).get("win_pct", 0) * 2.0 +
                r["elo_data"].get(1350, {}).get("win_pct", 0) +
                (r["elo_data"].get(1350, {}).get("mat_delta", 0) + r["elo_data"].get(1600, {}).get("mat_delta", 0)) * 0.01 +
                r["final_top1"] * 0.1
            )
        return r["final_top1"]

    ranked_results = sorted(results, key=sort_key, reverse=True)

    for rank, (cfg, res) in enumerate(ranked_results, 1):
        e1350 = res["elo_data"].get(1350, {})
        e1600 = res["elo_data"].get(1600, {})

        print(
            f"#{rank:<3} | {cfg['name']:<42} | {res['final_loss']:<7.4f} | {res['final_top1']:5.1f}% | {res['final_top3']:5.1f}% | "
            f"{res['throughput']:<13.0f} | {e1350.get('score_str', 'N/A'):<16} | {e1350.get('mat_str', 'N/A'):<10} | "
            f"{e1600.get('score_str', 'N/A'):<16} | {e1600.get('mat_str', 'N/A'):<10}"
        )
    print("=" * 135 + "\n")

    # Save champion model
    if args.save_best and best_model_ckpt is not None:
        save_path = "out/dialectic_patt_bot.pt"
        torch.save(best_model_ckpt, save_path)
        print(f"[Champion Model Saved] -> {save_path} (from {best_model_ckpt['arm_name']})\n")


if __name__ == "__main__":
    main()