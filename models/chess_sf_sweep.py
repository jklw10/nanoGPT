# =====================================================================
# models/chess_sweep_sf.py
# Pure Stockfish Architecture Sweep (Direct Cross-Entropy on SF Bestmove)
# Tracks: Live SF Training Rate (pos/s), CE Loss, SF Top-1, and Elo Match Play
#
# Run via: python -m models.chess_sweep_sf
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
    board_to_fen,
    uci_to_action,
    StockfishEngine,
    evaluate_model_elo,
)

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.02):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==========================================
# 1. EVALUATION AGAINST STOCKFISH ELO
# ==========================================

# ==========================================
# 2. PURE STOCKFISH ARM TRAINER
# ==========================================
def train_and_eval_pure_sf_arm(cfg, steps=250, batch_size=64, sf_engine=None, sf_train_nodes=50,
                              elos=(1350, 1600), games_per_elo=16, device="cuda"):
    model = DialecticPattChessBot(
        dim=cfg["dim"],
        num_patt_tokens=cfg["num_tokens"],
        rollout_steps=cfg["rollout_steps"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["peak_lr"], weight_decay=1e-4)
    warmup_steps = int(steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, steps, min_lr_ratio=0.02)
    env = GuardrailedTorchChess(batch_size, device=device, halfmove_limit=60)

    start_t = time.time()
    accum_loss, accum_top1 = 0.0, 0.0
    total_positions_trained = 0
    report_window = min(40, steps // 4)

    # Disable Elo throttling during training generation so Stockfish gives high-quality targets
    sf_engine._send(["setoption name UCI_LimitStrength value false", "isready"])
    while True:
        if "readyok" in sf_engine.proc.stdout.readline():
            break

    model.train()
    for step in range(1, steps + 1):
        l_mask = env.compute_legal_masks()
        is_w = (env.players == 0)

        # 1. Query Stockfish for ground-truth bestmove on the active batch
        train_idx, sf_targets = [], []
        for i in range(batch_size):
            legals = torch.nonzero(l_mask[i] > -1e8).squeeze(-1).tolist()
            if not legals:
                continue
            fen = board_to_fen(
                env.boards[i], 0 if is_w[i] else 1, env.castling_rights[i].tolist(),
                int(env.en_passant_sq[i].item()), int(env.halfmove_clock[i].item()), 1
            )
            bm = sf_engine.get_move(fen, movetime_ms=10)
            act = uci_to_action(bm) if bm else None
            if act is not None and act in legals:
                train_idx.append(i)
                sf_targets.append(act)

        if not train_idx:
            env.reset()
            continue

        t_idx = torch.tensor(train_idx, device=device)
        targets = torch.tensor(sf_targets, device=device)

        # 2. Student Cross-Entropy directly against Stockfish
        logits = model(env.boards[t_idx], is_w[t_idx], legal_mask=l_mask[t_idx])
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Telemetry
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            top1 = (preds == targets).float().mean().item()
            total_positions_trained += targets.numel()

            if step > (steps - report_window):
                accum_loss += loss.item()
                accum_top1 += top1

            # Step environment using the model + temperature exploration
            actions = torch.zeros(batch_size, dtype=torch.long, device=device)
            full_logits = model(env.boards, is_w, legal_mask=l_mask)
            actions = torch.multinomial(F.softmax(full_logits / 1.0, dim=-1), 1).squeeze(-1)

            dones = torch.zeros(batch_size, dtype=torch.bool, device=device)
            winners = torch.zeros(batch_size, dtype=torch.int8, device=device)
            dones, winners = env.step(actions, dones, winners, legal_mask=l_mask)
            if dones.any():
                env.reset(mask=dones)

    train_dt = time.time() - start_t
    train_rate = total_positions_trained / max(1e-4, train_dt)
    final_loss = accum_loss / max(1, report_window)
    final_top1 = (accum_top1 / max(1, report_window)) * 100.0

    # Phase 2: Elo Gauntlet Match Play
    elo_data = {}
    for elo in elos:
        elo_data[elo] = evaluate_model_elo(
            model, sf_engine, elo=elo, games=games_per_elo, max_plies=60, device=device
        )

    return {
        "final_loss": final_loss,
        "final_top1": final_top1,
        "throughput": train_rate,
        "total_pos": total_positions_trained,
        "elo_data": elo_data,
        "model": model
    }


# ==========================================
# 3. MAIN RUNNER
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Pure Stockfish Sweep for DialecticPattChessBot")
    parser.add_argument("--steps", type=int, default=250, help="Training steps per arm")
    parser.add_argument("--batch_size", type=int, default=64, help="Parallel boards per step")
    parser.add_argument("--sf_nodes", type=int, default=50, help="Nodes per Stockfish query during training")
    parser.add_argument("--games_per_elo", type=int, default=16, help="Evaluation games per Elo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    sf_exec = shutil.which("stockfish")
    if not sf_exec and os.path.exists("stockfish.exe"):
        sf_exec = "./stockfish.exe"

    if not sf_exec:
        print("[Error] Stockfish binary not found. A working Stockfish binary is required for this sweep.")
        return

    sf_engine = StockfishEngine(sf_exec, threads=2, hash_mb=64)
    test_elos = [1350, 1600]

    candidate_configs = [
        {"name": "0. Baseline (K=4, 128 tok, lr 2e-3)",     "rollout_steps": 4, "peak_lr": 2e-3, "num_tokens": 128, "dim": 64},
        {"name": "1. Deep Ponder (K=6, 128 tok, lr 2e-3)",  "rollout_steps": 6, "peak_lr": 2e-3, "num_tokens": 128, "dim": 64},
        {"name": "2. Shallow Ponder (K=2, 128 tok, lr 3e-3)","rollout_steps": 2, "peak_lr": 3e-3, "num_tokens": 128, "dim": 64},
        {"name": "3. High Memory (K=4, 256 tok, lr 2e-3)",   "rollout_steps": 4, "peak_lr": 2e-3, "num_tokens": 256, "dim": 64},
        {"name": "4. Aggressive LR (K=4, 128 tok, lr 4e-3)", "rollout_steps": 4, "peak_lr": 4e-3, "num_tokens": 128, "dim": 64},
    ]

    print("\n" + "=" * 115)
    print("                     PURE STOCKFISH DIALECTIC SWEEP (DIRECT CROSS-ENTROPY)")
    print(f" Train: {args.steps} Steps @ Batch {args.batch_size} (~{args.steps * args.batch_size:,} pos/arm) | Eval: {args.games_per_elo} Games vs SF {test_elos}")
    print("=" * 115)

    results = []
    for idx, cfg in enumerate(candidate_configs):
        print(f"\n[Arm {idx+1}/{len(candidate_configs)}] Running: {cfg['name']} ...")
        res = train_and_eval_pure_sf_arm(
            cfg,
            steps=args.steps,
            batch_size=args.batch_size,
            sf_engine=sf_engine,
            sf_train_nodes=args.sf_nodes,
            elos=test_elos,
            games_per_elo=args.games_per_elo,
            device=device
        )
        results.append((cfg, res))

        e1350 = res["elo_data"].get(1350, {})
        e1600 = res["elo_data"].get(1600, {})
        print(f"   --> SF Training Rate: {res['throughput']:5.0f} pos/s | CE Loss: {res['final_loss']:.4f} | SF Top-1: {res['final_top1']:4.1f}%")
        print(f"   --> SF Match Play   : Elo 1350: {e1350.get('score_str')} ({e1350.get('mat_str')}) | "
              f"Elo 1600: {e1600.get('score_str')} ({e1600.get('mat_str')})")

    sf_engine.close()

    # Leaderboard sorted by actual Elo 1350 + 1600 match performance
    print("\n\n" + "=" * 125)
    print("                                      PURE STOCKFISH SWEEP LEADERBOARD")
    print("=" * 125)
    header = (
        f"{'Rank':<4} | {'Arm Name':<38} | {'CE Loss':<8} | {'SF Top-1':<9} | "
        f"{'SF Rate':<11} | {'Elo 1350 (W/D/L)':<16} | {'ΔMat 1350':<10} | {'Elo 1600 (W/D/L)':<16} | {'ΔMat 1600':<10}"
    )
    print(header)
    print("-" * 125)

    def sort_score(item):
        c, r = item
        e13 = r["elo_data"].get(1350, {})
        e16 = r["elo_data"].get(1600, {})
        return (
            e16.get("win_pct", 0) * 2.0 +
            e13.get("win_pct", 0) +
            (e13.get("mat_delta", 0) + e16.get("mat_delta", 0)) * 0.02 +
            r["final_top1"] * 0.5
        )

    ranked_results = sorted(results, key=sort_score, reverse=True)
    for rank, (cfg, res) in enumerate(ranked_results, 1):
        e1350 = res["elo_data"].get(1350, {})
        e1600 = res["elo_data"].get(1600, {})
        print(
            f"#{rank:<3} | {cfg['name']:<38} | {res['final_loss']:<8.4f} | {res['final_top1']:5.1f}%   | "
            f"{res['throughput']:5.0f} pos/s | {e1350.get('score_str'):<16} | {e1350.get('mat_str'):<10} | "
            f"{e1600.get('score_str'):<16} | {e1600.get('mat_str'):<10}"
        )
    print("=" * 125 + "\n")


if __name__ == "__main__":
    main()