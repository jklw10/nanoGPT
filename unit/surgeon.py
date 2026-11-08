"""
nodechess_surgeon.py - Mechanistic Diagnostic & Argmax Inspector
Usage:
    python -m models.nodechess_surgeon --ckpt latest --thinking_steps 12
    python -m models.nodechess_surgeon --ckpt out/ckpt(10).pt --thinking_steps 32
"""

import argparse
import os
import shutil
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nodechess15 import (
    ChessNodeNet, 
    GuardrailedTorchChess, 
    HyperParamsConfig,
    StockfishEngine, 
    board_to_fen, 
    uci_to_action,
    find_latest_checkpoint
)

# =====================================================================
# 1. TACTICAL PROBE BENCH
# =====================================================================
TACTICAL_PROBES = [
    {
        "name": "Back-Rank Mate in 1",
        "fen": "6k1/5ppp/8/8/8/8/4QPPP/6K1 w - - 0 1",
        "good_moves": ["e2e8"],
        "blunders": ["e2d2", "e2e3", "g1f1"]
    },
    {
        "name": "Hanging Knight Capture",
        "fen": "r1bqk2r/pppp1ppp/2n5/4p3/1b2n3/2NP1N2/PPP1BPPP/R1BQK2R w KQkq - 0 1",
        "good_moves": ["d3e4"],
        "blunders": ["e1g1", "c1d2"]
    },
    {
        "name": "Defend King in Check",
        "fen": "rnb1k1nr/pppp1ppp/4p3/8/1b1P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 1",
        "good_moves": ["c1d2", "a2a3", "e2e3"],
        "blunders": ["g1f3", "d4d5"]
    },
    {
        "name": "Free Queen / Capture on e5",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1",
        "good_moves": ["c1g5", "d4e5", "f3e5"],
        "blunders": ["f3g5", "d4d5"]
    }
]

def uci_format(from_sq, to_sq):
    rf, cf = from_sq // 8, from_sq % 8
    rt, ct = to_sq // 8, to_sq % 8
    return f"{chr(ord('a') + cf)}{rf + 1}{chr(ord('a') + ct)}{rt + 1}"

def fen_to_board_tensor(fen, device="cuda"):
    parts = fen.split()
    board_str = parts[0]
    turn_char = parts[1]
    
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
    }
    
    board = torch.zeros((8, 8), dtype=torch.int8, device=device)
    rows = board_str.split('/')
    for r_idx, row in enumerate(rows):
        board_r = 7 - r_idx
        c_idx = 0
        for ch in row:
            if ch.isdigit():
                c_idx += int(ch)
            else:
                board[board_r, c_idx] = piece_map[ch]
                c_idx += 1
                
    player = 0 if turn_char == 'w' else 1
    return board, player

# =====================================================================
# 2. MECHANISTIC AUDIT: Factorization & Weight Spectrum
# =====================================================================
def audit_model_health(model):
    print("\n" + "=" * 70)
    print(" [1] MECHANISTIC WEIGHT & FACTORIZATION AUDIT")
    print("=" * 70)
    
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    w1_norm = torch.norm(raw.w1.weight, p=2, dim=-2)
    print(f"[*] w1 Predictor Norm: Mean: {w1_norm.mean().item():.4f} | Target Scale: {raw.w1.w_scale.item():.4f}")
    
    w_qk = raw.router.w_qk
    qk_norm = torch.norm(w_qk, p=2, dim=-1)
    print(f"[*] Router Key/Query Norm: Mean: {qk_norm.mean().item():.4f} | Std: {qk_norm.std().item():.4f}")
    
    s_norm = torch.norm(raw.sensory_proj.weight, p=2)
    r_norm = torch.norm(raw.reflex_proj.weight, p=2)
    print(f"[*] Sensory Proj Norm: {s_norm.item():.4f} | Reflex Proj Norm: {r_norm.item():.4f}")
    
    raw.update_cached_spatial()
    aff = raw.spatial_affinity
    print(f"[*] Spatial Affinity Matrix (64x64):")
    print(f"    Min: {aff.min().item():.4f} | Max: {aff.max().item():.4f} | Mean: {aff.mean().item():.4f} | Norm: {torch.norm(aff).item():.4f}")
    
    if aff.abs().max() < 0.25:
        print("    [!] NOTICE: Spatial affinity amplitude is low (|M| < 0.25).")
        print("        Action scoring relies predominantly on independent f(from) + g(to) sums.")

# =====================================================================
# 3. ARGMAX-AWARE TACTICAL BENCHMARK
# =====================================================================
def probe_tactical_competence(model, thinking_steps=12, device="cuda"):
    print("\n" + "=" * 70)
    print(f" [2] PROBING MODEL DECISIONS (ARGMAX EVALUATION @ {thinking_steps} STEPS)")
    print("=" * 70)
    
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    for probe in TACTICAL_PROBES:
        board, player = fen_to_board_tensor(probe["fen"], device=device)
        env = GuardrailedTorchChess(batch_size=1, device=device)
        env.boards[0] = board
        env.players[0] = player
        
        legal_mask = env.compute_legal_masks()
        obs, _ = env.get_obs(agent_noise=torch.tensor([0.01], device=device))
        
        state, out, err, lp = raw.init_states(batch_size=1)
        threat = torch.tensor([0.0], device=device)
        
        with torch.no_grad():
            # Deliberate for requested thinking steps (unpacks 7 values safely)
            _, state, out, err, lp, *_ = raw(
                obs, threat, state, out, err, lp, thinking_steps=thinking_steps
            )
            raw_logits = raw.get_action_logits(state)[0]
            masked_logits = raw_logits + legal_mask[0]
            
        legal_indices = torch.nonzero(legal_mask[0] > -1e8).squeeze(-1)
        num_legal = legal_indices.numel()
        
        # Rank ALL legal moves by logit (highest to lowest)
        sorted_scores, sorted_ranks = torch.sort(masked_logits[legal_indices], descending=True)
        sorted_acts = legal_indices[sorted_ranks].tolist()
        act_to_rank = {act: rank + 1 for rank, act in enumerate(sorted_acts)}
        
        # Determine actual move selected by torch.argmax
        best_act = sorted_acts[0]
        best_uci = uci_format(best_act // 64, best_act % 64)
        best_logit = sorted_scores[0].item()
        
        print(f"\nProbe: {probe['name']}")
        print(f"  FEN: {probe['fen']}")
        print(f"  Total Legal Moves: {num_legal}")
        print(f"  -------------------------------------------------------------")
        print(f"  >>> ACTUAL ARGMAX MOVE: {best_uci} (Logit: {best_logit:6.2f}) <<<")
        print(f"  -------------------------------------------------------------")
        
        # Top 5 legal moves overall
        print("  Top 5 Highest Legal Moves Available:")
        for r in range(min(5, num_legal)):
            a = sorted_acts[r]
            uci = uci_format(a // 64, a % 64)
            l_val = sorted_scores[r].item()
            is_best = " [ARGMAX PLAYED]" if r == 0 else ""
            print(f"    #{r+1:>2d}: Move {uci:5s} | Logit: {l_val:6.2f}{is_best}")
            
        print("  Candidate Benchmark Status:")
        for gm in probe["good_moves"]:
            act = uci_to_action(gm)
            if act is not None and act in act_to_rank:
                rnk = act_to_rank[act]
                l_val = masked_logits[act].item()
                flag = " [SELECTED!]" if rnk == 1 else ""
                print(f"    [GOOD]  Move {gm:5s} => Rank: {rnk:>2d}/{num_legal} | Logit: {l_val:6.2f}{flag}")
                
        for bm in probe["blunders"]:
            act = uci_to_action(bm)
            if act is not None and act in act_to_rank:
                rnk = act_to_rank[act]
                l_val = masked_logits[act].item()
                flag = " [CRITICAL BLUNDER PLAYED!]" if rnk == 1 else ""
                print(f"    [BLUND] Move {bm:5s} => Rank: {rnk:>2d}/{num_legal} | Logit: {l_val:6.2f}{flag}")

# =====================================================================
# 4. LIVE STOCKFISH BENCHMARK MATCHES
# =====================================================================
def test_against_stockfish(model, num_games=10, sf_nodes=500, thinking_steps=12, device="cuda"):
    print("\n" + "=" * 70)
    print(f" [3] PLAYING {num_games} LIVE GAMES AGAINST STOCKFISH ({sf_nodes} nodes | {thinking_steps} steps)")
    print("=" * 70)
    
    sf_exec = shutil.which("stockfish")
    if sf_exec is None:
        print("[!] Stockfish executable not found in PATH.")
        return
        
    sf = StockfishEngine(stockfish_path=sf_exec)
    env = GuardrailedTorchChess(batch_size=num_games, device=device)
    noise = torch.full((num_games,), 0.01, device=device)
    
    agent_white = torch.zeros(num_games, dtype=torch.bool, device=device)
    agent_white[:num_games // 2] = True
    
    dones = torch.zeros(num_games, dtype=torch.bool, device=device)
    winners = torch.zeros(num_games, dtype=torch.int8, device=device)
    plies = torch.zeros(num_games, dtype=torch.long, device=device)
    
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    state, out, err, lp = raw.init_states(batch_size=num_games)
    
    max_plies = 300
    while not dones.all() and (plies < max_plies).any():
        active = ~dones
        legal_masks = env.compute_legal_masks()
        turns = env.players
        is_agent_turn = active & ((turns == 0) == agent_white)
        is_sf_turn = active & ~is_agent_turn
        actions = torch.zeros(num_games, dtype=torch.long, device=device)
        
        if is_agent_turn.any():
            ag_idx = torch.nonzero(is_agent_turn).squeeze(-1)
            obs, threat = env.get_obs(agent_noise=noise)
            
            with torch.no_grad():
                _, s_next, o_next, e_next, l_next, *_ = raw(
                    obs[ag_idx], threat[ag_idx],
                    state[:, ag_idx, :], out[:, ag_idx, :],
                    err[:, ag_idx], lp[:, ag_idx],
                    thinking_steps=thinking_steps
                )
                state[:, ag_idx, :] = s_next
                out[:, ag_idx, :] = o_next
                err[:, ag_idx] = e_next
                lp[:, ag_idx] = l_next
                
                raw_logits = raw.get_action_logits(s_next)
                # Pure argmax execution
                actions[ag_idx] = torch.argmax(raw_logits + legal_masks[ag_idx], dim=-1)
                
        if is_sf_turn.any():
            sf_indices = torch.nonzero(is_sf_turn).squeeze(-1).tolist()
            for i in sf_indices:
                fen = board_to_fen(
                    env.boards[i], int(turns[i].item()), 
                    env.castling_rights[i].tolist(), 
                    ep_sq=int(env.en_passant_sq[i].item()),
                    halfmove_clock=int(env.halfmove_clock[i].item()),
                    fullmove_num=int(plies[i].item() // 2 + 1)
                )
                legal_acts = torch.nonzero(legal_masks[i] > -1e8).squeeze(-1).tolist()
                best_m = sf.get_move(fen, nodes=sf_nodes)
                act = uci_to_action(best_m) if best_m else None
                if act is None or act not in legal_acts:
                    act = legal_acts[0]
                actions[i] = act
                
        dones, winners = env.step(actions, dones, winners, legal_mask=legal_masks)
        plies[active] += 1
        
    sf.close()
    
    ag_wins = (((winners == 1) & agent_white) | ((winners == 2) & (~agent_white))).sum().item()
    sf_wins = (((winners == 2) & agent_white) | ((winners == 1) & (~agent_white))).sum().item()
    draws   = (winners == 3).sum().item()
    
    print(f"\nFinal Result vs Stockfish @ {sf_nodes} nodes:")
    print(f"  Agent Wins: {ag_wins} | Stockfish Wins: {sf_wins} | Draws: {draws}")
    print(f"  Score: {((ag_wins + 0.5 * draws) / num_games)*100:.1f}%")
    print(f"  Avg Plies Lasted: {plies.float().mean().item():.1f}")

# =====================================================================
# 5. MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Mechanistic Argmax Inspector for nodechess15")
    parser.add_argument("--ckpt", type=str, default="latest", help="Path to checkpoint or 'latest'")
    parser.add_argument("--thinking_steps", type=int, default=12, help="Thinking steps per ply (e.g. 12 or 32)")
    parser.add_argument("--sf_games", type=int, default=10, help="Number of games vs Stockfish")
    parser.add_argument("--sf_nodes", type=int, default=500, help="Stockfish search nodes")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running Diagnostic on {device}...")
    
    cfg = HyperParamsConfig(device)
    model = ChessNodeNet(num_nodes=32, dim=32, input_dim=100, rank_k=16, device=device).to(device)
    model.apply_hyperparams(cfg)
    
    ckpt_path = find_latest_checkpoint("out") if args.ckpt == "latest" else args.ckpt
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Restoring checkpoint: {ckpt_path}")
        ckpt_data = torch.load(ckpt_path, map_location=device)
        sd = ckpt_data.get('model_state_dict', ckpt_data)
        cleaned_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(cleaned_sd, strict=False)
        model.w1.normalize_weight()
        model.update_cached_spatial()
    else:
        print(f"[!] Checkpoint not found at {args.ckpt}. Using uninitialized model.")
        
    audit_model_health(model)
    probe_tactical_competence(model, thinking_steps=args.thinking_steps, device=device)
    test_against_stockfish(model, num_games=args.sf_games, sf_nodes=args.sf_nodes, thinking_steps=args.thinking_steps, device=device)

if __name__ == "__main__":
    main()