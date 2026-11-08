"""
diagnose_champion.py - Mechanistic Diagnostic Tool for 32-Node NodeChess
Usage: python -m models.diagnose_champion [--ckpt out/champion_agent.pt]
"""

import argparse
import math
import os
import torch
import torch.nn.functional as F

from models.nodechess15 import (
    ChessNodeNet, 
    GuardrailedTorchChess, 
    HyperParamsConfig
)

def uci_format(from_sq, to_sq):
    rf, cf = from_sq // 8, from_sq % 8
    rt, ct = to_sq // 8, to_sq % 8
    return f"{chr(ord('a') + cf)}{rf + 1}{chr(ord('a') + ct)}{rt + 1}"

def unpack_and_diagnose(ckpt_path="out/champion_agent.pt", device="cpu"):
    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint not found at: {ckpt_path}")
        return

    print("=" * 80)
    print(f" UNPACKING & DIAGNOSING CHAMPION AGENT: {ckpt_path}")
    print("=" * 80)

    ckpt = torch.load(ckpt_path, map_location=device)
    raw_sd = ckpt['model_state_dict']
    cleaned_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}

    # Architecture metadata
    N = 32
    D = 32
    K = 16
    cfg = HyperParamsConfig(device)
    model = ChessNodeNet(num_nodes=N, dim=D, input_dim=100, rank_k=K, device=device).to(device)
    model.load_state_dict(cleaned_sd, strict=False)
    model.eval()

    slot_id = ckpt.get('champion_slot_id', 'Unknown')
    noise = ckpt.get('champion_noise', torch.tensor([0.0])).item()
    state = ckpt.get('champion_state', None)       # (32, 1, 32)
    output = ckpt.get('champion_output', None)     # (32, 1, 32)
    error_ema = ckpt.get('champion_error_ema', None) # (32, 1)
    lp_ema = ckpt.get('champion_lp_ema', None)       # (32, 1)

    if state.dim() == 2:
        state = state.unsqueeze(1)
    if output.dim() == 2:
        output = output.unsqueeze(1)

    print(f"[*] Champion Slot:           #{slot_id}")
    print(f"[*] Allostatic Stress Level: {noise:.4f}")
    print(f"[*] Network Layout:          N={N} nodes | D={D} per node | Attention Rank K={K}")
    print(f"    - Nodes  0 -  3: Sensory Boundary (4 nodes)")
    print(f"    - Nodes  4 - 27: Cognitive Interneurons (24 nodes)")
    print(f"    - Nodes 28 - 31: Motor Boundary (4 nodes)")

    # =========================================================================
    # 1. COMPUTE THE 32x32 DIRECTED ATTENTION GRAPH (LinearBusRouting)
    # =========================================================================
    # In LinearBusRouting: Q = elu(raw_q) + 1, K = elu(raw_k) + 1
    # Node i reads from Node j with weight proportional to (Q_i . K_j)
    w_qk = model.router.w_qk # (N, D, 2K) = (32, 32, 32)
    raw_qk = torch.bmm(state, w_qk) # (32, 1, 32)
    raw_q, raw_k = raw_qk.squeeze(1).chunk(2, dim=-1) # (32, 16), (32, 16)
    Q = F.elu(raw_q) + 1.0 # (32, 16)
    K = F.elu(raw_k) + 1.0 # (32, 16)

    # Raw attention affinity matrix: A_raw[i, j] = Q_i . K_j
    # Row i = reader (node taking in info), Col j = writer (node sending info)
    A_raw = torch.mm(Q, K.T) # (32, 32)
    A_norm = A_raw / (A_raw.sum(dim=1, keepdim=True) + 1e-8) # Row-normalized percentages

    # =========================================================================
    # 2. NODE-BY-NODE HEALTH & COMMUNICATION PROFILE
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [1] 32-NODE ACTIVITY & COMMUNICATION SPECIALIZATION TABLE")
    print("=" * 80)
    print(f"{'Node':>4} | {'Role':<9} | {'State L2':>8} | {'Range [Min, Max]':<18} | {'ErrEMA':>6} | {'LP-EMA':>6} | {'Top Inbound (Listens To)':<24} | {'Top Outbound (Speaks To)'}")
    print("-" * 125)

    saturation_limit = math.sqrt(2.0 + math.sqrt(2.0)) # ~1.8477 buntanh bound
    pinned_count = 0

    for i in range(N):
        if i < 4:
            role = "SENSORY"
        elif i >= 28:
            role = "MOTOR"
        else:
            role = "THINKING"

        s_vec = state[i, 0]
        s_norm = torch.norm(s_vec, p=2).item()
        s_min, s_max = s_vec.min().item(), s_vec.max().item()
        e_val = error_ema[i].item() if error_ema is not None else 0.0
        lp_val = lp_ema[i].item() if lp_ema is not None else 0.0

        if abs(s_min) > saturation_limit * 0.95 or abs(s_max) > saturation_limit * 0.95:
            pinned_count += 1

        # Inbound attention: Which nodes does node i read from the most?
        in_weights = A_norm[i]
        top_in = torch.topk(in_weights, k=3).indices.tolist()
        in_str = ", ".join([f"N{src}({in_weights[src]*100:2.0f}%)" for src in top_in])

        # Outbound attention: Which nodes read from node i the most?
        out_weights = A_norm[:, i]
        top_out = torch.topk(out_weights, k=3).indices.tolist()
        out_str = ", ".join([f"N{dst}({out_weights[dst]*100:2.0f}%)" for dst in top_out])

        print(f"N{i:02d} | {role:<9} | {s_norm:8.4f} | [{s_min:6.2f}, {s_max:6.2f}] | {e_val:6.4f} | {lp_val:6.4f} | {in_str:<24} | {out_str}")

    print("-" * 125)
    print(f"[*] Nodes saturated near buntanh limit: {pinned_count} / {N}")

    # =========================================================================
    # 3. INTER-COMPARTMENT INFORMATION HIGHWAYS
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [2] MACRO INFORMATION FLOW (Inter-Compartment Density)")
    print("=" * 80)

    # Sensory: 0..3, Thinking: 4..27, Motor: 28..31
    sensory_idx = list(range(0, 4))
    thinking_idx = list(range(4, 28))
    motor_idx = list(range(28, 32))

    s_to_t = A_norm[thinking_idx][:, sensory_idx].mean().item() * 100
    t_to_m = A_norm[motor_idx][:, thinking_idx].mean().item() * 100
    s_to_m = A_norm[motor_idx][:, sensory_idx].mean().item() * 100
    t_to_t = A_norm[thinking_idx][:, thinking_idx].mean().item() * 100
    m_to_t = A_norm[thinking_idx][:, motor_idx].mean().item() * 100

    print(f"  * Sensory  --> Thinking Nodes:  {s_to_t:5.2f}% attention share")
    print(f"  * Thinking --> Thinking Nodes:  {t_to_t:5.2f}% (Internal loop)")
    print(f"  * Thinking --> Motor Nodes:     {t_to_m:5.2f}% (Cognitive policy drive)")
    print(f"  * Sensory  --> Motor Nodes:     {s_to_m:5.2f}% (Direct sensory reflex override)")
    print(f"  * Motor    --> Thinking Nodes:  {m_to_t:5.2f}% (Efference copy feedback)")

    if t_to_m < 1.0 and s_to_m < 1.0:
        print("\n  [!] CRITICAL SEVERANCE: Motor nodes are receiving < 1% attention!")
        print("      The motor boundary is completely decoupled from sensory and cognitive state.")

    # =========================================================================
    # 4. 32x32 ATTENTION ROUTING MAP (ASCII HEATMAP)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [3] 32x32 DIRECTED ATTENTION MAP (Rows = Readers, Cols = Writers)")
    print("     Legend: [ ] <1%  [.] 1-3%  [:] 3-6%  [*] 6-12%  [#] >12%")
    print("=" * 80)
    header = "     " + "".join([f"{c%10}" for c in range(32)])
    print(header)
    print("     " + "-" * 32)
    for r in range(32):
        row_chars = []
        for c in range(32):
            val = A_norm[r, c].item() * 100
            if val < 1.0:    ch = " "
            elif val < 3.0:  ch = "."
            elif val < 6.0:  ch = ":"
            elif val < 12.0: ch = "*"
            else:            ch = "#"
            row_chars.append(ch)
        prefix = f"N{r:02d}| "
        print(prefix + "".join(row_chars))

    # =========================================================================
    # 5. MOVE-1 ACTION LOGIT ANALYSIS (First Move Against Stockfish)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [4] PROBING MOVE-1 DECISION LOGITS ON STARTING POSITION")
    print("=" * 80)

    chess_env = GuardrailedTorchChess(batch_size=1, device=device)
    obs, threat = chess_env.get_obs(agent_noise=torch.tensor([0.01], device=device))
    legal_mask = chess_env.compute_legal_masks()

    # Get raw action logits from the champion's state
    with torch.no_grad():
        raw_logits = model.get_action_logits(state)[0] # (4096,)
        masked_logits = raw_logits + legal_mask[0]
        legal_probs = F.softmax(masked_logits, dim=-1)

    # 1. Legal Moves breakdown
    legal_indices = torch.nonzero(legal_mask[0] > -1e8).squeeze(-1)
    legal_scores, sorted_k = torch.sort(legal_probs[legal_indices], descending=True)
    sorted_legals = legal_indices[sorted_k]

    print(f"Total Legal Moves Available: {legal_indices.numel()} (Uniform baseline: {100.0/legal_indices.numel():.2f}%)")
    print("\nTop 5 Legal Moves Chosen by Champion:")
    for rank in range(min(5, len(sorted_legals))):
        act = sorted_legals[rank].item()
        f_sq, t_sq = act // 64, act % 64
        prob = legal_scores[rank].item() * 100
        raw_val = raw_logits[act].item()
        uci = uci_format(f_sq, t_sq)
        print(f"  #{rank+1}: Move {uci} (act {act:4d}) | Prob: {prob:5.2f}% | Raw Logit: {raw_val:7.2f}")

    # 2. Check if the model is trying to play illegal moves
    raw_sorted_scores, raw_sorted_indices = torch.sort(raw_logits, descending=True)
    print("\nTop 5 Absolute Highest Raw Logits (Checking for Illegal Move Attraction):")
    for rank in range(5):
        act = raw_sorted_indices[rank].item()
        f_sq, t_sq = act // 64, act % 64
        is_legal = legal_mask[0, act] > -1e8
        uci = uci_format(f_sq, t_sq)
        status = "[LEGAL]  " if is_legal else "[ILLEGAL]"
        print(f"  #{rank+1}: {status} Move {uci} (act {act:4d}) | Raw Logit: {raw_sorted_scores[rank].item():7.2f}")

    # 3. Check Spatial Affinity Matrix (64x64)
    model.update_cached_spatial()
    aff = model.spatial_affinity
    print("\n" + "=" * 80)
    print(" [5] SPATIAL TRANSITION AFFINITY (64x64 Matrix Health)")
    print("=" * 80)
    print(f"Norm: {torch.norm(aff).item():.4f} | Mean: {aff.mean().item():.4f} | Std: {aff.std().item():.4f}")
    print(f"Min:  {aff.min().item():.4f} | Max:  {aff.max().item():.4f}")
    if torch.allclose(aff, torch.zeros_like(aff), atol=1e-4):
        print("  [!] WARNING: Spatial affinity is 0! Joint scoring is completely broken.")
    else:
        print("  [+] Spatial affinity contains non-zero signal.")

    print("\n" + "=" * 80)
    print(" [END OF DIAGNOSTIC REPORT] Copy-paste the output above into the chat!")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose 32-Node Champion Checkpoint")
    parser.add_argument("--ckpt", type=str, default="out/champion_agent.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    unpack_and_diagnose(ckpt_path=args.ckpt, device=args.device)