"""
analyze_extreme_inits.py
Performs mathematical post-mortem on saved extreme seed initializations.
Computes spectral radii, spatial affinity breakdown, motor-sensory alignment,
and Quasizero distortion.
"""

import glob
import os
import torch
import torch.nn.functional as F

def analyze_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    seed = ckpt["seed"]
    mode = ckpt["failure_mode"]
    worst_mat = ckpt["worst_mat"]
    opening_move = ckpt["opening_move"]
    
    sq_enc = ckpt["normalized_square_enc"]         # (64, 64)
    affinity = ckpt["spatial_affinity"]            # (64, 64)
    w1 = ckpt["w1_weight"]                         # (32 nodes, 32, 64)
    w_rx = ckpt["w_rx"]                           # (32 nodes, 32, 32)
    w_tx = ckpt["w_tx"]                           # (32 nodes, 32, 32)
    
    # 1. Spatial Affinity Diagnostics
    diag_energy = torch.diag(affinity).abs().mean().item()
    off_diag_mask = ~torch.eye(64, dtype=torch.bool)
    off_diag_energy = affinity[off_diag_mask].abs().mean().item()
    spatial_cond = torch.linalg.cond(sq_enc).item()
    
    # 2. W1 Quasizero & Spectral Health
    # Block 0: Mean prediction (cols 0:32) | Block 1: Error prediction (cols 32:64)
    w1_mean_block = w1[:, :, :32]
    w1_err_block = w1[:, :, 32:]
    
    # Check deviation of block 0 from scaled identity (Quasizero preservation)
    target_id = torch.eye(32).unsqueeze(0).expand(32, -1, -1) * 0.8
    qz_preservation = (w1_mean_block - target_id).abs().mean().item()
    
    # Check noise inflation in block 1
    err_col_norms = torch.norm(w1_err_block, p=2, dim=1).mean().item()
    
    # Singular value spread (collapse check across all 32 nodes)
    svs = torch.linalg.svdvals(w1)  # (32, 32)
    sv_ratio = (svs[:, 0] / (svs[:, -1] + 1e-8)).mean().item()
    
    # 3. Bus Receptive Field Diversity
    # Measure cosine similarity between node 0 and node 31 w_tx
    tx_flat = w_tx.view(32, -1)
    tx_diversity = 1.0 - F.cosine_similarity(tx_flat[0:1], tx_flat[-1:]).item()
    
    # Loop Gain: Spectral radius of transmitter -> receiver cycle
    tx_rx_cycle = torch.bmm(w_tx, w_rx.transpose(-1, -2))
    cycle_svs = torch.linalg.svdvals(tx_rx_cycle)
    mean_cycle_gain = cycle_svs[:, 0].mean().item()
    
    return {
        "seed": seed, "mode": mode, "worst_mat": worst_mat, "move": opening_move,
        "diag_affinity": diag_energy, "off_diag_affinity": off_diag_energy,
        "spatial_cond": spatial_cond, "qz_error": qz_preservation,
        "err_block_norm": err_col_norms, "sv_cond": sv_ratio,
        "tx_diversity": tx_diversity, "cycle_gain": mean_cycle_gain
    }

def main():
    files = sorted(glob.glob("out/diagnostics/inits/*.pt"))
    if not files:
        print("No checkpoint files found in out/diagnostics/inits/")
        return

    print("=" * 115)
    print(f" MATHEMATICAL AUTOPSY OF EXTREME SEED INITIALIZATIONS ({len(files)} Artifacts)")
    print("=" * 115)
    print(f"{'Type/Seed':<18} | {'DiagAff':<8} | {'OffDiag':<8} | {'QZ-Err':<8} | {'ErrNorm':<8} | {'SV-Cond':<8} | {'BusDiv':<8} | {'WorstMat':<9} | {'Move'}")
    print("-" * 115)

    for f in files:
        fname = os.path.basename(f)
        tag = fname.split("_seed_")[0]
        stats = analyze_checkpoint(f)
        
        print(f"{tag:<18} | {stats['diag_affinity']:<8.4f} | {stats['off_diag_affinity']:<8.5f} | "
              f"{stats['qz_error']:<8.4f} | {stats['err_block_norm']:<8.4f} | {stats['sv_cond']:<8.2f} | "
              f"{stats['tx_diversity']:<8.5f} | {stats['worst_mat']:<+9d} | {stats['move']}")

    print("=" * 115)
    print("\nINTERPRETATION GUIDE:")
    print(" 1. OffDiag Affinity near 0.0000  => Spatial affinity matrix is degenerate (Identity matrix). No distance awareness.")
    print(" 2. ErrNorm near 0.8000           => Normalization blew up the random error block into full-scale noise.")
    print(" 3. BusDiv near 0.0000            => Nodes 0..31 are identical clones with 0 specialization.")
    print(" 4. SV-Cond >> 10.0               => Hidden node internal dynamics suffer severe ill-conditioning.")
    print("=" * 115)

if __name__ == "__main__":
    main()