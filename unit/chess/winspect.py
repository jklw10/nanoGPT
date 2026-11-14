import argparse
import json
import os
import re
import sys
import time
import torch
import numpy as np

torch.set_printoptions(
    threshold=sys.maxsize,
    linewidth=1000000,
    profile="full",
    precision=6,
    sci_mode=False
)
np.set_printoptions(threshold=sys.maxsize, linewidth=1000000)


def format_num(n: float) -> str:
    if abs(n) >= 1e9: return f"{n / 1e9:.2f}B"
    if abs(n) >= 1e6: return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3: return f"{n / 1e3:.2f}K"
    return f"{n:.2f}"


def round_floats(obj, precision=4):
    """Recursively round floats to limit token count while preserving fidelity."""
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, list):
        return [round_floats(x, precision) for x in obj]
    elif isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    return obj


def print_matrix_grid(t: torch.Tensor, precision=3):
    """Prints 2D matrices formatted cleanly with aligned columns."""
    if t.dim() != 2:
        print(t)
        return
    rows, cols = t.shape
    fmt = f"{{:> {precision + 5}.{precision}f}}"
    lines = []
    for r in range(rows):
        row_str = " ".join([fmt.format(t[r, c].item()) for c in range(cols)])
        lines.append(f"  [{r:02d}] {row_str}")
    print("\n".join(lines))


def inspect_pt(path: str, section: str = None, node_idx: int = None, 
               print_values: bool = False, full: bool = False, 
               precision: int = 4, export_dir: str = None, save_file: str = None):
    
    if not os.path.exists(path):
        print(f"Error: File not found at '{path}'")
        sys.exit(1)

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path)))

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint file: {e}")
        return

    if not isinstance(ckpt, dict):
        print(f"Checkpoint root object is of type '{type(ckpt)}' (expected dict).")
        return

    sd = ckpt.get('model_state_dict', None)
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()} if sd else {}

    # =========================================================================
    # A. TARGETED SURGICAL EXTRACTION (--section is specified)
    # =========================================================================
    if section:
        section_lower = section.lower()
        target_tensor = None
        target_name = None

        # 1. Match against known top-level buffers/meta
        if section_lower in ("noise", "agent_noise"):
            target_tensor = ckpt.get('agent_noise')
            target_name = "agent_noise"
        elif section_lower in ("error", "err_ema", "agent_error"):
            target_tensor = ckpt.get('agent_error')
            target_name = "agent_error"
        elif section_lower in ("lp", "lp_ema", "agent_lp"):
            target_tensor = ckpt.get('agent_lp')
            target_name = "agent_lp"
        elif section_lower in ("champion", "champ"):
            champ = ckpt.get('champion', {})
            print(f"\n--- CHAMPION SNAPSHOT (Slot #{champ.get('slot_id', 'N/A')}) ---")
            print(f"Score : {champ.get('score')}")
            print(f"Loss  : {champ.get('loss')}")
            print(f"Noise : {float(champ.get('noise', torch.tensor(0.0)).squeeze()):.5f}")
            if champ.get('state') is not None:
                print(f"State shape: {list(champ['state'].shape)}")
            return
        else:
            # 2. Match against state_dict keys (e.g., 'sensory', 'w1', 'router')
            for k, v in clean_sd.items():
                if section_lower in k.lower():
                    target_tensor = v
                    target_name = k
                    break

        if target_tensor is None:
            print(f"Error: Section '{section}' not found. Available keys: {list(clean_sd.keys())} + ['noise', 'error', 'lp', 'champ']")
            return

        # Slicing specific node if requested
        header = f"{target_name} | Shape: {list(target_tensor.shape)}"
        if node_idx is not None:
            if target_tensor.dim() >= 2 and target_tensor.shape[0] > node_idx:
                target_tensor = target_tensor[node_idx]
                header = f"{target_name}[Node {node_idx}] | Sliced Shape: {list(target_tensor.shape)}"
            else:
                print(f"Warning: Cannot index node {node_idx} on tensor with shape {list(target_tensor.shape)}")

        # Tensor statistics
        t_float = target_tensor.float()
        print("\n" + "=" * 80)
        print(f"  SURGICAL EXTRACTION: {header}")
        print(f"  Elements: {target_tensor.numel():,} | Mean: {t_float.mean():.4f} | Std: {t_float.std():.4f} | Min: {t_float.min():.4f} | Max: {t_float.max():.4f}")
        print("=" * 80 + "\n")

        # Save to specific file if requested
        if save_file:
            payload = round_floats(target_tensor.tolist(), precision=precision)
            with open(save_file, "w") as f:
                json.dump(payload, f, separators=(',', ':') if not full else None, indent=2 if full else None)
            print(f"  Saved extracted data to: '{save_file}'")
            return

        # Output to terminal
        if print_values or full:
            if target_tensor.dim() == 2 and not full:
                print_matrix_grid(t_float, precision=precision)
            else:
                payload = round_floats(target_tensor.tolist(), precision=precision)
                if full:
                    print(json.dumps(payload, indent=2))
                else:
                    print(json.dumps(payload))
        return

    # =========================================================================
    # B. FULL CHECKPOINT OVERVIEW (Default if no --section specified)
    # =========================================================================
    print("=" * 80)
    print(f"  CHECKPOINT INSPECTOR: {os.path.basename(path)}")
    print(f"  Path: {os.path.abspath(path)}")
    print(f"  File Size: {file_size_mb:.2f} MB | Last Modified: {mtime}")
    print("=" * 80)

    print("\n[1] METADATA & TRAINING STATUS")
    print("-" * 80)
    print(f"  Batch Index     : {ckpt.get('batch_idx', 'Not recorded')}")
    print(f"  Completed Games : {ckpt.get('completed_games', 0):,}")
    print(f"  Top-level Keys  : {list(ckpt.keys())}")

    print("\n[2] AGENT POPULATION ALLOSTASIS (Stress/Noise)")
    print("-" * 80)
    if 'agent_noise' in ckpt and ckpt['agent_noise'] is not None:
        noise = ckpt['agent_noise'].float()
        print(f"  Population Size : {noise.numel()} agents")
        print(f"  Somatic Noise   : Mean: {noise.mean():.4f} | Min: {noise.min():.4f} | Max: {noise.max():.4f} | Std: {noise.std():.4f}")
    
    if 'champion' in ckpt and ckpt['champion'] is not None:
        champ = ckpt['champion']
        print(f"  Champion Slot   : #{champ.get('slot_id', 'N/A')} | Tourney Score: {champ.get('score', 'N/A')} | Noise: {float(champ.get('noise', torch.tensor(0.0)).squeeze()):.5f}")

    print("\n[3] MODEL ARCHITECTURE & TENSORS")
    print("-" * 80)
    print(f"  {'Parameter / Buffer Name':<42} | {'Shape':<16} | {'Params':<10} | {'Mean ± Std':<16}")
    print("  " + "-" * 76)

    total_params = 0
    for name, tensor in clean_sd.items():
        t_float = tensor.float()
        cnt = tensor.numel()
        if tensor.is_floating_point(): total_params += cnt
        shape_str = str(list(tensor.shape))
        mean_std = f"{t_float.mean():.3f} ± {t_float.std():.3f}" if cnt > 1 else f"{t_float.item():.3f}"
        print(f"  {name:<42} | {shape_str:<16} | {format_num(cnt):<10} | {mean_std:<16}")

    print("  " + "-" * 76)
    print(f"  Total Floating-Point Parameters: {total_params:,}")

    # Export all layers to directory if requested
    if export_dir and clean_sd:
        os.makedirs(export_dir, exist_ok=True)
        print(f"\n[4] EXPORTING MATRICES TO DIRECTORY: '{export_dir}'")
        print("-" * 80)
        for name, tensor in clean_sd.items():
            clean_file_name = name.replace(".", "_") + ".json"
            out_path = os.path.join(export_dir, clean_file_name)
            payload = round_floats(tensor.tolist(), precision=precision)
            with open(out_path, "w") as f:
                json.dump(payload, f)
            print(f"  Saved: {out_path} (Shape: {list(tensor.shape)})")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Targeted Checkpoint Inspector & Surgical Weight Extractor")
    parser.add_argument("checkpoint_path", type=str, help="Path to the .pt file")
    
    # Surgical targeting flags
    parser.add_argument("-s", "--section", type=str, default=None, 
                        help="Target specific tensor: 'sensory', 'w1', 'router', 'noise', 'champ'")
    parser.add_argument("-n", "--node", type=int, default=None, 
                        help="Slice specific node index (0-31) for 3D tensors (e.g. w1, router)")
    parser.add_argument("-p", "--precision", type=int, default=4, 
                        help="Decimal precision for exported floats (default: 4 to preserve tokens)")
    
    # Display & Export flags
    parser.add_argument("--print_values", action="store_true", help="Print the targeted slice to stdout")
    parser.add_argument("--full", action="store_true", help="Pretty print with indentation (JSON format)")
    parser.add_argument("--save_file", type=str, default=None, help="Save the extracted slice to this specific file")
    parser.add_argument("--export_dir", type=str, default=None, help="Export all layers into this folder as .json")

    args = parser.parse_args()
    inspect_pt(
        args.checkpoint_path, 
        section=args.section, 
        node_idx=args.node, 
        print_values=args.print_values, 
        full=args.full, 
        precision=args.precision, 
        export_dir=args.export_dir,
        save_file=args.save_file
    )