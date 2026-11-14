import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Adjust imports to match your main filename (e.g. node_chess or nodechess17)

from models.nodenet.nodechess18 import ChessNodeNet, GuardrailedTorchChess, select_cuda_engine_move, board_to_fen


def sq_to_name(sq):
    return f"{chr(ord('a') + (sq % 8))}{(sq // 8) + 1}"


def action_to_str(act):
    return f"{sq_to_name(act // 64)}->{sq_to_name(act % 64)}"


def record_trajectory_for_position(model, env, agent_noise, thinking_steps=32):
    raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
    state, out, err_ema, lp_ema = raw_m.init_states(1)
    
    env_in, threat = env.get_obs(agent_noise)
    legal_mask = env.compute_legal_masks()

    with torch.no_grad():
        raw_s = raw_m.sensory_proj(env_in)
        lat = F.softsign(raw_s.view(1, raw_m.num_input_nodes, raw_m.D).permute(1, 0, 2))

    c_s, c_o, c_e, c_p = state.clone(), out.clone(), err_ema.clone(), lp_ema.clone()
    
    s_history = [c_s.squeeze(1).cpu().numpy()]
    vel_history = []
    logit_history = []
    
    prev_s = c_s.clone()
    for step in range(thinking_steps):
        sl, c_s, c_o, c_e, c_p, sv = raw_m._step(env_in, threat, c_s, c_o, c_e, c_p, precomputed=lat)
        
        vel = torch.norm(c_s - prev_s, dim=-1).mean().item()
        vel_history.append(vel)
        prev_s = c_s.clone()

        logits = raw_m.get_action_logits(c_s) + legal_mask
        logit_history.append(logits.squeeze(0).cpu().numpy())
        s_history.append(c_s.squeeze(1).cpu().numpy())

    return {
        'states': np.array(s_history),           # [T+1, N, D]
        'velocities': np.array(vel_history),     # [T]
        'logits': np.array(logit_history),       # [T, 4096]
        'legal_mask': legal_mask.squeeze(0).cpu().numpy()
    }


def play_and_capture_phases(model, device, target_plies=[1, 15, 35], thinking_steps=32):
    env = GuardrailedTorchChess(1, device=device)
    raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    state, out, err_ema, lp_ema = raw_m.init_states(1)
    agent_noise = torch.tensor([0.005], device=device)
    
    dones = torch.zeros(1, dtype=torch.bool, device=device)
    winners = torch.zeros(1, dtype=torch.int8, device=device)
    ply = 0
    
    captured_data = {}
    max_target = max(target_plies)

    print(f"\n[Rollout] Playing game to capture plies: {target_plies}...")

    while not dones.all() and ply <= max_target:
        current_ply = ply + 1
        is_agent_turn = (env.players[0] == 0)  # Agent plays White, Bot plays Black
        l_masks = env.compute_legal_masks()

        # Capture trajectory if current ply matches target
        if current_ply in target_plies and is_agent_turn:
            wm, bm = env.get_material()
            fen = board_to_fen(env.boards[0], int(env.players[0].item()), env.castling_rights[0].tolist(),
                               int(env.en_passant_sq[0].item()), int(env.halfmove_clock[0].item()), current_ply // 2 + 1)
            print(f"  -> Capturing Ply {current_ply:02d} | White Mat: {wm[0].item():.0f} | Black Mat: {bm[0].item():.0f}")
            print(f"     FEN: {fen}")

            traj = record_trajectory_for_position(model, env, agent_noise, thinking_steps=thinking_steps)
            captured_data[current_ply] = {
                'traj': traj,
                'fen': fen,
                'mat': (wm[0].item(), bm[0].item())
            }

        # Step game forward
        actions = torch.zeros(1, dtype=torch.long, device=device)
        if is_agent_turn:
            e_in, th = env.get_obs(agent_noise)
            _, ns, no, ne, nl, _, _ = model(e_in, th, state, out, err_ema, lp_ema, thinking_steps=8)
            state, out, err_ema, lp_ema = ns, no, ne, nl
            actions[0] = torch.argmax(model.get_action_logits(state) + l_masks, dim=-1)
        else:
            actions[0] = select_cuda_engine_move(
                env.boards, l_masks, torch.tensor([False], device=device), device, env=env, b_idx=torch.tensor([0], device=device)
            )

        dones, winners = env.step(actions, dones, winners, legal_mask=l_masks)
        ply += 1

    return captured_data


def plot_comparative_dashboard(captured_data, output_path="multi_move_trajectory.png"):
    phases = list(captured_data.keys())
    n_phases = len(phases)
    if n_phases == 0:
        print("No phases captured.")
        return

    fig, axes = plt.subplots(3, n_phases, figsize=(6 * n_phases, 14))
    fig.suptitle("Deliberation Dynamics Across Different Game Phases", fontsize=18, fontweight='bold')

    for col, ply in enumerate(phases):
        data = captured_data[ply]
        traj = data['traj']
        states = traj['states']      # [T+1, N, D]
        logits = traj['logits']      # [T, 4096]
        vels = traj['velocities']
        steps = len(vels)
        num_nodes = states.shape[1]

        phase_title = f"Ply {ply} (" + ("Opening" if ply <= 4 else ("Middlegame" if ply <= 24 else "Late Game")) + ")"
        wm, bm = data['mat']

        # -------------------------------------------------------------
        # ROW 1: NODE ACTIVITY ENERGY HEATMAP
        # -------------------------------------------------------------
        ax1 = axes[0, col] if n_phases > 1 else axes[0]
        energies = np.linalg.norm(states[1:], axis=-1).T  # [N, T]
        im1 = ax1.imshow(energies, aspect='auto', cmap='magma', origin='lower')
        ax1.set_title(f"{phase_title}\nNode Activity (Mat: W {wm:.0f} / B {bm:.0f})")
        ax1.set_ylabel("Node Index (0-1: Sensory, 28-31: Motor)")
        ax1.set_xlabel("Thinking Step")
        ax1.axhline(1.5, color='cyan', linestyle='--', linewidth=1)
        ax1.axhline(num_nodes - 4.5, color='lime', linestyle='--', linewidth=1)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Print active node summary to stdout
        mean_node_e = energies.mean(axis=1)
        active_nodes = np.where(mean_node_e > 0.15)[0].tolist()
        print(f"\n[Ply {ply:02d} Node Audit] Nodes Active (>0.15 norm): {active_nodes}")
        if 7 in active_nodes: print(f"  ==> Node 7 IS ACTIVE in Ply {ply}! (Norm: {mean_node_e[7]:.3f})")
        else: print(f"  --> Node 7 is silent in Ply {ply} (Norm: {mean_node_e[7]:.3f})")
        if 31 in active_nodes: print(f"  ==> Node 31 IS ACTIVE in Ply {ply}! (Norm: {mean_node_e[31]:.3f})")

        # -------------------------------------------------------------
        # ROW 2: ATTRACTOR VELOCITY (Settling Curve)
        # -------------------------------------------------------------
        ax2 = axes[1, col] if n_phases > 1 else axes[1]
        ax2.plot(range(1, steps + 1), vels, color='tab:blue', marker='o', markersize=3, linewidth=2)
        ax2.set_title(f"Settling Velocity (Final: {vels[-1]:.3f})")
        ax2.set_xlabel("Thinking Step")
        ax2.set_ylabel("||S_t - S_{t-1}||")
        ax2.set_ylim(0, 0.25)
        ax2.grid(True, linestyle=':', alpha=0.6)

        # -------------------------------------------------------------
        # ROW 3: TOP 5 CANDIDATE MOVE LOGITS
        # -------------------------------------------------------------
        ax3 = axes[2, col] if n_phases > 1 else axes[2]
        legal_indices = np.where(traj['legal_mask'] > -1e8)[0]
        if len(legal_indices) > 0:
            top_sub = np.argsort(logits[:, legal_indices].max(axis=0))[-5:]
            top_moves = legal_indices[top_sub]

            for mv in top_moves:
                ax3.plot(range(1, steps + 1), logits[:, mv], linewidth=2, label=action_to_str(mv))

            ax3.set_title(f"Deliberation (Winner: {action_to_str(top_moves[-1])})")
            ax3.set_xlabel("Thinking Step")
            ax3.set_ylabel("Action Logit")
            ax3.legend(loc="best", fontsize=8)
            ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"\n[Saved Multi-Position Dashboard] -> {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-Position Deliberation Trajectory Visualizer")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--plies", type=int, nargs="+", default=[1, 15, 35], help="Plies to capture (e.g. 1 15 35)")
    parser.add_argument("--thinking_steps", type=int, default=32)
    parser.add_argument("--num_nodes", type=int, default=32)
    parser.add_argument("--output", type=str, default="multi_move_trajectory.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNodeNet(num_nodes=args.num_nodes, dim=32, input_dim=64, rank_k=16, device=device).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', {})
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()

    captured = play_and_capture_phases(model, device, target_plies=args.plies, thinking_steps=args.thinking_steps)
    plot_comparative_dashboard(captured, output_path=args.output)


if __name__ == "__main__":
    main()