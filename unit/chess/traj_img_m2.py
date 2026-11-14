import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import models.nodenet.nodechess21 as nc21

def sq_to_name(sq):
    return f"{chr(ord('a') + (sq % 8))}{(sq // 8) + 1}"

def action_to_str(act):
    return f"{sq_to_name(act // 64)}->{sq_to_name(act % 64)}"

def record_nodechess21_trajectory(model, env, agent_noise, thinking_steps=32):
    state, out, err_ema, lp_ema = model.init_states(1)
    env_in, threat = env.get_obs(agent_noise)
    legal_mask = env.compute_legal_masks()

    raw_s = model.sensory_proj(env_in)
    lat = F.softsign(raw_s.view(1, model.num_input_nodes, model.D).permute(1, 0, 2))
    bus = lat.mean(dim=0)
    latched_em = model.router.emit(state)

    c_s, c_o, c_e, c_p = state.clone(), out.clone(), err_ema.clone(), lp_ema.clone()
    latched_s = c_s.clone()
    prev_s = c_s.clone()

    s_history = [c_s.squeeze(1).cpu().numpy()]
    vel_history = []
    logit_history = []

    cooldown = 0
    settled_counter = 0
    kick_steps = []
    latch_step = 0

    init_logits = model.get_action_logits(c_s)
    top2_init = torch.topk(init_logits, k=2, dim=-1).values
    best_score = (top2_init[:, 0] + (top2_init[:, 0] - top2_init[:, 1])).item()

    args = model.args

    for step in range(thinking_steps):
        overall_vel = torch.norm(c_s - prev_s, dim=-1).mean().item()
        vel_history.append(overall_vel)

        if overall_vel < args.settled_vel_thresh and cooldown == 0:
            settled_counter += 1
        else:
            settled_counter = 0

        if settled_counter >= args.settled_patience:
            p_logits = model.get_action_logits(c_s)
            top2_p = torch.topk(p_logits, k=2, dim=-1).values
            p_score = (top2_p[:, 0] + (top2_p[:, 0] - top2_p[:, 1])).item()
            if p_score > best_score:
                latched_s = c_s.clone()
                best_score = p_score
                latch_step = step + 1

            c_s = F.softsign(c_s + torch.randn_like(c_s) * args.dirac_pulse_sigma)
            cooldown = args.cooldown_steps
            settled_counter = 0
            kick_steps.append(step + 1)

        noise_gate = 0.0 if (cooldown > 0 or overall_vel > args.coasting_vel_thresh) else 1.0
        cooldown = max(0, cooldown - 1)
        prev_s = c_s.clone()

        n_hidden = model.N - model.num_input_nodes - model.motor_nodes_count
        h_mask = (torch.rand(n_hidden, 1, 1, device=c_s.device) < args.poisson_tick_prob).float()
        tick_mask = torch.cat([
            torch.ones(model.num_input_nodes, 1, 1, device=c_s.device),
            h_mask,
            torch.ones(model.motor_nodes_count, 1, 1, device=c_s.device)
        ], dim=0)

        sl, c_s, c_o, c_e, c_p, bus, latched_em = model._step(
            env_in, threat, c_s, c_o, c_e, c_p, bus,
            latched_emissions=latched_em, precomputed=lat,
            noise_gate=noise_gate, tick_mask=tick_mask, v_latch=args.v_latch
        )

        logits = model.get_action_logits(c_s) + legal_mask
        logit_history.append(logits.squeeze(0).cpu().numpy())
        s_history.append(c_s.squeeze(1).cpu().numpy())

    final_motor_vel = torch.norm(c_s[-model.motor_nodes_count:] - prev_s[-model.motor_nodes_count:], dim=-1).mean().item()
    if final_motor_vel < args.v_latch and cooldown == 0:
        f_logits = model.get_action_logits(c_s)
        top2_f = torch.topk(f_logits, k=2, dim=-1).values
        f_score = (top2_f[:, 0] + (top2_f[:, 0] - top2_f[:, 1])).item()
        if f_score > best_score:
            latched_s = c_s.clone()
            latch_step = thinking_steps

    latched_logits = (model.get_action_logits(latched_s) + legal_mask).squeeze(0).cpu().numpy()

    return {
        'states': np.array(s_history),
        'velocities': np.array(vel_history),
        'logits': np.array(logit_history),
        'latched_logits': latched_logits,
        'legal_mask': legal_mask.squeeze(0).cpu().numpy(),
        'kick_steps': kick_steps,
        'latch_step': latch_step
    }

def capture_champion_game(model, target_plies=[1, 15, 35], thinking_steps=32, device="cuda"):
    env = nc21.GuardrailedTorchChess(1, device=device)
    state, out, err_ema, lp_ema = model.init_states(1)
    agent_noise = torch.tensor([0.005], device=device)
    dones = torch.zeros(1, dtype=torch.bool, device=device)
    winners = torch.zeros(1, dtype=torch.int8, device=device)
    ply = 0

    captured_data = {}
    max_target = max(target_plies)

    print(f"\n[Rollout] Capturing plies: {target_plies} using {thinking_steps} thinking steps...")
    while not dones.all() and ply <= max_target:
        current_ply = ply + 1
        is_agent_turn = (env.players[0] == 0)
        l_masks = env.compute_legal_masks()

        if current_ply in target_plies and is_agent_turn:
            wm, bm = env.get_material()
            fen = nc21.board_to_fen(
                env.boards[0], int(env.players[0].item()), env.castling_rights[0].tolist(),
                int(env.en_passant_sq[0].item()), int(env.halfmove_clock[0].item()), current_ply // 2 + 1
            )
            print(f"  -> Capturing Ply {current_ply:02d} | White: {wm[0].item():.0f} cp | Black: {bm[0].item():.0f} cp")
            traj = record_nodechess21_trajectory(model, env, agent_noise, thinking_steps=thinking_steps)
            captured_data[current_ply] = {'traj': traj, 'fen': fen, 'mat': (wm[0].item(), bm[0].item())}

        actions = torch.zeros(1, dtype=torch.long, device=device)
        if is_agent_turn:
            e_in, th = env.get_obs(agent_noise)
            _, ns, no, ne, nl, _ = model(e_in, th, state, out, err_ema, lp_ema, thinking_steps=thinking_steps)
            state, out, err_ema, lp_ema = ns, no, ne, nl
            actions[0] = torch.argmax(model.get_action_logits(state) + l_masks, dim=-1)
        else:
            actions[0] = torch.nonzero(l_masks[0] > -1e8)[0][0]

        dones, winners = env.step(actions, dones, winners, legal_mask=l_masks)
        ply += 1

    return captured_data

def plot_dashboard(captured_data, output_path="champion21_deliberation.png"):
    phases = list(captured_data.keys())
    n_phases = len(phases)
    if n_phases == 0:
        print("No phases captured.")
        return

    fig, axes = plt.subplots(3, n_phases, figsize=(6.5 * n_phases, 14))
    fig.suptitle("NodeChess 21 Deliberation Dynamics [Harmonic Geometric Heads & Asynch Swarm]", fontsize=17, fontweight='bold')

    for col, ply in enumerate(phases):
        data = captured_data[ply]
        traj = data['traj']
        states = traj['states']
        logits = traj['logits']
        vels = traj['velocities']
        kicks = traj.get('kick_steps', [])
        l_step = traj.get('latch_step', 0)
        steps = len(vels)
        num_nodes = states.shape[1]

        phase_title = f"Ply {ply} (" + ("Opening" if ply <= 4 else ("Middlegame" if ply <= 24 else "Late Game")) + ")"
        wm, bm = data['mat']

        # ROW 1: HEATMAP
        ax1 = axes[0, col] if n_phases > 1 else axes[0]
        energies = np.linalg.norm(states[1:], axis=-1).T
        im1 = ax1.imshow(energies, aspect='auto', cmap='magma', origin='lower')
        ax1.set_title(f"{phase_title}\nNode Activity (Mat: W {wm:.0f} / B {bm:.0f})")
        ax1.set_ylabel("Node Index (0-1: Sens, 28-31: Motor)")
        ax1.set_xlabel("Thinking Step")
        ax1.axhline(1.5, color='cyan', linestyle='--', linewidth=1.2)
        ax1.axhline(num_nodes - 4.5, color='lime', linestyle='--', linewidth=1.2)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # ROW 2: VELOCITY
        ax2 = axes[1, col] if n_phases > 1 else axes[1]
        ax2.plot(range(1, steps + 1), vels, color='tab:blue', marker='o', markersize=3, linewidth=2)
        for k in kicks: ax2.axvline(k, color='red', linestyle=':', alpha=0.6)
        if l_step > 0:
            ax2.axvline(l_step, color='lime', linestyle='--', alpha=0.8, label=f"Latched @ t={l_step}")
            ax2.legend(loc="upper right", fontsize=8)
        ax2.set_title(f"Settling Velocity (Final: {vels[-1]:.3f})")
        ax2.set_xlabel("Thinking Step")
        ax2.set_ylabel("||S_t - S_{t-1}||")
        ax2.set_ylim(0, max(0.25, max(vels) * 1.1))
        ax2.grid(True, linestyle=':', alpha=0.6)

        # ROW 3: LOGIT COMPETITION
        ax3 = axes[2, col] if n_phases > 1 else axes[2]
        legal_indices = np.where(traj['legal_mask'] > -1e8)[0]
        if len(legal_indices) > 0:
            max_logits = logits[:, legal_indices].max(axis=0)
            top_moves = legal_indices[np.argsort(max_logits)[-5:]]
            latched_l = traj.get('latched_logits', logits[-1])
            final_winner = legal_indices[np.argmax(latched_l[legal_indices])]

            for mv in top_moves:
                is_win = (mv == final_winner)
                lbl = f"{action_to_str(mv)}" + (" ★" if is_win else "")
                ax3.plot(range(1, steps + 1), logits[:, mv], linewidth=2.5 if is_win else 1.5, label=lbl)

            for k in kicks: ax3.axvline(k, color='red', linestyle=':', alpha=0.4)
            if l_step > 0: ax3.axvline(l_step, color='lime', linestyle='--', alpha=0.5)

            ax3.set_title(f"Deliberation (Latched Move: {action_to_str(final_winner)})")
            ax3.set_xlabel("Thinking Step")
            ax3.set_ylabel("Action Logit")
            ax3.legend(loc="best", fontsize=8)
            ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"\n[Saved Dashboard] -> {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NodeChess 21 Champion Trajectory Plotter")
    parser.add_argument("--ckpt", type=str, default="out/champion_agent.pt", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="champion21_deliberation.png", help="Output PNG file")
    parser.add_argument("--thinking_steps", type=int, default=32)
    parser.add_argument("--plies", type=int, nargs="+", default=[1, 15, 35])
    parsed_vis = parser.parse_args()

    args = nc21.parse_nodechess_args()
    args.thinking_steps = parsed_vis.thinking_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nc21.ChessNodeNet(args, device=device).to(device)

    if os.path.exists(parsed_vis.ckpt):
        print(f"[Loader] Restoring weights from {parsed_vis.ckpt}...")
        ckpt = torch.load(parsed_vis.ckpt, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd, strict=False)
    else:
        print(f"[Warning] Checkpoint '{parsed_vis.ckpt}' not found. Using initialized harmonic model.")

    model.eval()
    captured = capture_champion_game(model, target_plies=parsed_vis.plies, thinking_steps=parsed_vis.thinking_steps, device=device)
    plot_dashboard(captured, output_path=parsed_vis.output)