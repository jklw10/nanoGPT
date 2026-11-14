"""
trace_deliberation.py
Traces the full 32-step dynamical trajectory from the legal chess opening.
Compares Golden (Isl 15: quasizero + prekick) vs Rotten (Isl 00: baseline + viscous).
"""

import math
import os
import torch
import torch.nn.functional as F

diag_dir = "out/diagnostics"
pt_files = [os.path.join(diag_dir, f) for f in os.listdir(diag_dir) if f.startswith("tensors_") and f.endswith(".pt")]
latest_pt = max(pt_files, key=os.path.getmtime)
payload = torch.load(latest_pt, map_location="cuda")
sd = payload["model_state_dict"]

from models.nodenet.nodediagnostic import BatchedIslandNet, sq_to_name, action_to_str
from models.nodenet.nodechess20 import GuardrailedTorchChess

env = GuardrailedTorchChess(1, device="cuda")
l_mask = env.compute_legal_masks()
legal_acts = torch.nonzero(l_mask[0] > -1e8).squeeze(-1).tolist()
max_ent = math.log(len(legal_acts))

# Instantiate model and restore saved diagnostic weights
model = BatchedIslandNet(
    num_islands=16, sub_batch_size=1,
    island_modes=payload["assigned_modes"],
    init_schemes=payload["assigned_inits"],
    seeds=payload["seeds"],
    device="cuda"
)
model.load_state_dict(sd)

GOLDEN, ROTTEN = 15, 0

def trace_island(isl_idx):
    mode = payload["assigned_modes"][isl_idx]
    init = payload["assigned_inits"][isl_idx]
    print("\n" + "=" * 90)
    print(f" DYNAMICAL TRACE: Island {isl_idx:02d} ({init} + {mode}) | Starting Chess Position")
    print("=" * 90)
    print(f"{'Step':<4} | {'Motor Vel':<9} | {'Swarm Vel':<9} | {'Latch Status':<14} | {'Legal Ent':<9} | {'Top Move (Prob)'}")
    print("-" * 90)

    # Initial sensory observation of standard board
    e_in, th = env.get_obs(torch.zeros(1, device="cuda"))
    raw_s = torch.matmul(e_in, model.sensory_proj_weight[isl_idx])
    s_lat = F.softsign(raw_s.view(1, model.num_input_nodes, model.D).permute(1, 0, 2))
    bus = s_lat.mean(dim=0)

    c_s = 1e-4 * torch.randn(model.N, 1, model.D, device="cuda")
    c_o = 1e-4 * torch.randn(model.N, 1, model.D, device="cuda")
    c_e = torch.full((model.N, 1), 0.05, device="cuda")
    c_p = torch.zeros((model.N, 1), device="cuda")
    latched_s = c_s.clone()
    prev_s = c_s.clone()

    w_rx = model.w_rx[isl_idx]
    w_tx = model.w_tx[isl_idx]
    w1_w = model.w1_weight[isl_idx]

    cooldown = 0
    settled_counter = 0

    for step in range(1, 33):
        # 1. Physical Motor Latch Check
        motor_vel = torch.norm(c_s[-model.motor_nodes_count:] - prev_s[-model.motor_nodes_count:], dim=-1).mean().item()
        latch_event = "Coasting"

        if mode == "viscous" and motor_vel < 0.025:
            latched_s = c_s.clone()
            latch_event = "CLUTCH ENGAGED"

        # 2. Settling & Sawtooth Prekick Check
        overall_vel = torch.norm(c_s - prev_s, dim=-1).mean().item()
        is_settled = (overall_vel < 0.018) and (cooldown == 0)
        settled_counter = (settled_counter + 1) if is_settled else 0

        trigger_pulse = (settled_counter >= 2)
        if mode == "prekick" and trigger_pulse:
            latched_s = c_s.clone()
            latch_event = "LATCH PRE-KICK"

        if trigger_pulse:
            c_s = F.softsign(c_s + torch.randn_like(c_s) * 0.35)
            cooldown = 8
            settled_counter = 0
            latch_event += " + PULSE"

        is_coasting = (cooldown > 0) or (overall_vel > 0.025)
        noise_gate = 0.0 if is_coasting else 1.0
        cooldown = max(0, cooldown - 1)
        prev_s = c_s.clone()

        # 3. Read -> Deliberate -> Write
        b_exp = bus.unsqueeze(0).expand(model.N, -1, -1)
        perceived = torch.tanh(torch.bmm(b_exp, w_rx) * 0.5) * math.sqrt(2.0 + math.sqrt(2.0))

        raw_boredom = math.pow(1.0 - min(max(F.relu(c_p).mean().item() / (c_e.mean().item() + 1e-6), 0.0), 1.0), 1.0)
        boredom = raw_boredom * noise_gate
        tgt = perceived + math.sqrt(0.05) * torch.matmul(torch.randn_like(c_o), model.circ_1_0) * boredom - (0.02) * c_s * (1.0 - boredom)
        tgt = torch.cat([s_lat, F.softsign(tgt)[model.num_input_nodes:]], dim=0)

        w1_in = torch.bmm(c_s, w1_w)
        mean, err_p = torch.chunk(torch.tanh(w1_in * 0.5) * math.sqrt(2.0 + math.sqrt(2.0)), 2, dim=-1)
        cur_err = F.mse_loss(mean, tgt, reduction='none').mean(dim=-1)
        c_e = c_e * (1.0 - model.lp_alpha) + cur_err * model.lp_alpha
        c_p = c_p * (1.0 - model.lp_alpha) + torch.log((c_e + 1e-6) / (cur_err + 1e-6)) * model.lp_alpha

        settled = F.softsign(c_s + 0.25 * (tgt - c_s))
        c_s = torch.cat([s_lat, settled[model.num_input_nodes:]], dim=0)

        raw_em = torch.tanh(torch.bmm(c_s, w_tx) * 0.5)
        bus = F.softsign(0.65 * bus + 0.35 * (0.60 * s_lat.mean(dim=0) + 0.40 * raw_em.mean(dim=0)))

        # Evaluate policy output on CURRENT latched state
        logits = model.get_action_logits_island(isl_idx, latched_s)
        legal_logits = logits[0, legal_acts]
        probs = F.softmax(legal_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()

        top_val, top_idx = probs.max(dim=-1)
        top_move_str = action_to_str(legal_acts[top_idx.item()])

        print(f" {step:02d}  | {motor_vel:8.4f}  | {overall_vel:8.4f}  | {latch_event:<14} | {entropy:6.3f}   | {top_move_str} ({top_val.item()*100:4.1f}%)")

trace_island(ROTTEN)
trace_island(GOLDEN)