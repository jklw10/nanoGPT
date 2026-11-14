"""
test_dust_muon.py
Tests DUST bootstrap with Muon-style Newton-Schulz orthogonalization
and Riemannian tangent projection on the weight sphere.
"""

import math
import torch
import torch.nn.functional as F
from models.nodenet.nodechess20 import ChessNodeNet, GuardrailedTorchChess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = 256
thinking_steps = 10
dust_lr = 0.02  # On an orthogonalized tangent sphere, 0.01 - 0.05 is the sweet spot

def batched_newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    norm = torch.linalg.matrix_norm(X, keepdim=True) + eps
    X = X / norm
    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(-2, -1)
        transposed = True
    for _ in range(steps):
        A = torch.bmm(X, X.transpose(-2, -1))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)
    if transposed:
        X = X.transpose(-2, -1)
    return X.type_as(G)

def project_to_sphere_tangent(G, W, dim=-2):
    dot_prod = (G * W).sum(dim=dim, keepdim=True)
    w_norm_sq = (W * W).sum(dim=dim, keepdim=True) + 1e-8
    return G - (dot_prod / w_norm_sq) * W

print(" DUST + MUON NEWTON-SCHULZ SPHERE BOOTSTRAP TEST")

model = ChessNodeNet(num_nodes=32, dim=32, input_dim=64, rank_k=16, device=device)

# Initialize with clean Quasizero Identity
with torch.no_grad():
    D = model.D
    eye_split = torch.cat([torch.eye(D, device=device), torch.zeros(D, D, device=device)], dim=1)
    model.w1.weight.data.copy_((eye_split.unsqueeze(0).expand(model.N, -1, -1) + torch.randn_like(model.w1.weight.data) * 0.01) * 0.5)
    model.w1.normalize_weight()  # Normalize BEFORE recording W_init
    model.action_logit_scale.data.fill_(2.5)

W_init = model.w1.weight.data.clone()

env = GuardrailedTorchChess(B, device=device)
agent_state, agent_output, agent_error, agent_lp = model.init_states(B)
agent_noise = torch.full((B,), 0.01, device=device)

dones = torch.zeros(B, dtype=torch.bool, device=device)
winners = torch.zeros(B, dtype=torch.int8, device=device)
active_idx = torch.arange(B, device=device)

print(f"{'Ply':<4} | {'Loss':<8} | {'Weight Drift ||W-W0||':<22} | {'Legal Ent':<9} | {'Top Move (Prob)'}")
print("-" * 90)

DUST_SIGMA = 0.04

for ply in range(1, 51):
    l_mask = env.compute_legal_masks()
    e_in, th = env.get_obs(agent_noise)
    PB = e_in.shape[0]

    with torch.no_grad():
        raw_s = model.sensory_proj(e_in)
        lat = F.softsign(raw_s.view(PB, model.num_input_nodes, model.D).permute(1, 0, 2))

    # 1. Deliberation trajectory
    traj_l, ns, no, ne, nl, omgs, s_traj = model(
        e_in, th, agent_state, agent_output, agent_error, agent_lp, 
        thinking_steps=thinking_steps, precomputed_latents=lat
    )

    # 2. Global kick switch (intended sum)
    s_magnitude = s_traj.abs().sum(dim=(1, 2), keepdim=True)
    hunger_switch = torch.randn_like(s_traj) / (1.0 + s_magnitude)

    # 3. Finite Difference Evaluation
    in2 = e_in.repeat(2, 1)
    th2 = th.repeat(2)
    s2 = s_traj.repeat(1, 2, 1)
    o2 = no.repeat(1, 2, 1)
    e2 = ne.repeat(1, 2)
    l2 = nl.repeat(1, 2)

    xi = torch.randn(model.N, PB, model.D * 2, device=device)
    lat2 = lat.repeat(1, 2, 1)

    _, _, _, _, _, l_both, _ = model(
        in2, th2, s2, o2, e2, l2,
        act_jitter=torch.cat([DUST_SIGMA * xi, -DUST_SIGMA * xi], dim=1),
        precomputed_latents=lat2,
        thinking_steps=1
    )
    lp, ln = l_both.chunk(2, dim=1)
    
    # 4. Raw gradient from DUST
    gh = (((lp - ln) / (2.0 * DUST_SIGMA)) * xi)
    gh_demand = gh.std(dim=-1, keepdim=True, correction=0)
    s_hungry_w1 = s_traj + hunger_switch + (torch.randn_like(s_traj) * gh_demand * 0.10)
    
    raw_grad = torch.bmm(s_hungry_w1.transpose(1, 2), gh) / PB

    # 5. MUON Orthogonalization + Tangent Projection
    ortho_grad = batched_newtonschulz5(raw_grad)
    tangent_grad = project_to_sphere_tangent(ortho_grad, model.w1.weight.data, dim=-2)

    # 6. Step along manifold + Retract
    model.w1.weight.data.sub_(dust_lr * tangent_grad)
    model.w1.normalize_weight()

    # Update states
    agent_state[:, active_idx, :] = ns
    agent_output[:, active_idx, :] = no
    agent_error[:, active_idx] = ne
    agent_lp[:, active_idx] = nl

    acts = torch.argmax(model.get_action_logits(ns) + l_mask, dim=-1)
    dones, winners = env.step(acts, dones, winners, legal_mask=l_mask)
    if dones.any():
        env.reset(mask=dones)
        dones.fill_(False)

    with torch.no_grad():
        w_drift = torch.norm(model.w1.weight.data - W_init).item()
        logits = model.get_action_logits(ns[:, 0:1, :])
        legal_acts = torch.nonzero(l_mask[0] > -1e8).squeeze(-1)
        legal_logits = logits[0, legal_acts]
        probs = F.softmax(legal_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
        top_prob, top_idx = probs.max(dim=-1)
        top_act = legal_acts[top_idx.item()].item()
        move_str = f"{chr(ord('a') + (top_act // 64 % 8))}{(top_act // 64 // 8) + 1}{chr(ord('a') + (top_act % 64 % 8))}{(top_act % 64 // 8) + 1}"

    if ply % 5 == 0 or ply == 1:
        print(f" {ply:02d}  | {traj_l.item():.4f}   | {w_drift:20.4f}   | {entropy:6.3f}    | {move_str} ({top_prob.item()*100:4.1f}%)")
