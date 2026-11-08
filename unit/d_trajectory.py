import torch
import torch.nn.functional as F
from models.nodechess15 import ChessNodeNet, GuardrailedTorchChess, HyperParamsConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = HyperParamsConfig(device)
model = ChessNodeNet(num_nodes=32, dim=32, input_dim=100, rank_k=16, device=device).to(device)

ckpt = torch.load("out/champion_agent.pt", map_location=device)
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(sd, strict=False)
model.eval()

# Starting chess position
env = GuardrailedTorchChess(batch_size=1, device=device)
obs, threat = env.get_obs(agent_noise=torch.tensor([0.01], device=device))
legal_mask = env.compute_legal_masks()[0]

state = ckpt['champion_state'].to(device)
if state.dim() == 2: state = state.unsqueeze(1)
out = ckpt['champion_output'].to(device)
if out.dim() == 2: out = out.unsqueeze(1)
err = ckpt['champion_error_ema'].to(device)
lp = ckpt['champion_lp_ema'].to(device)

print(f"{'Step':>4} | {'Sensory L2':>10} | {'Thinking L2':>11} | {'Motor L2':>9} | {'Top Logit':>9} | {'Logit Spread':>12} | {'Top Move Prob':>13}")
print("-" * 85)

with torch.no_grad():
    for step in range(1, 129):
        _, state, out, err, lp, _ = model._forward(obs, threat, state, out, err, lp)
        
        sensory_l2 = torch.norm(state[:4, 0], p=2).item()
        think_l2   = torch.norm(state[4:28, 0], p=2).item()
        motor_l2   = torch.norm(state[28:, 0], p=2).item()
        
        logits = model.get_action_logits(state)[0] + legal_mask
        probs = F.softmax(logits, dim=-1)
        
        legal_indices = torch.nonzero(legal_mask > -1e8).squeeze(-1)
        legal_logits = logits[legal_indices]
        top_prob = probs[legal_indices].max().item() * 100
        spread = (legal_logits.max() - legal_logits.min()).item()
        top_l = legal_logits.max().item()
        
        
        print(f"{step:4d} | {sensory_l2:10.4f} | {think_l2:11.4f} | {motor_l2:9.4f} | {top_l:9.4f} | {spread:12.4f} | {top_prob:12.2f}%")