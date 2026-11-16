import shutil
import torch
from models.chess_bot import GuardrailedTorchChess, StockfishEngine, uci_to_action
from models.chess_sweep4 import ExperimentTiedBot, fast_board_to_fen

device = "cuda" if torch.cuda.is_available() else "cpu"
ds = torch.load("data/chess/sf_dataset.pt", map_location=device, weights_only=False)

N = ds["targets"].shape[0]
boards = ds["boards"]
is_w = ds["is_white"]
targets = ds["targets"]
rights = ds.get("castling_rights", None)

def act_to_uci(a):
    fr, to = a // 64, a % 64
    return f"{chr(97 + fr % 8)}{fr // 8 + 1}{chr(97 + to % 8)}{to // 8 + 1}"

# 1. Check start position presence & model prediction on ply 0
env = GuardrailedTorchChess(1, device=device)
start_b = env.boards[0]
matches = (boards == start_b).all(dim=(1, 2)) & is_w
print(f"Dataset size: {N:,} | White startpos count: {matches.sum().item()} | White ratio: {is_w.float().mean().item():.3f}")
if matches.any():
    for idx in torch.nonzero(matches).squeeze(-1).tolist()[:3]:
        print(f"  startpos target in ds: {act_to_uci(int(targets[idx]))}")

# Check piece count distribution in dataset
piece_counts = (boards != 0).sum(dim=(1, 2)).float()
print(f"Piece counts -> mean: {piece_counts.mean():.1f}, min: {piece_counts.min():.0f}, max: {piece_counts.max():.0f}")

# 2. Check model's top-5 moves on startpos
ckpt = torch.load("out/sweep_arm_8.pt", map_location=device, weights_only=False)
cfg = ckpt["config"]
m = ExperimentTiedBot(
    dim=cfg["dim"], num_patt_tokens=cfg["num_tokens"], rollout_steps=cfg["rollout_steps"],
    num_heads=cfg.get("num_heads", 4),
    input_norm=cfg.get("input_norm", False), pollute_residual=cfg.get("pollute_residual", False),
    norm_order=cfg.get("norm_order", "pre_ln"), norm_type=cfg.get("norm_type", "layernorm"),
    use_wo=cfg.get("use_wo", False), use_qk_norm=cfg.get("use_qk_norm", False),
    use_swiglu=cfg.get("use_swiglu", False), cosine_patt=cfg.get("cosine_patt", False),
).to(device)
m.load_state_dict(ckpt["model_state_dict"])
m.has_castling = (rights is not None)
m.eval()

lm0 = env.compute_legal_masks()
with torch.no_grad():
    logits0 = m(env.boards, torch.tensor([True], device=device), castling_rights=env.castling_rights, legal_mask=lm0)[0]
top5 = torch.topk(logits0, 5)
print("Arm 8 top-5 on startpos:", [(act_to_uci(int(a)), round(float(v), 2)) for a, v in zip(top5.indices, top5.values)])

# 3. Inspect 5 random samples & compare against live Stockfish 10k
sf_exec = shutil.which("stockfish") or "./stockfish.exe"
sf = StockfishEngine(sf_exec, threads=1, hash_mb=32)
print("\n5 Random dataset samples vs live SF (10k nodes):")
for idx in torch.randint(0, N, (5,)).tolist():
    r_np = rights[idx].cpu().numpy() if rights is not None else [True, True, True, True]
    fen = fast_board_to_fen(boards[idx].cpu().numpy(), bool(is_w[idx].item()), r_np, -1, 0, 1)
    sf_bm = sf.get_teacher_move(fen, nodes=10000)
    ds_bm = act_to_uci(int(targets[idx].item()))
    print(f"  [{idx:>6}] ds={ds_bm} | sf={sf_bm} | {fen}")
sf.close()