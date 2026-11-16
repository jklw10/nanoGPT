import torch
from models.chess_bot import GuardrailedTorchChess

ds = torch.load("data/chess/sf_dataset.pt", map_location="cuda", weights_only=False)
N = ds["targets"].shape[0]
chunk = 25000
env = GuardrailedTorchChess(chunk, device="cuda")

total_hits = 0
for start in range(0, N, chunk):
    end = min(start + chunk, N)
    m = end - start
    if m != env.B:
        env = GuardrailedTorchChess(m, device="cuda")
    env.boards[:] = ds["boards"][start:end]
    env.players[:] = torch.where(ds["is_white"][start:end], 0, 1).to(torch.int8)
    if "castling_rights" in ds:
        env.castling_rights[:] = ds["castling_rights"][start:end]
    env.en_passant_sq[:] = -1
    lm = env.compute_legal_masks()
    first_legal = (lm > -1e8).float().argmax(dim=-1)
    hits = int((first_legal == ds["targets"][start:end]).sum().item())
    total_hits += hits
    print(f"[{start:>6}:{end:>6}] legals[0] rate: {100.0 * hits / m:5.1f}%")

print(f"\nOVERALL legals[0] rate: {100.0 * total_hits / N:5.1f}% ({total_hits:,} / {N:,})")