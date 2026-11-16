import os
import queue
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from models.chess_bot import GuardrailedTorchChess, StockfishEngine, uci_to_action
from models.chess_sweep4 import ExperimentTiedBot, fast_board_to_fen

device = "cuda" if torch.cuda.is_available() else "cpu"
sf_data = torch.load("data/chess/sf_dataset.pt", map_location=device, weights_only=False)
sf_exec = shutil.which("stockfish") or "./stockfish.exe"

num_workers = max(2, min((os.cpu_count() or 8) - 2, 16))
print(f"Starting Stockfish pool with {num_workers} workers...")
eng_queue = queue.Queue()
engines = []
for _ in range(num_workers):
    e = StockfishEngine(sf_exec, threads=1, hash_mb=32)
    engines.append(e)
    eng_queue.put(e)
pool = ThreadPoolExecutor(max_workers=num_workers)


def safe_teacher_move(eng, fen, nodes=10000):
    if eng.proc is None or eng.proc.poll() is not None:
        eng._spawn()
        eng.current_mode = None

    if eng.current_mode != "teacher":
        eng._send(["setoption name UCI_LimitStrength value false", "isready"])
        while True:
            line = eng.proc.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish EOF during readyok")
            if "readyok" in line:
                break
        eng.current_mode = "teacher"

    eng._send([f"position fen {fen}", f"go nodes {nodes}"])
    return eng._read_bestmove()


def parallel_elo_moves(tasks, elo=1350):
    def query(t):
        idx, fen, leg = t
        eng = eng_queue.get()
        try:
            bm = eng.get_elo_move(fen, elo=elo)
            act = uci_to_action(bm) if bm else None
            return idx, (act if (act is not None and act in leg) else leg[0])
        finally:
            eng_queue.put(eng)

    futs = [pool.submit(query, t) for t in tasks]
    out = {}
    for f in as_completed(futs, timeout=15.0):
        idx, act = f.result()
        out[idx] = act
    return out


def parallel_teacher_matches(samples, nodes=10000):
    print(f"  [Teacher] Scoring {len(samples)} midgame positions @ {nodes} nodes...")
    print(f"  [Teacher] Sample #0 FEN: {samples[0][1]}")

    def query(item):
        bot_act, fen = item
        eng = eng_queue.get()
        try:
            t_bm = safe_teacher_move(eng, fen, nodes=nodes)
            return 1 if (t_bm and uci_to_action(t_bm) == bot_act) else 0
        finally:
            eng_queue.put(eng)

    futs = [pool.submit(query, s) for s in samples]
    done = 0
    hits = 0
    for f in as_completed(futs, timeout=15.0):
        hits += f.result()
        done += 1
        if done == 1 or done % 40 == 0 or done == len(samples):
            print(f"    [Teacher] {done}/{len(samples)} scored (running Top-1: {100.0 * hits / done:5.1f}%)")
    return 100.0 * hits / max(1, len(samples))


def load_arm(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    m.has_castling = ("castling_rights" in sf_data)
    m.eval()
    return ckpt["arm_name"], m


@torch.no_grad()
def check_ckpt(ckpt_path):
    t0 = time.perf_counter()
    name, model = load_arm(ckpt_path)
    print(f"\n=== Checking: {name} ({ckpt_path}) ===")

    idx = torch.randperm(sf_data["targets"].shape[0], device=device)[:4096]
    c_ds = sf_data["castling_rights"][idx] if "castling_rights" in sf_data else None
    logits_ds = model(sf_data["boards"][idx], sf_data["is_white"][idx], castling_rights=c_ds)
    ds_top1 = (logits_ds.argmax(dim=-1) == sf_data["targets"][idx]).float().mean().item() * 100.0
    print(f"  [Dataset] sf_dataset Top-1 (4096 sample): {ds_top1:5.1f}%")

    games = 16
    env = GuardrailedTorchChess(games, device=device, halfmove_limit=100)
    bot_is_w = torch.arange(games, device=device) < (games // 2)
    dones = torch.zeros(games, dtype=torch.bool, device=device)
    winners = torch.zeros(games, dtype=torch.int8, device=device)

    ood_samples = []
    for ply in range(60):
        if dones.all():
            break
        lm = env.compute_legal_masks()
        acts = torch.zeros(games, dtype=torch.long, device=device)
        is_w = (env.players == 0)
        has_leg = (lm > -1e8).any(-1)

        bot_term = (~dones) & (is_w == bot_is_w) & (~has_leg)
        dones = dones | bot_term

        is_bot = (~dones) & (is_w == bot_is_w) & has_leg
        is_sf = (~dones) & (is_w != bot_is_w)

        if is_bot.any():
            b_acts = model.select_action(env.boards, is_w, lm, castling_rights=env.castling_rights, temperature=0.0)
            acts[is_bot] = b_acts[is_bot]
            if ply >= 6:
                b_np = env.boards.cpu().numpy()
                w_l = is_w.cpu().tolist()
                r_np = env.castling_rights.cpu().numpy()
                ep_l = env.en_passant_sq.cpu().tolist()
                hm_l = env.halfmove_clock.cpu().tolist()
                ba_l = b_acts.cpu().tolist()
                for i in torch.nonzero(is_bot).squeeze(-1).tolist():
                    fen = fast_board_to_fen(b_np[i], w_l[i], r_np[i], ep_l[i], hm_l[i], ply // 2 + 1)
                    ood_samples.append((ba_l[i], fen))

        if is_sf.any():
            sf_idx = torch.nonzero(is_sf).squeeze(-1).tolist()
            b_np = env.boards.cpu().numpy()
            w_l = is_w.cpu().tolist()
            r_np = env.castling_rights.cpu().numpy()
            ep_l = env.en_passant_sq.cpu().tolist()
            hm_l = env.halfmove_clock.cpu().tolist()
            nz = torch.nonzero(lm > -1e8).cpu()
            leg_map = {i: [] for i in sf_idx}
            sf_set = set(sf_idx)
            for bi, act in nz.tolist():
                if bi in sf_set:
                    leg_map[bi].append(act)

            tasks = []
            for i in sf_idx:
                if not leg_map[i]:
                    dones[i] = True
                else:
                    fen = fast_board_to_fen(b_np[i], w_l[i], r_np[i], ep_l[i], hm_l[i], ply // 2 + 1)
                    tasks.append((i, fen, leg_map[i]))
            if tasks:
                for idx_g, act in parallel_elo_moves(tasks, elo=1350).items():
                    acts[idx_g] = act

        dones, winners = env.step(acts, dones, winners, legal_mask=lm)
        if (ply + 1) % 10 == 0 or dones.all():
            print(f"  [Rollout] Ply {ply+1:02d}/60 | Active: {(~dones).sum().item()}/{games} | Collected OOD pos: {len(ood_samples)}")

    sample_n = min(200, len(ood_samples))
    ood_top1 = parallel_teacher_matches(ood_samples[:sample_n], nodes=10000)
    dt = time.perf_counter() - t0
    print(f"  --> RESULT ({dt:.1f}s) | sf_dataset Top-1: {ds_top1:5.1f}% | On-Policy Midgame Top-1 (SF 10k): {ood_top1:5.1f}%")


try:
    for p in ["out/sweep_arm_2.pt", "out/sweep_arm_8.pt"]:
        if os.path.exists(p):
            check_ckpt(p)
        else:
            print(f"[Skip] {p} not found.")
finally:
    pool.shutdown(wait=False)
    for e in engines:
        e.close()