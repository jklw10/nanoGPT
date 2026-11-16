# =====================================================================
# chess_sf_sweep2.py
# Transformer Residual / Norm / Attention Ablation Sweep
# for 2-Engine Alternating Tied Dialectic PATT Chess Bot
# =====================================================================

import argparse
import math
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.chess_bot import (
    DialecticPattChessBot,
    StockfishEngine,
    evaluate_model_elo,
)


# =====================================================================
# 1. MODERN TRANSFORMER & PATT BUILDING BLOCKS
# =====================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        out = (x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
        return out * self.weight if self.weight is not None else out


def make_norm(norm_type: str, dim: int, affine: bool = True) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim, elementwise_affine=affine)
    return nn.LayerNorm(dim, elementwise_affine=affine)


class FlexPattLayer(nn.Module):
    """PattLayer with optional Tokenformer-style cosine-normalized keys."""
    def __init__(self, input_dim: int, output_dim: int, num_tokens: int, cosine_patt: bool = False):
        super().__init__()
        self.cosine_patt = cosine_patt
        self.key_tokens = nn.Parameter(torch.empty(num_tokens, input_dim))
        self.val_tokens = nn.Parameter(torch.empty(num_tokens, output_dim))
        nn.init.kaiming_uniform_(self.key_tokens, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.val_tokens, a=math.sqrt(5))
        if cosine_patt:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            nn.init.normal_(self.val_tokens, std=1.0 / math.sqrt(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cosine_patt:
            x_n = F.normalize(x.float(), dim=-1).to(x.dtype)
            k_n = F.normalize(self.key_tokens.float(), dim=-1).to(x.dtype)
            scale = self.logit_scale.exp().clamp(max=50.0).to(x.dtype)
            scores = (x_n @ k_n.T) * scale
        else:
            scores = (x @ self.key_tokens.T) / math.sqrt(self.key_tokens.shape[0])
        attn = F.softmax(scores.float(), dim=-1).to(self.val_tokens.dtype)
        return attn @ self.val_tokens


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Parameter-matched to 4*dim standard MLP (~8/3 * dim)
        hidden = max(32, ((int(dim * 8 / 3) + 15) // 16) * 16)
        self.w13 = nn.Linear(dim, hidden * 2, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * val)


class ExperimentPattAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        num_patt_tokens: int = 128,
        num_heads: int = 4,
        norm_order: str = "pre_ln",     # pre_ln | post_ln | peri_ln | layerscale | adaln_zero
        norm_type: str = "layernorm",   # layernorm | rmsnorm
        use_wo: bool = False,
        use_qk_norm: bool = False,
        use_swiglu: bool = False,
        cosine_patt: bool = False,
        layerscale_init: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_order = norm_order

        self.patt_q = FlexPattLayer(dim, dim, num_patt_tokens, cosine_patt=cosine_patt)
        self.patt_k = FlexPattLayer(dim, dim, num_patt_tokens, cosine_patt=cosine_patt)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_o = nn.Linear(dim, dim, bias=False) if use_wo else nn.Identity()

        self.q_norm = make_norm(norm_type, self.head_dim) if use_qk_norm else nn.Identity()
        self.k_norm = make_norm(norm_type, self.head_dim) if use_qk_norm else nn.Identity()

        if use_swiglu:
            self.mlp = SwiGLUFFN(dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Linear(dim * 4, dim),
            )

        use_affine = (norm_order != "adaln_zero")
        self.norm1 = make_norm(norm_type, dim, affine=use_affine)
        self.norm2 = make_norm(norm_type, dim, affine=use_affine)

        if norm_order == "peri_ln":
            self.post_norm1 = make_norm(norm_type, dim)
            self.post_norm2 = make_norm(norm_type, dim)
        elif norm_order == "layerscale":
            self.ls1 = nn.Parameter(torch.full((dim,), layerscale_init))
            self.ls2 = nn.Parameter(torch.full((dim,), layerscale_init))
        elif norm_order == "adaln_zero":
            self.ada_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
            nn.init.zeros_(self.ada_mlp[1].weight)
            nn.init.zeros_(self.ada_mlp[1].bias)

    def _attn(self, h: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        q = self.patt_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.patt_k(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        ctx = F.scaled_dot_product_attention(q, k, v)
        ctx = ctx.transpose(1, 2).reshape(B, N, D)
        return self.proj_o(ctx)

    def forward(self, x: torch.Tensor, depth_token: torch.Tensor = None) -> torch.Tensor:
        if self.norm_order == "pre_ln":
            h = self.norm1(x)
            if depth_token is not None:
                h = h + depth_token
            x = x + self._attn(h)
            x = x + self.mlp(self.norm2(x))
            return x

        if self.norm_order == "post_ln":
            h = (x + depth_token) if depth_token is not None else x
            x = self.norm1(x + self._attn(h))
            x = self.norm2(x + self.mlp(x))
            return x

        if self.norm_order == "peri_ln":
            h = self.norm1(x)
            if depth_token is not None:
                h = h + depth_token
            x = x + self.post_norm1(self._attn(h))
            x = x + self.post_norm2(self.mlp(self.norm2(x)))
            return x

        if self.norm_order == "layerscale":
            h = self.norm1(x)
            if depth_token is not None:
                h = h + depth_token
            x = x + self.ls1 * self._attn(h)
            x = x + self.ls2 * self.mlp(self.norm2(x))
            return x

        if self.norm_order == "adaln_zero":
            s1, g1_s, g1, s2, g2_s, g2 = self.ada_mlp(depth_token).chunk(6, dim=-1)
            h1 = self.norm1(x) * (1.0 + g1_s) + s1
            x = x + g1 * self._attn(h1)
            h2 = self.norm2(x) * (1.0 + g2_s) + s2
            x = x + g2 * self.mlp(h2)
            return x

        raise ValueError(f"Unknown norm_order: {self.norm_order}")


class ExperimentTiedBot(DialecticPattChessBot):
    def __init__(
        self,
        dim: int = 64,
        num_patt_tokens: int = 128,
        rollout_steps: int = 6,
        head_dim: int = 32,
        num_heads: int = 4,
        input_norm: bool = False,
        pollute_residual: bool = False,
        norm_order: str = "pre_ln",
        norm_type: str = "layernorm",
        use_wo: bool = False,
        use_qk_norm: bool = False,
        use_swiglu: bool = False,
        cosine_patt: bool = False,
    ):
        super().__init__(dim=dim, num_patt_tokens=num_patt_tokens, rollout_steps=rollout_steps, head_dim=head_dim)
        del self.rollout_layers

        self.input_norm = input_norm
        self.pollute_residual = pollute_residual
        self.ply_embed = nn.Embedding(16, dim)
        self.register_buffer("ply_indices", torch.arange(16), persistent=False)

        if not input_norm:
            self.norm = nn.Identity()
        if norm_type == "rmsnorm":
            self.final_norm = RMSNorm(dim)

        layer_kwargs = dict(
            dim=dim,
            num_patt_tokens=num_patt_tokens,
            num_heads=num_heads,
            norm_order=norm_order,
            norm_type=norm_type,
            use_wo=use_wo,
            use_qk_norm=use_qk_norm,
            use_swiglu=use_swiglu,
            cosine_patt=cosine_patt,
        )
        self.thesis_engine = ExperimentPattAttentionLayer(**layer_kwargs)
        self.antithesis_engine = ExperimentPattAttentionLayer(**layer_kwargs)

    def forward(
        self,
        boards: torch.Tensor,
        is_white: torch.Tensor,
        castling_rights: torch.Tensor = None,
        legal_mask: torch.Tensor = None,
    ):
        B = boards.shape[0]
        flat = boards.view(B, 64).long().clamp(0, 12)
        turn_idx = torch.where(is_white, 0, 1).long()

        pos_embed = self.rank_embed(self.sq_ranks) + self.file_embed(self.sq_files)
        tokens = self.piece_embed(flat) + pos_embed.unsqueeze(0) + self.turn_embed(turn_idx).unsqueeze(1)

        if castling_rights is not None and getattr(self, "has_castling", True):
            tokens = tokens + self.castling_proj(castling_rights.float()).unsqueeze(1)

        x = self.norm(tokens)

        for ply in range(self.rollout_steps):
            depth_token = self.ply_embed(self.ply_indices[ply]).view(1, 1, self.dim)
            engine = self.thesis_engine if (ply % 2 == 0) else self.antithesis_engine
            if self.pollute_residual:
                x = engine(x + depth_token, depth_token=None)
            else:
                x = engine(x, depth_token=depth_token)

        x = self.final_norm(x)

        q_fr = self.proj_from(x)
        k_to = self.proj_to(x)
        pairwise = torch.bmm(q_fr, k_to.transpose(1, 2)) * (1.0 / math.sqrt(self.head_dim))

        b_fr = self.bias_from(x)
        b_to = self.bias_to(x).transpose(1, 2)

        logits = (pairwise + b_fr + b_to).view(B, 4096)
        if legal_mask is not None:
            logits = logits + legal_mask
        return logits


# =====================================================================
# 2. UTILITIES & TRAINING LOOP
# =====================================================================
def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.03):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_mat_cp(cell: dict) -> float:
    if not cell:
        return 0.0
    if "mat_lead" in cell:
        return float(cell["mat_lead"])
    raw = str(cell.get("mat_str", "0")).replace("cp", "").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def train_and_eval_arm(
    arm_cfg,
    dataset,
    sf_engine,
    epochs=10,
    default_batch=2048,
    elos=(1350, 1600),
    eval_temps=(0.0, 0.7),
    games_per_elo=100,
    max_plies=60,
    use_masked_ce=False,
    device="cuda",
):
    dim = arm_cfg["dim"]
    num_tokens = arm_cfg["num_tokens"]
    rollout_steps = arm_cfg["rollout_steps"]
    num_heads = arm_cfg.get("num_heads", 4)
    lr = arm_cfg.get("lr", 1.5e-3)
    wd = arm_cfg.get("weight_decay", 1e-4)
    batch_size = arm_cfg.get("batch_size", default_batch)
    clip_norm = arm_cfg.get("clip_norm", 1.0)

    model = ExperimentTiedBot(
        dim=dim,
        num_patt_tokens=num_tokens,
        rollout_steps=rollout_steps,
        num_heads=num_heads,
        input_norm=arm_cfg.get("input_norm", False),
        pollute_residual=arm_cfg.get("pollute_residual", False),
        norm_order=arm_cfg.get("norm_order", "pre_ln"),
        norm_type=arm_cfg.get("norm_type", "layernorm"),
        use_wo=arm_cfg.get("use_wo", False),
        use_qk_norm=arm_cfg.get("use_qk_norm", False),
        use_swiglu=arm_cfg.get("use_swiglu", False),
        cosine_patt=arm_cfg.get("cosine_patt", False),
    ).to(device)

    # Zero weight decay on embeddings, 1D norm weights, biases, and LayerScale parameters
    raw_model = getattr(model, "_orig_mod", model)
    no_decay_params = [
        p for n, p in raw_model.named_parameters()
        if p.ndim < 2 or "embed" in n or "castling_proj" in n
    ]
    no_decay_set = set(no_decay_params)
    decay_params = [p for p in raw_model.parameters() if p not in no_decay_set]

    optimizer = torch.optim.AdamW(
        [
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
            {"params": decay_params, "lr": lr, "weight_decay": wd},
        ],
        fused=(device.type == "cuda"),
    )

    boards = dataset["boards"]
    is_white = dataset["is_white"]
    targets = dataset["targets"]
    castling = dataset.get("castling_rights", None)
    legal_bool = dataset.get("legal_bool", None) if use_masked_ce else None

    has_castling = (castling is not None)
    model.has_castling = has_castling

    raw_dataset_size = targets.shape[0]
    effective_size = raw_dataset_size - (raw_dataset_size % batch_size)
    steps_per_epoch = effective_size // batch_size
    total_steps = steps_per_epoch * epochs
    total_positions = effective_size * epochs
    warmup_steps = max(1, int(total_steps * 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.03)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"   [Arch] Dim: {dim} | Tokens: {num_tokens} | Plies: {rollout_steps} | "
        f"Norm: {arm_cfg.get('norm_order')} ({arm_cfg.get('norm_type', 'layernorm')}) | "
        f"Params: {param_count:,} | LR: {lr:.1e}"
    )
    print(f"   [Volume] {effective_size:,} pos/epoch * {epochs} epochs = {total_positions:,} total pos ({total_steps:,} steps)")

    print("   [Warmup] Compiling model CUDA graphs...", end="", flush=True)
    compiled_model = torch.compile(model, mode="reduce-overhead")

    wb, ww, wy = boards[:batch_size], is_white[:batch_size], targets[:batch_size]
    wc = castling[:batch_size] if has_castling else None

    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            w_logits = compiled_model(wb, ww, castling_rights=wc)
            if legal_bool is not None:
                w_lm = legal_bool[:batch_size].clone()
                w_lm.scatter_(1, wy.unsqueeze(1), True)
                w_logits = w_logits.masked_fill(~w_lm, -1e4)
            w_loss = F.cross_entropy(w_logits, wy)
        w_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    model.load_state_dict(init_state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(" Ready.")

    arm_start_t = time.time()
    last_epoch_loss, last_epoch_correct = 0.0, 0
    last_epoch_samples = 0
    global_step = 0

    compiled_model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(effective_size, device=device)
        is_last_epoch = (epoch == epochs)

        for i in range(0, effective_size, batch_size):
            global_step += 1
            idx = perm[i:i + batch_size]
            b, w, y = boards[idx], is_white[idx], targets[idx]
            c = castling[idx] if has_castling else None

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits = compiled_model(b, w, castling_rights=c)
                if legal_bool is not None:
                    lm = legal_bool[idx].clone()
                    lm.scatter_(1, y.unsqueeze(1), True)
                    logits = logits.masked_fill(~lm, -1e4)
                loss = F.cross_entropy(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            scheduler.step()

            if is_last_epoch:
                with torch.no_grad():
                    correct = (torch.argmax(logits, dim=-1) == y).sum().item()
                    last_epoch_loss += loss.item() * batch_size
                    last_epoch_correct += correct
                    last_epoch_samples += batch_size

        elapsed = time.time() - arm_start_t
        processed_positions = global_step * batch_size
        rate = processed_positions / max(1e-4, elapsed)
        remaining_positions = max(0, total_positions - processed_positions)
        eta = remaining_positions / max(1e-4, rate)
        print(
            f"     Epoch [{epoch:02d}/{epochs:02d}] Step [{global_step:04d}/{total_steps:04d}] | "
            f"Loss: {loss.item():.4f} | Speed: {rate:7.0f} pos/s | "
            f"Elapsed: {format_eta(elapsed)} | ETA: {format_eta(eta)}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - arm_start_t
    train_rate = total_positions / max(1e-4, total_time)
    final_loss = last_epoch_loss / max(1, last_epoch_samples)
    final_top1 = (last_epoch_correct / max(1, last_epoch_samples)) * 100.0

    print(
        f"   [Arm Trained] {total_positions:,} pos in {format_eta(total_time)} ({train_rate:7.0f} pos/s) | "
        f"Top-1: {final_top1:4.1f}% | Loss: {final_loss:.4f}"
    )
    print(f"   [Gauntlet] Running Stockfish match play ({games_per_elo} games/temp/elo, max_plies={max_plies})...")

    orig_select = model.select_action
    elo_data = {}

    for temp in eval_temps:
        temp_tag = f"T={temp:.1f}"
        print(f"     -> Evaluating [{temp_tag}] ({'Deterministic Greedy' if temp == 0.0 else 'Stochastic Sampling'})...")

        def make_eval_wrapper(t_val):
            def eval_select_wrapper(boards, is_white, legal_mask, castling_rights=None, temperature=0.0, current_ply=0):
                c_rights = castling_rights if has_castling else None
                return orig_select(boards, is_white, legal_mask, castling_rights=c_rights, temperature=t_val)
            return eval_select_wrapper

        model.select_action = make_eval_wrapper(temp)
        elo_data[temp] = {}

        for elo in elos:
            res_elo = evaluate_model_elo(
                model, sf_engine, elo=elo, games=games_per_elo, max_plies=max_plies, device=device
            )
            elo_data[temp][elo] = res_elo
            print(f"        Elo {elo:<4d} | Score: {res_elo['score_str']:<14} | Mat: {res_elo['mat_str']}")

    model.select_action = orig_select

    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        torch.compiler.reset()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "final_loss": final_loss,
        "final_top1": final_top1,
        "throughput": train_rate,
        "params": param_count,
        "elo_data": elo_data,
        "model": model,
        "cfg": arm_cfg,
    }


# =====================================================================
# 3. TRANSFORMER LEARNINGS ABLATION SWEEP
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Transformer Learnings Sweep for Tied Dialectic Patt Bot")
    parser.add_argument("--dataset", type=str, default="data/chess/sf_dataset.pt", help="Path to cleaned SF dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per arm")
    parser.add_argument("--batch", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Base learning rate")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_tokens", type=int, default=128, help="PATT concept tokens")
    parser.add_argument("--rollout_steps", type=int, default=6, help="Tied dialectic rollout plies")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--games_per_elo", type=int, default=100, help="Evaluation games per Elo and Temperature")
    parser.add_argument("--max_plies", type=int, default=60, help="Max plies per evaluation game")
    parser.add_argument("--eval_temps", type=float, nargs="+", default=[0.0, 0.7], help="Evaluation temperatures")
    parser.add_argument("--masked_ce", action="store_true", help="Use legal_masks in CE loss if saved in dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    os.makedirs("out", exist_ok=True)

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Stockfish dataset not found at '{args.dataset}'")

    print(f"\nLoading Stockfish dataset from '{args.dataset}' into {device}...")
    sf_data = torch.load(args.dataset, map_location=device, weights_only=False)

    if "legal_masks" in sf_data:
        if args.masked_ce:
            sf_data["legal_bool"] = (sf_data.pop("legal_masks") > -1e8).contiguous()
            print("  -> Converted legal_masks to compact bool tensor for Masked CE.")
        else:
            del sf_data["legal_masks"]
            if device.type == "cuda":
                torch.cuda.empty_cache()
    elif args.masked_ce:
        print("  -> [Notice] --masked_ce requested, but 'legal_masks' not in dataset. Falling back to unmasked CE.")

    sf_data["boards"] = sf_data["boards"].contiguous()
    sf_data["is_white"] = sf_data["is_white"].contiguous()
    sf_data["targets"] = sf_data["targets"].contiguous()
    if "castling_rights" in sf_data:
        sf_data["castling_rights"] = sf_data["castling_rights"].contiguous()
        print("  -> Castling rights detected in dataset.")
    else:
        print("  -> [Notice] No castling rights in dataset. Forward will bypass castling projections.")

    print(f"Loaded {sf_data['targets'].shape[0]:,} positions successfully (Dataset cached in VRAM).")

    sf_exec = shutil.which("stockfish") or ("./stockfish.exe" if os.path.exists("stockfish.exe") else None)
    if not sf_exec:
        raise RuntimeError("Stockfish binary required for evaluation gauntlet.")

    sf_engine = StockfishEngine(sf_exec, threads=4, hash_mb=128)
    test_elos = [1350, 1600]

    base = {
        "dim": args.dim,
        "num_tokens": args.num_tokens,
        "rollout_steps": args.rollout_steps,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "batch_size": args.batch,
    }

    sweep_arms = [
        {
            **base,
            "name": "0. Baseline (DblNorm+Pollute+PreLN)",
            "input_norm": True, "pollute_residual": True, "norm_order": "pre_ln",
            "norm_type": "layernorm", "use_wo": False, "use_qk_norm": False,
            "use_swiglu": False, "cosine_patt": False,
        },
        {
            **base,
            "name": "1. Clean Residual (No DblNorm/Pollute)",
            "input_norm": False, "pollute_residual": False, "norm_order": "pre_ln",
            "norm_type": "layernorm", "use_wo": False, "use_qk_norm": False,
            "use_swiglu": False, "cosine_patt": False,
        },
        {
            **base,
            "name": "2. Post-LN (Universal Transformer)",
            "input_norm": False, "pollute_residual": False, "norm_order": "post_ln",
            "norm_type": "layernorm", "use_wo": False, "use_qk_norm": False,
            "use_swiglu": False, "cosine_patt": False,
        },
        {
            **base,
            "name": "3. Peri-LN (Sandwich / Gemma-2)",
            "input_norm": False, "pollute_residual": False, "norm_order": "peri_ln",
            "norm_type": "layernorm", "use_wo": False, "use_qk_norm": False,
            "use_swiglu": False, "cosine_patt": False,
        },
        {
            **base,
            "name": "4. Pre-LN + LayerScale (0.1)",
            "input_norm": False, "pollute_residual": False, "norm_order": "layerscale",
            "norm_type": "layernorm", "use_wo": False, "use_qk_norm": False,
            "use_swiglu": False, "cosine_patt": False,
        },
        {
            **base,
            "name": "5. AdaLN-Zero (Per-Ply Modulated)",
            "input_norm": False, "pollute_residual": False, "norm_order": "adaln_zero",
            "norm_type": "layernorm", "use_wo": False, "use_qk_norm": False,
            "use_swiglu": False, "cosine_patt": False,
        },
        {
            **base,
            "name": "6. Peri-LN + RMS + Wo + QKNorm + SwiGLU",
            "input_norm": False, "pollute_residual": False, "norm_order": "peri_ln",
            "norm_type": "rmsnorm", "use_wo": True, "use_qk_norm": True,
            "use_swiglu": True, "cosine_patt": False,
        },
        {
            **base,
            "name": "7. Arm 6 + CosinePatt Fix",
            "input_norm": False, "pollute_residual": False, "norm_order": "peri_ln",
            "norm_type": "rmsnorm", "use_wo": True, "use_qk_norm": True,
            "use_swiglu": True, "cosine_patt": True,
        },
        {
            **base,
            "name": "8. Post-LN + RMS+Wo+QKNorm+SwiGLU+Cos",
            "input_norm": False, "pollute_residual": False, "norm_order": "post_ln",
            "norm_type": "rmsnorm", "use_wo": True, "use_qk_norm": True,
            "use_swiglu": True, "cosine_patt": True,
        },
    ]

    print("\n" + "=" * 150)
    print("             TRANSFORMER ARCHITECTURE ABLATION SWEEP (RESIDUALS, NORMS, ATTENTION, COSINE-PATT)")
    print(f" {len(sweep_arms)} Configs | {args.epochs} Epochs/Arm | Gauntlet: {args.games_per_elo} Games @ Elos {test_elos} across Temps {args.eval_temps}")
    print("=" * 150)

    results = []
    try:
        for idx, arm in enumerate(sweep_arms):
            print(f"\n[Arm {idx+1}/{len(sweep_arms)}] Running: {arm['name']} ...")
            res = train_and_eval_arm(
                arm_cfg=arm,
                dataset=sf_data,
                sf_engine=sf_engine,
                epochs=args.epochs,
                default_batch=args.batch,
                elos=test_elos,
                eval_temps=args.eval_temps,
                games_per_elo=args.games_per_elo,
                max_plies=args.max_plies,
                use_masked_ce=args.masked_ce,
                device=device,
            )
            results.append((arm, res))

            arm_ckpt_path = f"out/sweep_arm_{idx}.pt"
            torch.save(
                {
                    "model_state_dict": res["model"].state_dict(),
                    "config": arm,
                    "arm_name": arm["name"],
                    "results": {"loss": res["final_loss"], "top1": res["final_top1"], "elo": res["elo_data"]},
                },
                arm_ckpt_path,
            )

    finally:
        sf_engine.close()

    if not results:
        print("[Notice] No arms completed.")
        return

    # =========================================================================
    # LEADERBOARD
    # =========================================================================
    print("\n\n" + "=" * 175)
    print("                                      TRANSFORMER LEARNINGS SWEEP LEADERBOARD (T=0.0 vs T=0.7)")
    print("=" * 175)
    header = (
        f"{'Rank / Arm':<42} | {'Params':<8} | {'Loss':<6} | {'Top-1':<6} | "
        f"{'1350 (T=0.0)':<20} | {'1350 (T=0.7)':<20} | {'1600 (T=0.0)':<20} | {'1600 (T=0.7)':<20}"
    )
    print(header)
    print("-" * 175)

    def sort_key(item):
        _, r = item
        score = r["final_top1"] * 0.05
        weights = {0.0: 1.0, 0.7: 0.8}

        for temp, t_data in r["elo_data"].items():
            tw = weights.get(temp, 1.0)
            e13 = t_data.get(1350, {})
            e16 = t_data.get(1600, {})
            tot13 = max(1, e13.get("wins", 0) + e13.get("draws", 0) + e13.get("losses", 0))
            tot16 = max(1, e16.get("wins", 0) + e16.get("draws", 0) + e16.get("losses", 0))

            w13 = e13.get("wins", 0) / tot13
            w16 = e16.get("wins", 0) / tot16
            d13 = e13.get("draws", 0) / tot13
            d16 = e16.get("draws", 0) / tot16
            m13 = parse_mat_cp(e13)
            m16 = parse_mat_cp(e16)

            score += tw * ((w16 * 4.0) + (w13 * 2.0) + ((d13 + d16) * 0.75) + ((m13 + m16) / 1000.0 * 1.5))
        return score

    def fmt_cell(t_dict, elo):
        if not t_dict or elo not in t_dict:
            return "N/A"
        cell = t_dict[elo]
        return f"{cell['score_str']} ({cell['mat_str'].strip()})"

    ranked_arms = sorted(results, key=sort_key, reverse=True)
    for rank, (arm, res) in enumerate(ranked_arms, 1):
        ed = res["elo_data"]
        d0 = ed.get(0.0, {})
        d7 = ed.get(0.7, {})

        label = f"#{rank:<2} {arm['name']}"
        print(
            f"{label:<42} | {res['params']:<8,d} | {res['final_loss']:<6.4f} | {res['final_top1']:4.1f}% | "
            f"{fmt_cell(d0, 1350):<20} | {fmt_cell(d7, 1350):<20} | {fmt_cell(d0, 1600):<20} | {fmt_cell(d7, 1600):<20}"
        )
    print("=" * 175 + "\n")

    best_arm, best_res = ranked_arms[0]
    champion_path = "out/dialectic_patt_bot.pt"
    torch.save(
        {
            "model_state_dict": best_res["model"].state_dict(),
            "config": best_arm,
            "arm_name": best_arm["name"],
        },
        champion_path,
    )
    print(f"[Champion Artifact Saved] -> {champion_path} (from {best_arm['name']})\n")


if __name__ == "__main__":
    main()