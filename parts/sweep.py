# parts/sweep.py
from contextlib import contextmanager
import math
import random
import time
import torch
import torch.nn as nn


# =====================================================================
# 1. Environment, Determinism & Precision
# =====================================================================
def setup_torch(matmul_precision: str = "high", disable_cpp_codegen: bool = True, allow_tf32: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision(matmul_precision)
    if device.type == "cuda":
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    else:
        amp_dtype = torch.float32

    if disable_cpp_codegen:
        try:
            import torch._inductor.config as inductor_cfg
            inductor_cfg.disable_cpp_codegen = True
        except Exception:
            pass
    return device, amp_dtype


def seed_everything(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_compiled():
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        torch.compiler.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
# 2. Experiment Registry / Arm Filtering
# =====================================================================
def select_arms(arms, only=None, skip=None):
    """
    Supports either:
      - list of dicts (respecting arm.get('enabled', True))
      - dict of {name: bool} or {name: cfg_dict}
    Optional CLI filters `only` / `skip` (string or list of substrings).
    """
    if isinstance(arms, dict):
        items = []
        for k, v in arms.items():
            if isinstance(v, bool):
                if v:
                    items.append({"name": k})
            elif isinstance(v, dict):
                if v.get("enabled", True):
                    items.append({"name": k, **v})
    else:
        items = [dict(a) for a in arms if a.get("enabled", True)]

    if isinstance(only, str):
        only = [only]
    if isinstance(skip, str):
        skip = [skip]

    if only:
        items = [a for a in items if any(s.lower() in a["name"].lower() for s in only)]
    if skip:
        items = [a for a in items if not any(s.lower() in a["name"].lower() for s in skip)]
    return items


# =====================================================================
# 3. Optimizer Grouping & Cosine Warmup Schedule
# =====================================================================
def split_decay_params(
    model: nn.Module,
    lr: float,
    weight_decay: float = 1e-4,
    embed_lr: float = None,
    extra_no_decay: tuple = ("embed", "castling_proj"),
):
    """
    Transformer-safe parameter grouping:
      - 0.0 weight decay on <2D params (norms, biases, LayerScale) and embeddings
      - Optional separate LR for >=2D embedding/projection weights
    """
    raw = getattr(model, "_orig_mod", model)
    embed_params, no_decay_params, decay_params = [], [], []
    seen = set()

    for name, p in raw.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_embed_mat = (p.ndim >= 2) and any(tok in name for tok in extra_no_decay)
        if is_embed_mat and embed_lr is not None:
            embed_params.append(p)
        elif p.ndim < 2 or is_embed_mat:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    if embed_params:
        groups.append({"params": embed_params, "lr": embed_lr, "weight_decay": 0.0})
    return groups


def cosine_warmup_scheduler(
    optimizer,
    total_steps: int,
    warmup_frac: float = 0.05,
    warmup_steps: int = None,
    min_lr_ratio: float = 0.02,
):
    w_steps = warmup_steps if warmup_steps is not None else int(total_steps * warmup_frac)

    def lr_lambda(step: int):
        if w_steps > 0 and step < w_steps:
            return float(step + 1) / float(w_steps)
        denom = max(1, total_steps - w_steps)
        progress = min(1.0, max(0.0, float(step - w_steps) / float(denom)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# 4. Compile + Clean Warmup & CUDA Timing
# =====================================================================
def cudagraph_step():
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()


def compile_and_warmup(
    model: nn.Module,
    step_fn,
    mode: str = "reduce-overhead",
    warmup_iters: int = 2,
    optimizer=None,
    restore_state: bool = True,
    clear_opt_state: bool = False,
    device: str = "cuda",
):
    """
    Compiles `model`, runs `warmup_iters` of `step_fn(compiled_model)` to prime
    static CUDA graphs, then restores initial weights so warmup doesn't leak updates.
    """
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if restore_state else None
    compiled = torch.compile(model, mode=mode)

    for _ in range(warmup_iters):
        cudagraph_step()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        step_fn(compiled)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        if clear_opt_state:
            optimizer.state.clear()
    if restore_state and init_state is not None:
        model.load_state_dict(init_state)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return compiled


@contextmanager
def cuda_timer(device="cuda"):
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = {"ms": 0.0}
    try:
        yield res
    finally:
        if is_cuda:
            torch.cuda.synchronize()
        res["ms"] = (time.perf_counter() - t0) * 1000.0


# =====================================================================
# 5. Throughput Meter, ETA & Leaderboard
# =====================================================================
def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"


class ThroughputMeter:
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.processed = 0
        self.t0 = time.perf_counter()

    def reset(self):
        self.processed = 0
        self.t0 = time.perf_counter()

    def update(self, n_items: int):
        self.processed += n_items

    def stats(self):
        elapsed = max(1e-4, time.perf_counter() - self.t0)
        rate = self.processed / elapsed
        eta = max(0, self.total_items - self.processed) / max(1e-4, rate)
        return rate, format_eta(elapsed), format_eta(eta)


class Leaderboard:
    def __init__(self, title: str, columns: list, name_width: int = 38):
        """
        columns: list of (header, key_or_fn, width, fmt_spec)
        """
        self.title = title
        self.columns = columns
        self.name_width = name_width
        self.rows = []

    def add(self, name: str, metrics: dict, score: float = 0.0, payload=None):
        self.rows.append({"name": name, "metrics": metrics, "score": score, "payload": payload})

    def render(self, top_k: int = None):
        ranked = sorted(self.rows, key=lambda r: r["score"], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]

        nw = self.name_width
        parts = [f"{'Rank':<4}", f"{'Arm':<{nw}}"] + [f"{h:<{max(w, len(h))}}" for h, _, w, _ in self.columns]
        header = " | ".join(parts)
        bar = "=" * len(header)

        print(f"\n{bar}\n{self.title.center(len(header))}\n{bar}\n{header}\n{'-' * len(header)}")
        for idx, row in enumerate(ranked, 1):
            cells = [f"#{idx:<3}", f"{row['name'][:nw]:<{nw}}"]
            m = row["metrics"]
            for h, getter, w, fmt in self.columns:
                col_w = max(w, len(h))
                val = getter(m) if callable(getter) else m[getter]
                if val is None:
                    s_val = "N/A"
                else:
                    try:
                        s_val = format(val, fmt)
                    except (ValueError, TypeError):
                        s_val = str(val)
                cells.append(f"{s_val:<{col_w}}")
            print(" | ".join(cells))
        print(bar + "\n")
        return ranked


# =====================================================================
# 6. Lockstep Multi-Model Runner
# =====================================================================
class LockstepArena:
    """Runs multiple models side-by-side on identical batches with per-model timing."""
    def __init__(
        self,
        models_dict: dict,
        optimizers_dict: dict,
        device="cuda",
        compile_models: bool = True,
        compile_mode: str = "default",
        grad_clip: float = None,
    ):
        self.raw = models_dict
        self.optimizers = optimizers_dict
        self.device = device
        self.grad_clip = grad_clip
        self.compiled = {
            k: (torch.compile(m, mode=compile_mode) if compile_models else m)
            for k, m in models_dict.items()
        }
        self.total_ms = {k: 0.0 for k in models_dict}

    def step_all(self, loss_fn):
        """loss_fn(name, model) -> scalar loss tensor."""
        out = {}
        for name, model in self.compiled.items():
            opt = self.optimizers[name]
            with cuda_timer(self.device) as t:
                cudagraph_step()
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(name, model)
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.raw[name].parameters(), max_norm=self.grad_clip)
                opt.step()
            self.total_ms[name] += t["ms"]
            out[name] = (float(loss.detach().item()), t["ms"])
        return out