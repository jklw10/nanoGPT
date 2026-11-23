"""
Benchmark: AdamW vs. UnifiedKreinEntropicOptimizer (optimizers.kreinadam)
Ablates GRMS, Krein space, Kronecker sandwich, and stacking with CurlFreeLinear.
"""

from dataclasses import dataclass, replace
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimizers.kreinadam import UnifiedKreinEntropicOptimizer
from models.non_ml.triton_kmeans import run_engine_v11_triton, warmup_graphs
from parts.modules import HungryPattLayer
import parts.utils as utils

torch._inductor.config.disable_cpp_codegen = True
torch._dynamo.config.force_parameter_static_shapes = False
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 1024

# =====================================================================
# 1. Unified Config & Experiments
# =====================================================================

@dataclass(frozen=True)
class GPTConfig:
    # Data & Runtime
    dataset: str = "openwebtext"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    batch_size: int = 64
    block_size: int = 64
    vocab_size: int = 256

    # Model Architecture
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False

    # Optimization & Schedule
    opt_type: str = "adamw"              # "adamw" | "krein" | "sgd"
    sgd_momentum: float = 0.9
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_iters: int = 100
    lr_decay_iters: int = 30000
    decay_iters: int = 5000
    max_iters: int = 50000
    eval_interval: int = 1000
    eval_iters: int = 200

    # KreinAdam Flags
    krein_beta1: float = 0.9
    krein_beta2: float = 0.999
    krein_beta_cov: float = 0.1
    krein_alpha: float = 1.0
    krein_gamma: float = 1e-4
    krein_damping: float = 1e-5
    krein_kron_freq: int = 10
    krein_steps_ns: int = 2
    krein_power_iters: int = 2
    krein_use_adam: bool = True
    krein_use_kron: bool = False
    krein_use_space: bool = False
    krein_use_grms: bool = True

    # Extra GPT Flags
    spl: bool = False
    soloblock: bool = False
    sspl: bool = False
    wnll: bool = False
    patt: bool = False
    pattcount: int = 512

    # Optional Curl-Free Linear Replacement (for stacking test)
    linear_type: str = "nn"              # "nn" | "curl_free"
    cg_target: str = "mlp_proj"          # "mlp_proj" | "all_blocks" | "all"
    cf_space: str = "x"                  # "x" | "g" | "both"
    cf_mode: str = "qr_deflate"          # "qr_deflate" | "woodbury_damp" | "rare_boost"
    cf_k: int = 16
    cf_sub_n: int = 1024
    cf_robust_cap_q: float | None = 0.95
    cf_half_x: bool = True
    cf_gamma: float = 0.25


BASE_CFG = GPTConfig()

EXPERIMENTS = {
    #"adamw_nn":      (True, replace(BASE_CFG, opt_type="adamw", linear_type="nn")),
    #"krein_base":    (True, replace(BASE_CFG, opt_type="krein", krein_use_grms=True,  krein_use_space=False, krein_use_kron=False)),
    #"krein_no_grms": (True, replace(BASE_CFG, opt_type="krein", krein_use_grms=False, krein_use_space=False, krein_use_kron=False)),
    #"krein_space":   (True, replace(BASE_CFG, opt_type="krein", krein_use_grms=True,  krein_use_space=True,  krein_use_kron=False)),
    #"krein_kron":       (True, replace(BASE_CFG, opt_type="krein", krein_use_grms=True,  krein_use_space=False, krein_use_kron=True)),
    #"krein_cf_qr":      (True, replace(BASE_CFG, opt_type="krein", krein_use_grms=True,  krein_use_space=False, krein_use_kron=False, linear_type="curl_free", cf_gamma=0.25)),
    "krein_all":   (True, replace(BASE_CFG, opt_type="krein", krein_use_grms=True,  krein_use_space=True, krein_use_kron=True)),
    "krein_kron_cf_qr":      (True, replace(BASE_CFG, opt_type="krein", krein_use_grms=True,  krein_use_space=False, krein_use_kron=True, linear_type="curl_free", cf_gamma=0.25)),
    "krein_all_cf_qr":      (True, replace(BASE_CFG, opt_type="krein", krein_use_grms=True,  krein_use_space=True, krein_use_kron=True, linear_type="curl_free", cf_gamma=0.25)),
}

decode, vocab_size = utils.get_meta(BASE_CFG.dataset, always_byte=True)
get_batch = utils.get_get_batch(
    BASE_CFG.block_size, BASE_CFG.batch_size, BASE_CFG.dataset, BASE_CFG.device, always_byte=True
)


def get_lr(it: int, cfg: GPTConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

# =====================================================================
# 2. Curl-Free Self-Adjoint Linear Operator (for krein_cf_qr)
# =====================================================================

@torch.no_grad()
def _apply_qr_deflate(dW: torch.Tensor, acts: torch.Tensor, side: str, cfg: GPTConfig) -> torch.Tensor:
    if acts.size(0) < max(2 * cfg.cf_k, 16) or cfg.cf_gamma <= 0.0:
        return dW

    x_f = torch.nan_to_num(acts.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    x_scale = x_f.pow(2).mean().sqrt().clamp(min=1e-12)
    x_n = x_f / x_scale

    C = run_engine_v11_triton(
        x_n, k=cfg.cf_k, sub_n=cfg.cf_sub_n, robust_cap_q=cfg.cf_robust_cap_q,
        half_x=cfg.cf_half_x, use_graph=True,
    )
    c_half = 0.5 * (C * C).sum(dim=1)
    assign = torch.argmax(torch.addmm(-c_half, x_n, C.T), dim=1)
    cnt = torch.bincount(assign, minlength=C.size(0)).float()

    c_norm = C.norm(dim=1, keepdim=True)
    C_hat = torch.where(c_norm > 1e-6, C / c_norm.clamp(min=1e-6), torch.zeros_like(C))
    active_C = C_hat[cnt > 0]
    if active_C.size(0) == 0:
        return dW

    V, _ = torch.linalg.qr(active_C.T, mode="reduced")
    if side == "right":
        return dW - cfg.cf_gamma * ((dW @ V) @ V.T)
    return dW - cfg.cf_gamma * (V @ (V.T @ dW))


class CurlFreeLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, cfg: GPTConfig = BASE_CFG):
        super().__init__(in_features, out_features, bias=bias)
        self.cfg = cfg
        self._last_x: torch.Tensor | None = None
        self._last_g: torch.Tensor | None = None
        self.weight.register_hook(self._weight_grad_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_hook = self.training and torch.is_grad_enabled()
        if use_hook and self.cfg.cf_space in ("x", "both"):
            self._last_x = x.detach().reshape(-1, x.size(-1))
        y = F.linear(x, self.weight, self.bias)
        if use_hook and self.cfg.cf_space in ("g", "both"):
            y.register_hook(self._capture_output_grad)
        return y

    def _capture_output_grad(self, g: torch.Tensor):
        self._last_g = g.detach().reshape(-1, g.size(-1))

    @torch.no_grad()
    def _weight_grad_hook(self, dW: torch.Tensor) -> torch.Tensor:
        out = dW.float()
        if self.cfg.cf_space in ("x", "both") and self._last_x is not None:
            out = _apply_qr_deflate(out, self._last_x, side="right", cfg=self.cfg)
        if self.cfg.cf_space in ("g", "both") and self._last_g is not None:
            out = _apply_qr_deflate(out, self._last_g, side="left", cfg=self.cfg)
        self._last_x = None
        self._last_g = None
        return out.to(dW.dtype)


def make_linear(cfg: GPTConfig, in_features: int, out_features: int, bias: bool = False, role: str = "other") -> nn.Module:
    use_cf = (
        cfg.linear_type == "curl_free"
        and (
            cfg.cg_target == "all"
            or (cfg.cg_target == "all_blocks" and role != "lm_head")
            or (cfg.cg_target == "mlp_proj" and role == "mlp_proj")
        )
    )
    if use_cf:
        return CurlFreeLinear(in_features, out_features, bias=bias, cfg=cfg)
    return nn.Linear(in_features, out_features, bias=bias)

# =====================================================================
# 3. GPT Model & Optimizer Factory
# =====================================================================

class LayerNorm(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Attention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.q_attn = make_linear(config, config.n_embd, config.n_embd, bias=config.bias, role="attn")
        self.k_attn = make_linear(config, config.n_embd, config.n_embd, bias=config.bias, role="attn")
        self.v_attn = make_linear(config, config.n_embd, config.n_embd, bias=config.bias, role="attn")
        self.c_proj = make_linear(config, config.n_embd, config.n_embd, bias=config.bias, role="attn")

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = self.n_embd // self.n_head
        self.qm = nn.Parameter(torch.ones(self.head_dim).normal_(mean=1, std=0.1))

    def forward(self, x, causal=True):
        B, T, C = x.size()
        q = self.q_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_attn(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q_scale = math.log(T) * self.qm
        y = F.scaled_dot_product_attention(
            q * q_scale, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.use_patt = config.patt
        up = 4

        self.c_fc = make_linear(config, config.n_embd, up * config.n_embd, bias=config.bias, role="mlp_fc")
        self.gelu = nn.GELU()
        self.c_proj = make_linear(config, up * config.n_embd, config.n_embd, bias=config.bias, role="mlp_proj")
        self.dropout = nn.Dropout(config.dropout)

        if self.use_patt:
            self.patt = HungryPattLayer(config.n_embd, config.n_embd, config.pattcount)

    def forward(self, x):
        if self.use_patt:
            return self.patt(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.use_spl = config.spl
        self.attn = Attention(config)
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.register_buffer("block_input", torch.zeros([config.block_size, config.n_embd]))

    def forward(self, x, causal=True, fwd=True):
        if fwd and self.use_spl:
            self.block_input = x
        x = x + self.attn(self.ln_1(x), causal)
        x = x + self.mlp(self.ln_2(x))
        return x

    def spl(self, x):
        predicted_output = self(-x, causal=False, fwd=False)
        return F.mse_loss(
            utils.rms_norm(predicted_output.flatten()),
            utils.rms_norm(self.block_input.flatten()),
        )


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.n_embd = config.n_embd
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config),
        ))

        self.lm_head = make_linear(config, config.n_embd, config.vocab_size, bias=False, role="lm_head")
        self.transformer.wte.weight = self.lm_head.weight
        self.wnl = utils.wnormlearnloss(config.n_embd)

        self.apply(self._init_weights)
        for pn, module in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(module, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None, decaying=0.0):
        device = idx.device
        b, t = idx.size()

        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        if self.training and self.config.wnll:
            wnl = self.wnl(x)
        x1 = x
        for block in self.transformer.h:
            x = block(x)

        if isinstance(x, tuple):
            x = x[0]

        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            reduce = "none" if self.config.wnll else "mean"
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction=reduce)

            if self.training:
                if self.config.wnll:
                    mean_surprise = ((1.0 - wnl.detach()) / 2.0).mean() * 0.99
                    loss = loss / (1 + wnl.detach() * mean_surprise)
                    loss = (loss + wnl).mean()
                if self.config.spl:
                    if self.config.sspl:
                        self.transformer.h[1].block_input = x1
                        loss = loss + self.transformer.h[1].spl(x)
                    else:
                        loss = loss + self.transformer.h[0].spl(x)
            else:
                if self.config.wnll:
                    loss = loss.mean()
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Embedding, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def create_optimizer(model: nn.Module, cfg: GPTConfig):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

    if cfg.opt_type == "krein":
        optim_groups = [
            {"params": nodecay_params, "weight_decay": 0.0, "gamma": 0.0},
            {"params": decay_params, "weight_decay": cfg.weight_decay, "gamma": cfg.krein_gamma},
        ]
        return UnifiedKreinEntropicOptimizer(
            optim_groups,
            lr=cfg.learning_rate,
            beta1=cfg.krein_beta1,
            beta2=cfg.krein_beta2,
            beta_cov=cfg.krein_beta_cov,
            alpha=cfg.krein_alpha,
            gamma=cfg.krein_gamma,
            damping=cfg.krein_damping,
            weight_decay=cfg.weight_decay,
            kron_freq=cfg.krein_kron_freq,
            steps_ns=cfg.krein_steps_ns,
            power_iters=cfg.krein_power_iters,
            use_adam=cfg.krein_use_adam,
            use_kron=cfg.krein_use_kron,
            use_krein_space=cfg.krein_use_space,
            use_grms=cfg.krein_use_grms,
        )

    optim_groups = [
        {"params": nodecay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": cfg.weight_decay},
    ]
    if cfg.opt_type == "sgd":
        return torch.optim.SGD(optim_groups, lr=cfg.learning_rate, momentum=cfg.sgd_momentum)
    return torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)

# =====================================================================
# 4. Benchmark Orchestrator
# =====================================================================

def evaluate_model(raw_model: GPT, val_batches):
    raw_model.eval()
    tot_ce, tot_gen = 0.0, 0.0
    with torch.no_grad():
        for vx, vy in val_batches:
            logits, _ = raw_model(vx, vy)
            tot_ce += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), vy.reshape(-1)).item()
            tot_gen += F.cross_entropy(logits[:, -1, :], vy[:, -1]).item()
    raw_model.train()
    n = len(val_batches)
    return tot_ce / n, tot_gen / n


def run_benchmarks():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    active = {
        k: replace(cfg, vocab_size=vocab_size)
        for k, (enabled, cfg) in EXPERIMENTS.items()
        if enabled
    }
    print(f"Device: {BASE_CFG.device} | Vocab: {vocab_size} | Experiments: {list(active.keys())}")

    val_batches = [get_batch("val", BASE_CFG.block_size) for _ in range(BASE_CFG.eval_iters)]

    if BASE_CFG.device == "cuda":
        graph_groups = {}
        for cfg in active.values():
            if cfg.linear_type == "curl_free":
                key = (cfg.cf_sub_n, cfg.cf_half_x)
                dims = graph_groups.setdefault(key, set())
                if cfg.cg_target == "mlp_proj":
                    if cfg.cf_space in ("x", "both"):
                        dims.add((cfg.cf_k, 4 * cfg.n_embd))
                    if cfg.cf_space in ("g", "both"):
                        dims.add((cfg.cf_k, cfg.n_embd))
                else:
                    dims.add((cfg.cf_k, cfg.n_embd))
                    dims.add((cfg.cf_k, 4 * cfg.n_embd))
                    if cfg.cg_target == "all":
                        dims.add((cfg.cf_k, cfg.vocab_size))
        for (sub_n, half_x), kd_set in graph_groups.items():
            if kd_set:
                print(f"Pre-recording Triton CUDA graph for {sorted(kd_set)} (sub_n={sub_n}, half_x={half_x})...")
                warmup_graphs(sorted(kd_set), sub_n=sub_n, half_x=half_x, device=BASE_CFG.device)

    raw_models, models, optimizers, cfgs = {}, {}, {}, {}
    failed = set()
    bx_sanity, by_sanity = get_batch("train", BASE_CFG.block_size)

    for name, cfg in active.items():
        torch.manual_seed(cfg.seed)
        m = GPT(cfg).to(cfg.device)
        raw_models[name] = m
        cfgs[name] = cfg
        optimizers[name] = create_optimizer(m, cfg)

        if cfg.linear_type == "nn":
            t0 = time.perf_counter()
            compiled_m = torch.compile(m)
            for _ in range(2):
                optimizers[name].zero_grad(set_to_none=True)
                _, loss = compiled_m(bx_sanity, by_sanity)
                loss.backward()
                optimizers[name].zero_grad(set_to_none=True)
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            models[name] = compiled_m
            print(f"  [COMPILED]  {name:<18} in {(time.perf_counter() - t0):.2f} s")
        else:
            models[name] = m
            print(f"  [EAGER+CF]  {name:<18} ready")

    col_widths = {n: max(len(n), 16) for n in models.keys()}
    print("=" * 135)
    print(f"{'Step':<6} | " + " | ".join([f"{n:<{col_widths[n]}}" for n in models.keys()]) + " | ms/step")
    print(f"{'':<6} | " + " | ".join([f"{' val_ce    gen':<{col_widths[n]}}" for n in models.keys()]) + " |")
    print("-" * 135)

    interval_timing = {n: 0.0 for n in models.keys()}

    for it in range(BASE_CFG.max_iters):
        bx, by = get_batch("train", BASE_CFG.block_size)
        step = it + 1
        is_eval = (step % BASE_CFG.eval_interval == 0 or step == BASE_CFG.max_iters)

        for name, model in models.items():
            if name in failed:
                continue

            cfg = cfgs[name]
            opt = optimizers[name]
            lr = get_lr(it, cfg)
            for pg in opt.param_groups:
                pg["lr"] = lr

            decaying = max(0.0, min(1.0, 1.0 - (it / cfg.decay_iters)))

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            _, loss = model(bx, by, decaying=decaying)

            if not torch.isfinite(loss):
                print(f"[NON-FINITE] {name} diverged at step {step}: loss={loss.item()}")
                failed.add(name)
                continue

            loss.backward()
            opt.step()

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            interval_timing[name] += (time.perf_counter() - t0) * 1000.0

        if is_eval:
            eval_cells, ms_cells = [], []
            for name, rm in raw_models.items():
                if name in failed:
                    eval_cells.append(f"{'FAILED':<{col_widths[name]}}")
                    ms_cells.append("  ---")
                else:
                    v_ce, v_gen = evaluate_model(rm, val_batches)
                    eval_cells.append(f"{f'{v_ce:7.4f} {v_gen:7.4f}':<{col_widths[name]}}")
                    ms_cells.append(f"{interval_timing[name] / BASE_CFG.eval_interval:5.1f}")
                    interval_timing[name] = 0.0
            print(f"{step:<6d} | " + " | ".join(eval_cells) + " | " + " / ".join(ms_cells))


if __name__ == "__main__":
    run_benchmarks()