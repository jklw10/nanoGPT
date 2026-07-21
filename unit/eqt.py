import torch
import random

from models.mazenode4 import (
        StatelessNodeNet3,
        MazeEnvTensor,
        ExecutionConfig,
        HyperParamsConfig,
        successful_params,
        step_kernel,
        raymarch_grid,
        get_obs_kernel
    )
import parts.utils
# CONFIGURATION
# ==========================================
# Match the exact total batch size used in the ES Rollout (12 population * 64 batch size)
BATCH_SIZE = 768  
MAZE_SIZE = 10
NUM_NODES = 32
DIM = 16
NUM_ITERS = 100
WARMUP_ITERS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmarking on: {device}")
print(f"Batch Size: {BATCH_SIZE} | Maze Size: {MAZE_SIZE}x{MAZE_SIZE}\n")

# Initialize Env & Model
hyper_cfg = HyperParamsConfig(device, init_dict=successful_params)
env = MazeEnvTensor(batch_size=BATCH_SIZE, size=MAZE_SIZE, food_density=0.15, device=device)
model = StatelessNodeNet3(num_nodes=NUM_NODES, dim=DIM, device=device).to(device)
model.apply_hyperparams(hyper_cfg)

# Initialize States
state, output, error_ema, lp_ema = model.init_states(BATCH_SIZE)
obs = env._get_obs()
actions = torch.randint(0, 4, (BATCH_SIZE,), device=device)

# Ensure no gradients are tracked during inference benchmarking
torch.set_grad_enabled(False)

# ==========================================
# TIMING UTILITY (CUDA-Native Timing)
# ==========================================
def benchmark_op(name, op_func, num_iters=NUM_ITERS, warmup=WARMUP_ITERS):
    # Warmup iterations to let CUDA compile & optimize kernels
    for _ in range(warmup):
        op_func()
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        op_func()
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / num_iters
    print(f"  {name:<40} | Avg: {avg_ms:8.3f} ms | Total ({num_iters} iterations): {elapsed_ms/1000:6.3f}s")
    return avg_ms

# ==========================================
# TEST RUNNERS FOR EACH PIECE
# ==========================================
def test_raymarch():
    raymarch_grid(env.agent_pos, env.grid, env.food_grid, env.size, max_dist=env.size)

def test_get_obs():
    get_obs_kernel(env.agent_pos, env.hunger, env.grid, env.food_grid, env.size)

def test_step_kernel():
    # Execute the underlying step physics and movement
    step_kernel(env.agent_pos, env.food_grid, env.hunger, env.steps, actions, env.grid, env.size)

def test_masked_reset():
    # Artificially trigger resets across half the batch to measure reset overhead
    env.needs_reset[:BATCH_SIZE // 2] = True
    env._masked_reset()

def test_env_step():
    env.step(actions)

def test_sparse_routing():
    # Profile routing and STE quantizations specifically
    route_logits = torch.randn(BATCH_SIZE, NUM_NODES, NUM_NODES, device=device)
    from models.mazenode4 import sparse_route_diffusion
    sparse_route_diffusion(route_logits, NUM_NODES)

def test_model_forward():
    model(obs, state, output, error_ema, lp_ema)

def test_action_selection():
    logits = model.get_action_logits(state, 4)
    torch.argmax(logits, dim=-1)

def test_full_rollout_step():
    global obs, state, output, error_ema, lp_ema
    from models.mazenode4 import model_and_env_step
    # Profile a unified network + environment step
    _, state, output, error_ema, lp_ema, _, _, obs = model_and_env_step(
        model, env, obs, state, output, error_ema, lp_ema
    )

# ==========================================
# EXECUTE BENCHMARKS
# ==========================================
print("=== Starting Detailed Component Profiling ===")

# 1. Environment & Physics Components
benchmark_op("1. step_kernel (Movement Logic)", test_step_kernel)
benchmark_op("2. raymarch_grid (Raycasting)", test_raymarch)
benchmark_op("3. get_obs_kernel (Observation Assembly)", test_get_obs)
benchmark_op("4. env._masked_reset (Conditional Resets)", test_masked_reset)
benchmark_op("5. env.step (Whole Env Lifecycle)", test_env_step)

print("-" * 70)

# 2. Model & Routing Components
benchmark_op("6. sparse_route_diffusion / STE Quantization", test_sparse_routing)
benchmark_op("7. model.forward (Neural Pipeline)", test_model_forward)
benchmark_op("8. action selection (Logits & Argmax)", test_action_selection)

print("-" * 70)

# 3. Complete Step Combined
benchmark_op("9. One Complete Rollout Step (Combined)", test_full_rollout_step)

print("=============================================")
print("Profiling completed. Analyze the average ms times to identify performance anomalies.")