import sys
import math
import importlib
import torch
import numpy as np

from study_platform import run_search
from study_generators import OptunaStudyWrapper
from parts import utils

def task_runner(mapped_vars, struct_kwargs, **task_args):
    """
    Generic task runner that maps dynamic and static variables to their clean config states,
    retranslates log-scale spaces, and runs short-horizon training.
    """
    batch_size = next(iter(mapped_vars.values())).shape[0]
    device = next(iter(mapped_vars.values())).device
    fitnesses = []
    
    # Extract inputs from task_args
    dataset = task_args['dataset']
    model_class = task_args['model_class']
    config_class = task_args['config_class']
    log_parameters = task_args['log_parameters']
    eval_steps = task_args.get('eval_steps', 40)

    for i in range(batch_size):
        # 1. Map continuous dynamic hyperparameters back to standard scales
        hps = {}
        for k, v in mapped_vars.items():
            val = v[i].item()
            if k in log_parameters:
                val = 10 ** val  # Revert from log10 base
            hps[k] = val
            
        # 2. Map static/structural parameters
        for k, val in struct_kwargs.items():
            hps[k] = val
            
        # 3. Instantiate model configuration
        cfg = config_class()
        for k, val in hps.items():
            if hasattr(cfg, k):
                setattr(cfg, k, val)
                
        # Handle StandardNodeNet3 tensor properties explicitly if present
        if 'untanh_slope' in hps:
            cfg.UNTANH = torch.tensor(hps['untanh_slope'])
        if 'c_moment' in hps:
            cfg.c_moment = torch.tensor(hps['c_moment'])
        if 'a_variance' in hps:
            cfg.a_variance = torch.tensor(hps['a_variance'])
            
        # 4. Instantiate model
        model = model_class(cfg).to(device)
        optimizer = model.configure_optimizers(lr=1e-3, wd=1e-1)
        
        losses = []
        nan_or_inf = False
        
        try:
            for step in range(eval_steps):
                X, Y = dataset('train')
                prefix_lens = torch.ones(X.size(0), dtype=torch.long, device=device)
                
                _, loss = model(X, prefix_lens, Y)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_or_inf = True
                    break
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                losses.append(loss.item())
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            nan_or_inf = True
            
        # Evaluate optimization fitness
        if nan_or_inf or len(losses) < 5:
            fitness = -999.0
        else:
            tail_steps = max(1, int(len(losses) * 0.3))
            avg_loss = sum(losses[-tail_steps:]) / tail_steps
            fitness = -avg_loss
            
        fitnesses.append(fitness)
        
        # Free runtime memory
        del model, optimizer
        torch.cuda.empty_cache()
        
    best_idx = np.argmax(fitnesses)
    hp_log_str = " | ".join([f"{k}={10**mapped_vars[k][best_idx].item() if k in log_parameters else mapped_vars[k][best_idx].item():.4f}" for k in mapped_vars.keys()])
    print(f"[Trial Batch] Max Fitness: {fitnesses[best_idx]:.4f} | Best HPs: {hp_log_str}")
          
    return np.array(fitnesses, dtype=np.float32)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m training_tools.hp_search <model_module>")
        sys.exit(1)
        
    module_name = sys.argv[1]
    print(f"Loading search environment module: {module_name}")
    model_module = importlib.import_module(module_name)
    
    # Retrieve decoupled model variables
    model_class = getattr(model_module, 'model')
    config_class = getattr(model_module, 'config')
    raw_config_space = getattr(model_module, 'config_space')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block_size = getattr(config_class, 'block_size', 64)
    batch_size = getattr(config_class, 'batch_size', 64)
    
    # 1. Translate the decoupled model spaces into the Study Platform's coordinates
    param_space = {}
    structural_space = {}
    defaults = {}
    bounds = {}
    log_parameters = {}
    
    for name, spec in raw_config_space.items():
        is_dynamic = (spec.get("type", "dynamic") == "dynamic")
        
        if is_dynamic:
            low = spec["low"]
            high = spec["high"]
            default = spec["default"]
            
            # Map log-scale spaces to linear parameters in the optimizer space
            if spec.get("scale", "linear") == "log" and low > 0:
                low_mapped = math.log10(low)
                high_mapped = math.log10(high)
                default_mapped = math.log10(default)
                log_parameters[name] = True
            else:
                low_mapped = low
                high_mapped = high
                default_mapped = default
                
            param_space[name] = {"low": low_mapped, "high": high_mapped}
            defaults[name] = default_mapped
            bounds[name] = {"low": low_mapped, "high": high_mapped}
        else:
            # Gather discrete static values
            values = spec.get("values", [spec.get("default", 0)])
            structural_space[name] = values
            defaults[name] = spec.get("default", values[0])
            bounds[name] = {"low": min(values), "high": max(values)}
            
    # Load dataset
    print(f"Initializing OpenWebText dataloader (Block size: {block_size}, Batch size: {batch_size})...")
    owt_get_batch = utils.get_get_batch(block_size, batch_size, "openwebtext", device, always_byte=True)
    
    task_args = {
        "dataset": owt_get_batch,
        "model_class": model_class,
        "config_class": config_class,
        "log_parameters": log_parameters,
        "eval_steps": 40
    }
    
    print("Beginning execution of continuous parameter space search...")
    run_search(
        task_runner=task_runner,
        param_space=param_space,
        structural_space=structural_space,
        defaults=defaults,
        bounds=bounds,
        device=torch.device(device),
        study_wrapper_cls=OptunaStudyWrapper,
        **task_args
    )