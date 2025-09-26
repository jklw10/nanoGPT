import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit

import quantizer

def benchmark(model, input_tensor, model_name):
    """Measures forward and backward pass time for a given model."""
    print(f"\n--- Benchmarking {model_name} ---")
    
    # Ensure model and tensor are on the correct device
    device = input_tensor.device
    model.to(device)
    
    # --- Forward Pass ---
    # Use torch.no_grad for a fair forward-only measurement
    with torch.no_grad():
        # Warm-up run
        for _ in range(10):
            _ = model(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        # Timed run
        start_time = timeit.default_timer()
        for _ in range(100):
            _ = model(input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        
        forward_ms = (end_time - start_time) / 100 * 1000
        print(f"Forward pass:  {forward_ms:.4f} ms")

    # --- Backward Pass ---
    # Warm-up run
    for _ in range(10):
        output = model(input_tensor)
        dummy_loss = output.mean()
        dummy_loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed run
    start_time = timeit.default_timer()
    for _ in range(100):
        # We must include the forward pass as it's part of the backward computation graph
        output = model(input_tensor)
        dummy_loss = output.mean()
        dummy_loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = timeit.default_timer()
    
    # This time includes both forward and backward
    full_pass_ms = (end_time - start_time) / 100 * 1000
    print(f"Forward + Backward pass: {full_pass_ms:.4f} ms")


if __name__ == '__main__':
    # --- Setup ---
    BATCH_SIZE = 512
    NUM_FEATURES = 1024
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {DEVICE}")

    input_data = torch.randn(BATCH_SIZE, NUM_FEATURES).to(DEVICE)

    # --- Models ---
    # Using degree=3 for B-spline, a common choice
    bspline_act = quantizer.BSplineActivation(in_features=NUM_FEATURES, n_basis=20, degree=3)
    taylor_act = quantizer.TaylorThresHotActivation(in_features=NUM_FEATURES, n_experts=20, degree=8)

    # --- Run Benchmarks ---
    benchmark(bspline_act, input_data, "B-Spline Activation (degree=3)")
    benchmark(taylor_act, input_data, "Taylor ThresHot Activation")

    # Let's see how much worse a higher degree is for B-spline
    bspline_act_deg5 = quantizer.BSplineActivation(in_features=NUM_FEATURES, n_basis=20, degree=8)
    benchmark(bspline_act_deg5, input_data, "B-Spline Activation (degree=8)")