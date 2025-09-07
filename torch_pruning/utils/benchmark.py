"""Benchmarking utilities for measuring model performance."""

import torch


def measure_latency(model, example_inputs, repeat=300, warmup=50, run_fn=None):
    """Measure model inference latency.
    
    Args:
        model: PyTorch model to benchmark.
        example_inputs: Input tensor(s) for the model.
        repeat: Number of inference runs for measurement. Defaults to 300.
        warmup: Number of warmup runs. Defaults to 50.
        run_fn: Custom function to run the model. If None, uses model(example_inputs).
        
    Returns:
        Tuple of (mean_latency_ms, std_latency_ms).
    """
    model.eval()
    latency = []
    
    # Warmup runs
    for _ in range(warmup):
        if run_fn is not None:
            _ = run_fn(model, example_inputs)
        else:
            _ = model(example_inputs)

    # Measurement runs
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if run_fn is not None:
            _ = run_fn(model, example_inputs)
        else:
            _ = model(example_inputs)
        end.record()
        torch.cuda.synchronize()
        latency.append(start.elapsed_time(end))
        
    latency = torch.tensor(latency)
    return latency.mean().item(), latency.std().item()

def measure_memory(model, example_inputs, device=None, run_fn=None):
    """Measure peak memory consumption during model inference.
    
    Reference: https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
    
    Args:
        model: PyTorch model to benchmark.
        example_inputs: Input tensor(s) for the model.
        device: CUDA device to measure memory on. Defaults to None.
        run_fn: Custom function to run the model. If None, uses model(example_inputs).
        
    Returns:
        Peak memory allocated in bytes.
    """
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    if run_fn is not None:
        _ = run_fn(model, example_inputs)
    else:
        _ = model(example_inputs)
    return torch.cuda.max_memory_allocated(device=device)


def measure_fps(model, example_inputs, repeat=300, warmup=50, run_fn=None):
    """Measure frames per second (FPS) of model inference.
    
    Args:
        model: PyTorch model to benchmark.
        example_inputs: Input tensor(s) for the model.
        repeat: Number of inference runs for measurement. Defaults to 300.
        warmup: Number of warmup runs. Defaults to 50.
        run_fn: Custom function to run the model. If None, uses model(example_inputs).
        
    Returns:
        Frames per second.
    """
    latency_mu, _ = measure_latency(model, example_inputs, repeat=repeat, warmup=warmup, run_fn=run_fn)
    fps = 1000.0 / latency_mu  # 1000 ms = 1 s
    return fps


def measure_throughput(model, example_inputs, repeat=300, warmup=50, run_fn=None):
    """Measure throughput (samples per second) of model inference.
    
    Args:
        model: PyTorch model to benchmark.
        example_inputs: Input tensor(s) for the model.
        repeat: Number of inference runs for measurement. Defaults to 300.
        warmup: Number of warmup runs. Defaults to 50.
        run_fn: Custom function to run the model. If None, uses model(example_inputs).
        
    Returns:
        Throughput in samples per second.
    """
    latency_mu, _ = measure_latency(model, example_inputs, repeat=repeat, warmup=warmup, run_fn=run_fn)
    throughput = example_inputs.shape[0] / (latency_mu / 1000)
    return throughput