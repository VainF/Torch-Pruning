import torch

# Latency
def measure_latency(model, example_inputs, repeat=300, warmup=50, run_fn=None):
    model.eval()
    latency = []
    for _ in range(warmup):
        if run_fn is not None:
            _ = run_fn(model, example_inputs)
        else:
            _ = model(example_inputs)

    for i in range(repeat):
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

# Memory Consumption
def measure_memory(model, example_inputs, device=None, run_fn=None):
    """ https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
    """
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    if run_fn is not None:
        _ = run_fn(model, example_inputs)
    else:
        _ = model(example_inputs)
    return torch.cuda.max_memory_allocated(device=device)

# Frame (Batch) per Second
def measure_fps(model, example_inputs, repeat=300, warmup=50, run_fn=None):
    latency_mu, latency_std = measure_latency(model, example_inputs, repeat=repeat, warmup=warmup, run_fn=run_fn)
    fps = 1000.0 / latency_mu # 1000 ms = 1 s
    return fps

# Throughput
def measure_throughput(model, example_inputs, repeat=300, warmup=50, run_fn=None):
    latency_mu, latency_std = measure_latency(model, example_inputs, repeat=repeat, warmup=warmup, run_fn=run_fn)
    throughput = example_inputs.shape[0] / (latency_mu/1000)
    return throughput