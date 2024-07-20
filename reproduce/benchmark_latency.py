from torchvision.models import resnet50 as model_entry
import sys, os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch_pruning as tp
import torch


def main():

    model = model_entry(pretrained=True).to('cuda:0')
    example_input = torch.rand(32, 3, 224, 224).to('cuda:0')
    importance = tp.importance.MagnitudeImportance(p=2)
    iterative_steps = 20
    pruner = tp.pruner.MagnitudePruner(
        model = model,
        example_inputs=example_input,
        importance=importance,
        pruning_ratio=1,
        iterative_steps=iterative_steps,
        round_to=2,
    )

    # Before Pruning
    macs, params = tp.utils.count_ops_and_params(model, example_input)
    latency_mu, latency_std = estimate_latency(model, example_input)
    # print all with .2f
    print(f"[Iter 0] \tPruning ratio: 0.00, \tMACs: {macs/1e9:.2f} G, \tParams: {params/1e6:.2f} M, \tLatency: {latency_mu:.2f} ms +- {latency_std:.2f} ms")

    for iter in range(iterative_steps):
        pruner.step()
        _macs, _params = tp.utils.count_ops_and_params(model, example_input)
        latency_mu, latency_std = estimate_latency(model, example_input)
        current_pruning_ratio = 1 / iterative_steps * (iter + 1)
        print(f"[Iter {iter+1}] \tPruning ratio: {current_pruning_ratio:.2f}, \tMACs: {_macs/1e9:.2f} G, \tParams: {_params/1e6:.2f} M, \tLatency: {latency_mu:.2f} ms +- {latency_std:.2f} ms")

        # uncomment the following lines to profile
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #    with torch.no_grad():
        #            for _ in range(50):
        #                _ = model(example_input)
        #print(prof)

def estimate_latency(model, example_inputs, repetitions=50):
    import numpy as np
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((repetitions,1))

    for _ in range(5):
        _ = model(example_inputs)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(example_inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn

if __name__=='__main__':
    main()