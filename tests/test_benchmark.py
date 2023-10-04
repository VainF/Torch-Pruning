import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torchvision.models import resnet18
import torch
import torch_pruning as tp

device = torch.device('cuda:0')
model = resnet18().eval().to(device)
example_inputs = torch.randn(32, 3, 224, 224).to(device)

# Test forward in eval mode
print("====== Forward (Inferece with torch.no_grad) ======")
with torch.no_grad():
    laterncy_mu, latency_std= tp.utils.benchmark.measure_latency(model, example_inputs, repeat=300)
    print('laterncy: {:.4f} +/- {:.4f} ms'.format(laterncy_mu, latency_std))

    memory = tp.utils.benchmark.measure_memory(model, example_inputs, device=device)
    print('memory: {:.4f} MB'.format(memory/ (1024)**2))

    example_inputs_bs1 = torch.randn(1, 3, 224, 224).to(device)
    fps = tp.utils.benchmark.measure_fps(model, example_inputs_bs1, repeat=300)
    print('fps: {:.4f}'.format(fps))

    example_inputs = torch.randn(256, 3, 224, 224).to(device)
    throughput = tp.utils.benchmark.measure_throughput(model, example_inputs, repeat=300)
    print('throughput (bz=256): {:.4f} images/s'.format(throughput))

print("====== Forward & Backward ======")
# Test forward & backward
def run_fn(model, example_inputs):
    output = model(example_inputs)
    loss = output.sum()
    loss.backward()
    return loss

laterncy_mu, latency_std= tp.utils.benchmark.measure_latency(model, example_inputs, repeat=300, run_fn=run_fn)
print('laterncy: {:.4f} +/- {:.4f} ms'.format(laterncy_mu, latency_std))

memory = tp.utils.benchmark.measure_memory(model, example_inputs, device=device, run_fn=run_fn)
print('memory: {:.4f} MB'.format(memory/ (1024)**2))

example_inputs_bs1 = torch.randn(1, 3, 224, 224).to(device)
fps = tp.utils.benchmark.measure_fps(model, example_inputs_bs1, repeat=300, run_fn=run_fn)
print('fps: {:.4f}'.format(fps))

example_inputs = torch.randn(256, 3, 224, 224).to(device)
throughput = tp.utils.benchmark.measure_throughput(model, example_inputs, repeat=300, run_fn=run_fn)
print('throughput (bz=256): {:.4f} images/s'.format(throughput))