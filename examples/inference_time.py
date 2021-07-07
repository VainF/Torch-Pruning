import sys, os
from torchvision.models.resnet import resnet50
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch_pruning as tp
import time

def measure_inference_time(net, input, repeat=100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        model(input)
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end-start) / repeat
 
device = torch.device('cuda') # or torch.device('cpu')
repeat = 100

# w/o rounding
model = resnet50(pretrained=True).eval()
fake_input = torch.randn(16,3,224,224)
model = model.to(device)
fake_input = fake_input.to(device)
inference_time_before_pruning = measure_inference_time(model, fake_input, repeat)
print("before pruning: inference time=%f s, parameters=%d"%(inference_time_before_pruning, tp.utils.count_params(model)))

model = resnet50(pretrained=True).eval()
strategy = tp.strategy.L1Strategy()
DG = tp.DependencyGraph()
fake_input = fake_input.cpu()
DG.build_dependency(model, example_inputs=fake_input)
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        pruning_idxs = strategy(m.weight, amount=0.2)
        pruning_plan = DG.get_pruning_plan( m, tp.prune_conv, idxs=pruning_idxs )
        pruning_plan.exec()
model = model.to(device)
fake_input = fake_input.to(device)
inference_time_without_rounding = measure_inference_time(model, fake_input, repeat)
print("w/o rounding: inference time=%f s, parameters=%d"%(inference_time_without_rounding, tp.utils.count_params(model)))
    
# w/ rounding
model = resnet50(pretrained=True).eval()
strategy = tp.strategy.L1Strategy()
DG = tp.DependencyGraph()
fake_input = fake_input.cpu()
DG.build_dependency(model, example_inputs=fake_input)
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        pruning_idxs = strategy(m.weight, amount=0.2, round_to=16)
        pruning_plan = DG.get_pruning_plan( m, tp.prune_conv, idxs=pruning_idxs )
        pruning_plan.exec()
model = model.to(device)
fake_input = fake_input.to(device)
inference_time_with_rounding = measure_inference_time(model, fake_input, repeat)
print("w/ rounding: inference time=%f s, parameters=%d"%(inference_time_with_rounding, tp.utils.count_params(model)))

