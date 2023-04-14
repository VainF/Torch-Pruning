import torch

from ultralytics import YOLO
import torch_pruning as tp
import torch.nn as nn
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck

def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)

def prune():
    # load trained yolov8x model
    model = YOLO('yolov8x.pt')

    for name, param in model.model.named_parameters():
        param.requires_grad = True
    
    replace_c2f_with_c2f_v2(model.model)

    # pruning
    model.model.eval()
    example_inputs = torch.randn(1, 3, 640, 640).to(model.device)
    imp = tp.importance.MagnitudeImportance(p=2)  # L2 norm pruning

    ignored_layers = []
    unwrapped_parameters = []

    modules_list = list(model.model.modules())
    for i, m in enumerate(modules_list):
        if isinstance(m, (Detect,)):
            ignored_layers.append(m)

    iterative_steps = 1  # progressive pruning
    pruner = tp.pruner.MagnitudePruner(
        model.model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=0.5,  # remove 50% channels
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters
    )
    print(model.model)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)
    for g in pruner.step(interactive=True):
        g.prune()

    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs)
    print(model.model)
    print("Before Pruning: MACs=%f G, #Params=%f M" % (base_macs / 1e9, base_nparams / 1e6))
    print("After Pruning: MACs=%f G, #Params=%f M" % (pruned_macs / 1e9, pruned_nparams / 1e6))

    # fine-tuning, TBD


if __name__ == "__main__":
    prune()