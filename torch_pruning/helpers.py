from turtle import forward
import torch.nn as nn
from . import prune
import numpy as np 
import torch

class _CustomizedOp(nn.Module):
    def __init__(self, op_class):
        self.op_cls = op_class

    def __repr__(self):
        return "CustomizedOp(%s)" % (str(self.op_cls))

######################################################
# Dummy module
class _ConcatOp(nn.Module):
    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_ConcatOp(%s)" % (self.offsets)


class _SplitOp(nn.Module):
    def __init__(self):
        super(_SplitOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_SplitOp(%s)" % (self.offsets)


class _ElementWiseOp(nn.Module):
    def __init__(self, grad_fn):
        super(_ElementWiseOp, self).__init__()
        self._grad_fn = grad_fn
    def __repr__(self):
        return "_ElementWiseOp(%s)"%(self._grad_fn)

######################################################
# Dummy Pruning fn
class DummyPruner(prune.BasePruner):
    def __call__(self, layer, *args, **kargs):
        return layer, 0
    def calc_nparams_to_prune(self, layer, idxs):
        return 0
    def prune(self, layer, idxs):
        return layer

class ConcatPruner(DummyPruner):
    pass 
class SplitPruner(DummyPruner):
    pass 
class ElementWiseOpPruner(DummyPruner):
    pass 

_prune_concat = ConcatPruner()
_prune_split = SplitPruner()
_prune_elementwise_op = ElementWiseOpPruner()


######################################################
# Index transform
class _FlattenIndexTransform(object):
    def __init__(self, stride=1, reverse=False):
        self._stride = stride
        self.reverse = reverse

    def __call__(self, idxs):
        new_idxs = []
        if self.reverse == True:
            for i in idxs:
                new_idxs.append(i // self._stride)
                new_idxs = list(set(new_idxs))
        else:
            for i in idxs:
                new_idxs.extend(list(range(i * self._stride, (i + 1) * self._stride)))
        return new_idxs


class _ConcatIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        
        if self.reverse == True:
            new_idxs = [
                i - self.offset[0]
                for i in idxs
                if (i >= self.offset[0] and i < self.offset[1])
            ]
        else:
            new_idxs = [i + self.offset[0] for i in idxs]
        return new_idxs


class _SplitIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse == True:
            new_idxs = [i + self.offset[0] for i in idxs]
        else:
            new_idxs = [
                i - self.offset[0]
                for i in idxs
                if (i >= self.offset[0] and i < self.offset[1])
            ]
        return new_idxs

class _GroupConvIndexTransform(object):
    def __init__(self, in_channels, out_channels, groups, reverse=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse == True:
            new_idxs = [i + self.offset[0] for i in idxs]
        else:
            group_histgram = np.histogram(idxs, bins=self.groups, range=(0, self.out_channels))
            max_group_size = int(group_histgram.max())
        return new_idxs

class GConv(nn.Module):
    def __init__(self, gconv):
        super(GConv, self).__init__()
        self.groups = gconv.groups
        self.convs = nn.ModuleList()
        oc_size = gconv.out_channels // self.groups
        ic_size = gconv.in_channels // self.groups
        for g in range(self.groups):
            self.convs.append(
                nn.Conv2d(
                    in_channels = oc_size,
                    out_channels = ic_size,
                    kernel_size = gconv.kernel_size,
                    stride = gconv.stride,
                    padding = gconv.padding,
                    dilation = gconv.dilation,
                    groups = 1,
                    bias = gconv.bias is not None,
                    padding_mode = gconv.padding_mode,
                )
            )
        # copy parameters
        group_size = gconv.out_channels // self.groups
        gconv_weight = gconv.weight
        for (i, conv) in enumerate(self.convs):
            conv.weight.data = gconv_weight.data[oc_size*i: oc_size*(i+1)]
            if gconv.bias is not None:
                conv.bias.data = gconv.bias.data[oc_size*i: oc_size*(i+1)]
    def forward(self, x):
        split_sizes = [ conv.in_channels for conv in self.convs ]
        xs = torch.split(x, split_sizes, dim=1)
        out = torch.cat([ conv(xi) for (conv, xi) in zip(self.convs, xs) ], dim=1)
        return out

def gconv2convs(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.groups>1 and module.groups!=module.in_channels:
        new_module = GConv(module)
    for name, child in module.named_children():
        new_module.add_module(name, gconv2convs(child))
    return new_module
