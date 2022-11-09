from pickletools import optimize
import torch.nn as nn
from enum import IntEnum


class DummyMHA(nn.Module):
    def __init__(self):
        super(DummyMHA, self).__init__()


class _CustomizedOp(nn.Module):
    def __init__(self, op_class):
        self.op_cls = op_class

    def __repr__(self):
        return "CustomizedOp({})".format(str(self.op_cls))


class _ConcatOp(nn.Module):
    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_ConcatOp({})".format(self.offsets)


class _SplitOp(nn.Module):
    def __init__(self):
        super(_SplitOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_SplitOp({})".format(self.offsets)


class _ElementWiseOp(nn.Module):
    def __init__(self, grad_fn):
        super(_ElementWiseOp, self).__init__()
        self._grad_fn = grad_fn

    def __repr__(self):
        return "_ElementWiseOp({})".format(self._grad_fn)


######################################################
# Dummy Pruners
class DummyPruner(object):
    def __call__(self, layer, *args, **kargs):
        return layer

    def prune_out_channels(self, layer, idxs):
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return None

    def get_in_channels(self, layer):
        return None


class ConcatPruner(DummyPruner):
    pass


class SplitPruner(DummyPruner):
    pass


class ElementWisePruner(DummyPruner):
    pass


# Standard Modules
TORCH_CONV = nn.modules.conv._ConvNd
TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
TORCH_LAYERNORM = nn.modules.normalization.LayerNorm
TORCH_PRELU = nn.PReLU
TORCH_LINEAR = nn.Linear
TORCH_EMBED = nn.Embedding
TORCH_PARAMETER = nn.Parameter
TORCH_LSTM = nn.LSTM
try:
    TORCH_MHA = nn.MultiheadAttention
except:
    TORCH_MHA = DummyMHA  # for pytorch w/o MultiHeadAttention
TORCH_OTHERS = None


class OPTYPE(IntEnum):
    CONV = 0
    BN = 1
    LINEAR = 2
    PRELU = 3
    DEPTHWISE_CONV = 4
    CONCAT = 5  # torch.cat
    SPLIT = 6  # torch.split
    CUSTOMIZED = 7  # customized module
    ELEMENTWISE = 8  # element-wise add, sub, etc.
    LN = 9  # nn.LayerNorm
    EMBED = 10  # nn.Embedding
    PARAMETER = 11  # nn.Parameter
    MHA = 12
    LSTM = 13


def module2type(module):
    if isinstance(module, TORCH_CONV):
        if module.groups == module.out_channels:
            return OPTYPE.DEPTHWISE_CONV
        else:
            return OPTYPE.CONV
    elif isinstance(module, TORCH_BATCHNORM):
        return OPTYPE.BN
    elif isinstance(module, TORCH_PRELU):
        return OPTYPE.PRELU
    elif isinstance(module, TORCH_LINEAR):
        return OPTYPE.LINEAR
    elif isinstance(module, _ConcatOp):
        return OPTYPE.CONCAT
    elif isinstance(module, _SplitOp):
        return OPTYPE.SPLIT
    elif isinstance(module, TORCH_LAYERNORM):
        return OPTYPE.LN
    elif isinstance(module, TORCH_EMBED):
        return OPTYPE.EMBED
    elif isinstance(module, _CustomizedOp):
        return OPTYPE.CUSTOMIZED
    elif isinstance(module, nn.Parameter):
        return OPTYPE.PARAMETER
    elif isinstance(module, TORCH_MHA):
        return OPTYPE.MHA
    elif isinstance(module, TORCH_LSTM):
        return OPTYPE.LSTM
    else:
        return OPTYPE.ELEMENTWISE


def type2class(op_type):
    if op_type == OPTYPE.CONV or op_type==OPTYPE.DEPTHWISE_CONV:
        return TORCH_CONV
    elif op_type == OPTYPE.BN:
        return TORCH_BATCHNORM
    elif op_type == OPTYPE.PRELU:
        return TORCH_PRELU
    elif op_type == OPTYPE.LINEAR:
        return TORCH_LINEAR
    elif op_type == OPTYPE.CONCAT:
        return _ConcatOp
    elif op_type == OPTYPE.SPLIT:
        return _SplitOp
    elif op_type == OPTYPE.LN:
        return TORCH_LAYERNORM
    elif op_type == OPTYPE.EMBED:
        return TORCH_EMBED
    elif op_type == OPTYPE.CUSTOMIZED:
        return _CustomizedOp
    elif op_type == OPTYPE.PARAMETER:
        return TORCH_PARAMETER
    elif op_type == OPTYPE.MHA:
        return TORCH_MHA
    elif op_type == OPTYPE.LSTM:
        return TORCH_LSTM
    else:
        return _ElementWiseOp

