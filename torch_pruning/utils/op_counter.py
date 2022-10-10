""" Modified from https://github.com/Lyken17/pytorch-OpCounter
"""

from distutils.version import LooseVersion
import logging
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
import numpy as np
import warnings

if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
    logging.warning(
        "You are using an old version PyTorch {version}, which THOP does NOT support.".format(
            version=torch.__version__
        )
    )


def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res

def l_sum(in_list):
    res = 0
    for _ in in_list:
        res += _
    return res


def calculate_parameters(param_list):
    total_params = 0
    for p in param_list:
        total_params += torch.DoubleTensor([p.nelement()])
    return total_params


def calculate_zero_ops():
    return torch.DoubleTensor([int(0)])

def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    # n, out_c, oh, ow = output_size
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])


def calculate_conv(bias, kernel_size, output_size, in_channel, group):
    warnings.warn("This API is being deprecated.")
    """inputs are all numbers!"""
    return torch.DoubleTensor([output_size * (in_channel / group * kernel_size + bias)])


def calculate_norm(input_size):
    """input is a number not a array or tensor"""
    return torch.DoubleTensor([2 * input_size])

def calculate_relu_flops(input_size):
    # x[x < 0] = 0
    return 0
    

def calculate_relu(input_size: torch.Tensor):
    warnings.warn("This API is being deprecated")
    return torch.DoubleTensor([int(input_size)])


def calculate_softmax(batch_size, nfeatures):
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return torch.DoubleTensor([int(total_ops)])


def calculate_avgpool(input_size):
    return torch.DoubleTensor([int(input_size)])


def calculate_adaptive_avg(kernel_size, output_size):
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_op * output_size)])


def calculate_upsample(mode: str, output_size):
    total_ops = output_size
    if mode == "linear":
        total_ops *= 5
    elif mode == "bilinear":
        total_ops *= 11
    elif mode == "bicubic":
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops *= ops_solve_A + ops_solve_p
    elif mode == "trilinear":
        total_ops *= 13 * 2 + 5
    return torch.DoubleTensor([int(total_ops)])


def calculate_linear(in_feature, num_elements):
    return torch.DoubleTensor([int(in_feature * num_elements)])


def counter_matmul(input_size, output_size):
    input_size = np.array(input_size)
    output_size = np.array(output_size)
    return np.prod(input_size) * output_size[-1]


def counter_mul(input_size):
    return input_size


def counter_pow(input_size):
    return input_size


def counter_sqrt(input_size):
    return input_size


def counter_div(input_size):
    return input_size

multiply_adds = 1


def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += torch.DoubleTensor([p.numel()])
    m.total_params[0] = calculate_parameters(m.parameters())


def zero_ops(m, x, y):
    m.total_ops += calculate_zero_ops()


def count_convNd(m: _ConvNd, x, y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    m.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.weight.shape),
        groups = m.groups,
        bias = m.bias
    )
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    # m.total_ops += calculate_conv(
    #     bias_ops,
    #     torch.zeros(m.weight.size()[2:]).numel(),
    #     y.nelement(),
    #     m.in_channels,
    #     m.groups,
    # )


def count_convNd_ver2(m: _ConvNd, x, y: torch.Tensor):
    x = x[0]

    # N x H x W (exclude Cout)
    output_size = torch.zeros((y.size()[:1] + y.size()[2:])).numel()
    # # Cout x Cin x Kw x Kh
    # kernel_ops = m.weight.nelement()
    # if m.bias is not None:
    #     # Cout x 1
    #     kernel_ops += + m.bias.nelement()
    # # x N x H x W x Cout x (Cin x Kw x Kh + bias)
    # m.total_ops += torch.DoubleTensor([int(output_size * kernel_ops)])
    m.total_ops += calculate_conv(m.bias.nelement(), m.weight.nelement(), output_size)


def count_normalization(m: nn.modules.batchnorm._BatchNorm, x, y):
    # TODO: add test cases
    # https://github.com/Lyken17/pytorch-OpCounter/issues/124
    # y = (x - mean) / sqrt(eps + var) * weight + bias
    x = x[0]
    # bn is by default fused in inference
    flops = calculate_norm(x.numel())
    if (getattr(m, 'affine', False) or getattr(m, 'elementwise_affine', False)):
        flops *= 2
    m.total_ops += flops


# def count_layer_norm(m, x, y):
#     x = x[0]
#     m.total_ops += calculate_norm(x.numel())


# def count_instance_norm(m, x, y):
#     x = x[0]
#     m.total_ops += calculate_norm(x.numel())


def count_prelu(m, x, y):
    x = x[0]

    nelements = x.numel()
    if not m.training:
        m.total_ops += calculate_relu(nelements)


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_ops += calculate_relu_flops(list(x.shape))


def count_softmax(m, x, y):
    x = x[0]
    nfeatures = x.size()[m.dim]
    batch_size = x.numel() // nfeatures

    m.total_ops += calculate_softmax(batch_size, nfeatures)


def count_avgpool(m, x, y):
    # total_add = torch.prod(torch.Tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    num_elements = y.numel()
    m.total_ops += calculate_avgpool(num_elements)


def count_adap_avgpool(m, x, y):
    kernel = torch.div(
        torch.DoubleTensor([*(x[0].shape[2:])]), 
        torch.DoubleTensor([*(y.shape[2:])])
    )
    total_add = torch.prod(kernel)
    num_elements = y.numel()
    m.total_ops += calculate_adaptive_avg(total_add, num_elements)


# TODO: verify the accuracy
def count_upsample(m, x, y):
    if m.mode not in (
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
    ):  # "trilinear"
        logging.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
        m.total_ops += 0
    else:
        x = x[0]
        m.total_ops += calculate_upsample(m.mode, y.nelement())


# nn.Linear
def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += calculate_linear(total_mul, num_elements)


import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


def _count_rnn_cell(input_size, hidden_size, bias=True):
    # h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})
    total_ops = hidden_size * (input_size + hidden_size) + hidden_size
    if bias:
        total_ops += hidden_size * 2

    return total_ops


def count_rnn_cell(m: nn.RNNCell, x: torch.Tensor, y: torch.Tensor):
    total_ops = _count_rnn_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def _count_gru_cell(input_size, hidden_size, bias=True):
    total_ops = 0
    # r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
    # z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
    state_ops = (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 2

    # n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
    total_ops += (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        total_ops += hidden_size * 2
    # r hadamard : r * (~)
    total_ops += hidden_size

    # h' = (1 - z) * n + z * h
    # hadamard hadamard add
    total_ops += hidden_size * 3

    return total_ops


def count_gru_cell(m: nn.GRUCell, x: torch.Tensor, y: torch.Tensor):
    total_ops = _count_gru_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def _count_lstm_cell(input_size, hidden_size, bias=True):
    total_ops = 0

    # i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
    # f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
    # o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
    # g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
    state_ops = (input_size + hidden_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 4

    # c' = f * c + i * g \\
    # hadamard hadamard add
    total_ops += hidden_size * 3

    # h' = o * \tanh(c') \\
    total_ops += hidden_size

    return total_ops


def count_lstm_cell(m: nn.LSTMCell, x: torch.Tensor, y: torch.Tensor):
    total_ops = _count_lstm_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_rnn(m: nn.RNN, x, y):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if isinstance(x[0], PackedSequence):
        batch_size = torch.max(x[0].batch_sizes)
        num_steps = x[0].batch_sizes.size(0)
    else:
        if m.batch_first:
            batch_size = x[0].size(0)
            num_steps = x[0].size(1)
        else:
            batch_size = x[0].size(1)
            num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_rnn_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_rnn_cell(input_size, hidden_size, bias)

    for i in range(num_layers - 1):
        if m.bidirectional:
            total_ops += _count_rnn_cell(hidden_size * 2, hidden_size, bias) * 2
        else:
            total_ops += _count_rnn_cell(hidden_size, hidden_size, bias)

    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_gru(m: nn.GRU, x, y):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if isinstance(x[0], PackedSequence):
        batch_size = torch.max(x[0].batch_sizes)
        num_steps = x[0].batch_sizes.size(0)
    else:
        if m.batch_first:
            batch_size = x[0].size(0)
            num_steps = x[0].size(1)
        else:
            batch_size = x[0].size(1)
            num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_gru_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_gru_cell(input_size, hidden_size, bias)

    for i in range(num_layers - 1):
        if m.bidirectional:
            total_ops += _count_gru_cell(hidden_size * 2, hidden_size, bias) * 2
        else:
            total_ops += _count_gru_cell(hidden_size, hidden_size, bias)

    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_lstm(m: nn.LSTM, x, y):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if isinstance(x[0], PackedSequence):
        batch_size = torch.max(x[0].batch_sizes)
        num_steps = x[0].batch_sizes.size(0)
    else:
        if m.batch_first:
            batch_size = x[0].size(0)
            num_steps = x[0].size(1)
        else:
            batch_size = x[0].size(1)
            num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_lstm_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_lstm_cell(input_size, hidden_size, bias)

    for i in range(num_layers - 1):
        if m.bidirectional:
            total_ops += _count_lstm_cell(hidden_size * 2, hidden_size, bias) * 2
        else:
            total_ops += _count_lstm_cell(hidden_size, hidden_size, bias)

    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


default_dtype = torch.float64

register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,
    nn.BatchNorm1d: count_normalization,
    nn.BatchNorm2d: count_normalization,
    nn.BatchNorm3d: count_normalization,
    nn.LayerNorm: count_normalization,
    nn.InstanceNorm1d: count_normalization,
    nn.InstanceNorm2d: count_normalization,
    nn.InstanceNorm3d: count_normalization,
    nn.PReLU: count_prelu,
    nn.Softmax: count_softmax,
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,
    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,
    nn.RNNCell: count_rnn_cell,
    nn.GRUCell: count_gru_cell,
    nn.LSTMCell: count_lstm_cell,
    nn.RNN: count_rnn,
    nn.GRU: count_gru,
    nn.LSTM: count_lstm,
    nn.Sequential: zero_ops,
    nn.PixelShuffle: zero_ops,
}

if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
    register_hooks.update({nn.SyncBatchNorm: count_normalization})


def profile(
    model: nn.Module,
    inputs,
    custom_ops=None,
    verbose=True,
    ret_layer_info=False,
    report_missing=False,
):
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        # overwrite `verbose` option when enable report_missing
        verbose = True

    def add_hooks(m: nn.Module):
        if 'total_ops' in m._buffers:
            return

        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        # for p in m.parameters():
        #     m.total_params += torch.DoubleTensor([p.numel()])

        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and report_missing:
                print(
                    "[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params."
                    % m_type
                )

        if fn is not None:
            handler_collection[m] = (
                m.register_forward_hook(fn),
                m.register_forward_hook(count_parameters),
            )
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    def dfs_count(module: nn.Module, known: set(), prefix="\t"):
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            next_dict = {}
            if m in handler_collection and not isinstance(
                m, (nn.Sequential, nn.ModuleList)
            ):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, known, prefix=prefix + "\t")
            
            if id(m) not in known:
                ret_dict[n] = (m_ops, m_params, next_dict)
                total_ops += m_ops
                known.add(id(m))
            total_params += m_params
        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(model, set())

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        
    for m in model.modules():
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    if ret_layer_info:
        return total_ops, total_params, ret_dict
    return total_ops, total_params