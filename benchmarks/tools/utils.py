from contextlib import contextmanager
import logging
import os, sys
from termcolor import colored
import copy
import numpy as np
import torch
import torchvision

def flatten_dict(dic):
    flattned = dict()
    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten( k, v )
                else:
                    _flatten( prefix+'/%s'%k, v )
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[ prefix+'/%s'%k ] = v
        
    _flatten(None, dic)
    return flattned

@contextmanager
def dummy_ctx(*args, **kwds):
    try:
        yield None
    finally:
        pass

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)

        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        
        return prefix + " " + log

def get_logger(name='train', output=None, color=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # STDOUT
    stdout_handler = logging.StreamHandler( stream=sys.stdout )
    stdout_handler.setLevel( logging.DEBUG )

    plain_formatter = logging.Formatter( 
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S" )
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S")
    else:
        formatter = plain_formatter
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    # FILE
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            os.makedirs(os.path.dirname(output), exist_ok=True)
            filename = output
        else:
            os.makedirs(output, exist_ok=True)
            filename = os.path.join(output, "log.txt")
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(plain_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    return logger

def get_n_flops(model=None, img_size=(224,224), n_channel=3, count_adds=True, idx_scale=None):
    '''Only count the FLOPs of conv and linear layers (no BN layers etc.). 
    Only count the weight computation (bias not included since it is negligible)
    '''
    if hasattr(img_size, '__len__'):
        height, width = img_size
    else:
        assert isinstance(img_size, int)
        height, width = img_size, img_size

    model = copy.deepcopy(model)
    list_conv = []
    def conv_hook(self, input, output):
        flops = np.prod(self.weight.data.shape) * output.size(2) * output.size(3) / self.groups
        list_conv.append(flops)

    list_linear = []
    def linear_hook(self, input, output):
        flops = np.prod(self.weight.data.shape)
        list_linear.append(flops)

    def register_hooks(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            return
        for c in childrens:
            register_hooks(c)

    register_hooks(model)
    use_cuda = next(model.parameters()).is_cuda
    input = torch.rand(1, n_channel, height, width)
    if use_cuda:
        input = input.cuda()
    
    # forward
    try:
        model(input)
    except:
        model(input, {'idx_scale': idx_scale})
        # @mst (TODO): for SR network, there may be an extra argument for scale. Here set it to 2 to make it run normally. 
        # -- An ugly solution. Probably will be improved later.
    
    total_flops = (sum(list_conv) + sum(list_linear))
    if count_adds:
        total_flops *= 2
    return total_flops

def get_n_params(model):
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    return total
