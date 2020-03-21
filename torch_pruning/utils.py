from .dependency import TORCH_CONV, TORCH_BATCHNORM, TORCH_PRELU, TORCH_LINEAR

def count_prunable_params(module):
    if isinstance( module, ( TORCH_CONV, TORCH_LINEAR) ):
        num_params = module.weight.numel()
        if module.bias is not None:
            num_params += module.bias.numel()
        return num_params
    elif isinstance( module, TORCH_BATCHNORM ):
        num_params = module.running_mean.numel() + module.running_var.numel()
        if module.affine:
            num_params+= module.weight.numel() + module.bias.numel()
    elif isinstance( module, TORCH_PRELU ):
        if len( module.weight )==1:
            return 0
        else:
            return module.weight.numel
