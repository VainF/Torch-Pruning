import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# classification
from torchvision.models.alexnet import alexnet
from torchvision.models.densenet import densenet121, densenet169, densenet201, densenet161
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152 #, \
#    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2 # not supported
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models.googlenet import googlenet

#from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0 # not supported
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, \
    mnasnet1_3

# segmentation
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, \
    deeplabv3_resnet50, deeplabv3_resnet101


if __name__=='__main__':

    entries = globals().copy()

    import torch
    import torch.nn as nn
    import torch_pruning as tp
    import random

    def random_prune(model, example_inputs, output_transform):
        model.cpu().eval()
        prunable_module_type = ( nn.Conv2d, nn.BatchNorm2d )
        prunable_modules = [ m for m in model.modules() if isinstance(m, prunable_module_type) ]
        ori_size = tp.utils.count_params( model )
        DG = tp.DependencyGraph().build_dependency( model, example_inputs=example_inputs, output_transform=output_transform )
        for layer_to_prune in prunable_modules:
            # select a layer
    
            if isinstance( layer_to_prune, nn.Conv2d ):
                prune_fn = tp.prune_conv
            elif isinstance(layer_to_prune, nn.BatchNorm2d):
                prune_fn = tp.prune_batchnorm

            ch = tp.utils.count_prunable_channels( layer_to_prune )
            rand_idx = random.sample( list(range(ch)), min( ch//2, 10 ) )
            plan = DG.get_pruning_plan( layer_to_prune, prune_fn, rand_idx)
            plan.exec()

        print(model)
        with torch.no_grad():
            out = model( example_inputs )
            if output_transform:
                out = output_transform(out)
            print(model_name)
            print( "  Params: %s => %s"%( ori_size, tp.utils.count_params(model) ) )
            print( "  Output: ", out.shape )
            print("------------------------------------------------------\n")

    for model_name, entry in entries.items():
        if not callable(entry):
            continue
        if 'inception' in model_name:
            example_inputs = torch.randn(1,3,299,299)
        else:
            example_inputs = torch.randn(1,3,256,256)
        
        if 'googlenet' in model_name or 'inception' in model_name:
            model = entry(aux_logits=False)
        elif 'fcn' in model_name or 'deeplabv3' in model_name:
            model = entry(aux_loss=None)
        else:
            model = entry() 
        
        if 'fcn' in model_name or 'deeplabv3' in model_name:
            output_transform = lambda x: x['out']
        else:
            output_transform = None
        print(model_name)
        random_prune(model, example_inputs=example_inputs, output_transform=output_transform)
