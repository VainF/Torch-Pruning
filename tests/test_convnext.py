from torchvision.models.convnext import convnext_tiny, convnext_small, convnext_base, convnext_large
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__=='__main__':
    entries = globals().copy()
    import torch
    import torch.nn as nn
    import torch_pruning as tp
    import random

    def random_prune(model, example_inputs, output_transform):
        model.cpu().eval()

        model = tp.helpers.gconv2convs(model)
        from torchvision.models.convnext import CNBlock
        user_defined_parameters = []
        for m in model.modules():
            if isinstance(m, CNBlock):
                user_defined_parameters.append(m.layer_scale)
        tp.prune.prune_parameter.dim = 0
        
        prunable_module_type = ( nn.Conv2d, nn.BatchNorm2d )
        prunable_modules = [ m for m in model.modules() if isinstance(m, prunable_module_type) ]
        ori_size = tp.utils.count_params( model )
        DG = tp.DependencyGraph().build_dependency( model, example_inputs=example_inputs, output_transform=output_transform, user_defined_parameters=user_defined_parameters )
        for layer_to_prune in prunable_modules:
            # select a layer
    
            if isinstance( layer_to_prune, nn.Conv2d ):
                prune_fn = tp.prune_conv_out_channel
            elif isinstance(layer_to_prune, nn.BatchNorm2d):
                prune_fn = tp.prune_batchnorm

            ch = tp.utils.count_prunable_channels( layer_to_prune )
            rand_idx = random.sample( list(range(ch)), min( ch//2, 10 ) )
            plan = DG.get_pruning_plan( layer_to_prune, prune_fn, rand_idx)
            plan.exec()

            for m in model.modules():
                if isinstance(m, CNBlock):
                    print(m.layer_scale.shape, m)

        print(model)
        with torch.no_grad():
            out = model( example_inputs )
            if output_transform:
                out = output_transform(out)
            print(model_name)
            print( "  Params: %s => %s"%( ori_size, tp.utils.count_params(model) ) )
            print( "  Output:", out.shape )
            print("------------------------------------------------------\n")

    for model_name, entry in entries.items():
        if not callable(entry):
            continue
        example_inputs = torch.randn(1,3,256,256)
        model = entry() 
        
        if 'fcn' in model_name or 'deeplabv3' in model_name:
            output_transform = lambda x: x['out']
        else:
            output_transform = None
        print(model_name)
        random_prune(model, example_inputs=example_inputs, output_transform=output_transform)
