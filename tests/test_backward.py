import os, sys
import torchvision
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

# torchvision==0.13.1

###########################################
# Prunable Models
############################################
try:
    from torchvision.models.vision_transformer import (
        vit_b_16,
        vit_b_32,
        vit_l_16,
        vit_l_32,
        vit_h_14,
    )
    from torchvision.models.convnext import (
        convnext_tiny,
        convnext_small,
        convnext_base,
        convnext_large,
    )
except:
    pass

from torchvision.models.densenet import (
    densenet121,
    densenet169,
    densenet201,
    densenet161,
)
from torchvision.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_s,
    efficientnet_v2_m,
    efficientnet_v2_l,
)
from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models.regnet import (
    regnet_y_400mf,
    regnet_y_800mf,
    regnet_y_1_6gf,
    regnet_y_3_2gf,
    regnet_y_8gf,
    regnet_y_16gf,
    regnet_y_32gf,
    regnet_y_128gf,
)
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.vgg import (
    vgg11,
    vgg13,
    vgg16,
    vgg19,
    vgg11_bn,
    vgg13_bn,
    vgg16_bn,
    vgg19_bn,
)


if __name__ == "__main__":

    entries = globals().copy()

    import torch
    import torch.nn as nn
    import torch_pruning as tp
    import random

    def my_prune(model, example_inputs, output_transform, model_name):
        
        from torchvision.models.vision_transformer import VisionTransformer

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ori_size = tp.utils.count_params(model)
        model.cpu().eval()
        ignored_layers = []
        for p in model.parameters():
            p.requires_grad_(True)
        #########################################
        # Ignore unprunable modules
        #########################################
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1000:
                ignored_layers.append(m)
            #elif isinstance(m, nn.modules.linear.NonDynamicallyQuantizableLinear):
            #    ignored_layers.append(m) # this module is used in Self-Attention
        if 'ssd' in model_name:
            ignored_layers.append(model.head)
        if model_name=='raft_large':
            ignored_layers.extend(
                [model.corr_block, model.update_block, model.mask_predictor]
            )
        # For ViT: Rounding the number of channels to the nearest multiple of num_heads
        round_to = None
        try:
            if isinstance( model, torchvision.models.vision_transformer.VisionTransformer): round_to = model.encoder.layers[0].num_heads
        except:
            pass
        #########################################
        # (Optional) Register unwrapped nn.Parameters 
        # TP will automatically detect unwrapped parameters and prune the last dim for you by default.
        # If you want to prune other dims, you can register them here.
        #########################################
        unwrapped_parameters = None
        #if model_name=='ssd300_vgg16':
        #    unwrapped_parameters=[ (model.backbone.scale_weight, 0) ] # pruning the 0-th dim of scale_weight
        #if isinstance( model, VisionTransformer):
        #    unwrapped_parameters = [ (model.class_token, 0), (model.encoder.pos_embedding, 0)]
        #elif isinstance(model, ConvNeXt):
        #    unwrapped_parameters = []
        #    for m in model.modules():
        #        if isinstance(m, CNBlock):
        #            unwrapped_parameters.append( (m.layer_scale, 0) )

        #########################################
        # Build network pruners
        #########################################
        importance = tp.importance.MagnitudeImportance(p=1)
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=1,
            pruning_ratio=0.5,
            round_to=round_to,
            unwrapped_parameters=unwrapped_parameters,
            ignored_layers=ignored_layers,
        )


        #########################################
        # Pruning 
        #########################################
        print("Pruning: {}...".format(model_name))
        pruner.step()
        try:
            if isinstance(
                model, torchvision.models.vision_transformer.VisionTransformer
            ):  # Torchvision relies on the hidden_dim variable for forwarding, so we have to modify this varaible after pruning s
                model.hidden_dim = model.conv_proj.out_channels
                print(model.hidden_dim, model.class_token.shape, model.encoder.pos_embedding.shape)
        except: pass
        model.train()
        def hook(m, gi, go):
            print(m)
            print("GI:", [g.shape for g in gi if g is not None])
            print("GO:", [g.shape for g in go if g is not None])
            print("--------------------")
        
        #for m in model.modules():
        #    m.register_full_backward_hook(hook)
        out = model(example_inputs)
        loss = torch.nn.functional.cross_entropy(out, torch.randint(0, 1000, (1,)))
        loss.backward()
        #########################################
        # Testing 
        #########################################
        with torch.no_grad():
            print("{} Pruning: ".format(model_name))
            print("  Params: %s => %s" % (ori_size, tp.utils.count_params(model)))
            if isinstance(out, (dict,list,tuple)):
                print("  Output:")
                for o in tp.utils.flatten_as_list(out):
                    print(o.shape)
            else:
                print("  Output:", out.shape)
            print("------------------------------------------------------\n")

    successful = []
    unsuccessful = []
    for model_name, entry in entries.items():
        if not callable(entry):
            continue
        if "inception" in model_name:
            example_inputs = torch.randn(1, 3, 299, 299)
        elif 'fasterrcnn' in model_name:
            example_inputs = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        else:
            example_inputs = torch.randn(1, 3, 224, 224)

        if "googlenet" in model_name or "inception" in model_name:
            model = entry(aux_logits=False)
        elif "fcn" in model_name or "deeplabv3" in model_name:
            model = entry(aux_loss=None)
        else:
            model = entry()

        if "fcn" in model_name or "deeplabv3" in model_name:
            output_transform = lambda x: x["out"]
        else:
            output_transform = None

        #try:
        my_prune(
                model, example_inputs=example_inputs, output_transform=output_transform, model_name=model_name
            )
        successful.append(model_name)
        #except Exception as e:
        #    print(e)
        #    unsuccessful.append(model_name)
        print("Successful Pruning: %d Models\n"%(len(successful)), successful)
        print("")
        print("Unsuccessful Pruning: %d Models\n"%(len(unsuccessful)), unsuccessful)
        sys.stdout.flush()
        assert len(unsuccessful)==0, "Unsuccessful Pruning: {} Models: {}\n".format(len(unsuccessful), unsuccessful)