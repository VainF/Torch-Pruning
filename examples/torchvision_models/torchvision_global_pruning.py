import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

# torchvision==0.13.1

###########################################
# Prunable Models
############################################
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn, 
    fasterrcnn_resnet50_fpn_v2, 
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn
)
from torchvision.models.detection.fcos import fcos_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn
from torchvision.models.alexnet import alexnet

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
from torchvision.models.segmentation import (
    fcn_resnet50,
    fcn_resnet101,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
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



###########################################
# Failue cases in this script
############################################
from torchvision.models.optical_flow import raft_large
from torchvision.models.swin_transformer import swin_t, swin_s, swin_b # TODO: support Swin ops 
from torchvision.models.shufflenetv2 import ( # TODO: support channel shuffling
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)


if __name__ == "__main__":

    entries = globals().copy()

    import torch
    import torch.nn as nn
    import torch_pruning as tp
    import random

    def my_prune(model, example_inputs, output_transform, model_name):
        
        from torchvision.models.vision_transformer import VisionTransformer
        from torchvision.models.convnext import CNBlock, ConvNeXt

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
        if 'fasterrcnn' in model_name:
            ignored_layers.extend([ 
                 model.rpn.head.cls_logits, model.rpn.head.bbox_pred, model.backbone.fpn, model.roi_heads
            ])
        if model_name=='fcos_resnet50_fpn':
            ignored_layers.extend([model.head.classification_head.cls_logits, model.head.regression_head.bbox_reg, model.head.regression_head.bbox_ctrness])
        if model_name=='keypointrcnn_resnet50_fpn':
            ignored_layers.extend([model.rpn.head.cls_logits, model.backbone.fpn.layer_blocks, model.rpn.head.bbox_pred, model.roi_heads.box_head, model.roi_heads.box_predictor, model.roi_heads.keypoint_predictor])
        if model_name=='maskrcnn_resnet50_fpn_v2':
            ignored_layers.extend([model.rpn.head.cls_logits, model.rpn.head.bbox_pred, model.roi_heads.box_predictor, model.roi_heads.mask_predictor])
        if model_name=='retinanet_resnet50_fpn_v2':
            ignored_layers.extend([model.head.classification_head.cls_logits, model.head.regression_head.bbox_reg])
        # For ViT: Rounding the number of channels to the nearest multiple of num_heads
        round_to = None
        channel_groups = {}
        if isinstance( model, VisionTransformer): 
            for m in model.modules():
                if isinstance(m, nn.MultiheadAttention):
                    channel_groups[m] = m.num_heads
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
            global_pruning=True,
            round_to=round_to,
            unwrapped_parameters=unwrapped_parameters,
            ignored_layers=ignored_layers,
            channel_groups=channel_groups,
        )


        #########################################
        # Pruning 
        #########################################
        print("==============Before pruning=================")
        print("Model Name: {}".format(model_name))
        print(model)

        layer_channel_cfg = {}
        for module in model.modules():
            if module not in pruner.ignored_layers:
                #print(module)
                if isinstance(module, nn.Conv2d):
                    layer_channel_cfg[module] = module.out_channels
                elif isinstance(module, nn.Linear):
                    layer_channel_cfg[module] = module.out_features

        pruner.step()
        if isinstance(
            model, VisionTransformer
        ):  # Torchvision relies on the hidden_dim variable for forwarding, so we have to modify this varaible after pruning
            model.hidden_dim = model.conv_proj.out_channels
            print(model.class_token.shape, model.encoder.pos_embedding.shape)
        print("==============After pruning=================")
        print(model)

        #########################################
        # Testing 
        #########################################
        with torch.no_grad():
            if isinstance(example_inputs, dict):
                out = model(**example_inputs)
            else:
                out = model(example_inputs)
            if output_transform:
                out = output_transform(out)
            print("{} Pruning: ".format(model_name))
            params_after_prune = tp.utils.count_params(model)
            print("  Params: %s => %s" % (ori_size, params_after_prune))

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
        if 'swin' in model_name.lower() or 'raft' in model_name.lower() or 'shufflenet' in model_name.lower(): # stuck
            unsuccessful.append(model_name)
            continue

        if not callable(entry):
            continue
        if "inception" in model_name:
            example_inputs = torch.randn(1, 3, 299, 299)
        elif "raft" in model_name:
            example_inputs = {
                "image1": torch.randn(1, 3, 224, 224),
                "image2": torch.randn(1, 3, 224, 224),
            }
        elif 'fasterrcnn' in model_name:
            example_inputs = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        else:
            example_inputs = torch.randn(1, 3, 224, 224)

        if "googlenet" in model_name or "inception" in model_name:
            model = entry(aux_logits=False)
        elif "fcn" in model_name or "deeplabv3" in model_name:
            model = entry(aux_loss=None)
        elif 'fasterrcnn' in model_name:
            model = entry(weights_backbone=None, trainable_backbone_layers=5) # TP does not support FrozenBN.
        elif 'fcos' in model_name:
            model = entry(weights_backbone=None, trainable_backbone_layers=5) # TP does not support FrozenBN.
        elif 'rcnn' in model_name:
            model = entry(weights=None, weights_backbone=None, trainable_backbone_layers=5) # TP does not support FrozenBN.
        else:
            model = entry()

        if "fcn" in model_name or "deeplabv3" in model_name:
            output_transform = lambda x: x["out"]
        else:
            output_transform = None

        try:
            my_prune(
                model, example_inputs=example_inputs, output_transform=output_transform, model_name=model_name
            )
            successful.append(model_name)
        except Exception as e:
            print(e)
            unsuccessful.append(model_name)
        print("Successful Pruning: %d Models\n"%(len(successful)), successful)
        print("")
        print("Unsuccessful Pruning: %d Models\n"%(len(unsuccessful)), unsuccessful)
        sys.stdout.flush()

print("Finished!")

print("Successful Pruning: %d Models\n"%(len(successful)), successful)
print("")
print("Unsuccessful Pruning: %d Models\n"%(len(unsuccessful)), unsuccessful)