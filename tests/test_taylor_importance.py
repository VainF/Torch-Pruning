import torch
from torchvision.models import resnet18
import torch_pruning as tp

def test_taylor():
    model = resnet18(pretrained=True)

    # Importance criteria
    example_inputs = torch.randn(1, 3, 224, 224)
    imp = tp.importance.TaylorImportance()

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    iterative_steps = 1 # progressive pruning
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        global_pruning=True,
        pruning_ratio=0.1, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        pruning_ratio_dict={model.layer1: 0.5, (model.layer2, model.layer3): 0.5},
        ignored_layers=ignored_layers,
    )
    print(model)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        if isinstance(imp, tp.importance.TaylorImportance):
            # loss = F.cross_entropy(model(images), targets)
            loss = model(example_inputs).sum() # a dummy loss for TaylorImportance
            loss.backward()
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        # finetune your model here
        # finetune(model)
        # ...
    print(model)
    

if __name__=="__main__":
    test_taylor()