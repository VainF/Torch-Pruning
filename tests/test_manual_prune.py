import torch
from torchvision.models import resnet18
import torch_pruning as tp

def test_manual_pruning():
    model = resnet18(pretrained=True)

    # Importance criteria
    example_inputs = torch.randn(1, 3, 224, 224)
    imp = tp.importance.OBDImportance()

    target_layers = [model.layer1[0].conv1, model.layer3[0].conv1]
    iterative_steps = 1 # progressive pruning
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        target_layers=target_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        if isinstance(imp, tp.importance.OBDImportance):
            # loss = F.cross_entropy(model(images), targets)
            dummy_inputs = torch.randn(10, 3, 224, 224)
            output = model(dummy_inputs) 
            # compute loss for each sample
            loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 1000, (len(dummy_inputs),)), reduction='none').to(output.device)
            imp.zero_grad() # clear accumulated gradients
            for l in loss:
                model.zero_grad() # clear gradients
                l.backward(retain_graph=True) # simgle-sample gradient
                imp.accumulate_grad(model) # accumulate g^2
          
        for g in pruner.step(interactive=True):
            print(g)
            g.prune()
        
        assert model.layer1[0].conv1.out_channels == 32 and model.layer1[0].conv2.in_channels == 32
        assert model.layer3[0].conv1.out_channels == 128 and model.layer3[0].conv2.in_channels == 128

        print(model)
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"MACs: {macs/base_macs:.2f}, #Params: {nparams/base_nparams:.2f}")
        # finetune your model here
        # finetune(model)
        # ...

if __name__=="__main__":
    test_manual_pruning()