import torch
from torchvision.models import resnet18
import torch_pruning as tp

def test_OBD():
    model = resnet18(pretrained=True)

    # Importance criteria
    example_inputs = torch.randn(1, 3, 224, 224)
    imp = tp.importance.OBDImportance()

    ignored_layer_outputs = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layer_outputs.append(m) # DO NOT prune the final classifier!

    iterative_steps = 1 # progressive pruning
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layer_outputs=ignored_layer_outputs,
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
        
        #for g in pruner.DG.get_all_groups(ignored_layer_outputs=pruner.ignored_layer_outputs, target_layer_types=pruner.target_layer_types):
        #    print(len(imp(g)) == len(imp2(g)))
            
        for g in pruner.step(interactive=True):
            g.prune()

        print(model)
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"MACs: {macs/base_macs:.2f}, #Params: {nparams/base_nparams:.2f}")
        # finetune your model here
        # finetune(model)
        # ...

if __name__=="__main__":
    test_OBD()