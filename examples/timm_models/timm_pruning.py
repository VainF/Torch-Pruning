import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

import torch
import timm
import torch_pruning as tp

# timm==0.9.2
# torch==1.12.1

timm_models = timm.list_models()
print(timm_models)
example_inputs = torch.randn(1,3,224,224)
imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
prunable_list = []
unprunable_list = []
problem_with_input_shape = []
for i, model_name in enumerate(timm_models):
    print("Pruning %s..."%model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #if 'rexnet' in model_name or 'sequencer' in model_name or 'botnet' in model_name:  # pruning process stuck with that architectures - skip them.
    #    unprunable_list.append(model_name)
    #    continue
    try:
        model = timm.create_model(model_name, pretrained=False, no_jit=True).eval().to(device)
    except: # out of memory error
        model = timm.create_model(model_name, pretrained=False, no_jit=True).eval()
        device = 'cpu'

    input_size = model.default_cfg['input_size']
    example_inputs = torch.randn(1, *input_size).to(device)
    test_output = model(example_inputs)

    print(model)
    prunable = True
    try:
        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        pruner = tp.pruner.MagnitudePruner(
                        model, 
                        example_inputs, 
                        global_pruning=False, # If False, a uniform sparsity will be assigned to different layers.
                        importance=imp, # importance criterion for parameter selection
                        iterative_steps=1, # the number of iterations to achieve target sparsity
                        ch_sparsity=0.5,
                        ignored_layers=[],
                    )
        pruner.step()
        test_output = model(example_inputs)
        pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
        print("Base MACs: %d, Pruned MACs: %d"%(base_macs, pruned_macs))
        print("Base Params: %d, Pruned Params: %d"%(base_params, pruned_params))
    except Exception as e:
        prunable = False
    
    if prunable:
        prunable_list.append(model_name)
    else:
        unprunable_list.append(model_name)
    
    print("Prunable: %d models, \n %s\n"%(len(prunable_list), prunable_list))
    print("Unprunable: %d models, \n %s\n"%(len(unprunable_list), unprunable_list))