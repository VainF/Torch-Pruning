#load all
import torch
import matplotlib.pyplot as plt
params_record, loss_record, acc_record = torch.load('record.pth')

# change the plt style
plt.style.use('bmh')

color_dict = {
    'Group Hessian': "C0",
    'Single-layer Hessian': "C0",

    'Random': "C1",

    'Group L1': "C2",
    'Single-layer L1': "C2",
    
    'Group Slimming': "C3",
    'Single-layer Slimming': "C3",
    
    'Group Taylor': "C4",
    'Single-layer Taylor': "C4" 
}

plt.figure()
for imp_name in params_record.keys():
    # use dash if 'single-layer' is in the name, use the same color as the group version
    plt.plot(params_record[imp_name], acc_record[imp_name], label=imp_name, linestyle='--' if 'Single-layer' in imp_name else '-', color=color_dict[imp_name])
plt.xlabel('#Params')
plt.ylabel('Accuracy')
plt.legend()
# remove white space
plt.tight_layout()
plt.savefig(f'params_acc_final.png')

plt.figure()
for imp_name in params_record.keys():
    plt.plot(params_record[imp_name], loss_record[imp_name], label=imp_name, linestyle='--' if 'Single-layer' in imp_name else '-', color=color_dict[imp_name])
plt.xlabel('#Params')
plt.ylabel('Loss')
plt.legend()
# remove white space
plt.tight_layout()
plt.savefig(f'params_loss_final.png')

