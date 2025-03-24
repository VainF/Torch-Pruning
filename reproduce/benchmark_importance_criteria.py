import torch
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch_pruning as tp
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

N_batchs = 10
imagenet_root = 'data/imagenet'
print('Parsing dataset...')
train_dst = ImageFolder(os.path.join(imagenet_root, 'train'), transform=T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
)
val_dst = ImageFolder(os.path.join(imagenet_root, 'val'), transform=T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
)
train_loader = torch.utils.data.DataLoader(train_dst, batch_size=64, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dst, batch_size=128, shuffle=False, num_workers=4)

def validate_model(model, val_loader):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)

# Importance criteria
imp_dict = {
    'Group Hessian': tp.importance.HessianImportance(group_reduction='mean'),
    'Single-layer Hessian': tp.importance.HessianImportance(group_reduction='first'),

    'Group Taylor': tp.importance.TaylorImportance(group_reduction='mean'),
    'Single-layer Taylor': tp.importance.TaylorImportance(group_reduction='first'),    

    'Group L1': tp.importance.MagnitudeImportance(p=1, group_reduction='mean'),
    'Single-layer L1': tp.importance.MagnitudeImportance(p=1, group_reduction='first'),
    
    'Group Slimming': tp.importance.BNScaleImportance(group_reduction='mean'),
    'Single-layer Slimming': tp.importance.BNScaleImportance(group_reduction='first'),

    'Random': tp.importance.RandomImportance(),
}

params_record = {}
loss_record = {}
acc_record = {}
macs_record = {}

model = resnet50(pretrained=True).eval().cuda()
example_inputs = torch.randn(1, 3, 224, 224).cuda()
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
base_val_acc, base_val_loss = validate_model(model, val_loader)
print(f"MACs: {base_macs/base_macs:.2f}, #Params: {base_nparams/base_nparams:.2f}, Acc: {base_val_acc:.4f}, Loss: {base_val_loss:.4f}")

for imp_name, imp in imp_dict.items():
    print(imp_name)
    if imp_name not in params_record:
        loss_record[imp_name] = []
        acc_record[imp_name] = []
        params_record[imp_name] = []
        macs_record[imp_name] = []

    model = resnet50(pretrained=True).eval().cuda()
    example_inputs = torch.randn(1, 3, 224, 224).cuda()
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m) # DO NOT prune the final classifier!
    
    iterative_steps = 5
    pruner = tp.pruner.BasePruner(
        model,
        example_inputs,
        iterative_steps=iterative_steps,
        importance=imp,
        pruning_ratio=0.3, 
        ignored_layers=ignored_layers,
    )

    print(f"MACs: {base_macs/base_macs:.2f}, #Params: {base_nparams/base_nparams:.2f}, Acc: {base_val_acc:.4f}, Loss: {base_val_loss:.4f}")

    params_record[imp_name].append(base_nparams)
    loss_record[imp_name].append(base_val_loss)
    acc_record[imp_name].append(base_val_acc)
    macs_record[imp_name].append(base_macs)

    for i in range(iterative_steps):
        if isinstance(imp, tp.importance.HessianImportance):
            # loss = F.cross_entropy(model(images), targets)
            for k, (imgs, lbls) in enumerate(train_loader):
                if k>=N_batchs: break
                imgs = imgs.cuda()
                lbls = lbls.cuda()
                output = model(imgs) 
                # compute loss for each sample
                loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                imp.zero_grad() # clear accumulated gradients
                for l in loss:
                    model.zero_grad() # clear gradients
                    l.backward(retain_graph=True) # simgle-sample gradient
                    imp.accumulate_grad(model) # accumulate g^2
        elif isinstance(imp, tp.importance.TaylorImportance):
            # loss = F.cross_entropy(model(images), targets)
            for k, (imgs, lbls) in enumerate(train_loader):
                if k>=N_batchs: break
                imgs = imgs.cuda()
                lbls = lbls.cuda()
                output = model(imgs)
                loss = torch.nn.functional.cross_entropy(output, lbls)
                loss.backward()

        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        #continue
        val_acc, val_loss = validate_model(model, val_loader)
        print(f"MACs: {macs/base_macs:.2f}, #Params: {nparams/base_nparams:.2f}, Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")
        params_record[imp_name].append(nparams)
        loss_record[imp_name].append(val_loss)
        acc_record[imp_name].append(val_acc)
        macs_record[imp_name].append(macs)
        
    #continue
    # Draw all curves in an image
    plt.figure()
    for imp_name in params_record.keys():
        # use dash if 'single-layer' is in the name, use the same color as the group version
        plt.plot(params_record[imp_name], acc_record[imp_name], label=imp_name, linestyle='--' if 'Single-layer' in imp_name else '-', color='C'+str(list(params_record.keys()).index(imp_name)))
    plt.xlabel('#Params')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'params_acc_final.png')

    plt.figure()
    for imp_name in params_record.keys():
        plt.plot(params_record[imp_name], loss_record[imp_name], label=imp_name, linestyle='--' if 'Single-layer' in imp_name else '-', color='C'+str(list(params_record.keys()).index(imp_name)))
    plt.xlabel('#Params')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'params_loss_final.png')

    plt.figure()
    for imp_name in params_record.keys():
        # follow the same rule
        plt.plot(macs_record[imp_name], acc_record[imp_name], label=imp_name, linestyle='--' if 'Single-layer' in imp_name else '-', color='C'+str(list(params_record.keys()).index(imp_name)))
    plt.xlabel('MACs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'macs_acc_final.png')

    plt.figure()
    for imp_name in params_record.keys():
        plt.plot(macs_record[imp_name], loss_record[imp_name], label=imp_name, linestyle='--' if 'Single-layer' in imp_name else '-', color='C'+str(list(params_record.keys()).index(imp_name)))
    plt.xlabel('MACs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'macs_loss_final.png')

    torch.save([params_record, loss_record, acc_record], 'record.pth')
        



