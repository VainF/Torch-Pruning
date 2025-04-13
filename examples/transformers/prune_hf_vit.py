import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import torch.nn.functional as F
import torch_pruning as tp
from transformers import ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTSelfOutput
import warnings
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='ViT Pruning')
parser.add_argument('--model_name', default='google/vit-base-patch16-224', type=str, help='model name')
parser.add_argument('--data_path', default='data/imagenet', type=str, help='model name')
parser.add_argument('--taylor_batchs', default=10, type=int, help='number of batchs for taylor criterion')
parser.add_argument('--pruning_ratio', default=0.5, type=float, help='prune ratio')
parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
parser.add_argument('--pruning_type', default='l1', type=str, help='pruning type', choices=['random', 'taylor', 'l1'])
parser.add_argument('--test_accuracy', default=False, action='store_true', help='test accuracy')
parser.add_argument('--global_pruning', default=False, action='store_true', help='global pruning')

parser.add_argument('--train_batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--val_batch_size', default=128, type=int, help='val batch size')
parser.add_argument('--save_as', default=None, type=str, help='save as')
args = parser.parse_args()

def prepare_imagenet(imagenet_root, train_batch_size=64, val_batch_size=128, num_workers=4):
    """The imagenet_root should contain train and val folders.
    """

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
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
example_inputs = torch.randn(1,3,224,224).to(device)

if args.pruning_type == 'random':
    imp = tp.importance.RandomImportance()
elif args.pruning_type == 'taylor':
    imp = tp.importance.TaylorImportance()
elif args.pruning_type == 'l1':
    imp = tp.importance.MagnitudeImportance(p=1)
else: raise NotImplementedError

if args.pruning_type=='taylor' or args.test_accuracy:
    train_loader, val_loader = prepare_imagenet(args.data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size)

# Load the model
model = ViTForImageClassification.from_pretrained(args.model_name).to(device)
base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
print(base_macs/1e9, base_params/1e6)

if args.test_accuracy:
    print("Testing accuracy of the original model...")
    acc_ori, loss_ori = validate_model(model, val_loader, device)
    print("Accuracy: %.4f, Loss: %.4f"%(acc_ori, loss_ori))

print("Pruning %s..."%args.model_name)
num_heads = {}
ignored_layers = [model.classifier]
# All heads should be pruned simultaneously, so we group channels by head.
for m in model.modules():
    if isinstance(m, ViTSelfAttention):
        num_heads[m.query] = m.num_attention_heads
        num_heads[m.key] = m.num_attention_heads
        num_heads[m.value] = m.num_attention_heads
    if args.bottleneck and isinstance(m, ViTSelfOutput):
        ignored_layers.append(m.dense)

pruner = tp.pruner.BasePruner(
                model, 
                example_inputs, 
                global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                pruning_ratio=args.pruning_ratio, # target pruning ratio
                ignored_layers=ignored_layers,
                output_transform=lambda out: out.logits.sum(),
                num_heads=num_heads,
                prune_head_dims=True,
                prune_num_heads=False,
                head_pruning_ratio=0.5, # disabled when prune_num_heads=False
)

if isinstance(imp, tp.importance.TaylorImportance):
    model.zero_grad()
    print("Accumulating gradients for taylor pruning...")
    for k, (imgs, lbls) in enumerate(train_loader):
        if k>=args.taylor_batchs: break
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        output = model(imgs).logits
        loss = torch.nn.functional.cross_entropy(output, lbls)
        loss.backward()

for g in pruner.step(interactive=True):
    g.prune()

# Modify the attention head size and all head size aftering pruning
for m in model.modules():
    if isinstance(m, ViTSelfAttention):
        print(m)
        print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size, '=>')
        m.num_attention_heads = pruner.num_heads[m.query]
        m.attention_head_size = m.query.out_features // m.num_attention_heads
        m.all_head_size = m.query.out_features
        print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size)
        print()
print(model)

if args.test_accuracy:
    print("Testing accuracy of the pruned model...")
    acc_pruned, loss_pruned = validate_model(model, val_loader, device)
    print("Accuracy: %.4f, Loss: %.4f"%(acc_pruned, loss_pruned))

print("----------------------------------------")
print("Summary:")
pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
print("Base MACs: %.2f G, Pruned MACs: %.2f G"%(base_macs/1e9, pruned_macs/1e9))
print("Base Params: %.2f M, Pruned Params: %.2f M"%(base_params/1e6, pruned_params/1e6))
if args.test_accuracy:
    print("Base Loss: %.4f, Pruned Loss: %.4f"%(loss_ori, loss_pruned))
    print("Base Accuracy: %.4f, Pruned Accuracy: %.4f"%(acc_ori, acc_pruned))

if args.save_as is not None:
    print("Saving the pruned model to %s..."%args.save_as)
    os.makedirs(os.path.dirname(args.save_as), exist_ok=True)
    model.zero_grad()
    torch.save(model, args.save_as)