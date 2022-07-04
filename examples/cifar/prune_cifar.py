import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
import registry
import models

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--restore_from', type=str, default=None)

args = parser.parse_args()


def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

def train_model(model, train_loader, test_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    for epoch in range(args.total_epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i%10==0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f"%(epoch, args.total_epochs, acc))
        if best_acc<acc:
            os.makedirs('checkpoints/pruned', exist_ok=True)
            torch.save( model, 'checkpoints/pruned/%s-%s-round%d.pth'%(args.dataset, args.model, args.round) )
            best_acc=acc
        scheduler.step()
    print("Best Acc=%.4f"%(best_acc))

def prune_model(model, args):
    model.cpu()
    userdefined_parameters=[model.pos_embedding, model.cls_token] if 'vit' in args.model else None
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32), userdefined_parameters=userdefined_parameters )
    
    def prune_linear(layer, amount=0.4):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(layer.weight, amount=amount)
        plan = DG.get_pruning_plan(layer, tp.prune_linear_out_channel, pruning_index)
        plan.exec()

    def prune_conv(layer, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(layer.weight, amount=amount)
        plan = DG.get_pruning_plan(layer, tp.prune_conv_out_channel, pruning_index)
        plan.exec()

    for (i, m) in enumerate(model.modules()):
        if isinstance( m, models.vit.FeedForward ): # do not prune the to_patch_embedding layers! 
            prune_linear(m.net[0], 0.2)
            prune_linear(m.net[0], 0.2)
        elif isinstance(m, nn.Conv2d):
            prune_conv(m, 0.1)
    return model    

def main():
    num_classes, train_dst, val_dst = registry.get_dataset(args.dataset, data_root='data')
    train_loader = torch.utils.data.DataLoader(
        train_dst, batch_size=args.batch_size, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        val_dst,batch_size=args.batch_size, num_workers=4)
    args.num_classes = num_classes
    
    if args.restore_from is not None:
        loaded = torch.load(args.restore_from, map_location='cpu')
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model = registry.get_model(args.model, num_classes=num_classes)
            model.load_state_dict(loaded['state_dict'])

    if args.mode=='train':
        args.round=0
        train_model(model, train_loader, test_loader)
    elif args.mode=='prune':
        print("Pruning round %d, load model from %s"%( args.round, args.restore_from ))
        prune_model(model, args)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        train_model(model, train_loader, test_loader)
    elif args.mode=='test':
        print("Load model from %s"%( args.restore_from ))
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc))

if __name__=='__main__':
    main()
