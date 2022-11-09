import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import time
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import argparse, os

import torch_pruning as tp
from functools import partial
from engine.utils import count_ops_and_params

# Basic options
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "prune", "test"])
parser.add_argument('--output-dir', default='run/agnews', help='path where to save')

# For pruning
parser.add_argument("--method", type=str, default=None)
parser.add_argument("--speed-up", type=float, default=2)
parser.add_argument("--max-sparsity", type=float, default=1.0)
parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
parser.add_argument("--reg", type=float, default=1e-4)

parser.add_argument("--restore", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--global-pruning", action="store_true", default=False)
parser.add_argument("--sl-total-epochs", type=int, default=100, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
parser.add_argument("--sl-restore", action="store_true", default=False)
parser.add_argument("--iterative-steps", default=400, type=int)

args = parser.parse_args()

args.output_dir = os.path.join(args.output_dir, args.mode)
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 5000
LEARNING_RATE = 1e-3 if args.mode=='pretrain' else 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 50
DROPOUT = 0.5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 128
BIDIRECTIONAL = True
HIDDEN_DIM = 256
NUM_LAYERS = 2
OUTPUT_DIM = 4


df = pd.read_csv('data/ag_news_csv/train.csv', header=None, index_col=None)
df.columns = ['classlabel', 'title', 'content']
df['classlabel'] = df['classlabel']-1
df.head()
df[['classlabel', 'content']].to_csv('data/ag_news_csv/train_prepocessed.csv', index=None)

df = pd.read_csv('data/ag_news_csv/test.csv', header=None, index_col=None)
df.columns = ['classlabel', 'title', 'content']
df['classlabel'] = df['classlabel']-1
df.head()
df[['classlabel', 'content']].to_csv('data/ag_news_csv/test_prepocessed.csv', index=None)

del df

TEXT = data.Field(sequential=True,
                  tokenize='spacy',
                  include_lengths=True) # necessary for packed_padded_sequence

LABEL = data.LabelField(dtype=torch.float)

fields = [('classlabel', LABEL), ('content', TEXT)]

train_dataset = data.TabularDataset(
    path="data/ag_news_csv/train_prepocessed.csv", format='csv',
    skip_header=True, fields=fields)

test_dataset = data.TabularDataset(
    path="data/ag_news_csv/test_prepocessed.csv", format='csv',
    skip_header=True, fields=fields)

train_data, valid_data = train_dataset.split(
    split_ratio=[0.95, 0.05],
    random_state=random.seed(RANDOM_SEED))

print(f'Num Train: {len(train_data)}')
print(f'Num Valid: {len(valid_data)}')

TEXT.build_vocab(train_data,
                 max_size=VOCABULARY_SIZE,
                 vectors='glove.6B.100d',
                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab(train_data)

print(f'Vocabulary size: {len(TEXT.vocab)}')
print(f'Number of classes: {len(LABEL.vocab)}')

train_loader, valid_loader, test_loader = data.BucketIterator.splits(
    (train_data, valid_data, test_dataset), 
    batch_size=BATCH_SIZE,
    sort_within_batch=True, # necessary for packed_padded_sequence
    sort_key=lambda x: len(x.content),
    device=DEVICE)

print('Train')
for batch in train_loader:
    print(f'Text matrix size: {batch.content[0].size()}')
    print(f'Target vector size: {batch.classlabel.size()}')
    break
    
print('\nValid:')
for batch in valid_loader:
    print(f'Text matrix size: {batch.content[0].size()}')
    print(f'Target vector size: {batch.classlabel.size()}')
    break
    
print('\nTest:')
for batch in test_loader:
    print(f'Text matrix size: {batch.content[0].size()}')
    print(f'Target vector size: {batch.classlabel.size()}')
    break

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, bidirectional, hidden_dim, num_layers, output_dim, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.ModuleList(
            [ nn.LSTM(embedding_dim,
                      hidden_dim,
                      num_layers=1,
                      bidirectional=bidirectional, 
                      dropout=dropout),
              nn.LSTM(hidden_dim*2, 
                      hidden_dim,
                      num_layers=1,
                      bidirectional=bidirectional, 
                      dropout=dropout)]
            )

        self.fc1 = nn.Linear(hidden_dim * num_layers, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_length):
        embedded = self.dropout(self.embedding(text))
        packed_output = nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))
        for rnn_layer in self.rnn:
            packed_output, (hidden, cell) = rnn_layer(packed_output)    
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        hidden = self.fc1(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return hidden

def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            text, text_lengths = batch_data.content
            logits = model(text, text_lengths)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += batch_data.classlabel.size(0)
            correct_pred += (predicted_labels.long() == batch_data.classlabel.long()).sum()
        return correct_pred.float()/num_examples * 100


INPUT_DIM = len(TEXT.vocab)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
torch.manual_seed(RANDOM_SEED)
model = RNN(INPUT_DIM, EMBEDDING_DIM, BIDIRECTIONAL, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT, PAD_IDX)
if args.restore is not None:
    model.load_state_dict(torch.load(args.restore))
    print("loading model from {}".format(args.restore))

def progressive_pruning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    print(model)
    for name, p in model.rnn.named_parameters():
        print(name, p.shape)
    print("")
    while current_speed_up < speed_up:
        pruner.step()
        print(model)
        for name, p in model.rnn.named_parameters():
            print(name, p.shape)
        print("")
        pruned_ops, _ = count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
       
    return current_speed_up

def get_pruner(model, example_inputs):
    sparsity_learning = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1_group_conv":
        imp = tp.importance.GroupConvImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2, to_group=False)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance(to_group=False)
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2,  normalizer=tp.importance.RelativeNormalizer(args.soft_keeping_ratio))
        pruner_entry = partial(tp.pruner.GroupNormPruner, soft_keeping_ratio=args.soft_keeping_ratio, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, normalizer=tp.importance.RelativeNormalizer(args.soft_keeping_ratio))
        pruner_entry = partial(tp.pruner.GroupNormPruner, soft_keeping_ratio=args.soft_keeping_ratio, reg=args.reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError
    
    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    args.sparsity_learning = sparsity_learning
    ignored_layers = []
    ch_sparsity_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == OUTPUT_DIM:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == OUTPUT_DIM:
            ignored_layers.append(m)
    
    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the target speed_up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=args.iterative_steps,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=args.max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner

def train(model):
    model = model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_time = time.time()
    best_acc = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):
            
            text, text_lengths = batch_data.content
            
            ### FORWARD AND BACK PROP
            logits = model(text, text_lengths)
            cost = F.cross_entropy(logits, batch_data.classlabel.long())
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 50:
                print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                    f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                    f'Cost: {cost:.4f}')

        with torch.set_grad_enabled(False):
            train_acc = compute_accuracy(model, train_loader, DEVICE)
            val_acc = compute_accuracy(model, valid_loader, DEVICE)
            print(f'training accuracy: '
                f'{train_acc:.2f}%'
                f'\nvalid accuracy: '
                f'{val_acc:.2f}%')
            if best_acc<val_acc:
                best_acc = val_acc
                os.makedirs(args.output_dir, exist_ok=True)
                if args.mode == "prune":
                    save_as = os.path.join( args.output_dir, "agnews_lstm_{}.pth".format(args.method) )
                    if save_state_dict_only:
                        torch.save(model.state_dict(), save_as)
                    else:
                        torch.save(model, save_as)
                elif args.mode == "pretrain":
                    save_as = os.path.join(args.output_dir, "agnews_lstm.pth")
                    torch.save(model.state_dict(), save_as)
        print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
    print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
    print(f'Test accuracy: {best_acc:.2f}%')


text, text_lengths = next(iter(train_loader)).content
example_inputs = (text.to(DEVICE), text_lengths.to(DEVICE))
if args.mode=='pretrain':
    model.eval().to(DEVICE)
    ops, params = count_ops_and_params(
        model, example_inputs=example_inputs,
    )
    print("Params: {:.2f} M".format(params / 1e6))
    print("MACs: {:.2f} G".format(ops / 1e9))
    train(model)
elif args.mode=='prune':
    model = model.eval().to(DEVICE)
    print(model)
    ori_ops, ori_size = count_ops_and_params(model, example_inputs=example_inputs)
    ori_acc = compute_accuracy(model, valid_loader, DEVICE)
    pruner = get_pruner(model, example_inputs=example_inputs)

    # 0. Sparsity Learning
    if args.sparsity_learning:
        reg_pth = "reg_{}_{}_{}_{}.pth".format(args.dataset, args.model, args.method, args.reg)
        reg_pth = os.path.join( os.path.join(args.output_dir, reg_pth) )
        if not args.sl_restore:
            print("Regularizing...")
            train_model(model)
        print("Loading sparsity model from {}...".format(reg_pth))
        model.load_state_dict( torch.load( reg_pth, map_location=DEVICE) )
    
    # 1. Pruning
    model.eval()
    print("Pruning...")
    progressive_pruning(pruner, model, speed_up=args.speed_up, example_inputs=example_inputs)
    del pruner # remove reference
    print(model)
    pruned_ops, pruned_size = count_ops_and_params(model, example_inputs=example_inputs)
    pruned_acc = compute_accuracy(model, valid_loader, DEVICE)
    print(
        "Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
            ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100
        )
    )
    print(
        "MACs: {:.2f} G => {:.2f} G ({:.2f}%, {:.2f}X )".format(
            ori_ops / 1e9,
            pruned_ops / 1e9,
            pruned_ops / ori_ops * 100,
            ori_ops / pruned_ops,
        )
    )
    print("Acc: {:.4f} => {:.4f}".format(ori_acc, pruned_acc))
    
    # 2. Finetuning
    print("Finetuning...")
    train_model(
        model,
        epochs=args.total_epochs,
        lr=args.lr,
        lr_decay_milestones=args.lr_decay_milestones,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        save_state_dict_only=False,
    )