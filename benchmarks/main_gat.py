import argparse
from doctest import Example
from pyexpat import model
import time

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch_pruning as tp
from functools import partial

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.optim import Adam

from engine.models.graph.gat import GAT, GATLayer
from engine.utils.graph_utils import constants, data_loading
from engine.utils.graph_utils import utils
from engine.utils import count_ops_and_params, MagnitudeRecover

def prune_to_target_ops(pruner, model, speed_up, example_inputs, device):
    model.eval()
    ori_ops, _ = tp.utils.count_ops_and_params(
        model,
        example_inputs=example_inputs,
    )
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(
            model,
            example_inputs=example_inputs,
        )
        current_speed_up = float(ori_ops) / pruned_ops
        print(current_speed_up)
    return current_speed_up

def get_pruner(model, example_inputs, config):
    unwrapped_parameters = []
    sparsity_learning = False
    is_accum_importance = False
    if config['method'] == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=config['global_pruning'])
    elif config['method'] == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=config['global_pruning'])
    elif config['method'] == "lamp":
        imp = tp.importance.LAMPImportance(p=2, to_group=False)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=config['global_pruning'])
    elif config['method'] == "slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance(to_group=False)
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=config['reg'], global_pruning=config['global_pruning'])
    elif config['method'] == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2, normalizer=tp.importance.RelativeNormalizer(percentage=config['soft_keeping_ratio']))
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=config['global_pruning'])
    elif config['method'] == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, normalizer=tp.importance.RelativeNormalizer(percentage=config['soft_keeping_ratio']))
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=config['reg'], global_pruning=config['global_pruning'])
    else:
        raise NotImplementedError
    
    config['is_accum_importance'] = is_accum_importance
    config['sparsity_learning'] = sparsity_learning
    ignored_layers = [model.gat_net[-1]]
    ch_sparsity_dict = {}

    channel_groups = {}
    for m in model.modules():
        if isinstance(m, GATLayer):
            channel_groups[m.linear_proj] = m.num_of_heads
            unwrapped_parameters.append(m.scoring_fn_source)
            unwrapped_parameters.append(m.scoring_fn_target)
            unwrapped_parameters.append(m.bias)

    # here we set iterative_steps=50 to prune the model with small steps until it satisfied required MACs.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=config['iterative_steps'],
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        channel_groups=channel_groups,
    )
    return pruner

# Simple decorator function so that I don't have to pass arguments that don't change from epoch to epoch
def train_one_epoch(phase, gat, sigmoid_cross_entropy_loss, optimizer, data_loader, epoch=0, regularizer=None, recover=None):
    device = next(gat.parameters()).device  # fetch the device info from the model instead of passing it as a param
    mean_score = []
    # Certain modules behave differently depending on whether we're training the model or not.
    # e.g. nn.Dropout - we only want to drop model weights during the training.
    if phase == constants.LoopPhase.TRAIN:
        gat.train()
    else:
        gat.eval()

    # Iterate over batches of graph data (2 graphs per batch was used in the original paper for the PPI dataset)
    # We merge them into a single graph with 2 connected components, that's the main idea. After that
    # the implementation #3 is agnostic to the fact that those are multiple and not a single graph!
    for batch_idx, (node_features, gt_node_labels, edge_index) in enumerate(data_loader):
        # Push the batch onto GPU - note PPI is to big to load the whole dataset into a normal GPU
        # it takes almost 8 GBs of VRAM to train it on a GPU
        #print(edge_index.shape, node_features.shape)
        edge_index = edge_index.to(device)
        node_features = node_features.to(device)
        gt_node_labels = gt_node_labels.to(device)

        # I pack data into tuples because GAT uses nn.Sequential which expects this format
        graph_data = (node_features, edge_index)

        # Note: [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        # shape = (N, C) where N is the number of nodes in the batch and C is the number of classes (121 for PPI)
        # GAT imp #3 is agnostic to the fact that we actually have multiple graphs
        # (it sees a single graph with multiple connected components)
        nodes_unnormalized_scores = gat(graph_data)[0]

        # Example: because PPI has 121 labels let's make a simple toy example that will show how the loss works.
        # Let's say we have 3 labels instead and a single node's unnormalized (raw GAT output) scores are [-3, 0, 3]
        # What this loss will do is first it will apply a sigmoid and so we'll end up with: [0.048, 0.5, 0.95]
        # next it will apply a binary cross entropy across all of these and find the average, and that's it!
        # So if the true classes were [0, 0, 1] the loss would be (-log(1-0.048) + -log(1-0.5) + -log(0.95))/3.
        # You can see that the logarithm takes 2 forms depending on whether the true label is 0 or 1,
        # either -log(1-x) or -log(x) respectively. Easy-peasy. <3
        loss = sigmoid_cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)

        if phase == constants.LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            if regularizer is not None:
                regularizer(gat)
            if recover is not None:
                recover(gat)
            optimizer.step()  # apply the gradients to weights
        pred = (nodes_unnormalized_scores > 0).float().cpu().numpy()
        gt = gt_node_labels.cpu().numpy()
        micro_f1 = f1_score(gt, pred, average='micro')
        mean_score.append(micro_f1)
    return sum(mean_score) / len(mean_score)  # in the case of test phase we just report back the test micro_f1

def train_gat_ppi(config):
    """
    Very similar to Cora's training script. The main differences are:
    1. Using dataloaders since we're dealing with an inductive setting - multiple graphs per batch
    2. Doing multi-class classification (BCEWithLogitsLoss) and reporting micro-F1 instead of accuracy
    3. Model architecture and hyperparams are a bit different (as reported in the GAT paper)
    """
    global BEST_VAL_PERF, BEST_VAL_LOSS

    # Checking whether you have a strong GPU. Since PPI training requires almost 8 GBs of VRAM
    # I've added the option to force the use of CPU even though you have a GPU on your system (but it's too weak).
    device = torch.device("cuda" if torch.cuda.is_available() and not config['force_cpu'] else "cpu")
    config['device'] = device

    # Step 1: prepare the data loaders
    data_loader_train, data_loader_val, data_loader_test = data_loading.load_graph_data(config, device)

    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    )
    print(gat)
    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    #The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops

    if config['restore'] is not None:
        loaded = torch.load( config['restore'] )
        if isinstance(loaded, dict):
            gat.load_state_dict( loaded['state_dict'] )
        else:
            gat = loaded
            
    if config['test_only']:
        micro_f1 = train_one_epoch(phase=constants.LoopPhase.TEST, gat=gat, sigmoid_cross_entropy_loss=loss_fn, optimizer=optimizer,  data_loader=data_loader_test, epoch=0)
        config['test_perf'] = micro_f1
        print('*' * 50)
        print(f'Test micro-F1 = {micro_f1}')
        return 

    if config['prune']:
        norm_recover = MagnitudeRecover(gat, reg=2*config['weight_decay'])
        os.makedirs('run/ppi/prune', exist_ok=True)
        node_features, gt_node_labels, edge_index = next(iter(data_loader_test))
        example_inputs = ( node_features, edge_index )
        gat.eval()
        base_macs, base_size = tp.utils.count_ops_and_params(
            gat,
            example_inputs=example_inputs,
        )
        pruner = get_pruner(gat, example_inputs=example_inputs, config=config)
        if config['sparsity_learning']:
            gat.to(device)
            optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            best_f1 = train(  gat=gat, 
                                sigmoid_cross_entropy_loss=loss_fn,
                                optimizer=optimizer,
                                main_loop=train_one_epoch, 
                                data_loader_train=data_loader_train, 
                                data_loader_test=data_loader_test, 
                                data_loader_val = data_loader_val,
                                config=config, 
                                save_as = os.path.join('run/ppi/prune', 'best_pruned_gat.pth'),
                                regularizer=pruner.regularize,)

        gat.to('cpu')
        prune_to_target_ops(pruner, gat, config['speed_up'], example_inputs=example_inputs, device=device)
        pruned_macs, pruned_size = tp.utils.count_ops_and_params(
            gat,
            example_inputs=example_inputs,
        )
        
        # The GAT net caches some information during backward, which causes shape incompatiblility after pruning.
        # Here we re-construct a new network to avoid this issue.
        config['num_features_per_layer'][1] = gat.gat_net[0].linear_proj.out_features // config['num_heads_per_layer'][0]
        config['num_features_per_layer'][2] = gat.gat_net[1].linear_proj.out_features // config['num_heads_per_layer'][1]
        state_dict = gat.state_dict()
        new_gat = GAT(
            num_of_layers=config['num_of_layers'],
            num_heads_per_layer=config['num_heads_per_layer'],
            num_features_per_layer=config['num_features_per_layer'],
            add_skip_connection=config['add_skip_connection'],
            bias=config['bias'],
            dropout=config['dropout'],
            layer_type=config['layer_type'],
            log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
        ) 
        new_gat.load_state_dict(state_dict)
        gat = new_gat
        print(gat)
        print(
            "Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
                base_size / 1e6, pruned_size / 1e6, pruned_size / base_size * 100
            )
        )
        print(
            "MACs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                base_macs / 1e6,
                pruned_macs / 1e6,
                pruned_macs / base_macs * 100,
                base_macs / pruned_macs,
            )
        )
        del pruner

        gat.to(device)
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
        best_f1 = train(  gat=gat, 
                            sigmoid_cross_entropy_loss=loss_fn,
                            optimizer=optimizer,
                            main_loop=train_one_epoch, 
                            data_loader_train=data_loader_train, 
                            data_loader_test=data_loader_test, 
                            data_loader_val = data_loader_val,
                            config=config, 
                            save_as = os.path.join('run/ppi/prune', 'best_pruned_gat.pth'),
                            recover=norm_recover.regularize if config['prune'] else None,)
        print(f'The best f1: {best_f1}')
    else:
        gat.to(device)
        os.makedirs('run/ppi/pretrain', exist_ok=True)
        best_f1 = train(  gat=gat, 
                            sigmoid_cross_entropy_loss=loss_fn,
                            optimizer=optimizer,
                            main_loop=train_one_epoch, 
                            data_loader_train=data_loader_train, 
                            data_loader_test=data_loader_test, 
                            data_loader_val = data_loader_val,
                            config=config, 
                            save_as = os.path.join('run/ppi/pretrain', 'best_gat.pth'))
        gat.load_state_dict( torch.load( os.path.join('run/ppi/pretrain', 'best_gat.pth') )['state_dict'] )
        micro_f1 = train_one_epoch(phase=constants.LoopPhase.TEST, data_loader=data_loader_test)
        print(f'Test micro-F1 = {micro_f1}')

def train(gat, main_loop, sigmoid_cross_entropy_loss, optimizer, data_loader_train, data_loader_test, data_loader_val, config, save_as, regularizer=None, recover=None):
    best_f1 = 0
    for epoch in range(config['num_of_epochs']):
        # Training loop
        train_one_epoch(phase=constants.LoopPhase.TRAIN, gat=gat, sigmoid_cross_entropy_loss=sigmoid_cross_entropy_loss, optimizer=optimizer,  data_loader=data_loader_train, epoch=epoch, regularizer=regularizer, recover=recover)
        micro_f1 = train_one_epoch(phase=constants.LoopPhase.VAL, gat=gat, sigmoid_cross_entropy_loss=sigmoid_cross_entropy_loss, optimizer=optimizer,  data_loader=data_loader_val, epoch=epoch)
        print(f'Epoch = {epoch} Test micro-F1 = {micro_f1}')
        if micro_f1 >= best_f1:
            # Save the best GAT in the binaries directory
            torch.save(
                utils.get_training_state(config, gat),
                save_as,
            )
            best_f1 = micro_f1
    return best_f1

def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num-of-epochs", type=int, help="number of training epochs", default=200)
    parser.add_argument("--patience-period", type=int, help="number of epochs with no improvement on val before terminating", default=100)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight-decay", type=float, help="L2 regularization on model weights", default=0)
    parser.add_argument("--test-only", action='store_true', help='should test the model on the test dataset? (no by default)')
    parser.add_argument("--force-cpu", action='store_true', help='use CPU if your GPU is too small (no by default)')
    parser.add_argument("--restore", default=None, help='restore model from checkpoints')


    # Dataset related (note: we need the dataset name for metadata and related stuff, and not for picking the dataset)
    parser.add_argument("--dataset-name", choices=[el.name for el in constants.DatasetType], help='dataset to use for training', default=constants.DatasetType.PPI.name)
    parser.add_argument("--batch-size", type=int, help='number of graphs in a batch', default=2)
    parser.add_argument("--should-visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable-tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console-log-freq", type=int, help="log to output console (batch) freq (None for no logging)", default=10)
    parser.add_argument("--checkpoint-freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=5)

    parser.add_argument("--prune", action='store_true', help='prune the model')
    parser.add_argument(
        "--method", type=str, default='l1',
    )
    parser.add_argument(
        "--global-pruning",
        action="store_true",
    )
    parser.add_argument(
        "--speed-up", type=float, default=2.0,
    )
    parser.add_argument(
        "--soft-keeping-ratio", type=float, default=0.0,
    )
    parser.add_argument(
        "--reg", type=float, default=1e-5,
    )
    parser.add_argument("--iterative_steps", default=50, type=int)

    
    args = parser.parse_args()

    # I'm leaving the hyperparam values as reported in the paper, but I experimented a bit and the comments suggest
    # how you can make GAT achieve an even higher micro-F1 or make it smaller
    gat_config = {
        # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_of_layers": 3,  # PPI has got 42% of nodes with all 0 features - that's why 3 layers are useful
        "num_heads_per_layer": [4, 4, 6],  # other values may give even better results from the reported ones
        "num_features_per_layer": [constants.PPI_NUM_INPUT_FEATURES, 256, 256, constants.PPI_NUM_CLASSES],  # 64 would also give ~0.975 uF1!
        "add_skip_connection": True,  # skip connection is very important! (keep it otherwise micro-F1 is almost 0)
        "bias": True,  # bias doesn't matter that much
        "dropout": 0.0,  # dropout hurts the performance (best to keep it at 0)
        "layer_type": constants.LayerType.IMP3  # the only implementation that supports the inductive setting
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['ppi_load_test_only'] = False  # load both train/val/test data loaders (don't change it)

    # Add additional config information
    training_config.update(gat_config)

    return training_config


if __name__ == '__main__':

    # Train the graph attention network (GAT)
    train_gat_ppi(get_training_args())