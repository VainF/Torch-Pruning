
import re
import os


#import git
import numpy as np


#from .constants import CHECKPOINTS_PATH, LayerType


def convert_adj_to_edge_index(adjacency_matrix):
    """
    Handles both adjacency matrices as well as connectivity masks used in softmax (check out Imp2 of the GAT model)
    Connectivity masks are equivalent to adjacency matrices they just have -inf instead of 0 and 0 instead of 1.
    I'm assuming non-weighted (binary) adjacency matrices here obviously and this code isn't meant to be as generic
    as possible but a learning resource.
    """
    assert isinstance(adjacency_matrix, np.ndarray), f'Expected NumPy array got {type(adjacency_matrix)}.'
    height, width = adjacency_matrix.shape
    assert height == width, f'Expected square shape got = {adjacency_matrix.shape}.'

    # If there are infs that means we have a connectivity mask and 0s are where the edges in connectivity mask are,
    # otherwise we have an adjacency matrix and 1s symbolize the presence of edges.
    active_value = 0 if np.isinf(adjacency_matrix).any() else 1

    edge_index = []
    for src_node_id in range(height):
        for trg_nod_id in range(width):
            if adjacency_matrix[src_node_id, trg_nod_id] == active_value:
                edge_index.append([src_node_id, trg_nod_id])

    return np.asarray(edge_index).transpose()  # change shape from (N,2) -> (2,N)


def name_to_layer_type(name):
    if name == LayerType.IMP1.name:
        return LayerType.IMP1
    elif name == LayerType.IMP2.name:
        return LayerType.IMP2
    elif name == LayerType.IMP3.name:
        return LayerType.IMP3
    else:
        raise Exception(f'Name {name} not supported.')


def get_training_state(training_config, model):
    training_state = {
        # Training details
        #"dataset_name": training_config['dataset_name'],
        #"num_of_epochs": training_config['num_of_epochs'],
        #"test_perf": training_config['test_perf'],

        # Model structure
        #"num_of_layers": training_config['num_of_layers'],
        #"num_heads_per_layer": training_config['num_heads_per_layer'],
        #"num_features_per_layer": training_config['num_features_per_layer'],
        #"add_skip_connection": training_config['add_skip_connection'],
        #"bias": training_config['bias'],
        #"dropout": training_config['dropout'],
        #"layer_type": training_config['layer_type'].name,

        # Model state
        "state_dict": model.state_dict()
    }

    return training_state


def get_available_binary_name(dataset_name='unknown'):
    prefix = f'gat_{dataset_name}'

    def valid_binary_name(binary_name):
        # First time you see raw f-string? Don't worry the only trick is to double the brackets.
        pattern = re.compile(rf'{prefix}_[0-9]{{6}}\.pth')
        return re.fullmatch(pattern, binary_name) is not None

    # Just list the existing binaries so that we don't overwrite them but write to a new one
    valid_binary_names = list(filter(valid_binary_name, os.listdir(CHECKPOINTS_PATH)))
    if len(valid_binary_names) > 0:
        last_binary_name = sorted(valid_binary_names)[-1]
        new_suffix = int(last_binary_name.split('.')[0][-6:]) + 1  # increment by 1
        return f'{prefix}_{str(new_suffix).zfill(6)}.pth'
    else:
        return f'{prefix}_000000.pth'


def print_model_metadata(training_state):
    header = f'\n{"*"*5} Model training metadata: {"*"*5}'
    print(header)

    for key, value in training_state.items():
        if key != 'state_dict':  # don't print state_dict it's a bunch of numbers...
            print(f'{key}: {value}')
    print(f'{"*" * len(header)}\n')