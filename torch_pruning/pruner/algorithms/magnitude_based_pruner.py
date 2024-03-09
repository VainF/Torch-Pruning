from .metapruner import MetaPruner

class MagnitudePruner(MetaPruner):
    """ Prune the smallest magnitude weights

    Args:
    
        # Basic
        * model (nn.Module): A to-be-pruned model
        * example_inputs (torch.Tensor or List): dummy inputs for graph tracing.
        * importance (Callable): importance estimator. 
        * global_pruning (bool): enable global pruning. Default: False.
        * pruning_ratio (float): global channel sparisty. Also known as pruning ratio. Default: 0.5.
        * pruning_ratio_dict (Dict[nn.Module, float]): layer-specific pruning ratio. Will cover pruning_ratio if specified. Default: None.
        * max_pruning_ratio (float): the maximum pruning ratio. Default: 1.0.
        * iterative_steps (int): number of steps for iterative pruning. Default: 1.
        * iterative_pruning_ratio_scheduler (Callable): scheduler for iterative pruning. Default: linear_scheduler.
        * ignored_layers (List[nn.Module | typing.Type]): ignored modules. Default: None.
        * round_to (int): round channels to the nearest multiple of round_to. E.g., round_to=8 means channels will be rounded to 8x. Default: None.
        
        # Adavanced
        * in_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer input. Default: dict().
        * out_channel_groups (Dict[nn.Module, int]): The number of channel groups for layer output. Default: dict().
        * num_heads (Dict[nn.Module, int]): The number of heads for multi-head attention. Default: dict().
        * prune_num_heads (bool): remove entire heads in multi-head attention. Default: False.
        * prune_head_dims (bool): remove head dimensions in multi-head attention. Default: True.
        * head_pruning_ratio (float): head pruning ratio. Default: 0.0.
        * head_pruning_ratio_dict (Dict[nn.Module, float]): layer-specific head pruning ratio. Default: None.
        * customized_pruners (dict): a dict containing module-pruner pairs. Default: None.
        * unwrapped_parameters (dict): a dict containing unwrapped parameters & pruning dims. Default: None.
        * root_module_types (list): types of prunable modules. Default: [nn.Conv2d, nn.Linear, nn.LSTM].
        * forward_fn (Callable): A function to execute model.forward. Default: None.
        * output_transform (Callable): A function to transform network outputs. Default: None.

        # Deprecated
        * channel_groups (Dict[nn.Module, int]): output channel grouping. Default: dict().
        * ch_sparsity (float): the same as pruning_ratio. Default: None.
        * ch_sparsity_dict (Dict[nn.Module, float]): the same as pruning_ratio_dict. Default: None.
    """
    pass
        