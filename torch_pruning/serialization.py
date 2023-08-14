import torch
from torch.serialization import DEFAULT_PROTOCOL
import pickle

load = torch.load
save = torch.save

def state_dict(model: torch.nn.Module):
    """ Returns a dictionary containing the state, attributions of a module.
    """
    full_state_dict = {}
    attributions = {}
    for name, module in model.named_modules():
        # state dicts
        full_state_dict[name] = module.__dict__.copy()
        module_attr = {}

        # attributes
        for attr_name in dir(module):
            attr_value = getattr(module, attr_name)
            if attr_name=='T_destination':
                continue
            if not callable(attr_value) and (not attr_name.startswith('__')) and (not attr_name.startswith('_')):
                if not isinstance(attr_value, torch.nn.Parameter) and not isinstance(attr_value, torch.Tensor):
                    module_attr[attr_name] = attr_value
        attributions[name] = module_attr
    return {'full_state_dict': full_state_dict, 'attributions': attributions}

def load_state_dict(model: torch.nn.Module, state_dict: dict):
    """ Load a model given a state_dict.  
    """

    full_state_dict = state_dict['full_state_dict']
    attributions = state_dict['attributions']
    for name, module in model.named_modules():
        # load state dicts
        if name in full_state_dict:
            module.__dict__.update(full_state_dict[name])
        # load attributes
        if name in attributions:
            for attr_name, attr_value in attributions[name].items():
                setattr(module, attr_name, attr_value)
    return model
