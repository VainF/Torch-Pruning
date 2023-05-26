import torch
from torch.serialization import DEFAULT_PROTOCOL
import pickle

load = torch.load
save = torch.save

def state_dict(model: torch.nn.Module):
    full_state_dict = {}
    for name, module in model.named_modules():
        # keep all attributes
        full_state_dict[name] = module.__dict__.copy()
    return full_state_dict

def load_state_dict(model: torch.nn.Module, state_dict: dict):
    for name, module in model.named_modules():
        if name in state_dict:
            module.__dict__.update(state_dict[name])
    return model
