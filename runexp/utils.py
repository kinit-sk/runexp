import torch
import random
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import get_method

# allow the use of eval: and method: in the configs
# eval: allows for the evaluation of python code
# method: allows for the use of functions from the configs
#
# Example:
# emb_dim: ${args.emb_dim}
# emb_dim2: ${eval:'${args.emb_dim}*3'}
# eval_method: ${method:routines.eval_method}

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("method", get_method)

#-------------------------------------------------------------------------------
# Flatten dict
#-------------------------------------------------------------------------------

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

#-------------------------------------------------------------------------------
# Random State Management
#-------------------------------------------------------------------------------

class RandomState:
    def __init__(self):
        self.save_state()

    def save_state(self):
        self.random_state = random.getstate()
        self.np_random_state = np.random.get_state()
        self.torch_random_state = torch.get_rng_state()
        
        if torch.cuda.is_available():
            self.torch_cuda_random_state = torch.cuda.get_rng_state()

    def restore_state(self):
        random.setstate(self.random_state)
        np.random.set_state(self.np_random_state)
        torch.set_rng_state(self.torch_random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(self.torch_cuda_random_state)

class RandomStatePreserver:
    def __init__(self):
        self.my_state = RandomState()
        self.global_state = RandomState()
        self.nested = False

    def __enter__(self):
        if not self.nested:
            self.global_state.save_state()
            self.my_state.restore_state()
            self.nested = True

    def __exit__(self, exc_type, exc_value, traceback):
        if self.nested:
            self.my_state.save_state()
            self.global_state.restore_state()
            self.nested = False

def preserve_random_state(func):
    def wrapper(self, *args, **kwargs):
        with self._random_state_preserver:
            return func(self, *args, **kwargs)
    return wrapper
