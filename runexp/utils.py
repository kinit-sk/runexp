import os
import torch
import random
import importlib
import numpy as np
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.utils import get_method, get_object

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
OmegaConf.register_new_resolver("object", get_object)

#-------------------------------------------------------------------------------
# Conditional import
#-------------------------------------------------------------------------------

def conditional_import(module_name, class_name, package=None):
    """
    Attempts to import a class from a module. Returns the class if found, else None.
    """
    try:
        module = importlib.import_module(module_name, package)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        return None

# Example usage:
# TrackerTrainerCallback = conditional_import('.trackers', 'TrackerTrainerCallback', __package__)

#-------------------------------------------------------------------------------
# Flatten/unflatten dict
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

def unflatten_dict(d):
    result = {}
    for key, value in d.items():
        parts = key.split('.')
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]

        d[parts[-1]] = value
    return result

#-------------------------------------------------------------------------------
# Config diff
#-------------------------------------------------------------------------------

def dict_diff(dict1, dict2):
    """Computes the difference between two dicts. Note: you can use
    flatten_dict() to flatten the dicts first."""
    diff_dict = {}

    for key in dict1:
        if key not in dict2 or dict1[key] != dict2[key]:
            diff_dict[key] = dict1[key]

    return diff_dict

def maybe_load_config(config, hydra_kwargs=None):
    """
    Loads a config if a string is provided, else returns the config as is.
    Config can be either a loaded config or a config name passed as a string.
    If a config name is passed, hydra_kwargs can be used to specify the
    hydra initialize_config_dir parameters (e.g., config_dir).
    """
    if isinstance(config, str):
        if hydra_kwargs is None:
            hydra_kwargs = dict(
                version_base=None,
                config_dir=os.path.join(os.getcwd(), "configs")
            )

        with initialize_config_dir(**hydra_kwargs):
            config = compose(config_name=config)

    return config

def config_diff(config1, config2, hydra_kwargs=None):
    """
    Computes the difference between two configs.
    
    Configs can be either loaded configs or config names passed as strings.
    If config names are passed, hydra_kwargs can be used to specify the
    hydra initialize_config_dir parameters (e.g., config_dir).
    """
    config1 = maybe_load_config(config1, hydra_kwargs=hydra_kwargs)
    config2 = maybe_load_config(config2, hydra_kwargs=hydra_kwargs)

    if type(config1) is not dict:
        config1 = OmegaConf.to_container(config1, resolve=False)

    if type(config2) is not dict:
        config2 = OmegaConf.to_container(config2, resolve=False)

    config1 = flatten_dict(config1)
    config2 = flatten_dict(config2)

    diff_dict = dict_diff(config1, config2)
    diff_dict = unflatten_dict(diff_dict)

    return diff_dict

#-------------------------------------------------------------------------------
# Random State Management
#-------------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
