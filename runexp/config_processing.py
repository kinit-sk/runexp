import abc
from omegaconf import OmegaConf

class ProcessStep(abc.ABC):
    @abc.abstractmethod
    def __call__(self, config) -> None:
        """
        Processes the config in-place in some way.
        """
        raise NotImplementedError

class RemoveNegKeysStep(ProcessStep):
    def __init__(self, missing_ok=False):
        self.missing_ok = missing_ok
    
    def __call__(self, config):
        for key in list(config.keys()):
            if key.startswith("~"):
                config.pop(key, None)

                if self.missing_ok:
                    config.pop(key[1:], None)
                else:
                    del config[key[1:]]

class MakePerTargetArgsStep(ProcessStep):
    def __call__(self, config):
        per_target = config.get("_per_target_", None)        
        if per_target is None: return

        target = config.get("_target_", None)

        if not target is None:
            per_target_args = per_target.get(target, {})

            for k, v in per_target_args.items():
                config[k] = v

        del config["_per_target_"]

class ConfigProcessor:
    def __init__(self, steps):
        self.steps = steps

    def _proc_config(self, config):
        for step in self.steps:
            step(config)

        for val in config.values():
            if hasattr(val, 'keys'):
                self._proc_config(val)

        return config

    def __call__(self, config):
        config = OmegaConf.to_container(config, resolve=False)
        config = self._proc_config(config)
        return OmegaConf.create(config)
