import hydra
from omegaconf import OmegaConf
from .utils import flatten_dict
from .config_processing import ConfigProcessor, RemoveDoubleNegKeysStep, RemoveNegKeysStep, MakePerTargetArgsStep
from collections.abc import Mapping

_base_config = """
build_order: ???

run_args:
  project_name: ???
  experiment_name: ???
  description: null
  experiment_class: ???
"""

class ExperimentRunner:
    @staticmethod
    def make_base_config(flattened=False):
        base_config = OmegaConf.create(_base_config)

        if flattened:
            base_config = flatten_dict(OmegaConf.to_container(base_config))

        return base_config

    def __init__(
        self,
        config,
        experiment_name=None,
        description=None,
        main=None,
        config_preprocessor='default',
        config_postprocessor='default'
    ):
        if config_preprocessor == 'default':
            config_preprocessor = ConfigProcessor([
                RemoveNegKeysStep(),
                MakePerTargetArgsStep()
            ])

        if config_postprocessor == 'default':
            config_postprocessor = ConfigProcessor([
                RemoveDoubleNegKeysStep()
            ])

        base_config = self.__class__.make_base_config()

        if isinstance(config, str):
            config = OmegaConf.create(config)

        config = OmegaConf.merge(base_config, config)
        self.config = config.copy()
        
        if not experiment_name is None:
            experiment_name = experiment_name.strip()
            config.run_args.experiment_name = experiment_name

        if not description is None:
            description = description.strip()
            config.run_args.description = description

        if not main is None:
            config.run_args.main = main

        # preprocess the config
        if not config_preprocessor is None:
            config = config_preprocessor(config)

        build_order = config.build_order

        # If build_order is a dict (mapping), sort its items by value and
        # assign the sorted keys as a list
        if isinstance(build_order, Mapping):
            build_order = [k for k, v in sorted(build_order.items(), key=lambda item: item[1])]

        config._set_flag(flags=["allow_objects"], values=[True])

        # instantiate everything according to the build order
        for k in build_order:
            val = config.get(k, None)
            if not val is None:
                config[k] = hydra.utils.instantiate(val, _convert_="object")

        self.config_built = config

        # postprocess the config
        if not config_postprocessor is None:
            self.config_built = config_postprocessor(self.config_built)

        missing_keys = OmegaConf.missing_keys(self.config_built)
        if missing_keys:
            raise ValueError(f"Missing keys in config: {missing_keys}")

    def __call__(self):
        experiment_class = self.config_built.run_args.experiment_class
        config = OmegaConf.to_container(self.config)
        experiment = experiment_class(config, self.config_built)
        return experiment()
    