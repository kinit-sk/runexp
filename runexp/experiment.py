import abc
from omegaconf import OmegaConf

class BaseExperiment(abc.ABC):
    def __init__(self, config, config_built):
        self.config = config
        self.config_built = config_built

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

class LoggedExperiment(BaseExperiment):
    def __init__(self, config, config_built):
        super().__init__(config, config_built)
        run_args = self.config_built.run_args

        self.logger = run_args.logger_class(
            project_name=run_args.project_name,
            experiment_name=run_args.experiment_name,
            description=run_args.description,
            config=self.config
        )

class DummyExperiment(LoggedExperiment):
    def __call__(self):
        print("This is a dummy experiment.")
        print("Config:")
        print(OmegaConf.to_yaml(self.config))
        print("Log dir:", self.logger.log_dir)
