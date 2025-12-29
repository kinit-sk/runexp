import abc
from omegaconf import OmegaConf
from .trackers import DummyTracker

class BaseExperiment(abc.ABC):
    def __init__(self, config, config_built):
        self.config = config
        self.config_built = config_built

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

class TrackedExperiment(BaseExperiment):
    def __init__(self, config, config_built):
        super().__init__(config, config_built)
        self.tracker = self.config_built.run_args.tracker

        if self.tracker is None:
            self.tracker = DummyTracker(
                self.config_built.run_args.project_name,
                self.config_built.run_args.experiment_name,
                self.config_built.run_args.description
            )
        else:
            self.tracker.setup(self.config)

class DummyExperiment(TrackedExperiment):
    def __call__(self):
        print("This is a dummy experiment.")
        print("Config:")
        print(OmegaConf.to_yaml(self.config))
        print("Log dir:", self.tracker.log_dir)
