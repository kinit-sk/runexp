import abc
from omegaconf import OmegaConf
from .trackers import DummyTracker

class BaseExperiment(abc.ABC):
    def __init__(self, config, config_built):
        self.config = config
        self.config_built = config_built
        self._finished = False

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    def finish(self):
        self._finished = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()
        return False

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

    def trackers(self):
        return [] if self.tracker is None else [self.tracker]

    def finish(self):
        if self._finished:
            return

        errors = []
        for tracker in self.trackers():
            try:
                tracker.finish()
            except Exception as exc:
                errors.append(exc)

        super().finish()

        if errors:
            raise errors[0]

class DummyExperiment(TrackedExperiment):
    def __call__(self):
        print("This is a dummy experiment.")
        print("Config:")
        print(OmegaConf.to_yaml(self.config))
        print("Log dir:", self.tracker.log_dir)
