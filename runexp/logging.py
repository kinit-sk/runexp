import os
import tempfile
from abc import ABCMeta, abstractmethod
from .utils import flatten_dict, RandomStatePreserver, preserve_random_state

class ExperimentLoggerMeta(ABCMeta):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        methods = ['initialize', 'log_dir', 'log_metrics', 'log_summary', 'log_artifact', 'finish']

        for method in methods:
            value = dct.get(method)
            if isinstance(value, property):
                setattr(cls, method, property(preserve_random_state(value.fget)))
            else:
                setattr(cls, method, preserve_random_state(value))

class ExperimentLogger(metaclass=ExperimentLoggerMeta):
    def __init__(self, project_name, experiment_name, description, config):
        self._random_state_preserver = RandomStatePreserver()
        self.initialize(project_name, experiment_name, description, config)

    def initialize(self, project_name, experiment_name, description, config):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def log_dir(self):
        raise NotImplementedError
    
    @abstractmethod
    def log_metrics(self, metrics, step=None, commit=None):
        raise NotImplementedError

    @abstractmethod
    def log_summary(self, metrics):
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, path, name=None):
        raise NotImplementedError

    @abstractmethod
    def finish(self):
        raise NotImplementedError
    
    def __del__(self):
        try:
            self.finish()
        except Exception:
            pass

class DummyLogger(ExperimentLogger):
    def __init__(self,
        project_name,
        experiment_name,
        description,
        config,
        checkpoint_path='outputs'
    ):
        super().__init__(project_name, experiment_name, description, config)
        os.makedirs(checkpoint_path, exist_ok=True)
        self._log_dir = tempfile.TemporaryDirectory(dir=checkpoint_path)

    def initialize(self, project_name, experiment_name, description, config):
        pass

    @property
    def log_dir(self):
        return self._log_dir.name

    def log_metrics(self, metrics, step=None, commit=None):
        pass

    def log_summary(self, metrics):
        pass

    def log_artifact(self, path, name=None):
        pass

    def finish(self):
        self._log_dir.cleanup()

class WandbLogger(ExperimentLogger):
    def initialize(self, project_name, experiment_name, description, config):
        import wandb

        self.wandb_run = wandb.init(
            project=project_name,
            name=experiment_name,
            notes=description,
            config=flatten_dict(config)
        )

    @property
    def log_dir(self):
        return self.wandb_run.dir
    
    def log_metrics(self, metrics, step=None, commit=None):
        return self.wandb_run.log(metrics, step=step, commit=commit)

    def log_summary(self, metrics):
        for name, value in metrics.items():
            self.wandb_run.summary[name] = value

    def log_artifact(self, path, name=None):
        return self.wandb_run.log_artifact(path, name=name)

    def finish(self):
        self.wandb_run.finish()

class MLFlowLogger(ExperimentLogger):
    def __init__(self,
        project_name,
        experiment_name,
        description,
        config,
        checkpoint_path='outputs',
        tracking_uri='http://localhost:5000' # 'sqlite:///mlruns/mlruns.db'
    ):
        self.tracking_uri = tracking_uri
        super().__init__(project_name, experiment_name, description, config)
        os.makedirs(checkpoint_path, exist_ok=True)
        self._log_dir = tempfile.TemporaryDirectory(dir=checkpoint_path)

    def initialize(self, project_name, experiment_name, description, config):
        from mlflow.tracking import MlflowClient

        if self.tracking_uri.startswith('sqlite'):
            base_path = os.path.dirname(self.tracking_uri.split('///')[-1])
            if len(base_path.strip()) > 0:
                os.makedirs(base_path, exist_ok=True)

        self.step = 0
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        project = self.client.get_experiment_by_name(project_name)

        if project is None:
            self.project_id = self.client.create_experiment(project_name)
        else:
            self.project_id = project.experiment_id

        run = self.client.create_run(self.project_id, run_name=experiment_name)
        self.run_id = run.info.run_id

        # incorporate description and config
        self.client.set_tag(self.run_id, "experiment_name", experiment_name)
        self.client.set_tag(self.run_id, "description", description)

        config = flatten_dict(config)

        # TODO: we need to be able to restore the ~ to - in the keys
        # when reconstructing the nested version of the config
        for k, v in config.items():
            k = k.replace("~", "-") # mlflow does not allow ~ in keys
            self.client.log_param(self.run_id, k, v)

    @property
    def log_dir(self):
        return self._log_dir.name

    def log_metrics(self, metrics, step=None, commit=None):
        if step is None:
            if commit is None:
                commit = True

            step = self.step
        else:
            if commit is None:
                commit = False
            
        for k, v in metrics.items():
            self.client.log_metric(self.run_id, k, v, step=step)

        if commit is True:
            self.step += 1

    def log_summary(self, metrics):
        for k, v in metrics.items():
            self.client.log_metric(self.run_id, "summary/" + k, v)
    
    def log_artifact(self, path, name=None):
        return self.client.log_artifact(self.run_id, path, artifact_path=name)
            
    def finish(self):
        self.client.set_terminated(self.run_id)
        self._log_dir.cleanup()