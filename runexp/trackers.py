import logging
import os
import re
import tempfile
import importlib.util
from abc import ABCMeta, abstractmethod
from .utils import flatten_dict, RandomStatePreserver, preserve_random_state

class ExperimentTrackerMeta(ABCMeta):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        methods = ['setup', 'log_dir', 'log_metrics', 'log_summary', 'log_artifact', 'finish']

        for method in methods:
            value = dct.get(method)
            if isinstance(value, property):
                setattr(cls, method, property(preserve_random_state(value.fget)))
            else:
                setattr(cls, method, preserve_random_state(value))

class ExperimentTracker(metaclass=ExperimentTrackerMeta):
    def __init__(self, project_name, experiment_name, description):
        self._random_state_preserver = RandomStatePreserver()
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.description = description

    def setup(self, config):
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

class DummyTracker(ExperimentTracker):
    def __init__(self,
        project_name,
        experiment_name,
        description,
        checkpoint_path='outputs'
    ):
        super().__init__(project_name, experiment_name, description)
        os.makedirs(checkpoint_path, exist_ok=True)
        self._log_dir = tempfile.TemporaryDirectory(dir=checkpoint_path)

    def setup(self, config=None):
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

class WandbTracker(ExperimentTracker):
    def setup(self, config=None):
        import wandb

        self.wandb_run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            notes=self.description,
            config=flatten_dict(config) if config is not None else {},
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
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact path does not exist: {path}")
        
        return self.wandb_run.log_artifact(path, name=name)

    def finish(self):
        self.wandb_run.finish()

class MLFlowTracker(ExperimentTracker):
    def __init__(self,
        project_name,
        experiment_name,
        description,
        checkpoint_path='outputs',
        tracking_uri='http://localhost:5000' # 'sqlite:///mlruns/mlruns.db'
    ):
        self.tracking_uri = tracking_uri
        self.checkpoint_path = checkpoint_path
        super().__init__(project_name, experiment_name, description)

    def setup(self, config=None):
        from mlflow.tracking import MlflowClient

        os.makedirs(self.checkpoint_path, exist_ok=True)
        self._log_dir = tempfile.TemporaryDirectory(dir=self.checkpoint_path)

        if self.tracking_uri.startswith('sqlite'):
            base_path = os.path.dirname(self.tracking_uri.split('///')[-1])
            if len(base_path.strip()) > 0:
                os.makedirs(base_path, exist_ok=True)

        self.step = 0
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        project = self.client.get_experiment_by_name(self.project_name)

        if project is None:
            self.project_id = self.client.create_experiment(self.project_name)
        else:
            self.project_id = project.experiment_id

        run = self.client.create_run(self.project_id, run_name=self.experiment_name)
        self.run_id = run.info.run_id

        # incorporate description and config
        self.client.set_tag(self.run_id, "experiment_name", self.experiment_name)
        self.client.set_tag(self.run_id, "description", self.description)

        if config is not None:
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
            # sanitize metric names to replace unsupported characters
            # like parentheses (following the transformers package)
            k = re.sub(r"[^0-9A-Za-z_\-\.\ :/]", "_", k)
            self.client.log_metric(self.run_id, k, v, step=step)

        if commit is True:
            self.step += 1

    def log_summary(self, metrics):
        for k, v in metrics.items():
            self.client.log_metric(self.run_id, "summary/" + k, v)
 
    @staticmethod
    def _infer_name(path: str) -> str:
        inferred = os.path.basename(os.path.normpath(path))
        if not inferred:
            raise ValueError(f"Could not infer artifact name from path: {path}")
        return inferred

    def log_artifact(self, path, name=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact path does not exist: {path}")
        artifact_name = name if name is not None else self._infer_name(path)

        if os.path.isfile(path):
            # Store file under the artifact folder, no rename
            return self.client.log_artifact(
                run_id=self.run_id,
                local_path=path,
                artifact_path=artifact_name,
            )
        elif os.path.isdir(path):
            return self.client.log_artifacts(
                run_id=self.run_id,
                local_dir=path,
                artifact_path=artifact_name,
            )
        else:
            raise ValueError(f"Path is neither file nor directory: {path}")
            
    def finish(self):
        self.client.set_terminated(self.run_id)
        self._log_dir.cleanup()

#------------------------------------------------------------------------------
# Integration with transformers loggers
#------------------------------------------------------------------------------

if importlib.util.find_spec("transformers") is not None:
    from transformers.trainer_callback import TrainerCallback
    from transformers.integrations.integration_utils import rewrite_logs
    
    class TrackerTrainerCallback(TrainerCallback):
        """
        A [`TrainerCallback`] that sends the logs to a RunExp logger.
        """

        def __init__(self, tracker: ExperimentTracker, log_artifacts: bool = True):
            super().__init__()
            self.tracker = tracker
            self._log_artifacts = log_artifacts

        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            single_value_scalars = [
                "train_runtime",
                "train_samples_per_second",
                "train_steps_per_second",
                "train_loss",
                "total_flos",
            ]

            if state.is_world_process_zero:
                # these should go into summary
                summary = {}
                metrics = {}

                for k, v in logs.items():
                    if k in single_value_scalars:
                        summary[k] = v
                    else:
                        metrics[k] = v
                
                metrics = rewrite_logs(metrics)
                self.tracker.log_summary(summary)
                self.tracker.log_metrics(metrics, step=state.global_step)

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero and self._log_artifacts:
                ckpt_dir = f"checkpoint-{state.global_step}"
                artifact_path = os.path.join(args.output_dir, ckpt_dir)
                self.tracker.log_artifact(artifact_path)
