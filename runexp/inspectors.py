import os
from abc import ABC, abstractmethod
import numpy as np
from omegaconf import OmegaConf
from .utils import config_diff, unflatten_dict
from pathlib import Path

class LogInspector(ABC):
    @abstractmethod
    def set_project(self, project_name):
        raise NotImplementedError

    @abstractmethod
    def get_project_names(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_runs(self, experiment_name=None):
        raise NotImplementedError
    
    @abstractmethod
    def get_experiment_names(self, runs=None):
        raise NotImplementedError
    
    @abstractmethod
    def make_config(self, run, base_for_diff=None, hydra_kwargs=None):
        """
        This reconstructs the config used for a given run.

        Arguments:
          - base_for_diff: If specified, the returned config will only contain
            the differences compared to this base config. This can be either
            a config or a the name of the config to be loaded from the hydra
            config directory.
          - hydra_kwargs: Arguments to pass to hydra.initialize_config_dir if
            the base config needs to be loaded; e.g. config_path, job_name, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self, experiment_name, metrics="test/recall_1"):
        raise NotImplementedError

    @abstractmethod
    def list_artifacts(self, run, return_raw=False):
        raise NotImplementedError
    
    @abstractmethod
    def download_artifact(self, run, name, dest_path):
        """
        Downloads the specified artifact from the given run into
        the destination path.
        
        To obtain a clean version of the artifact, it is best to clear the
        dest_path beforehand as different backends may choose to redownload
        and overwrite existing files and others may leave them.
        """
        raise NotImplementedError

class WandbLogInspector(LogInspector):
    def __init__(self, entity=None, project_name: str|None = None, wandb_api=None):
        super().__init__()
        
        if wandb_api is None:
            import wandb
            wandb_api = wandb.Api()

        self.api = wandb_api
        self.entity = entity
        self.project_name = project_name

    def set_project(self, project_name):
        self.project_name = project_name

    def get_project_names(self):
        return [p.name for p in self.api.projects(self.entity)]

    def get_runs(self, experiment_name=None, project_name=None):
        if project_name is None:
            project_name = self.project_name

        if self.entity is None:
            path = f"{project_name}"
        else:
            path = f"{self.entity}/{project_name}"

        return self.api.runs(
            path=path,
            filters={"display_name": experiment_name} if experiment_name else None
        )
    
    def get_experiment_names(self, runs=None):
        if runs is None:
            runs = self.get_runs()

        return sorted(set(run.name for run in runs))

    def get_metrics(self, experiment_name, metrics="test/recall_1"):
        if isinstance(metrics, str):
            metrics = [metrics]
            return_dict = False
        else:
            return_dict = True

        runs = self.get_runs(experiment_name)
        metrics = {metric: [] for metric in metrics}

        for run in runs:
            if run.state != "finished":
                continue

            for metric in metrics:
                metrics[metric].append(run.summary[metric])

        for metric in metrics:
            metrics[metric] = np.array(metrics[metric])

        if return_dict:
            return metrics
        else:
            return next(iter(metrics.values()))
        
    def get_history(self, run, metrics="val/recall_1"):
        if isinstance(metrics, str):
            metrics = [metrics]
            return_array = True
        else:
            return_array = False

        history = {}

        for metric in metrics:
            history_scan = run.scan_history(keys=["_step", metric])

            steps = []
            values = []
        
            for batch in history_scan:
                steps.append(batch["_step"])
                values.append(batch[metric])

            history[metric] = {'step': np.array(steps), 'value': np.array(values)}

        if return_array:
            return history[metrics[0]]
        else:
            return history

    def make_config(self, run, base_for_diff=None, hydra_kwargs=None):
        config = unflatten_dict(run.config)
        
        if base_for_diff is not None:
            config = config_diff(config, base_for_diff, hydra_kwargs=hydra_kwargs)

        return OmegaConf.create(config)

    def list_artifacts(self, run, return_raw=False, **kwargs):
        artifacts = run.logged_artifacts(**kwargs)

        if return_raw:
            return artifacts

        return [a.name for a in artifacts]
    
    def download_artifact(self, run, name, dest_path):
        root = Path(dest_path)
        root.mkdir(parents=True, exist_ok=True)

        if self.entity is None:
            artifact_path = f"{self.project_name}/{name}"
        else:
            artifact_path = f"{self.entity}/{self.project_name}/{name}"
    
        artifact = self.api.artifact(artifact_path)
        return artifact.download(root=dest_path)

class MLFlowLogInspector(LogInspector):
    def __init__(self, client, project_name: str|None = None):
        super().__init__()
        self.client = client
        self.project_name = project_name

        if project_name is not None:
            self.set_project(project_name)  

    def set_project(self, project_name: str):
        self.project_name = project_name
        projects = self.client.search_experiments(filter_string=f"name='{project_name}'")
        self.project_ids = [p.experiment_id for p in projects]

    def get_project_names(self):
        projects = self.client.search_experiments()
        return [p.name for p in projects]

    def get_runs(self, experiment_name=None):
        if experiment_name is None:
            runs = self.client.search_runs(self.project_ids)
        else:
            runs = self.client.search_runs(
                self.project_ids,
                filter_string=f"run_name='{experiment_name}'"
            )
            
        runs = sorted(runs, key=lambda run: run.info.start_time)
        return runs

    def get_run_id(self, run):
        return run.info.run_id

    def get_experiment_names(self, runs=None):
        if runs is None:
            runs = self.get_runs()

        return sorted(set(run.info.run_name for run in runs))

    def get_metrics(self, experiment_name, metrics="test/recall_1"):
        if isinstance(metrics, str):
            metrics = [metrics]
            return_dict = False
        else:
            return_dict = True

        runs = self.get_runs(experiment_name)
        metrics = {metric: [] for metric in metrics}

        for run in runs:
            if run.info.end_time is None:
                continue

            for metric in metrics:
                metrics[metric].append(
                    run.data.metrics['summary/' + metric])

        for metric in metrics:
            metrics[metric] = np.array(metrics[metric])

        if return_dict:
            return metrics
        else:
            return next(iter(metrics.values()))
        
    def get_history(self, run, metrics="val/recall_1"):        
        if isinstance(metrics, str):
            metrics = [metrics]
            return_array = True
        else:
            return_array = False

        history = {}

        for metric in metrics:
            h = self.client.get_metric_history(run.info.run_id, metric)

            steps = []
            values = []

            for x in h:
                steps.append(x.step)
                values.append(x.value)

            history[metric] = {'step': np.array(steps), 'value': np.array(values)}

        if return_array:
            return history[metrics[0]]
        else:
            return history

    def _fix_config_key(self, k):
        parts = k.split('.')
        if parts[-1][0] == '-':
            parts[-1] = '~' + parts[-1][1:]
        
        return '.'.join(parts)

    def make_config(self, run, base_for_diff=None, hydra_kwargs=None):
        config = run.data.params
        config = {self._fix_config_key(k): v for k, v in config.items()}

        # fix string representation of values
        for key, value in config.items():
            if isinstance(value, str):
                if value[0] == '[':
                    value = OmegaConf.create(value)
                    value = OmegaConf.to_container(value, resolve=False)
                    config[key] = value
                elif value == 'None':
                    config[key] = None
                elif value in ['True', 'False']:
                    config[key] = value == 'True'
                elif value in ['true', 'false']:
                    config[key] = value == 'true'
                else:
                    try:
                        config[key] = int(value)
                    except (ValueError, TypeError):
                        try:
                            config[key] = float(value)
                        except ValueError:
                            pass

        config = unflatten_dict(config)
        
        if base_for_diff is not None:
            config = config_diff(config, base_for_diff, hydra_kwargs=hydra_kwargs)

        return OmegaConf.create(config)
    
    def list_artifacts(self, run, return_raw=False):
        artifacts = self.client.list_artifacts(run.info.run_id)

        if return_raw:
            return artifacts

        return [a.path for a in artifacts]
    
    def download_artifact(self, run, name, dest_path, add_enclosing_directory=False):
        root = Path(dest_path)
        root.mkdir(parents=True, exist_ok=True)

        if not name.endswith('/') and not add_enclosing_directory:
            name += '/'
        
        return self.client.download_artifacts(run.info.run_id, name, dest_path)
