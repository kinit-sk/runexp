import logging
import os
import re
import shutil
import tempfile
import importlib.util
import concurrent.futures
import threading
import time
from abc import ABCMeta, abstractmethod
from .utils import flatten_dict, RandomStatePreserver, preserve_random_state

logger = logging.getLogger(__name__)

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
    def __init__(
        self,
        project_name,
        experiment_name,
        description,
        artifact_mode="files",
        artifact_upload_workers=2,
        artifact_progress_interval=10,
        namespace_summary: bool = True,
    ):
        if artifact_mode not in {"files", "artifacts"}:
            raise ValueError("artifact_mode must be either 'files' or 'artifacts'")
        self.artifact_mode = artifact_mode
        self.artifact_upload_workers = artifact_upload_workers
        self.artifact_progress_interval = artifact_progress_interval
        self.namespace_summary = namespace_summary
        self._artifact_upload_executor = None
        self._artifact_uploads = []
        self._artifact_staging_dirs = []
        self._artifact_upload_lock = threading.Lock()
        self._finished = False
        super().__init__(project_name, experiment_name, description)

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
        summary = {}
        for name, value in metrics.items():
            key = "summary/" + name if self.namespace_summary else name
            summary[key] = value
        if summary:
            self.wandb_run.summary.update(summary)

    def log_artifact(self, path, name=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact path does not exist: {path}")

        if self.artifact_mode == "artifacts":
            return self.wandb_run.log_artifact(path, name=name)

        artifact_name = name if name is not None else MLFlowTracker._infer_name(path)
        return self._log_artifact_as_files(path, artifact_name)

    @staticmethod
    def _relative_files(path):
        if os.path.isfile(path):
            yield os.path.basename(path), path
            return

        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    yield os.path.relpath(file_path, path), file_path
            return

        raise ValueError(f"Path is neither file nor directory: {path}")

    @staticmethod
    def _hardlink_or_copy(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    @staticmethod
    def _format_bytes(size):
        value = float(size)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024 or unit == "TiB":
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TiB"

    @staticmethod
    def _logical_path(*parts):
        return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))

    @staticmethod
    def _progress_bar(done, total, width=24):
        if total <= 0:
            return "[" + (" " * width) + "]   0.0%"
        fraction = min(1.0, max(0.0, done / total))
        filled = int(round(fraction * width))
        return "[" + ("#" * filled) + ("." * (width - filled)) + f"] {fraction * 100:5.1f}%"

    @staticmethod
    def _format_duration(seconds):
        seconds = int(max(0, seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:d}:{seconds:02d}"

    def _get_artifact_upload_executor(self):
        if self._artifact_upload_executor is None:
            self._artifact_upload_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.artifact_upload_workers,
                thread_name_prefix="runexp-wandb-upload",
            )
        return self._artifact_upload_executor

    def _make_artifact_staging_dir(self, path):
        parent = os.path.dirname(os.path.abspath(self.log_dir))
        try:
            staging_dir = tempfile.TemporaryDirectory(
                prefix=".runexp-wandb-upload-",
                dir=parent,
            )
        except OSError:
            staging_dir = tempfile.TemporaryDirectory(prefix=".runexp-wandb-upload-")

        self._artifact_staging_dirs.append(staging_dir)
        return staging_dir.name

    def _upload_run_file(self, upload):
        from wandb.sdk.internal.internal_api import Api as InternalApi
        from wandb.sdk.lib.paths import LogicalPath

        api = InternalApi(
            default_settings={
                "entity": upload["entity"],
                "project": upload["project"],
            }
        )
        api.set_current_run_id(upload["run_id"])

        def progress(_chunk_bytes, uploaded_bytes):
            with self._artifact_upload_lock:
                upload["uploaded"] = uploaded_bytes

        try:
            with open(upload["path"], "rb") as file:
                api.push(
                    {LogicalPath(upload["name"]): file},
                    run=upload["run_id"],
                    entity=upload["entity"],
                    project=upload["project"],
                    progress=progress,
                )
        except Exception as exc:
            with self._artifact_upload_lock:
                upload["failed"] = True
                upload["error"] = exc
            logger.exception("Failed to upload W&B run file %s", upload["name"])
            raise
        else:
            with self._artifact_upload_lock:
                upload["uploaded"] = upload["size"]
                upload["done"] = True
            return upload["name"]

    def _log_artifact_as_files(self, path, artifact_name):
        saved = []
        staging_dir = self._make_artifact_staging_dir(path)
        executor = self._get_artifact_upload_executor()
        files = sorted(self._relative_files(path), key=lambda item: item[0])

        logger.info(
            "Queueing W&B run-file upload path=%s name=%s files=%s size=%s",
            path,
            artifact_name,
            len(files),
            self._format_bytes(sum(os.path.getsize(file_path) for _, file_path in files)),
        )

        for rel_path, file_path in files:
            save_name = self._logical_path(artifact_name, rel_path)
            staged_path = os.path.join(staging_dir, save_name)
            self._hardlink_or_copy(file_path, staged_path)
            upload = {
                "name": save_name,
                "path": staged_path,
                "size": os.path.getsize(staged_path),
                "uploaded": 0,
                "done": False,
                "failed": False,
                "error": None,
                "run_id": self.wandb_run.id,
                "entity": self.wandb_run.entity,
                "project": self.wandb_run.project,
            }
            self._artifact_uploads.append(upload)
            upload["future"] = executor.submit(self._upload_run_file, upload)
            saved.append(save_name)

        return saved

    def _artifact_upload_summary(self):
        with self._artifact_upload_lock:
            total = sum(upload["size"] for upload in self._artifact_uploads)
            uploaded = sum(upload["uploaded"] for upload in self._artifact_uploads)
            done = sum(1 for upload in self._artifact_uploads if upload["done"])
            failed = [upload for upload in self._artifact_uploads if upload["failed"]]
            count = len(self._artifact_uploads)
        return uploaded, total, done, count, failed

    def _wait_for_artifact_uploads(self):
        if not self._artifact_uploads:
            return

        started = time.monotonic()
        progress = None
        last_uploaded = 0
        next_log = 0
        try:
            from tqdm.auto import tqdm

            _, total, _, count, _ = self._artifact_upload_summary()
            progress = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="W&B upload",
                dynamic_ncols=True,
                leave=True,
            )
            progress.set_postfix_str(f"files 0/{count}")
        except Exception:
            progress = None

        try:
            while True:
                uploaded, total, done, count, failed = self._artifact_upload_summary()
                now = time.monotonic()
                delta = max(0, uploaded - last_uploaded)
                if progress is not None and delta:
                    progress.update(delta)
                    progress.set_postfix_str(f"files {done}/{count}")
                last_uploaded = uploaded

                if failed:
                    break
                if done >= count:
                    break
                if progress is None and now >= next_log:
                    elapsed = now - started
                    rate = uploaded / elapsed if elapsed > 0 else 0
                    remaining = (total - uploaded) / rate if rate > 0 else 0
                    logger.info(
                        "W&B run-file upload progress %s %s/%s, files %s/%s, elapsed %s, eta %s",
                        self._progress_bar(uploaded, total),
                        self._format_bytes(uploaded),
                        self._format_bytes(total),
                        done,
                        count,
                        self._format_duration(elapsed),
                        self._format_duration(remaining),
                    )
                    next_log = now + self.artifact_progress_interval
                time.sleep(0.5)
        finally:
            if progress is not None:
                uploaded, _, done, count, _ = self._artifact_upload_summary()
                delta = max(0, uploaded - last_uploaded)
                if delta:
                    progress.update(delta)
                progress.set_postfix_str(f"files {done}/{count}")
                progress.close()

        for upload in self._artifact_uploads:
            upload["future"].result()

        uploaded, total, done, count, _ = self._artifact_upload_summary()
        logger.info(
            "W&B run-file uploads complete %s %s/%s, files %s/%s",
            self._progress_bar(uploaded, total),
            self._format_bytes(uploaded),
            self._format_bytes(total),
            done,
            count,
        )

    def finish(self):
        if self._finished:
            return
        self._finished = True
        upload_error = None
        try:
            try:
                self._wait_for_artifact_uploads()
            except Exception as exc:
                upload_error = exc
            if self._artifact_upload_executor is not None:
                self._artifact_upload_executor.shutdown(
                    wait=upload_error is None,
                    cancel_futures=upload_error is not None,
                )
            self.wandb_run.finish()
        finally:
            while self._artifact_staging_dirs:
                self._artifact_staging_dirs.pop().cleanup()
        if upload_error is not None:
            raise upload_error

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

        def __init__(
            self,
            tracker,
            log_artifacts: bool = True,
            log_final_model: bool = False,
            log_best_model: bool = False,
            final_name: str = "final-checkpoint",
            best_name: str = "best-checkpoint",
        ):
            super().__init__()
            self.tracker = tracker
            self._log_artifacts = log_artifacts
            self._log_final_model = log_final_model
            self._log_best_model = log_best_model
            self._final_name = final_name
            self._best_name = best_name

            self._logged_final = False

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
            if not state.is_world_process_zero:
                return
                
            if self._log_artifacts:
                ckpt_dir = f"checkpoint-{state.global_step}"
                artifact_path = os.path.join(args.output_dir, ckpt_dir)
                self.tracker.log_artifact(artifact_path)

            if self._log_final_model and state.global_step >= state.max_steps and not self._logged_final:
                ckpt_dir = f"checkpoint-{state.global_step}"
                artifact_path = os.path.join(args.output_dir, ckpt_dir)
                self.tracker.log_artifact(artifact_path, name=self._final_name)
                self._logged_final = True

        def on_step_end(self, args, state, control, **kwargs):
            if self._log_final_model and state.global_step >= state.max_steps:
                control.should_save = True
            return control
        
        def on_train_end(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return

            if self._log_best_model:
                best_ckpt = getattr(state, "best_model_checkpoint", None)
                if best_ckpt:
                    self.tracker.log_artifact(best_ckpt, name=self._best_name)

            best_step = getattr(state, "best_global_step", None)

            if best_step:
                self.tracker.log_summary({"best_global_step": best_step})

#------------------------------------------------------------------------------
# Integration with torchrl loggers
#------------------------------------------------------------------------------

if importlib.util.find_spec("torchrl") is not None:
    from collections.abc import Sequence

    from torch import Tensor
    from torchrl.record.loggers.common import Logger

    class TorchRLTrackerLogger(Logger):
        """torchrl Logger wrapper for runexp ExperimentTracker."""

        def __init__(
            self,
            tracker: ExperimentTracker,
            *,
            setup: bool = False,
            config=None,
            video_fps: int = 30,
        ) -> None:
            self.tracker = tracker
            self.video_fps = video_fps
            if setup:
                self.tracker.setup(config)
            super().__init__(exp_name=tracker.experiment_name, log_dir=tracker.log_dir)

        def _create_experiment(self) -> ExperimentTracker:
            return self.tracker

        def log_scalar(self, name: str, value: float, step: int | None = None) -> None:
            self.tracker.log_metrics({name: value}, step=step)

        def log_video(
            self, name: str, video: Tensor, step: int | None = None, **kwargs
        ) -> None:
            try:
                import torchvision
            except ImportError as exc:
                raise ImportError(
                    "Logging a video requires torchvision to be installed."
                ) from exc

            if video.ndim == 5:
                video = video[-1]  # N T C H W -> T C H W
            video = video.detach().cpu().permute(0, 2, 3, 1)  # T C H W -> T H W C
            if video.size(dim=-1) != 3:
                raise ValueError("Only videos with 3 color channels are supported.")

            fps = kwargs.pop("fps", self.video_fps)
            with tempfile.TemporaryDirectory() as temp_dir:
                filename = f"{name}_step_{step:04}.mp4" if step is not None else f"{name}.mp4"
                path = os.path.join(temp_dir, filename)
                torchvision.io.write_video(filename=path, video_array=video, fps=fps)
                self.tracker.log_artifact(path, name=filename)

        def log_hparams(self, cfg) -> None:
            if cfg is None:
                return

            cfg_dict = cfg
            if type(cfg) is not dict:
                try:
                    from omegaconf import OmegaConf

                    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
                except Exception:
                    cfg_dict = {"hparams": cfg}

            if isinstance(cfg_dict, dict):
                flat = flatten_dict(cfg_dict)
            else:
                flat = {"hparams": cfg_dict}

            numeric = {
                k: v for k, v in flat.items() if isinstance(v, (int, float))
            }
            if numeric:
                self.tracker.log_summary(
                    {f"hparams/{k}": v for k, v in numeric.items()}
                )

            try:
                import json

                with tempfile.TemporaryDirectory() as temp_dir:
                    path = os.path.join(temp_dir, "hparams.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(
                            cfg_dict,
                            f,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=True,
                            default=str,
                        )
                    self.tracker.log_artifact(path, name="hparams.json")
            except Exception:
                pass

        def __repr__(self) -> str:
            return (
                f"TorchRLTrackerLogger(exp_name={self.exp_name}, "
                f"tracker={self.tracker.__class__.__name__})"
            )

        def log_histogram(self, name: str, data: Sequence, **kwargs):
            raise NotImplementedError("Histogram logging is not supported.")
