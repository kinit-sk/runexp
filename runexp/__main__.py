import sys
import os
import hydra
from .runner import ExperimentRunner
from omegaconf import DictConfig

@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="config"
)
def main(config: DictConfig):
    experiment = ExperimentRunner(config)
    experiment()

if __name__ == "__main__":
    if "--help" in sys.argv:
        try:
            main()
        except Exception:
            print("Use --hydra-help to view Hydra specific help.")
            raise
    else:
        main()