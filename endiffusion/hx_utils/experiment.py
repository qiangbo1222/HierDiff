import logging
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class Experiment:

    def __init__(self, logdir, wandb_entity="") -> None:
        self.logdir = Path(logdir).expanduser()
        self.wandb_entity=wandb_entity
        self._load_config()

    @property
    def wandb_run_path(self):
        return f"{self.wandb_entity}/{self.wandb_project}/{self.wandb_run_id}"

    @property
    def wandb_run_id(self):
        return self.config.logging.wandb.version

    @property
    def wandb_project(self):
        return self.config.logging.wandb.project

    @property
    def ckpt_dir(self):
        return self.logdir / "log" / "checkpoints"

    @property
    def best_ckpt_path(self):
        ckpts = [
            ckpt
            for ckpt in list(self.ckpt_dir.glob("*.ckpt"))
            if "last" not in ckpt.name
        ]
        return ckpts[-1]

    @property
    def last_ckpt_path(self):
        return self.ckpt_dir / "last.ckpt"

    def _load_config(self):
        self.config_path = self.logdir / "log" / "csv" / "hparams.yaml"
        if self.config_path.exists():
            self.config = OmegaConf.load(self.config_path)["cfg"]
        else:
            # read hydra config
            hydra_config_path = self.logdir / ".hydra" / "hydra.yaml"
            hydra_config = OmegaConf.load(hydra_config_path)

            # read config
            names = hydra_config.hydra.run.dir.split("/")
            time_fmt = "%Y-%m-%d_%H-%M-%S"
            log_time = datetime.strptime(names[-1], time_fmt)
            OmegaConf.register_new_resolver("now", log_time.strftime)
            self.config_path = self.logdir / ".hydra" / "config.yaml"
            self.config = OmegaConf.load(self.config_path)
        logger.info(f"config has been read from {self.config_path}")

    def get_model(self, ckpt="best"):
        if ckpt == "best":
            ckpt_path = self.best_ckpt_path
        else:
            ckpt_path = self.last_ckpt_path
        logger.info(f"loading {ckpt_path}")
        pipeline_cfg = dict(self.config["pipeline"])
        pipeline_cfg[
            "_target_"
        ] = f"{pipeline_cfg['_target_']}.load_from_checkpoint"
        pipeline_cfg["checkpoint_path"] = ckpt_path
        model = instantiate(pipeline_cfg)
        model.cfg = self.config
        return model