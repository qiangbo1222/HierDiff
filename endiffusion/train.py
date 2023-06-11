"""
Resume Training:
    python train.py <previous args ...> hydra.run.dir=<previous_dir>
"""
import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.plugins import DDPPlugin

from hx_utils.log import print_config, save_lr_finder

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


logger = logging.getLogger(__name__)
pl._logger.handlers = []
pl._logger.propagate = True

from dataset.mol_tree import *


def element_exists(element):
    try:
        _ = element
        return True
    except:
        return False


def try_resume(cfg, hard_resume=False):
    """resume from previous checkpoint.

    Note
    ----------
    Triggered when `hydra.run.dir` is set to a previous working directory.

    Parameters
    ----------
    cfg : [type]
        [description]
    hard_resume : bool, optional
        If true, cfg will be replaced by `log/csv/hparams.yaml`, by default
        False
    """
    path_cwd = Path(Path(".").resolve())
    logger.info(f"Working directory: {path_cwd}")

    path_last_config = path_cwd / "log" / "csv" / "hparams.yaml"
    if path_last_config.exists():
        logger.info(f"Find previous config: {path_last_config}")
        pre_cfg = OmegaConf.load(path_last_config).cfg
        if hard_resume:
            logger.warning(f"Replace current config with: {path_last_config}.")
            cfg = pre_cfg
        elif element_exists(pre_cfg.logging.wandb) and element_exists(
            cfg.logging.wandb
        ):
            # restore wandb state
            logger.info(f"Will use previous wandb name and version.")
            pre_wandb_name = pre_cfg.logging.wandb.name
            pre_wandb_version = pre_cfg.logging.wandb.version
            logger.info(f"Previous wandb name: {pre_wandb_name}")
            logger.info(f"Previous wandb version: {pre_wandb_version}")
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.logging.wandb.name = pre_wandb_name
                cfg.logging.wandb.version = pre_wandb_version

    path_last_ckpt = path_cwd / "log" / "checkpoints" / "last.ckpt"
    if path_last_ckpt.exists():
        logger.info(f"Find previous checkpoint: {path_last_ckpt}")
        logger.info(
            f"Will use `pl_trainer.resume_from_checkpoint={path_last_ckpt}`"
        )

        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.pl_trainer.resume_from_checkpoint = str(path_last_ckpt)

    return cfg


def set_seed(cfg):
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)


def init_trainer(cfg):
    cfg_trainer = dict(cfg.trainer)

    # migrate pytorch-lightning 1.2.5 -> 1.5.6
    '''
    if pl.__version__ != "1.2.5" and cfg_trainer["accelerator"] == "ddp":
        cfg_trainer["accelerator"] = "gpu"
        cfg_trainer["strategy"] = "ddp"
        cfg_trainer["amp_level"] = None
        cfg_trainer["deterministic"] = False
    '''

    if "logging" in cfg:
        loggers = []
        for _, cfg_log in cfg.logging.items():
            loggers.append(instantiate(cfg_log))
        cfg_trainer["logger"] = loggers
    # print("logging" in cfg)
    # print(cfg_trainer["logger"])
    if cfg.callbacks:
        callbacks = []
        for _, cfg_callback in cfg.callbacks.items():
            callbacks.append(instantiate(cfg_callback))
        cfg_trainer["callbacks"] = callbacks
    '''
    if cfg_trainer["accelerator"] == "ddp" and cfg_trainer["precision"] < 32:
        cfg_trainer["plugins"] = DDPPlugin(find_unused_parameters=False)
        #TODO: modify here
    elif cfg_trainer["strategy"] == "ddp":
        cfg_trainer["strategy"] = DDPPlugin(find_unused_parameters=False)
    '''
    trainer = pl.Trainer(**cfg_trainer)
    return trainer


def init_model(cfg):
    model = instantiate(cfg.model, cfg=cfg, _recursive_=False)
    return model


def init_data(cfg):
    datamodule = instantiate(cfg.dataset)
    return datamodule


def find_lr(trainer, model, datamodule):
    lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
    save_lr_finder(lr_finder)
    logger.info(f"Suggestion: {lr_finder.suggestion()}")


def train_model(trainer, model, datamodule):
    trainer.fit(model, datamodule)


def test_model(trainer, datamodule, ckpt_path="best"):
    trainer.test(datamodule=datamodule, ckpt_path=ckpt_path)

def sample_model(model, num_samples):
    return model.sample(num_samples)


@hydra.main(config_path="conf", config_name="launch")
def main(cfg: DictConfig) -> None:
    cfg = try_resume(cfg, hard_resume=False)
    print_config(cfg)

    set_seed(cfg)
    model = init_model(cfg)
    trainer = init_trainer(cfg)
    datamodule = init_data(cfg)

    if cfg.mode == "find_lr":
        find_lr(trainer, model, datamodule)
    else:
        train_model(trainer, model, datamodule)

        if cfg.run_test:
            test_model(trainer, datamodule)


if __name__ == "__main__":
    main()
