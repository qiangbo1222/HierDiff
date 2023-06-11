import logging
import os
import numpy as np
import pickle
from pathlib import Path
import tqdm

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
import rdkit
from rdkit import Chem
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.plugins import DDPPlugin

from hx_utils.log import print_config, save_lr_finder


def init_model(cfg):
    model = instantiate(cfg.model, cfg=cfg, _recursive_=False)
    return model

@hydra.main(config_path="conf", config_name="sample")
def sample(cfg):
    model = init_model(cfg)#bug on cfg when using load_from_checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(cfg.checkpoint)['state_dict']
    else:
        checkpoint = torch.load(cfg.checkpoint, map_location=torch.device('cpu'))['state_dict']
    for key in list(checkpoint):
        checkpoint[key.replace('model.', '')] = checkpoint.pop(key)

    model.load_state_dict(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #SAS_context_range = [i/10 for i in range(-4, 50)]
    results = model.sample_batches(**cfg.sample, device=device, context_range=None)
    with open('sample_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    

if __name__ == "__main__":
    sample()
