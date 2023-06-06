import argparse
import logging
import os
import pickle
import time
from os.path import join

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from yaml import Dumper, Loader

from data_utils.mol_tree import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils import data
from torch.utils.data import DataLoader, random_split

import data_utils.dataset_refine as dataset
import train_utils
from trainmodule.Refine import Refine

parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    type=str,
                    default='conf')

args = parser.parse_args()
cfg = edict({
    'dataset':
    yaml.load(open(join(args.config_path, 'dataset/refine.yaml')),
              Loader=Loader),
    'model':
    yaml.load(open(join(args.config_path, 'model/refine.yaml')),
              Loader=Loader),
    'optim':
    yaml.load(open(join(args.config_path, 'optim/adamw.yaml')), Loader=Loader),
    'scheduler':
    yaml.load(open(join(args.config_path, 'scheduler/step.yaml')),
              Loader=Loader),
    'trainer':
    yaml.load(open(join(args.config_path, 'trainer/default.yaml')),
              Loader=Loader),
    'analyze':
    join(args.config_path, 'analyze/crossdock.yaml'),
})
#read in the config for model and training


class crossdock_data_module(LightningDataModule):
    def __init__(self, cfg):
        with open(cfg.dataset.dataset.split_dir, 'rb') as f:
            split = pickle.load(f)

    def train_dataloader(self):
        crossdock_data_train = dataset.mol_Tree_pos(cfg.dataset.dataset,
                                                    dataname='crossdock',
                                                    split=split['train'])
        print('train data size:', len(crossdock_data_train))
        dataloader_train = data.DataLoader(
            crossdock_data_train,
            **cfg.dataset,
            collate_fn=lambda batch: dataset.PadCollate(batch, args))
        return dataloader_train

    def val_dataloader(self):
        crossdock_data_valid = dataset.mol_Tree_pos(cfg.dataset.dataset,
                                                    dataname='crossdock',
                                                    split=split['valid'])
        dataloader_valid = data.DataLoader(
            crossdock_data_valid,
            **cfg.dataset,
            collate_fn=lambda batch: dataset.PadCollate(batch, args))
        return dataloader_valid

class GEOM_data_module(LightningDataModule):
    def __init__(self):
        super().__init__()
        with open(cfg.dataset.vocab_path, "r") as f:
            vocab = [x.strip() for x in f.readlines()]
        vocab = Vocab(vocab, fp_df=pd.read_csv(cfg.dataset.vocab_fp_path, index_col=0))
        whole_dataset = dataset.mol_Tree_pos(cfg.dataset.dataset, dataname='GEOM_drug', vocab=vocab)
        torch.manual_seed(2022)
        train_size, valid_size = int(len(whole_dataset) * 0.8), int(len(whole_dataset) * 0.1)
        test_size = len(whole_dataset) - train_size - valid_size
        dataset_list = torch.utils.data.random_split(whole_dataset, [train_size, valid_size, test_size])
        self.dataloader_train = DataLoader(dataset_list[0], **cfg.dataset.dataloader, collate_fn=lambda batch: dataset.PadCollate(batch, cfg.dataset))
        self.dataloader_valid = DataLoader(dataset_list[1], **cfg.dataset.dataloader, collate_fn=lambda batch: dataset.PadCollate(batch, cfg.dataset))
    def train_dataloader(self):
        return self.dataloader_train
    def val_dataloader(self):
        return self.dataloader_valid

tb_logger = TensorBoardLogger(cfg.trainer.default_root_dir, name='refine_element_f')
trainer = Trainer(**cfg.trainer, logger=tb_logger, callbacks=[EarlyStopping(monitor="val_accuracy", mode="max")])
model = Refine(cfg)
trainer.fit(model, GEOM_data_module())