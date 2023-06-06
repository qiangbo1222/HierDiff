from typing import Any, Dict

import torch
from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import DataLoader, random_split

from dataset.blur_utils import PadCollate, mol_Tree_pos
from dataset.mol_tree import *


class GEOM_data(LightningDataModule):
    def __init__(self, 
        name,
        args: Dict[str,Any]
        ):
        super().__init__()
        with open(args.vocab_path, "r") as f:
            vocab = [x.strip() for x in f.readlines()]
        vocab = Vocab(vocab, fp_df=pd.read_csv(args.vocab_fp_path, index_col=0))
        whole_dataset = mol_Tree_pos(args = args, dataname=args.data_name, vocab=vocab, split="train")
        torch.manual_seed(2022)
        train_size, valid_size = int(len(whole_dataset) * 0.9), int(len(whole_dataset) * 0.1)
        test_size = len(whole_dataset) - train_size - valid_size
        dataset_list = torch.utils.data.random_split(whole_dataset, [train_size, valid_size, test_size])
        self.dataloader_train = DataLoader(dataset_list[0], **args.dataloader_config, collate_fn=lambda batch: PadCollate(batch, args))
        self.dataloader_valid = DataLoader(dataset_list[1], **args.dataloader_config, collate_fn=lambda batch: PadCollate(batch, args))
        
    def train_dataloader(self):
        return self.dataloader_train

    def val_dataloader(self):
        return self.dataloader_valid
