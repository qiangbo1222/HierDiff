import os
import pickle
import random
import sys

import numpy as np
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import torch
import torch.nn as nn
from rdkit.Chem import (QED, AllChem, Descriptors, Descriptors3D, RDConfig,
                        rdMolDescriptors)
from torch.utils import data
from torch.utils.data import DataLoader

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
#import sascorer

#from dataset.kinase.eval_scorer import *
from dataset.mol_tree import MolTree

# from data_utils.mol_tree import MolTree

RESIDUE_LIST = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
#only consider the position of the tree node (without angle and torision)
class mol_Tree_pos(data.Dataset):

    def __init__(self, args, dataname, split=None, vocab=None, mode='flow'):

        self.dataname = dataname
        self.mode = mode
        self.cwd = '../../../../../../../'
        args.data_dir = os.path.join(self.cwd, args.data_dir)
        self.args = args
        
        print(args)
        if dataname == 'crossdock':
            self.split = split
            self.datapath_list = os.listdir(self.args.data_dir)
            
        elif dataname == 'GEOM_drug':
            self.datapath_list = os.listdir(self.args.data_dir)

        elif dataname == 'crossdock_cond':
            self.datapath_list = os.listdir(self.args.data_dir)
            self.datapath_list = [p for p in self.datapath_list if p.startswith(split)]

        else:
            raise ValueError('dataname not supported')
        self.vocab = vocab
        self.node_coarse_type = args.node_coarse_type
    
    def __len__(self):
        if self.dataname == 'crossdock':
            return len(self.split)
        elif self.dataname ==  'GEOM_drug':
            return len(self.datapath_list)
        elif self.dataname == 'crossdock_cond':
            return len(self.datapath_list)
    
    def __getitem__(self, index):
        if self.dataname == 'crossdock':
            datapath = self.datapath_list[self.split[index]]
            with open(os.path.join(self.args.data_dir, datapath), 'rb') as f:
                tree = pickle.load(f)
        elif self.dataname == 'GEOM_drug':
            datapath = self.datapath_list[index]
            with open(os.path.join(self.args.data_dir, datapath), 'rb') as f:
                tree = pickle.load(f)
        elif self.dataname == 'crossdock_cond':
            datapath = self.datapath_list[index]
            with open(os.path.join(self.args.data_dir, datapath), 'rb') as f:
                tree, protein_data = pickle.load(f)
        
        if isinstance(tree, list):
            random.shuffle(tree)
            tree = tree[0]
        for node in tree.nodes:
            if self.node_coarse_type == 'prop':
                fp_fix = np.array(self.vocab.fp_df.loc[node.smiles])
                contribute_TPSA = rdMolDescriptors._CalcTPSAContribs(tree.mol3D)
                contribute_ASA = rdMolDescriptors._CalcLabuteASAContribs(tree.mol3D)
                tpsa = sum([contribute_TPSA[i] for i in node.clique])/10
                asa = (sum([list(contribute_ASA[0])[i] for i in node.clique]) + contribute_ASA[1])/10
                node.fp = np.concatenate((np.array([node.hbd]), fp_fix, np.array([tpsa]), np.array([asa])))
            elif self.node_coarse_type == 'elem':
                node.fp = fp_fix
        feature_tensor = []
        pos_tensor = []
        for node in tree.nodes:
            #size = node.mol.GetNumHeavyAtoms()
            #feature_tensor.append(torch.cat([torch.tensor(node.fp), torch.tensor([size])]))
            context = 0#change to any scalar property you want to condition on
            feature_tensor.append(torch.tensor(node.fp))
            pos_tensor.append(torch.tensor(node.pos))
        feature_tensor = torch.stack(feature_tensor)
        pos_tensor = torch.stack(pos_tensor)
        if self.dataname != "crossdock_cond":
            return {'feat': feature_tensor, 'position': pos_tensor, 'adj_matrix': tree.adj_matrix, 'context': context}
        else:
            protein_feat = protein_data["residue_type"]
            protein_pos = np.array(protein_data["coord"])
            protein_feat = torch.tensor([RESIDUE_LIST.index(x) + 1 for x in protein_feat])
            protein_pos = torch.tensor(protein_pos)
            if protein_pos.shape[0] == 0:
                return self.__getitem__(index + 1)#TODO replace this later with filtered data
            return {'feat': feature_tensor, 'position': pos_tensor, 'adj_matrix': tree.adj_matrix, 'context': context, 'protein_feat': protein_feat.long(), 'protein_pos': protein_pos}
        
def PadCollate(batch, args):
    max_len = max([x['feat'].shape[0] for x in batch])
    max_pos = max([x['position'].shape[0] for x in batch])
    feat_tensor = torch.zeros([len(batch), max_len, args.feature_size])
    context_tensor = torch.zeros([len(batch), max_len, 1])
    feat_mask = torch.zeros([feat_tensor.shape[0],feat_tensor.shape[1],1])
    pos_tensor = torch.zeros([len(batch), max_pos, 3])
    pos_mask = torch.zeros(pos_tensor.shape)
    adj_matrix_tensor = torch.zeros([len(batch), max_len, max_len])
    adj_matrix_mask = torch.zeros(adj_matrix_tensor.shape)
    if 'protein_feat' in batch[0].keys():
        max_protein_len = max([x['protein_feat'].shape[0] for x in batch])
        protein_feat_tensor = torch.zeros([len(batch), max_protein_len], dtype=torch.long)
        protein_pos_tensor = torch.zeros([len(batch), max_protein_len, 3])
        protein_feat_mask = torch.zeros([protein_feat_tensor.shape[0],protein_feat_tensor.shape[1],1])
        protein_edge_mask = torch.zeros([protein_feat_tensor.shape[0],protein_feat_tensor.shape[1],protein_feat_tensor.shape[1]])
    for i, sample in enumerate(batch):
        feat_tensor[i, :sample['feat'].shape[0], :] = sample['feat']
        context_tensor[i, :sample['feat'].shape[0], :] = sample['context']
        feat_mask[i, :sample['feat'].shape[0], :] = 1
        pos_tensor[i, :sample['position'].shape[0], :] = sample['position']
        pos_mask[i, :sample['position'].shape[0], :] = 1
        adj_matrix_tensor[i, :sample['adj_matrix'].shape[0], :sample['adj_matrix'].shape[1]] = torch.tensor(sample['adj_matrix'])
        adj_matrix_mask[i, :sample['adj_matrix'].shape[0], :sample['adj_matrix'].shape[1]] = 1 - torch.eye(sample['adj_matrix'].shape[0])
        if 'protein_feat' in sample.keys():
            protein_feat_tensor[i, :sample['protein_feat'].shape[0]] = sample['protein_feat']
            protein_pos_tensor[i, :sample['protein_pos'].shape[0], :] = sample['protein_pos']
            protein_feat_mask[i, :sample['protein_feat'].shape[0], :] = 1
            protein_edge_mask[i, :sample['protein_feat'].shape[0], :sample['protein_feat'].shape[0]] = 1 - torch.eye(sample['protein_feat'].shape[0])
    
    feat_mask = feat_mask.bool()
    adj_matrix_mask = adj_matrix_mask.bool()

    if 'protein_feat' in batch[0].keys():
        return {'node_feature':feat_tensor, 
                'atom_mask':feat_mask, 
                'positions':pos_tensor, 
                'pos_mask':pos_mask, 
                'edge_mask':adj_matrix_mask, 
                'context':context_tensor, 
                'protein_feat':protein_feat_tensor, 
                'protein_pos':protein_pos_tensor, 
                'protein_feat_mask':protein_feat_mask.bool(),
                'protein_edge_mask':protein_edge_mask.bool()}
                
    return {'node_feature':feat_tensor, 'atom_mask':feat_mask, 'positions':pos_tensor, 'edge_mask':adj_matrix_mask, 'context':context_tensor}
