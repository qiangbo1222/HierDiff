import os
import pickle
import random

import numpy as np
import rdkit
import rdkit.Chem as Chem
import torch
import torch.nn as nn
from rdkit.Chem import AllChem, rdMolDescriptors
from torch.utils import data
from torch.utils.data import DataLoader

from data_utils.mol_tree import MolTree


#only consider the position of the tree node (without angle and torision)
class mol_Tree_pos(data.Dataset):

    def __init__(self, args, dataname, split=None, vocab=None, mode='flow'):

        self.dataname = dataname
        self.mode = mode
        self.args = args
        if dataname == 'crossdock':
            self.split = split
            self.datapath_list = os.listdir(args.data_dir)
            self.vocab = vocab
        elif dataname == 'GEOM_drug':
            self.datapath_list = os.listdir(args.data_dir)
            self.vocab = vocab
        else:
            raise ValueError('dataname not supported')
        
        self.node_coarse_type = args.node_coarse_type
    
    def __len__(self):
        if self.dataname == 'crossdock':
            return len(self.split)
        elif self.dataname ==  'GEOM_drug':
            return len(self.datapath_list)
    
    def __getitem__(self, index):
        if self.dataname == 'crossdock':
            datapath = self.datapath_list[self.split[index]]
        else:
            datapath = self.datapath_list[index]
        with open(os.path.join(self.args.data_dir, datapath), 'rb') as f:
            tree = pickle.load(f)
        if isinstance(tree, list):
            random.shuffle(tree)
            tree = tree[0]
        random_choose = random.randint(0, len(tree.nodes)-1)
        node_chose = tree.nodes[random_choose]
        feature_tensor = []
        vocab_tensor = []
        size_tensor = []
        pos_tensor = []
        for i, node in enumerate(tree.nodes):
            pos_tensor.append(torch.tensor(node.pos))
            if self.node_coarse_type == 'prop':
                fp_fix = np.array(self.vocab.fp_df.loc[node.smiles])
                contribute_TPSA = rdMolDescriptors._CalcTPSAContribs(tree.mol3D)
                contribute_ASA = rdMolDescriptors._CalcLabuteASAContribs(tree.mol3D)
                tpsa = sum([contribute_TPSA[i] for i in node.clique])/10
                asa = (sum([list(contribute_ASA[0])[i] for i in node.clique]) + contribute_ASA[1])/10
                node.fp = np.concatenate((np.array([node.hbd]), fp_fix, np.array([tpsa]), np.array([asa])))
            elif self.node_coarse_type == 'elem':
                fp_fix = np.array(self.vocab.fp_df.loc[node.smiles])
                node.fp = fp_fix

            size_tensor.append(torch.tensor(Chem.MolFromSmiles(node.smiles).GetNumHeavyAtoms()))
            if i == random_choose:
                vocab_tensor.append(torch.tensor(780))
                feature_tensor.append(torch.zeros(node.fp.shape))
            else:
                vocab_tensor.append(torch.tensor(node.wid))
                feature_tensor.append(torch.tensor(node.fp))
        feature_tensor = torch.stack(feature_tensor)
        vocab_tensor = torch.stack(vocab_tensor)
        size_tensor = torch.stack(size_tensor)
        pos_tensor = torch.stack(pos_tensor)
        edges = torch.tensor(tree.adj_matrix).nonzero().T.tolist()
        edges = get_bfs_depth_edges(edges, random_choose, feature_tensor.shape[0], sample=True)
        return feature_tensor, vocab_tensor, torch.tensor(node_chose.wid), size_tensor, pos_tensor, edges, tree.adj_matrix, random_choose
            
        
def PadCollate(batch, args):
    max_len = max([d[0].shape[0] for d in batch])
    feature_tensor = torch.zeros(len(batch), max_len, args.int_feature_size + args.num_continutes_feature)
    vocab_tensor = torch.zeros(len(batch), max_len, dtype=torch.long)
    size_tensor = torch.zeros(len(batch),max_len, dtype=torch.long)
    label_tensor = torch.zeros(len(batch), dtype=torch.long)
    pos_tensor = torch.zeros(len(batch), max_len, 3)
    mask_tensor = torch.zeros(len(batch), max_len, 1)
    max_depth = max([len(d[5]) for d in batch])
    edges_pad = [[[], []] for _ in range(max_depth)]
    predict_idx = []
    val = torch.zeros(len(batch))
    for i, (feature, vocab, label, size, pos, edges, adj_matrix, pred) in enumerate(batch):
        feature_tensor[i, :feature.shape[0], :] = feature
        vocab_tensor[i, :vocab.shape[0]] = vocab
        size_tensor[i, :size.shape[0]] = size
        label_tensor[i] = label
        mask_tensor[i, :feature.shape[0]] = 1
        pos_tensor[i, :pos.shape[0], :] = pos
        for j, edge in enumerate(edges):
            edges_pad[j][0].extend(list_add(edge[0], i * max_len))
            edges_pad[j][1].extend(list_add(edge[1], i * max_len))
        predict_idx.append(pred)
        val[i] = torch.sum(torch.tensor(adj_matrix), dim=1)[pred]
    return {'feature': feature_tensor, 
            'pos': pos_tensor,
            'vocab': vocab_tensor, 
            'label': label_tensor, 
            'size':size_tensor, 
             'mask': mask_tensor,
             'edges': edges_pad,
             'predict_idx': predict_idx,
             'val': val}

def get_bfs_depth_edges(edges, center, n_nodes, sample=False):
    depth = [0] * n_nodes
    depth[center] = 1
    queue = [center]
    while len(queue) > 0:
        cur = queue.pop(0)
        for i in range(len(edges[0])):
            if edges[0][i] == cur and depth[edges[1][i]] == 0:
                depth[edges[1][i]] = depth[edges[0][i]] + 1
                queue.append(edges[1][i])
    edges_depth = [[[], []]  for _ in range(max(depth) - 1)]
    for i in range(len(edges[0])):
        if depth[edges[0][i]] < depth[edges[1][i]]:
            edges_depth[depth[edges[1][i]] - 2][0].append(edges[1][i])
            edges_depth[depth[edges[1][i]] - 2][1].append(edges[0][i])
    edges_depth.reverse()
    if sample:
        walk_nodes = random_walk(edges, center, random.randint(0, n_nodes - 1))
        edges_walk = [[[], []]  for _ in range(max(depth) - 1)]
        for layer in range(len(edges_depth)):
            for e in range(len(edges_depth[layer][0])):
                if edges_depth[layer][0][e] in walk_nodes and edges_depth[layer][1][e] in walk_nodes:
                    edges_walk[layer][0].append(edges_depth[layer][0][e])
                    edges_walk[layer][1].append(edges_depth[layer][1][e])
        edges_depth = [edges_walk[i] for i in range(len(edges_walk)) if len(edges_walk[i][0]) > 0]
    return edges_depth

def list_add(list1, add):
    return [list1[i] + add for i in range(len(list1))]

def random_walk(edges, start, length):
    walk = [start]
    cur = start
    stop_walk = [0 for _ in range(length)]
    while len(walk) < length:
        cur = random.choice(walk)
        next_node = [edges[1][i] for i in range(len(edges[1])) if edges[0][i] == cur and edges[1][i] not in walk]
        if len(next_node) == 0:
            stop_walk[walk.index(cur)] = 1
            if sum(stop_walk) == len(walk):
                break
            continue
        cur = random.choice(next_node)
        walk.append(cur)
    return walk
