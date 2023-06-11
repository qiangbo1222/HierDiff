import argparse
import copy
import itertools
import logging
import os
import pickle
import random
import sys
import warnings
from queue import PriorityQueue


import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append(".")
sys.path.append("../generation/jtnn")

import data_utils.dataset_denoise as dataset
import numpy as np
import rdkit
import torch
import torch.nn as nn
import tqdm
import yaml
from data_utils.data_diffuse import get_dfs_order
from data_utils.mol_tree import *
from data_utils.MPNN_pattern import dfs_bidirection
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, random_split
from yaml import Dumper, Loader

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)


from models.edge_denoise import Edge_denoise
from models.model_refine import Node2Vec

from jtnn.jtnn_dec import can_assemble

parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    type=str,
                    default='conf')
parser.add_argument('--start_num',
                    type=int,
                    default=0)
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

cfg = edict({
    'model':
    yaml.load(open(os.path.join(args.config_path, 'model/edge_denoise.yaml')),
            Loader=Loader),
    'model_refine':
    yaml.load(open(os.path.join(args.config_path, 'model/refine.yaml')),
            Loader=Loader),
    'dataset':
        yaml.load(open(os.path.join(args.config_path, 'dataset/denoise.yaml')),
                Loader=Loader),
    'generation':
    yaml.load(open(os.path.join(args.config_path, 'generation/edge_denoise.yaml')), Loader=Loader)
})

def prepare_data(feat, position, size, adj_matrix, node_types={}, vocab_size=780):
    feat_vocab = torch.tensor([node_types[i] if i in node_types.keys() else vocab_size for i in range(feat.shape[0])]).unsqueeze(0).T
    feat = torch.cat([feat, feat_vocab], dim=1)
    return {'feat': feat, 
            'position': torch.tensor(position),
            'size': torch.tensor(size),
            'adj_matrix': torch.tensor(adj_matrix)}


def pad_data(data, device):
    node_max = max([d['feat'].shape[0] for d in data])
    bs = len(data)
    node_feat = torch.zeros(bs, node_max, data[0]['feat'].shape[1])
    node_mask = torch.zeros(bs, node_max, data[0]['feat'].shape[1])
    node_size = torch.zeros(bs, node_max, 1)
    node_position = torch.zeros(bs, node_max, data[0]['position'].shape[1])
    node_adj_matrix = torch.zeros(bs, node_max, node_max)
    edge_mask = torch.zeros(bs, node_max, node_max)
    for i, d in enumerate(data):
        node_feat[i, :d['feat'].shape[0], :] = d['feat']
        node_mask[i, :d['feat'].shape[0], :] = 1
        node_size[i, :d['feat'].shape[0], :] = d['size']
        node_position[i, :d['position'].shape[0], :] = d['position']
        node_adj_matrix[i, :d['adj_matrix'].shape[0], :d['adj_matrix'].shape[0]] = d['adj_matrix']
        edge_mask[i, :d['adj_matrix'].shape[0], :d['adj_matrix'].shape[0]] = 1 - torch.eye(d['adj_matrix'].shape[0])
    return {'node_feat': [node_feat.to(device), node_mask.to(device)],
            'node_size': node_size.to(device),
            'node_pos': node_position.to(device),
            'search_adj_matrix': node_adj_matrix.to(device),
            'edge_mask': edge_mask.to(device)}


def tree_to_data(tree: MolTree, vocab_size=780, feat_size=3, context_size=0):#TODO remember to move into config files
    """
    Convert a tree to a dict of tensors.
    """
    node_types = {}
    node_feat = torch.zeros([len(tree.nodes), feat_size + context_size + 1])
    node_pos = torch.zeros([len(tree.nodes), 3])
    node_size = torch.zeros([len(tree.nodes), 1])
    for i, node in enumerate(tree.nodes):
        if node.wid is not None:
            node_types[i] = node.wid
            dis_type = 1
        else:
            dis_type = 0
        if context_size > 0:
            node_feat[i] = torch.cat([torch.tensor(node.fp[:feat_size]), torch.tensor([dis_type], dtype=torch.float), torch.tensor(node.fp[feat_size:])])
        else:
            node_feat[i] = torch.cat([torch.tensor(node.fp), torch.tensor([dis_type], dtype=torch.float)])
        node_pos[i] = torch.tensor(node.pos)
        node_size[i] = torch.tensor(node.size)
    return prepare_data(node_feat, node_pos, node_size, tree.adj_matrix, node_types, vocab_size)


def handle_wrong_sizes(size, vocab):
    size_perm = [vocab.get_size(size + perm) for perm in [-1, 1, -2, 2]]
    size_perm_l = [len(p) for p in size_perm]
    size_perm = size_perm[size_perm_l.index(max(size_perm_l))]
    if max(size_perm_l) == 0:
        return random.choice(vocab.mol_sizes)
    else:
        return size_perm 

def handle_wrong_array(array, vocab):
    size = np.sum(array)
    size_inds = vocab.get_size(size)
    if len(size_inds) > 0:
        return size_inds
    else:
        return vocab.get_size(handle_wrong_sizes(size, vocab))

def update_trees(model, model_refine,trees, vocab, beam_size=5, device=torch.device('cpu'), refine=False):
    """
    return updated trees(add edge, new denoised node) and the logp of the new trees
    parrall the trees into model
    """
    #refine trees
    if refine:
        trees = [model_refine.check_tree(t, vocab, device) for t in trees]
        trees, logp_refine, refined = [d[0] for d in trees], [d[1] for d in trees], [d[2] for d in trees]
        refined_trees = [t for i, t in enumerate(trees) if refined[i]]
        logp_refine = [p for i, p in enumerate(logp_refine) if refined[i]]
        trees = [t for i, t in enumerate(trees) if not refined[i]]
    else:
        logp_refine = []
        refined_trees = []
    if len(trees) > 0:
        data_batch = [tree_to_data(t.tree) for t in trees]
        data_batch = pad_data(data_batch, model.device)
        edges_result, node_predict, node_array_predict, adj_matrix = model.sample_AR(data_batch)
        #print(edges_result)
        new_trees = [[copy.deepcopy(trees[i]) for _ in range(beam_size)] for i in range(len(trees))]
        logp_batch = []
        for i, t in enumerate(trees):
            array_inds = node_array_predict[i]
            predict = nn.LogSoftmax(dim=-1)(node_predict[i, array_inds].detach()).cpu()
            if len(array_inds) < beam_size:
                beam_array_cut = len(array_inds)
            else:
                beam_array_cut = beam_size
            max_ind = torch.topk(predict, beam_array_cut)[1]
            max_ind = [array_inds[i] for i in max_ind]
            cand_smiles = [vocab.get_smiles(i) for i in max_ind]
            cand_nodes = []
            for j in range(beam_array_cut):
                if len(edges_result[i]) == 2:
                    n = MolTreeNode(cand_smiles[j], trees[i].tree.nodes[edges_result[i][1]].pos, hbd=trees[i].tree.nodes[edges_result[i][1]].fp[0], vocab=vocab)
                    n.size = Chem.MolFromSmiles(cand_smiles[j]).GetNumHeavyAtoms()
                    n.fp = vocab.fp_df.loc[cand_smiles[j]].values
                    #n.pos = n.pos + x_perturb[i].cpu().detach().numpy()
                    if n.fp.shape < trees[i].tree.nodes[edges_result[i][1]].fp.shape:#add context
                        n.fp = np.concatenate([n.fp, [trees[i].tree.nodes[edges_result[i][1]].fp[-1]]])
                    cand_nodes.append(n)
                else:
                    n = MolTreeNode(cand_smiles[j], trees[i].tree.nodes[edges_result[i][0]].pos, hbd=trees[i].tree.nodes[edges_result[i][0]].fp[0], vocab=vocab)
                    n.size = Chem.MolFromSmiles(cand_smiles[j]).GetNumHeavyAtoms()
                    n.fp = vocab.fp_df.loc[cand_smiles[j]].values
                    #n.pos = n.pos + x_perturb[i].cpu().detach().numpy()
                    if n.fp.shape < trees[i].tree.nodes[edges_result[i][0]].fp.shape:#add context
                        n.fp = np.concatenate([n.fp, [trees[i].tree.nodes[edges_result[i][0]].fp[-1]]])
                    cand_nodes.append(n)
            logp = list(- (predict[[array_inds.index(m) for m in max_ind]]))
            

            #update nodes
            for j, n in enumerate(cand_nodes):
                if len(edges_result[i]) == 2:
                    new_trees[i][j].tree.nodes[edges_result[i][1]] = n
                else:
                    new_trees[i][j].tree.nodes[edges_result[i][0]] = n
                    #print(f'{j}_update: {new_trees[i][j].index_}: {[n.wid for n in new_trees[i][j].tree.nodes if isinstance(n, MolTreeNode)]}')
            
            #update edges
            for t_ind in range(len(new_trees[i])):
                if t_ind >= len(cand_nodes):
                    new_trees[i][t_ind] = None
                    continue
                if len(edges_result[i]) == 2:
                    new_trees[i][t_ind].tree.adj_matrix[0, 0] = 0
                    new_trees[i][t_ind].tree.add_edge(edges_result[i][0], edges_result[i][1])
                    new_trees[i][t_ind].last_focal = [new_trees[i][t_ind].tree.nodes[edges_result[i][0]], new_trees[i][t_ind].tree.nodes[edges_result[i][1]]]
                    if not can_assemble(new_trees[i][t_ind].tree.nodes[edges_result[i][0]]):
                        new_trees[i][t_ind] = None
                else:
                    new_trees[i][t_ind].tree.adj_matrix[0, 0] = 1#mark root for discovered
            logp = [logp[j] for j, t in enumerate(new_trees[i]) if t is not None]
            new_trees[i] = [t for t in new_trees[i] if t is not None]
            logp_batch.append(np.array(logp))

        #wrap and flat the list

        new_trees = [t for t in itertools.chain.from_iterable(new_trees)]

        logp_batch = list(np.concatenate(logp_batch)) + logp_refine
        new_trees = new_trees + refined_trees
        return new_trees, logp_batch
    else:
        return refined_trees, logp_refine

class beam_tree(object):
    def __init__(self, tree, index_, logp=0, end=False):
        self.tree = tree
        self.index_ = index_
        self.logp = logp
        self.end = end
        self.last_focal = None
    
    def check_end(self):
        res = True
        for node in self.tree.nodes:
            if not isinstance(node, MolTreeNode):
                res = False
                break
        self.end = res

def remove_queue_dup(q, ind, keep, pool=None, check_assemb=False):
    clean_q = PriorityQueue()
    collect_q_list = []
    count = 0
    while not q.empty():
        t = q.get()
        if t[1].index_ != ind:
            clean_q.put(t)
            #print(f'0: {t}')
        elif t[1].last_focal is not None:
            count_exact = len([n for n in t[1].tree.nodes if isinstance(n, MolTreeNode)])
            collect_q_list.append(t[1])
        else:
            count += 1
            clean_q.put(t)

    if count < keep:
        if check_assemb:
            count_exact = len([n for n in t[1].tree.nodes if isinstance(n, MolTreeNode)])
            collect_q_checked = pool.map(lambda x: sum([can_assemble(n) for n in x.tree.nodes if isinstance(n, MolTreeNode)]) == count_exact, collect_q_list)
            collect_q_checked = [x for i, x in enumerate(collect_q_list) if collect_q_checked[i]]
        else:
            collect_q_checked = collect_q_list
    else:
        collect_q_checked = collect_q_list

    collect_q_checked.sort(key=lambda x: x.logp)
    collect_q_checked = collect_q_checked[:keep - count]
    for t in collect_q_checked:
        clean_q.put((t.logp, t))

    return clean_q



def sample_trees_from_blur(jts, model, model_refine, vocab, cfg, device, context_norm=1):
    '''
    sample trees from [blur, pos]
    beam search on node type and check assemble every step
    '''
    pool = Pool(cfg.generation.beam_size ** 2)
    with torch.no_grad():
        #put blur trees in queue
        q = PriorityQueue()
        for i, jt in enumerate(jts):
            if 'context' in jt.keys():
                jt['h'] = torch.cat([jt['h'], jt['context'] * context_norm], dim=-1)
            nodes = [MolTreeNode_blur(jt['h'][i], jt['x'][i], jt['size'][i]) for i in range(jt['h'].shape[0])]
            tree = beam_tree(MolTree(mol=None, nodes=nodes), index_=i)
            tree.logp += random.uniform(0, 1e-8)
            q.put((tree.logp, tree))
        
        #beam search
        results = []
        tree_batch = []
        while not q.empty():
            p, tree = q.get()
            if tree.end:
                results.append(tree)
                #print(f'success for tree {len(results) + 1}')
                q = remove_queue_dup(q, tree.index_, cfg.generation.beam_size, pool)
                if len(results) == len(jts):
                    return results
                continue
            #update tree
            if not q.empty():
                if len(tree_batch) < len(jts) - 1:
                    tree_batch.append(tree)
                else:
                    tree_batch.append(tree)
                    new_trees, logp_batch = update_trees(model, model_refine, tree_batch, vocab, cfg.generation.beam_size, device)
                    for i, new_t in enumerate(new_trees):
                        new_t.check_end()
                        new_t.logp += (logp_batch[i] + random.uniform(0, 1e-8))
                        q.put((new_t.logp, new_t))
                    q = remove_queue_dup(q, new_t.index_, cfg.generation.beam_size, pool)
                    tree_batch = []
            else:
                tree_batch.append(tree)
                new_trees, logp_batch = update_trees(model, model_refine, tree_batch, vocab, cfg.generation.beam_size, device)
                for i, new_t in enumerate(new_trees):
                    new_t.check_end()
                    new_t.logp += (logp_batch[i] + random.uniform(0, 1e-8))
                    q.put((new_t.logp, new_t))
                q = remove_queue_dup(q, new_t.index_, cfg.generation.beam_size, pool)
                tree_batch = []
    #print('failed')
    return results


if __name__ == '__main__':
    with open(args.input_path, 'rb') as f:
        data = pickle.load(f)

    model = Edge_denoise(**cfg.model)
    model_refine = Node2Vec(**cfg.model_refine)
    if torch.cuda.is_available():
        checkpoint = torch.load(cfg.generation.model_path)['state_dict']
        checkpoint_refine = torch.load(cfg.generation.model_refine_path)['state_dict']
    else:
        checkpoint = torch.load(cfg.generation.model_path, map_location=torch.device('cpu'))['state_dict']
        checkpoint_refine = torch.load(cfg.generation.model_refine_path, map_location=torch.device('cpu'))['state_dict']
    for key in list(checkpoint):
        checkpoint[key.replace('model.', '')] = checkpoint.pop(key)
    for key in list(checkpoint_refine):
        checkpoint_refine[key.replace('model.', '')] = checkpoint_refine.pop(key)

    with open(cfg.generation.vocab_path, "r") as f:
        vocab = [x.strip() for x in f.readlines()]

    if cfg.generation.node_coarse_type == 'prop':
        vocab_fp_path = cfg.generation.vocab_fp_path_prop
    elif cfg.generation.node_coarse_type == 'elem':
        vocab_fp_path = cfg.generation.vocab_fp_path_elem

    vocab_fp = pd.read_csv(vocab_fp_path, index_col=0)
    vocab = Vocab(vocab, vocab_fp)
    random.seed(2022)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint)
    model_refine.load_state_dict(checkpoint_refine)
    model.to(device)
    model_refine.to(device)
    model.device = device
    model.eval()
    model_refine.eval()

    #sample trees
    sample_batch_size = 1
    trees_output = []
    
    data = data[args.start_num: args.start_num+1000]
    for i in range(len(data)):
        data[i]['size'] = torch.round(data[i]['h'][:, 3]).int()
        data[i]['h'] = torch.round(data[i]['h'][:, :3]).int()

    for i in tqdm.tqdm(range(0, len(data), sample_batch_size)):
        #logp_batch = logP_context_range[(i + args.start_num // 32) % len(logP_context_range)]
        #try:
        r = sample_trees_from_blur(data[i: i + sample_batch_size], model, model_refine, vocab, cfg, device, context_norm=1)#remember to change the norm to 10 for SAS
        #except:
        #    print('DDPM generate impossible atom dict')
        #    continue
        #need the above code to skip impossible fragments when using hard constraint
        if len(r) > 0:
            for t in r:
                if 'context' in data[i].keys():
                    t.context = data[i]['context'][0].item()
                trees_output.append(t)
                break
    print(f'{len(trees_output)} trees sampled')
    with open(os.path.join(args.output_path, f'sampled_trees_from_atom_embed_{args.start_num}-{args.start_num+1000}__{cfg.generation.beam_size}.pkl'), 'wb') as f:
        pickle.dump(trees_output, f)
    
