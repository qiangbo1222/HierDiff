import os
import pickle
import random

import numpy as np
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import torch
import torch.nn as nn
from rdkit.Chem import QED, AllChem, Descriptors, RDConfig, rdMolDescriptors

from data_utils.data_diffuse import get_bfs_order_new, get_dfs_order

"""
from models.edge_denoise import (adj_matrix_to_edges_dfs,
                                 adj_matrix_to_edges_flat,
                                 attach_to_adj_matrix, concat_edges,
                                 pad_adj_matrix, split_edges, split_nodes,
                                 strip_adj_matrix)
                                 """
from torch.utils import data
from torch.utils.data import DataLoader

from data_utils.mol_tree import MolTree, Vocab
# from data_utils.data_diffuse import tree_to_train_graph
from data_utils.MPNN_pattern import (bfs_multi_layer_direction,
                                     bfs_single_layer_direction,
                                     bfs_single_step_direction,
                                     dfs_bidirection, dfs_direction,
                                     tree_to_search_tree)


# only consider the position of the tree node (without angle and torision)
class mol_Tree_pos(data.Dataset):
    def __init__(self, args, dataname, split=None, vocab=None, x_perturb=1.0):

        self.dataname = dataname
        self.args = args
        if dataname == "crossdock":
            self.split = split
            self.datapath_list = os.listdir(args.data_dir)
            self.vocab = vocab
        elif dataname == "GEOM_drug":
            with open(args.split, "rb") as f:
                split_ = pickle.load(f)
            self.datapath_list = os.listdir(args.data_dir)
            self.datapath_list.sort()
            self.datapath_list = [self.datapath_list[ind] for ind in split_]
            self.vocab = vocab
            self.array_dict = pickle.load(open(args.array_dict, "rb"))
        elif dataname == "QM9":
            with open(args.split, "rb") as f:
                split_ = pickle.load(f)
            self.datapath_list = os.listdir(args.data_dir)
            self.datapath_list.sort()
            self.datapath_list = [self.datapath_list[ind] for ind in split_]
            self.vocab = vocab
            self.array_dict = pickle.load(open(args.array_dict, "rb"))
        else:
            raise ValueError("dataname not supported")

        search_mode = args.search_mode
        self.search_priority = args.search_priority
        if search_mode == "bfs_multi_layer_direction":
            self.search_mode = bfs_multi_layer_direction
        elif search_mode == "bfs_single_layer_direction":
            self.search_mode = bfs_single_layer_direction
        elif search_mode == "bfs_single_step_direction":
            self.search_mode = bfs_single_step_direction
        elif search_mode == "dfs_direction":
            self.search_mode = dfs_direction
        elif search_mode == "dfs_bidirection":
            self.search_mode = dfs_bidirection
        else:
            raise ValueError("search_mode not supported")
        
        #self.perturb_dist = torch.distributions.normal.Normal(0, x_perturb)
    

    def __len__(self):
        if self.dataname == "crossdock":
            return len(self.split)
        elif self.dataname == "GEOM_drug" or self.dataname == "QM9":
            return len(self.datapath_list)

    def __getitem__(self, index):
        if self.dataname == "crossdock":
            datapath = self.datapath_list[self.split[index]]
        else:
            datapath = self.datapath_list[index]
        with open(os.path.join(self.args.data_dir, datapath), "rb") as f:
            tree = pickle.load(f)
        if isinstance(tree, list):
            random.shuffle(tree)
            tree = tree[0]
            context = Descriptors.MolLogP(tree.mol3D)
            for node in tree.nodes:
                if self.args.context_nf:#only suited for element bag feature
                    node.fp = np.concatenate(
                        [
                            np.array(self.vocab.fp_df.loc[node.smiles]),
                            np.array([context]),
                        ]
                    )
                else:
                    fp_fix = np.array(self.vocab.fp_df.loc[node.smiles])
                    contribute_TPSA = rdMolDescriptors._CalcTPSAContribs(tree.mol3D)
                    contribute_ASA = rdMolDescriptors._CalcLabuteASAContribs(tree.mol3D)
                    tpsa = sum([contribute_TPSA[i] for i in node.clique])/10
                    asa = (sum([list(contribute_ASA[0])[i] for i in node.clique]) + contribute_ASA[1])/10
                    node.fp = np.concatenate((np.array([node.hbd]), fp_fix, np.array([tpsa]), np.array([asa])))
                    #node.fp = fp_fix
            tree = tree_to_search_tree(
                tree,
                self.search_mode,
                self.vocab,
                self.args.int_feature_size,
                priority=False,
            )
        feature_tensor = []
        pos_tensor = []
        array_tensor = []
        #perturb_tensor = []
        #size_tensor = []
        for node_id, node in enumerate(tree.nodes):
            feature_tensor.append(torch.tensor(node.fp))
            pos_tensor.append(torch.tensor(node.pos))
            #if node_id not in tree.undiscovered:
            #    perturb_tensor.append(torch.zeros_like(torch.tensor(node.pos)))
            #else:
            #    perturb_tensor.append(self.perturb_dist.sample(node.pos.shape))
            if not self.args.full_softmax:
                if self.args.context_nf:
                    array_tensor.append(
                        torch.tensor(check_array_in_list(node.fp[:-2], self.array_dict[0]))
                    )
                else:
                    array_tensor.append(
                        torch.tensor(check_array_in_list(node.fp[:-1], self.array_dict[0]))
                    )
            #size_tensor.append(torch.tensor(Chem.MolFromSmiles(node.smiles).GetNumHeavyAtoms(), dtype=torch.long))
        feature_tensor = torch.stack(feature_tensor)
        pos_tensor = torch.stack(pos_tensor)
        #perturb_tensor = torch.stack(perturb_tensor)
        #pos_tensor = pos_tensor + perturb_tensor
        #size_tensor = torch.stack(size_tensor)
        if not self.args.full_softmax:
            array_tensor = torch.stack(array_tensor)
        label_smiles = tree.nodes[tree.search_ind].smiles
        label = self.vocab.get_index(label_smiles)

        val_miss = set(
            list((tree.adj_matrix - tree.search_adj_matrix_org).sum(1).nonzero()[0])
        )
        discover = set(list(tree.search_adj_matrix_org.sum(1).nonzero()[0]))
        focal = list(val_miss.intersection(discover))

        return {
            "feat": feature_tensor,
            "position": pos_tensor,
            "adj_matrix": tree.adj_matrix,
            "search_adj_matrix_org": tree.search_adj_matrix_org,
            "search_adj_matrix": tree.search_adj_matrix,
            "label": label,
            "array": array_tensor,
            "focal": focal,
            "discover": discover,
            "undiscovered": tree.undiscovered,
            "predict_idx": tree.search_ind,
            "last_ind": tree.last_ind,

        }#"perturb": perturb_tensor[tree.search_ind]
        #'search_edges': tree.search_edges,#'size': size_tensor


def PadCollate_onehot(batch, args):
    max_len = max([x["feat"].shape[0] for x in batch])
    max_pos = max([x["position"].shape[0] for x in batch])
    batch_size = len(batch)
    # feat_tensor = torch.zeros([len(batch), max_len, args.int_feature_size + args.num_continutes_feature + 1 + args.vocab_size])#add node vel feature
    feat_tensor = torch.zeros(
        [
            len(batch),
            max_len,
            args.int_feature_size
            + args.num_continutes_feature
            + args.context_nf
            + 1
            + 1,
        ]
    )  # only use index for vocab feature
    feat_mask = torch.zeros(feat_tensor.shape).bool()
    #size_tensor = torch.zeros([len(batch), max_len], dtype=torch.long)
    array_tensor = torch.zeros([len(batch), max_len], dtype=torch.long)
    pos_tensor = torch.zeros([len(batch), max_pos, 3])
    # pos_mask = torch.zeros(pos_tensor.shape)
    edge_search = []
    # edge_pad = torch.zeros([len(batch)*max_len, len(batch)*max_len])
    search_adj_matrix = torch.zeros(len(batch), max_len, max_len).bool()
    search_adj_matrix_org = torch.zeros(len(batch), max_len, max_len).bool()
    adj_matrix = torch.zeros(len(batch), max_len, max_len).bool()
    edge_mask = torch.zeros(
        len(batch), max_len, max_len
    ).bool()  # only use for fully connected graph

    predict_idx = []
    last_ind = []
    label = torch.zeros(len(batch), dtype=torch.long)
    focal = []
    focal_cand = []
    undiscovered = []

    for i, sample in enumerate(batch):
        feat_tensor[
            i,
            : sample["feat"].shape[0],
            : args.int_feature_size + args.num_continutes_feature,
        ] = sample["feat"][:, : args.int_feature_size + args.num_continutes_feature]
        feat_tensor[
            i,
            : sample["feat"].shape[0],
            args.int_feature_size + args.num_continutes_feature,
        ] = torch.tensor(
            [int(ind in sample["discover"]) for ind in range(sample["feat"].shape[0])]
        )
        feat_tensor[
            i,
            : sample["feat"].shape[0],
            args.int_feature_size
            + args.num_continutes_feature
            + 1 : args.int_feature_size
            + args.num_continutes_feature
            + args.context_nf
            + 1,
        ] = sample["feat"][
            :,
            args.int_feature_size
            + args.num_continutes_feature : args.int_feature_size
            + args.num_continutes_feature
            + args.context_nf,
        ]
        feat_tensor[
            i,
            : sample["feat"].shape[0],
            args.int_feature_size + args.num_continutes_feature + args.context_nf + 1,
        ] = sample["feat"][:, -1]
        feat_mask[i, : sample["feat"].shape[0], :] = 1
        #size_tensor[i, :sample['size'].shape[0]] = sample['size']
        if sample['array'] != []:
            array_tensor[i, : sample["array"].shape[0]] = sample["array"]
        pos_tensor[i, : sample["position"].shape[0], :] = sample["position"]
        # pos_mask[i, :sample['position'].shape[0], :] = 1
        # edge_pad[i*max_len: i*max_len + sample['feat'].shape[0], i*max_len: i*max_len + sample['feat'].shape[0]] = torch.tensor(sample['adj_matrix'])
        # edge_search.append(flat_add(sample['search_edges'], i * max_len))
        search_adj_matrix_org[
            i, : sample["adj_matrix"].shape[0], : sample["adj_matrix"].shape[1]
        ] = torch.tensor(sample["search_adj_matrix_org"])
        search_adj_matrix[
            i, : sample["adj_matrix"].shape[0], : sample["adj_matrix"].shape[1]
        ] = torch.tensor(sample["search_adj_matrix"])
        adj_matrix[
            i, : sample["adj_matrix"].shape[0], : sample["adj_matrix"].shape[1]
        ] = torch.tensor(sample["adj_matrix"])

        edge_mask[
            i, : sample["adj_matrix"].shape[0], : sample["adj_matrix"].shape[1]
        ] = 1 - torch.eye(sample["adj_matrix"].shape[0])
        label[i] = sample["label"]

        predict_idx.append(sample["predict_idx"])
        last_ind.append(sample["last_ind"])
        focal_cand.extend([ind + i * max_len for ind in sample["discover"]])
        focal.extend([ind + i * max_len for ind in sample["focal"]])
        undiscovered.append(sample["undiscovered"])

    # max_search_depth = max([len(layers[0]) for layers in edge_search])
    """
    edge_search_pad = [[] for _ in range(max_search_depth)]
    edge_search_pad_orig = [[] for _ in range(max_search_depth - 1)]
    edge_search_flat = []
    real_focal = []
    for e in edge_search:
        for i, l in enumerate(e[0]):
            edge_search_pad[i].append(l)

        if len(e[0]) > 0:
            real_focal.append(e[0][-1][0])
            for i, l in enumerate(e[0][: -1]):
                edge_search_flat.append(l)
                edge_search_pad_orig[i].append(l)
    
    edge_depth = len(edge_search_flat)
    for i in range(edge_depth):
        e = edge_search_flat[i]
        if (e[1], e[0]) not in edge_search_flat:
            edge_search_flat.append((e[1], e[0]))
    edge_search_flat = list(set(edge_search_flat))
    """
    node_nums_batch = feat_mask[:, :, 0].sum(1)
    if search_adj_matrix_org.sum() > 0:
        edge_search_flat = []
        for i, adj_matrix_slice in enumerate(search_adj_matrix_org):
            edge_search_flat.append(
                adj_matrix_to_edges_flat(
                    strip_adj_matrix(adj_matrix_slice, node_nums_batch[i])
                )
            )
        edge_search_flat = concat_edges(edge_search_flat, max_len)  # only one layer
        edge_search_flat = [
            torch.tensor([item for sublist in edge_search_flat[0] for item in sublist]),
            torch.tensor([item for sublist in edge_search_flat[1] for item in sublist]),
        ]
    else:
        edge_search_flat = [torch.tensor([]), torch.tensor([])]

    if search_adj_matrix_org.sum() > 0:
        edge_search_origin = []
        for i, adj_matrix_slice in enumerate(search_adj_matrix_org):
            strip_m = strip_adj_matrix(adj_matrix_slice, node_nums_batch[i])
            if predict_idx[i] >= 0:
                edge_search_origin.append(
                    adj_matrix_to_edges_bfs(strip_m, blur_feature=None, end=last_ind[i])
                )
            else:
                edge_search_origin.append([])

        edge_search_origin = concat_edges(edge_search_origin, max_len)
    else:
        edge_search_origin = []

    if search_adj_matrix.sum() > 0:
        edge_search_pad = []
        for i, adj_matrix_slice in enumerate(search_adj_matrix):
            strip_m = strip_adj_matrix(adj_matrix_slice, node_nums_batch[i])
            if predict_idx[i] >= 0:
                edge_search_pad.append(
                    adj_matrix_to_edges_bfs(
                        strip_m, blur_feature=None, end=predict_idx[i]
                    )
                )
            else:
                edge_search_pad.append([])

        edge_search_pad = concat_edges(edge_search_pad, max_len)
    else:
        edge_search_pad = []

    focal = [1 if f in focal else 0 for f in focal_cand]

    return {
        "node_feat": [feat_tensor, feat_mask],
        "node_array": array_tensor,
        "node_pos": pos_tensor,
        "focal": torch.tensor(focal),  # nodes that can be connected
        "focal_cand": focal_cand,
        "real_focal": [
            l + i * max_len for i, l in enumerate(last_ind) if l >= 0
        ],  # nodes that connected this time
        "edge_search_pad": edge_search_pad,  # ordered edges
        "edge_search_pad_orig": edge_search_origin,  # ordered edges [:-1]
        "edge_search_flat": edge_search_flat,  # ordered edges [:-1] flat
        "search_adj_matrix": search_adj_matrix_org,  # adj matrix from edges [:-1]
        "edge_mask": edge_mask,
        "predict_idx": predict_idx,
        "label": label,
        "undiscovered": undiscovered,
    }  #"perturb_x": torch.stack([sample["perturb"] for sample in batch])'node_size': size_tensor,


"""
def flat_add(edge_list, add_num):
    for layer in edge_list:
        for edge_id, _ in enumerate(layer):
            layer[edge_id] = (layer[edge_id][0] + add_num, layer[edge_id][1] + add_num)
    return edge_list
"""


def pad_adj_matrix(adj_matrix):
    # pad bs, n_nodes, n_nodes into bs*n_nodes, bs*n_nodes
    bs, n_nodes = adj_matrix.shape[:2]
    adj_matrix_pad = torch.zeros(bs * n_nodes, bs * n_nodes).to(adj_matrix.device)
    for i in range(bs):
        adj_matrix_pad[
            i * n_nodes : (i + 1) * n_nodes, i * n_nodes : (i + 1) * n_nodes
        ] = adj_matrix[i]
    return adj_matrix_pad


def strip_adj_matrix(adj_matrix_pad, n_nodes):
    # n_nodes = torch.sum(adj_matrix_pad, dim=0).nonzero().max() + 1
    return adj_matrix_pad[:n_nodes, :n_nodes]


def adj_matrix_to_edges_flat(adj_matrix):
    return adj_matrix.nonzero().T.tolist()


def adj_matrix_to_edges_dfs(adj_matrix, blur_feature, end, priority=False):
    # adj_matrix need to remove the padding first
    if adj_matrix.sum() == 0:
        return [[]]
    else:
        edges = np.array(adj_matrix.nonzero().cpu())
        val = np.sum(np.array(adj_matrix.cpu()), axis=-1)
        num_nodes = adj_matrix.shape[0]
        graph = [[] for i in range(num_nodes)]
        for edge in edges:
            if edge[1] not in graph[edge[0]]:
                graph[edge[0]].append(edge[1])
            if edge[0] not in graph[edge[1]]:
                graph[edge[1]].append(edge[0])
        dfs_result = get_dfs_order(graph, end)
        dfs_order, dfs_paths = dfs_result["order"], dfs_result["path"]
        dfs_depth = max([d[1] for d in dfs_order])
        dfs_paths_reverse = []
        for e in dfs_paths[:dfs_depth]:
            dfs_paths_reverse = [[e[1], e[0]]] + dfs_paths_reverse
        return dfs_paths_reverse


def adj_matrix_to_edges_bfs(adj_matrix, blur_feature, end, priority=False):
    # adj_matrix need to remove the padding first
    if adj_matrix.sum() == 0:
        return [[]]
    else:
        edges = np.array(adj_matrix.nonzero().cpu())
        # val = np.sum(np.array(adj_matrix.cpu()), axis=-1)
        node_count = set()
        for e0, e1 in edges:
            node_count.add(e0)
            node_count.add(e1)
        num_nodes = len(node_count)
        bfs_result = get_bfs_order_new(edges, num_nodes, end)
        return bfs_result


def attach_to_adj_matrix(adj_matrix, edges):
    for e in edges:
        adj_matrix[e[0], e[1]] = 1
    return adj_matrix


def concat_edges(edges, n_nodes):
    max_depth = max([len(e) for e in edges])
    concat = [[] for _ in range(max_depth)]
    for i, e in enumerate(edges):
        e = flat_add(e, i * n_nodes)
        for l_i, l in enumerate(e):
            if len(l) > 0:
                if isinstance(l[0], int):
                    concat[l_i].append(l)
                else:
                    concat[l_i].extend(l)
            else:
                concat[l_i].extend(l)
    return concat


def flat_add(edge_list, add_num):
    for layer in edge_list:
        for edge_id, _ in enumerate(layer):
            if isinstance(layer[edge_id], list):
                layer[edge_id] = (
                    layer[edge_id][0] + add_num,
                    layer[edge_id][1] + add_num,
                )
            else:
                layer[edge_id] += add_num
    return edge_list


def check_array_in_list(array, list_a):
    for ind, ref in enumerate(list_a):
        if ((array - ref) ** 2).sum() == 0:
            return ind
    return None
