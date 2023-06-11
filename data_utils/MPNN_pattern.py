import copy
import random
from functools import cmp_to_key

import numpy as np
import torch

from data_utils.data_diffuse import get_bfs_order, get_dfs_order


def forfor(a):
        return [item for sublist in a for item in sublist]
        

def dfs_bidirection(adj_matrix, blur_feature, sampling=None):
    edges = np.array(adj_matrix.nonzero()).T
    val = np.sum(adj_matrix, axis=-1)
    num_nodes = adj_matrix.shape[0]
    graph = [[] for i in range(num_nodes)]
    for edge in edges:
        if edge[1] not in graph[edge[0]]:
            graph[edge[0]].append(edge[1])
        if edge[0] not in graph[edge[1]]:
            graph[edge[1]].append(edge[0])
    dfs_result = get_dfs_order(graph, 0)
    dfs_order, dfs_paths = dfs_result['order'], dfs_result['path']
    if sampling is not None:
        random_sample_id = sampling
    else:
        random_sample_id = random.randint(0, len(dfs_order)-1)
        #random_sample_id = 7
    if random_sample_id == 0:
        return [[]], [i for i in range(adj_matrix.shape[0])], 0, -1
    search_ind = dfs_order[random_sample_id][0]
    search_depth = dfs_order[random_sample_id][1]
    dfs_depth = [d[1] for d in dfs_order]
    last_ind = dfs_order[dfs_depth.index(search_depth) - 1][0]
    undiscovered = [dfs_order[i][0] for i in range(len(dfs_order)) if dfs_order[i][1] > search_depth]
    if sampling:
        return [dfs_paths[:search_depth]], undiscovered, search_ind, dfs_order
    else:
        return [dfs_paths[:search_depth]], undiscovered, search_ind, last_ind


#function that mask all the undiscovered nodes and mark the search node
def tree_to_search_tree(tree, search_method, vocab, noise_nf, add_noise=False):#search_method should be one of the functions above
    feature = torch.stack([torch.tensor(n.fp) for n in tree.nodes])
    #add noise for the int feature
    if add_noise:
        noise = torch.rand_like(feature[:, :noise_nf]) - 0.5
        feature[:, :noise_nf] = feature[:, :noise_nf] + noise
    edges, undiscovered, search_ind, last_ind = search_method(tree.adj_matrix, feature)
    tree.search_edges = edges
    search_adj_matrix = np.array(tree.adj_matrix)
    search_adj_matrix[undiscovered + [search_ind, ], :] = 0
    search_adj_matrix[:, undiscovered + [search_ind, ]] = 0
    tree.search_adj_matrix_org = copy.deepcopy(search_adj_matrix)
    search_adj_matrix[last_ind, search_ind] = 1
    search_adj_matrix[search_ind, last_ind] = 1
    tree.search_adj_matrix = search_adj_matrix


    for i, node in enumerate(tree.nodes):
        node.fp = feature[i]

    for n in tree.nodes:
        n.fp = np.append(n.fp, np.array(vocab.get_index(n.smiles)))
    undiscovered_fp = vocab.size()
    search_fp = vocab.size()
    if len(undiscovered) > 0:
        for un in undiscovered:
            tree.nodes[un].fp[-1] = undiscovered_fp
    tree.nodes[search_ind].fp[-1] = undiscovered_fp

    if search_ind not in undiscovered:
        tree.undiscovered = undiscovered + [search_ind, ]
    else:
        tree.undiscovered = undiscovered
        
    tree.search_ind = search_ind
    tree.last_ind = last_ind
    
    return tree
