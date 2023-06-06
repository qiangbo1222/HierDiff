import copy
import random
from functools import cmp_to_key

import numpy as np
import torch

from data_utils.data_diffuse import get_bfs_order, get_dfs_order


def forfor(a):
        return [item for sublist in a for item in sublist]
        
#these functions are used to generate edges, return [0]layers of edges, [1]undiscovered nodes, [2]predict node index
def bfs_single_layer_direction(adj_matrix, blur_feature, priority=None):
    edges = adj_matrix.nonzero()
    bfs_order, bfs_paths = get_bfs_order(edges, adj_matrix.shape[0])
    bfs_order, bfs_paths = forfor(bfs_order), forfor(bfs_paths)
    random_sample_id = random.randint(0, len(bfs_order)-1)
    search_ind = bfs_order[random_sample_id]
    search_depth = bfs_order.index(search_ind)
    undiscovered = bfs_order[search_depth + 1:]
    return [bfs_paths[:search_depth]], undiscovered, search_ind

#TODO rewrite priority for full current layer to match sampling strategy
def bfs_single_step_direction(adj_matrix, blur_feature, priority=False):
    edges = adj_matrix.nonzero()
    val = torch.sum(adj_matrix, axis=-1)
    bfs_order, bfs_paths = get_bfs_order(edges, adj_matrix.shape[0])
    random_layer_id = random.randint(0, len(bfs_order)-1)
    random_sample_id = random.randint(0, len(bfs_order[random_layer_id])-1)
    if not priority:
        search_ind = bfs_order[random_layer_id][random_sample_id]
        bfs_order, bfs_paths = forfor(bfs_order), forfor(bfs_paths)
        search_depth = bfs_order.index(search_ind)
        undiscovered = bfs_order[search_depth + 1:]
        bfs_edges = []
        if search_depth > 0:
            bfs_edges = [[edge[0], edge[1]] for edge in bfs_paths[:search_depth]]
        return bfs_edges, undiscovered, search_ind
    else:
        #rank all the previous full layers according to priority
        for i in range(random_layer_id):
            rank_ind = priority_rank(blur_feature[bfs_order[i]], val[bfs_order[i]])
            bfs_order[i] = bfs_order[i][rank_ind]
            bfs_paths[i] = bfs_paths[i][rank_ind]
        #rank the current layer according to priority
        rank_ind = priority_rank(blur_feature[bfs_order[random_layer_id][:random_sample_id]], val[bfs_order[random_layer_id][:random_sample_id]])
        bfs_order[random_layer_id][:random_sample_id] = bfs_order[random_layer_id][rank_ind]
        bfs_paths[random_layer_id][:random_sample_id] = bfs_paths[random_layer_id][rank_ind]
        search_ind = bfs_order[random_layer_id][random_sample_id]
        search_depth = forfor(bfs_order).index(search_ind)
        undiscovered = forfor(bfs_order)[search_depth + 1:]
        bfs_edges = [edges_layer for edges_layer in bfs_paths[:random_layer_id]]
        bfs_edges.append(bfs_paths[random_layer_id][:random_sample_id])
        return forfor(bfs_edges), undiscovered, search_ind
            


def dfs_direction(adj_matrix, blur_feature, priority=False):
    edges = adj_matrix.nonzero()
    val = torch.sum(adj_matrix, axis=-1)
    num_nodes = adj_matrix.shape[0]
    graph = [[] for i in range(num_nodes)]
    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    if priority:
        for i in range(num_nodes):
            rank_ind = priority_rank(blur_feature[graph[i]], val[graph[i]])
            graph[i] = graph[i][rank_ind]
    dfs_order, dfs_paths = get_dfs_order(graph, 0)
    random_sample_id = random.randint(0, len(dfs_order)-1)
    search_ind = dfs_order[random_sample_id]
    search_depth = dfs_order.index(search_ind)
    undiscovered = dfs_order[search_depth + 1:]
    return dfs_paths[:search_depth], undiscovered, search_ind

def dfs_bidirection(adj_matrix, blur_feature, priority=False, sampling=None):
    edges = np.array(adj_matrix.nonzero()).T
    val = np.sum(adj_matrix, axis=-1)
    num_nodes = adj_matrix.shape[0]
    graph = [[] for i in range(num_nodes)]
    for edge in edges:
        if edge[1] not in graph[edge[0]]:
            graph[edge[0]].append(edge[1])
        if edge[0] not in graph[edge[1]]:
            graph[edge[1]].append(edge[0])
    if priority:
        for i in range(num_nodes):
            rank_ind = priority_rank(blur_feature[graph[i]], val[graph[i]])
            graph[i] = [graph[i][r] for r in rank_ind]
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


def bfs_multi_layer_direction(adj_matrix, blur_feature, priority=False):#all messages i --> i + 1 that have the same depth pass together
    edges = adj_matrix.nonzero()
    val = np.sum(adj_matrix, axis=-1)
    bfs_order, bfs_paths = get_bfs_order(edges, adj_matrix.shape[0])
    if len(bfs_order) > 1:
        random_layer_id = random.randint(0, len(bfs_order)-1)
    else:
        return [], [i for i in range(adj_matrix.shape[0])], 0
    if random_layer_id == 0:
        return [], [i for i in range(adj_matrix.shape[0])], 0
    if len(bfs_order[random_layer_id]) > 1:
        random_sample_id = random.randint(0, len(bfs_order[random_layer_id])-1)
    else:
        random_sample_id = 0
    if not priority:
        search_ind = bfs_order[random_layer_id][random_sample_id]
        search_depth = forfor(bfs_order).index(search_ind)
        undiscovered = forfor(bfs_order)[search_depth + 1:]
        bfs_edges = [edges_layer for edges_layer in bfs_paths[:random_layer_id]]
        bfs_edges.append(bfs_paths[random_layer_id][:random_sample_id+1])
        return bfs_edges, undiscovered, search_ind

    else:
        #rank all the previous full layers according to priority
        for i in range(random_layer_id):
            rank_ind = priority_rank(blur_feature[bfs_order[i]], val[bfs_order[i]])
            bfs_order[i] = [bfs_order[i][r] for r in rank_ind]
            if i > 0:
                bfs_paths[i] = [bfs_paths[i][r] for r in rank_ind]
        #rank the current layer according to priority
        rank_ind = priority_rank(blur_feature[bfs_order[random_layer_id][:random_sample_id + 1]], val[bfs_order[random_layer_id][:random_sample_id + 1]])
        bfs_order[random_layer_id][:random_sample_id + 1] = [bfs_order[random_layer_id][r] for r in rank_ind]
        bfs_paths[random_layer_id][:random_sample_id + 1] = [bfs_paths[random_layer_id][r] for r in rank_ind]

        search_ind = bfs_order[random_layer_id][random_sample_id]
        search_depth = forfor(bfs_order).index(search_ind)
        undiscovered = forfor(bfs_order)[search_depth + 1:]
        bfs_edges = [edges_layer for edges_layer in bfs_paths[:random_layer_id]]
        bfs_edges.append(bfs_paths[random_layer_id][:random_sample_id + 1])
        return bfs_edges, undiscovered, search_ind


def priority_rank(blur_feature, val):#--> rank according to 1) ring type 2) ring valence 3) charge 3) hetoatoms 
    indices = [i for i in range(blur_feature.shape[0])]
    indices.sort(key=cmp_to_key(lambda x, y: priority(blur_feature[x], blur_feature[y], val[x], val[y])))
    return indices

def priority(f1, f2, val1, val2):
    # check ring type aromatic
    if f1[4] > 0.5 and f2[4] > 0.5:
        if val1 > val2:
            return -1
        else:
            return 1
    elif f1[4] > 0.5 and f2[4] < 0.5:
        return -1
    elif f1[4] < 0.5 and f2[4] > 0.5:
        return 1
    # check ring valence
    if f1[5] > 0.5 and f2[5] > 0.5:
        if val1 > val2:
            return -1
        else:
            return 1
    elif f1[5] > 0.5 and f2[5] < 0.5:
        return -1
    elif f1[5] < 0.5 and f2[5] > 0.5:
        return 1
    # check charge
    f1[3] = f1[3].abs()
    f2[3] = f2[3].abs()
    if f1[3] > 0.5 and f2[3] > 0.5:
        if f1[2] > f2[2]:
            return -1
        else:
            return 1
    elif f1[3] > 0.5 and f2[3] < 0.5:
        return -1
    elif f1[3] < 0.5 and f2[3] > 0.5:
        return 1
    # check hetoatoms
    if f1[2] > f2[2]:
        return -1
    else:
        return 1

#function that mask all the undiscovered nodes and mark the search node
def tree_to_search_tree(tree, search_method, vocab, noise_nf, priority, add_noise=False):#search_method should be one of the functions above
    feature = torch.stack([torch.tensor(n.fp) for n in tree.nodes])
    #add noise for the int feature
    #if add_noise:
    #    noise = torch.rand_like(feature[:, :noise_nf]) - 0.5
    #    feature[:, :noise_nf] = feature[:, :noise_nf] + noise
    edges, undiscovered, search_ind, last_ind = search_method(tree.adj_matrix, feature, priority=False)
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
