import argparse
import copy
import os
import pickle
import sys
from collections import deque
from multiprocessing import Pool

import numpy as np
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import torch
import torch.nn as nn
import tqdm

sys.path.append(".")

from data_utils.mol_tree import *

class bfs_node(object):
    def __init__(self, idx, links):
        self.idx = idx
        self.links = links
        self.depth = None


def get_bfs_order(edges, n_nodes):
    edges = list(zip(*edges))
    bfs_links = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        for link_parent, link_son in edges:
            if link_parent == i:
                bfs_links[i].append(link_son)
            elif link_son == i:
                bfs_links[i].append(link_parent)
    bfs_nodes = [bfs_node(idx, links) for idx, links in enumerate(bfs_links)]
    queue = deque([bfs_nodes[0]])
    visited = set([bfs_nodes[0].idx])
    bfs_nodes[0].depth = 0
    order1,order2 = [[],], [[],]
    bfs_order = [[0] ,]
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.links:
            y = bfs_nodes[y]
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1) - 1:
                    order1.append([])
                    order2.append([])
                    bfs_order.append([])
                order1[y.depth].append( (x.idx, y.idx) )
                order2[y.depth].append( (y.idx, x.idx) )
                bfs_order[y.depth].append(y.idx)
    return bfs_order, order1

def get_bfs_order_new(edges, n_nodes, start):
    visited = set()
    visited.add(start)
    search_edges = []
    bfs_order = []
    while len(visited) < n_nodes:
        depth_edges = []
        depth_nodes = []
        cache = []
        for e in edges:
            if e[0] in visited and e[1] not in visited:
                cache.append(e[1])
                depth_edges.append([e[1], e[0]])
                depth_nodes.append(e[1])
        for n in cache:
            visited.add(n)
        bfs_order.append(depth_nodes)
        search_edges.append(depth_edges)
    search_edges.reverse()
    return search_edges
            

    
def get_dfs_order(graph, start, visited=None, result=None):#if priority graph(dic should be sorted)
    if visited is None:
        visited = set()
    if result is None:
        result = {'order': [], 'path': []}
    result['order'].append((start, len(result['path'])))
    visited.add(start)
    for next_ in graph[start]:
        if next_ not in visited:
            visited.add(next_)
            result['path'].append((start, next_))
            get_dfs_order(graph, next_, visited, result)
            result['path'].append((next_, start))
    return result

    

def tree_to_train_graph(tree, vocab, return_type='get_all'):
    vocab = vocab.vocab
    vocab_size = len(vocab)
    edges = tree.adj_matrix.nonzero()
    bfs_order, bfs_paths = get_bfs_order(edges, len(tree.nodes))

    for i in range(len(tree.nodes)):
        smiles_id = vocab.index(tree.nodes[i].smiles)
        tree.nodes[i].fp = np.append(tree.nodes[i].fp, np.array([smiles_id]))

    if return_type == 'get_all':
        full_graph = [copy.deepcopy(tree) for _ in range(len(bfs_order))]
        #get all graphs in auto-regressive order
        collect_train_graph = []

        for search_ind, jt in zip(bfs_order, full_graph):
            search_depth = bfs_order.index(search_ind)
            undiscovered = bfs_order[search_depth + 1:]

            for node_id in range(len(jt.nodes)):
                if node_id == search_ind:
                    search_fp = len(vocab)#vocab + predict-token + undiscover-token
                    jt.nodes[node_id].fp[-1] = search_fp
                elif node_id in undiscovered:
                    undiscover_fp = len(vocab) + 1#vocab + predict-token + undiscover-token
                    jt.nodes[node_id].fp[-1] = undiscover_fp
            decode_adj_matrix = np.zeros(tree.adj_matrix.shape)
            if search_depth > 0:
                for edge in bfs_paths[:search_depth]:
                    decode_adj_matrix[edge[0], edge[1]] = 1
            jt.decode_adj_matrix = decode_adj_matrix
            collect_train_graph.append(jt)
        return collect_train_graph
    else:
        #randomly sample a graph from all graphs
        random_sample_id = random.randint(0, len(bfs_order)-1)
        search_ind = bfs_order[random_sample_id]
        jt = copy.deepcopy(tree)
        search_depth = bfs_order.index(search_ind)
        undiscovered = bfs_order[search_depth + 1:]

        for node_id in range(len(jt.nodes)):
            if node_id == search_ind:
                search_fp = len(vocab)#vocab + predict-token + undiscover-token
                jt.nodes[node_id].fp[-1] = search_fp
            elif node_id in undiscovered:
                undiscover_fp = len(vocab) + 1#vocab + predict-token + undiscover-token
                jt.nodes[node_id].fp[-1] = undiscover_fp
        bfs_edges = []
        if search_depth > 0:
            bfs_edges = [[edge[0], edge[1]] for edge in bfs_paths[:search_depth]]
                
        jt.bfs_edges = np.array(bfs_edges)
        return jt