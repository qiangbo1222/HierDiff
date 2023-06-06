import argparse
import os
import sys

sys.path.append('..')
import pickle
import random
from multiprocessing import Pool

import torch
import tqdm
from data_utils.chemutils import mol_equal
from data_utils.data_diffuse import get_dfs_order
from data_utils.mol_tree import *


#use args to set the data_dir_base and save_dir
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir_base', type=str, default='data/crossdock_data/trees')
parser.add_argument('--save_dir', type=str, default='data/denoise_split.pkl')
args = parser.parse_args()

data_dir = os.listdir(args.data_dir_base)
data_dir.sort()
data_dir = [os.path.join(args.data_dir_base, d) for d in data_dir]

def check(dir):
    with open(dir, 'rb') as f:
        trees = pickle.load(f)
    check_list = []
    for tree in trees:
        edges = np.array(tree.adj_matrix.nonzero()).T
        num_nodes = tree.adj_matrix.shape[0]
        graph = [[] for i in range(num_nodes)]
        for edge in edges:
            if edge[1] not in graph[edge[0]]:
                graph[edge[0]].append(edge[1])
            if edge[0] not in graph[edge[1]]:
                graph[edge[1]].append(edge[0])
        dfs_order = get_dfs_order(graph, 0)['order']
        check_list.append(num_nodes == len(dfs_order))
    return sum(check_list) == len(trees), dir

        


pool = Pool(12)
splits = []
for check_result, dir in tqdm.tqdm(pool.imap_unordered(check, data_dir), total=len(data_dir)):
    if check_result:
        splits.append(data_dir.index(dir))


with open(args.save_dir, 'wb') as f:
    pickle.dump(splits, f)
print(f'len splits:{len(splits)}')
