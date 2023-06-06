import copy
import pickle
import sys

import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import torch
import torch.nn as nn
from data_utils.mol_tree import *

sys.path.append('/home/songyuxuan/Projects/3Dmolgen/generation/jtnn')
from generation.jtnn.jtnn_dec import can_assemble

from models.egnn.gcl import E_GCL


class Node2Vec(nn.Module):

    def __init__(self, size_dict, vocab_size, feature_size, hidden_size, n_layers=3):
        super(Node2Vec, self).__init__()
        self.size_dict = pickle.load(open(size_dict, 'rb'))
        self.feature_size = feature_size
        self.v_embedding = nn.Embedding(vocab_size + 1, hidden_size)
        self.f_embedding = nn.Sequential(
                            nn.Linear(feature_size, hidden_size),
                            nn.SiLU(),
                            nn.Linear(hidden_size, hidden_size))
        self.projection = nn.Sequential(
                              nn.Linear(hidden_size * 3, hidden_size * 3),
                              nn.SiLU(),
                              nn.Linear(hidden_size * 3, hidden_size),
                              nn.SiLU(),
                              nn.Linear(hidden_size, hidden_size))
        self.size_embedding = nn.Embedding(26, hidden_size)
        self.n_layers = n_layers

        for i in range(0, n_layers):
            self.add_module("gcl_collect%d" % i, E_GCL(hidden_size, hidden_size, hidden_size, edges_in_d=1, act_fn=nn.SiLU(), recurrent=True, attention=True, tanh=True, coords_range=30, agg='sum', coord_update=True, edge_update=False))
            self.add_module("gcl_reverse%d" % i, E_GCL(hidden_size, hidden_size, hidden_size, edges_in_d=1, act_fn=nn.SiLU(), recurrent=True, attention=True, tanh=True, coords_range=30, agg='sum', coord_update=True, edge_update=False))
            self.add_module("gcl_back%d" % i, E_GCL(hidden_size, hidden_size, hidden_size, edges_in_d=1, act_fn=nn.SiLU(), recurrent=True, attention=True, tanh=True, coords_range=30, agg='sum', coord_update=True, edge_update=False))

        self.output = nn.Sequential(
                            nn.Linear(hidden_size + 1, hidden_size),
                            nn.SiLU(),
                            nn.Linear(hidden_size, vocab_size))
    
    def message(self, edges, h, x, mask=None):
        for edge_depth in edges:
            for i in range(0, self.n_layers):
                edge_depth = torch.tensor(edge_depth).to(h.device)
                edge_attr = torch.sum((x[edge_depth[0]] - x[edge_depth[1]]) ** 2, dim=1, keepdim=True)
                h, x = self._modules["gcl_collect%d" % i](h, edge_depth, x, edge_attr=edge_attr, node_mask=mask)
        
        edges_reverse = copy.deepcopy(edges)
        edges_reverse.reverse()
        for i in range(len(edges_reverse)):
            edges_reverse[i] = [edges_reverse[i][1], edges_reverse[i][0]]

        for edge_depth in edges_reverse:
            for i in range(0, self.n_layers):
                edge_depth = torch.tensor(edge_depth).to(h.device)
                edge_attr = torch.sum((x[edge_depth[0]] - x[edge_depth[1]]) ** 2, dim=1, keepdim=True)
                h, x = self._modules["gcl_reverse%d" % i](h, edge_depth, x, edge_attr=edge_attr, node_mask=mask)
        
        for edge_depth in edges:
            for i in range(0, self.n_layers):
                edge_depth = torch.tensor(edge_depth).to(h.device)
                edge_attr = torch.sum((x[edge_depth[0]] - x[edge_depth[1]]) ** 2, dim=1, keepdim=True)
                h, x = self._modules["gcl_back%d" % i](h, edge_depth, x, edge_attr=edge_attr, node_mask=mask)
        return h, x

    def forward(self, batch):
        f = batch['feature']
        v = batch['vocab']
        size = batch['size']
        x = batch['pos']
        edges = batch['edges']
        mask = batch['mask']
        label = batch['label']
        predict_idx = batch['predict_idx']
        val = batch['val']
        bs, n_nodes = f.shape[:2]

        v_embedding = self.v_embedding(v)
        f_embedding = self.f_embedding(f)
        size_embedding = self.size_embedding(size)
        combined = torch.cat((v_embedding, f_embedding, size_embedding), dim=2)
        combined = self.projection(combined)
        h = combined * mask
        
        h = h.view(bs*n_nodes, -1)
        x = x.view(bs*n_nodes, -1)
        mask = mask.view(bs*n_nodes, -1)
        
        h, x = self.message(edges, h, x, mask)

        h = h.view(bs, n_nodes, -1)
        h = torch.stack([h[i, predict_idx[i]] for i in range(bs)])
        output = self.output(torch.cat([h, val.unsqueeze(1)], dim=1))
        accuracy = 0
        loss = torch.tensor(0.0).to(output.device)
        size = size.view(bs, n_nodes)
        size = torch.stack([size[i, pr] for i, pr in enumerate(predict_idx)])
        for i in range(bs):
            node_predict_i = output[i, self.size_dict[size[i].item()]].unsqueeze(0)
            label_i = torch.tensor(self.size_dict[size[i].item()].index(label[i].item()), dtype=torch.long).to(h.device).unsqueeze(-1)
            loss += nn.CrossEntropyLoss()(node_predict_i, label_i)
            if torch.argmax(output[i, self.size_dict[size[i].item()]]) == self.size_dict[size[i].item()].index(label[i].item()):
                accuracy += 1
        return {'loss': loss, 'accuracy': torch.tensor(accuracy) / bs}
    

    @torch.no_grad()
    def check_node(self, vocab, nodes, edges, pad_idx, pad_wid, device, check_num=1):
        bs = len(pad_idx)
        x = torch.stack([torch.tensor(n.pos, dtype=torch.float) for n in nodes]).to(device)
        #clip out the context part
        f = torch.stack([torch.tensor(n.fp[:self.feature_size], dtype=torch.float) for n in nodes]).to(device)
        v = torch.tensor([n.wid for n in nodes], dtype=torch.long).to(device)
        size = torch.tensor([n.size for n in nodes], dtype=torch.long).to(device)
        if bs > 1:
            x = torch.stack([x for _ in range(bs)])
            f = torch.stack([f for _ in range(bs)])
            v = torch.stack([v for _ in range(bs)])
            size = torch.stack([size for _ in range(bs)])
            for i in range(len(pad_idx)):
                v[i, pad_idx[i]] = 780
        else:
            x, f, v = x.unsqueeze(0), f.unsqueeze(0), v.unsqueeze(0)
            size = size.unsqueeze(0)
            v[0, pad_idx[0]] = 780

        f_embedding = self.f_embedding(f)
        v_embedding = self.v_embedding(v)
        size_embedding = self.size_embedding(size)
        combined = torch.cat((v_embedding, f_embedding, size_embedding), dim=2)
        h = self.projection(combined)
        val = torch.tensor([torch.sum(torch.tensor(edges[0]) == pad_idx[i]) for i in range(bs)]).to(device)
        '''
        #code for using old embedding
        h_v = torch.stack([torch.tensor([n.wid], dtype=torch.long) for n in nodes])
        v_embedding = self.v_embedding(h_v)
        f_embedding = self.f_embedding(h)
        combined = torch.cat((v_embedding, f_embedding), dim=2)
        h = self.projection(combined)
        '''
        edges = [get_bfs_depth_edges(edges, pad_idx[i], x.shape[1]) for i in range(bs)]
        edges = flat_add_and_concat(edges, len(nodes))

        h = h.view(-1, h.shape[-1])
        x = x.view(-1, x.shape[-1])
        h, x = self.message(edges, h, x)
        h = h.view(bs, -1, h.shape[-1])
        x = x.view(bs, -1, x.shape[-1])

        h_predict = torch.stack([h[i, pad_idx[i]] for i in range(bs)])
        output = self.output(torch.cat([h_predict, val.unsqueeze(1)], dim=1))
        results = []
        for i in range(bs):
            predict_size = nodes[pad_idx[i]].size
            size_ind = vocab.get_size(predict_size)
            if len(size_ind) == 0:
                predict_size = handle_wrong_sizes(predict_size)
                size_ind = vocab.get_size(predict_size)
            if len(size_ind) < check_num:
                check_num = len(size_ind)
            max_p = [size_ind[i] for i in torch.topk(output[i, size_ind], check_num)[1]]
            if check_num == 1:
                results.append((nn.LogSoftmax()(output[i, size_ind])[size_ind.index(pad_wid[i])], (max_p[0] == pad_wid[i], max_p[0])))
            else:
                results.append((nn.LogSoftmax()(output[i, size_ind])[size_ind.index(pad_wid[i])], [(p == pad_wid[i], p) for p in max_p]))
        return results

    def check_tree(self, beam_tree, vocab, device, check_num=0.1):
        tree = beam_tree.tree
        edges = torch.tensor(tree.adj_matrix).nonzero().T.tolist()
        nodes_exact = [n for n in tree.nodes if isinstance(n, MolTreeNode)]
        if len(nodes_exact) * check_num <= 1:
            return beam_tree, 0.0, False
        nodes_exact_idx = {}
        exact_count = 0
        for i, n in enumerate(tree.nodes):
            if isinstance(n, MolTreeNode):
                nodes_exact_idx[i] = exact_count
                exact_count += 1
        nodes_exact_idx_reverse = {v: k for k, v in nodes_exact_idx.items()}
        edges = [[nodes_exact_idx[e] for e in edges[0]],
                 [nodes_exact_idx[e] for e in edges[1]]]

        
        pad_wid = [nodes_exact[i].wid for i in range(len(nodes_exact))]
        pad_idx = [i for i in range(len(nodes_exact))]
        check_results = self.check_node(vocab, nodes_exact, edges, pad_idx, pad_wid, device)
        p = torch.tensor([r[0] for r in check_results])
        sum_p = torch.sum(p)
        sort_p_ind = torch.argsort(p)
        check_num = int(len(nodes_exact) * check_num)
        if sort_p_ind.shape[0] > check_num:
            sort_p_ind = sort_p_ind[: check_num]
        sort_p_ind = [p for p in sort_p_ind if p < len(nodes_exact) * 0.5]
        
        #check_skips = [(i, can_assemble(nodes_exact[i])) for i in range(len(nodes_exact))]
        #check_skips = [ind for ind, assem in check_skips if not assem]
        #len_assem_check = len(check_skips)
        #if len(check_skips) > 0:
        #    sort_p_ind = check_skips + sort_p_ind.tolist()
        if len(sort_p_ind) == 0:
            return beam_tree, 0.0, False
        
        pertube_p_sum = 0
        for c, i in enumerate(sort_p_ind):
            if not check_results[i][1][0]:
                idx_to_change = i
                nodes_exact_pertube = copy.deepcopy(nodes_exact)
                node_to_pertube = nodes_exact_pertube[i]
                wid_to_change = check_results[i][1][1]
                #node_to_pertube.fp = vocab.fp_df.loc[vocab.get_smiles(wid_to_change)]
                node_to_pertube.wid = wid_to_change
                node_to_pertube.smiles = vocab.get_smiles(wid_to_change)
                mol = Chem.MolFromSmiles(node_to_pertube.smiles)
                Chem.Kekulize(mol)
                node_to_pertube.mol = mol
                nodes_exact_pertube[i] = node_to_pertube

                pad_wid = [nodes_exact_pertube[j].wid for j in range(len(nodes_exact_pertube))]
                pad_idx = [j for j in range(len(nodes_exact_pertube))]
                check_results_pertube = self.check_node(vocab, nodes_exact_pertube, edges, pad_idx, pad_wid, device)

                p_pertube = torch.sum(torch.tensor([r[0] for r in check_results_pertube]))
                check_assembles = [nodes_exact_pertube[i]] + nodes_exact_pertube[i].neighbors
                check_assembles_neis = sum([can_assemble(n) for n in check_assembles]) == len(check_assembles)
                if p_pertube > sum_p and check_assembles_neis:
                    node_pertube_confirm = tree.nodes[nodes_exact_idx_reverse[int(idx_to_change)]]
                    #node_pertube_confirm.fp = vocab.fp_df.loc[vocab.get_smiles(wid_to_change)]
                    node_pertube_confirm.wid = wid_to_change
                    node_pertube_confirm.smiles = vocab.get_smiles(wid_to_change)
                    mol = Chem.MolFromSmiles(node_pertube_confirm.smiles)
                    Chem.Kekulize(mol)
                    node_pertube_confirm.mol = mol
                    tree.nodes[nodes_exact_idx_reverse[int(idx_to_change)]] = node_pertube_confirm

                    beam_tree.tree = tree
                    pertube_p_sum += (- p_pertube.cpu().item() + sum_p.cpu().item())
                    #if c > len_assem_check:
                    return beam_tree, pertube_p_sum, True
        
        beam_tree.tree = tree
        return beam_tree, 0.0, False
    
    
    def check_final_tree(self, beam_tree, vocab, device, check_num=10):
        tree = beam_tree.tree
        edges = torch.tensor(tree.adj_matrix).nonzero().T.tolist()
        check_skips = [(i, can_assemble(tree.nodes[i])) for i in range(len(tree.nodes))]
        check_skips = [ind for ind, assem in check_skips if not assem]
        if len(check_skips) == 0:
            return beam_tree
        elif len(check_skips) > 0.2 * len(tree.nodes):
            return None
        else:
            corrected = 0
            pad_wid = [tree.nodes[i].wid for i in check_skips]
            check_results = self.check_node(vocab, tree.nodes, edges, check_skips, pad_wid, device, check_num)
            sum_p = torch.sum(torch.tensor([r[0] for r in check_results]))
            for i , result in enumerate(check_results):
                if check_num > len(result[1]):
                    check_num_cut = len(result[1])
                else:
                    check_num_cut = check_num
                for j in range(check_num_cut):
                    if not result[1][j][0]:
                        idx_to_change = check_skips[i]
                        nodes_pertube = copy.deepcopy(tree.nodes)
                        node_to_pertube = nodes_pertube[idx_to_change]
                        wid_to_change = result[1][j][1]
                        #node_to_pertube.fp = vocab.fp_df.loc[vocab.get_smiles(wid_to_change)]
                        node_to_pertube.wid = wid_to_change
                        node_to_pertube.smiles = vocab.get_smiles(wid_to_change)
                        mol = Chem.MolFromSmiles(node_to_pertube.smiles)
                        Chem.Kekulize(mol)
                        node_to_pertube.mol = mol
                        nodes_pertube[idx_to_change] = node_to_pertube

                        pad_wid = [nodes_pertube[j].wid for j in range(len(nodes_pertube))]
                        pad_idx = [j for j in range(len(nodes_pertube))]
                        check_results_pertube = self.check_node(vocab, nodes_pertube, edges, pad_idx, pad_wid, device)

                        p_pertube = torch.sum(torch.tensor([r[0] for r in check_results_pertube]))

                        if can_assemble(node_to_pertube) and p_pertube > sum_p:
                            tree.nodes = nodes_pertube
                            beam_tree.tree = tree
                            corrected += 1
                            break
            if corrected == len(check_skips):
                return beam_tree
            else:
                return None


def get_bfs_depth_edges(edges, center, n_nodes):
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
    if len(edges_depth[0]) == 0:
        edges_depth = [edges_depth]
    for i in range(len(edges[0])):
        if depth[edges[0][i]] < depth[edges[1][i]]:
            edges_depth[depth[edges[1][i]] - 2][0].append(edges[1][i])
            edges_depth[depth[edges[1][i]] - 2][1].append(edges[0][i])
    edges_depth.reverse()
    return edges_depth

def flat_add_and_concat(edges, n_nodes):
    for batch_id in range(len(edges)):
        for layer_id in range(len(edges[batch_id])):
            for i in range(len(edges[batch_id][layer_id][0])):
                edges[batch_id][layer_id][0][i] += batch_id * n_nodes
                edges[batch_id][layer_id][1][i] += batch_id * n_nodes
    
    max_depth = max([len(e) for e in edges])
    if max_depth > 1:
        flat_edges = [[[], []] for _ in range(max_depth)]
    elif max_depth == 1:
        flat_edges = [[[], []]]
    else:
        return [[[], []]]
    
    for batch_id in range(len(edges)):
        for layer_id in range(len(edges[batch_id])):
            flat_edges[layer_id][0].extend(edges[batch_id][layer_id][0])
            flat_edges[layer_id][1].extend(edges[batch_id][layer_id][1])
    return flat_edges

def handle_wrong_sizes(size, vocab):
    size_perm = [vocab.get_size(size + perm) for perm in [-1, 1, -2, 2]]
    size_perm_l = [len(p) for p in size_perm]
    size_perm = size_perm[size_perm_l.index(max(size_perm_l))]
    if max(size_perm_l) == 0:
        return random.choice(vocab.molsizes)
    else:
        return size_perm 
