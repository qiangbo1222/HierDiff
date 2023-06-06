import pickle

import numpy as np
import torch
import torch.nn as nn
from data_utils.data_diffuse import (get_bfs_order, get_bfs_order_new,
                                     get_dfs_order)
from data_utils.MPNN_pattern import priority_rank

from models.egnn.egnn_new import EGNN
from models.egnn.gcl import E_GCL
from models.flows.utils import remove_mean, remove_mean_with_mask


class Edge_denoise(nn.Module):
    def __init__(self, vocab_size, in_node_nf, hidden_nf, out_node_nf, array_dict, context_nf=0,
                    in_edge_nf=1, n_layers_full=3, n_layers_focal=3,
                    focal_loss=1, edge_loss=1, node_loss=1, perturb_loss=1, full_softmax=False):
        super(Edge_denoise, self).__init__()
        if not full_softmax:
            self.array_dict = pickle.load(open(array_dict, 'rb'))
        else:
            self.array_dict = None
        self.in_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.context_nf = context_nf
        self.n_layers_full = n_layers_full
        self.n_layers_focal = n_layers_focal
        self.feature_embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.vocab_embedding = nn.Embedding(vocab_size, self.hidden_nf)
        #self.size_embedding = nn.Embedding(26, self.hidden_nf)
        self.edge_embedding = nn.Linear(in_edge_nf + 1, self.hidden_nf)
        self.node_embedding = nn.Linear(self.hidden_nf * 2, self.hidden_nf)

        for i in range(0, n_layers_full):
            self.add_module("gcl_full_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, context_nf=context_nf, edges_in_d=hidden_nf, act_fn=nn.SiLU(), recurrent=True, attention=True, tanh=True, coords_range=30, agg='sum', coord_update=True, edge_update=True))
        #self.egnn_full = EGNN(self.hidden_nf * 2, self.hidden_nf * 2, hidden_nf * 2, out_node_nf=self.hidden_nf * 2,
        #                      n_layers=self.n_layers_full, act_fn=nn.SiLU(), attention=True, tanh=True, 
        #                      coords_range=30, aggregation_method='sum', sin_embedding=False, edge_update=True)
        
        for i in range(0, n_layers_focal):
            self.add_module("gcl_focal_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, context_nf=context_nf, edges_in_d=hidden_nf, act_fn=nn.SiLU(), recurrent=True, attention=False, tanh=True, coords_range=30, agg='sum', coord_update=True, edge_update=True))
        self.add_module("gcl_edge", E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, context_nf=context_nf, edges_in_d=1, act_fn=nn.SiLU(), recurrent=True, attention=False, tanh=True, coords_range=30, agg='sum', coord_update=True, edge_update=False))
        self.add_module("gcl_denoise", E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, context_nf=context_nf, edges_in_d=1, act_fn=nn.SiLU(), recurrent=True, attention=False, tanh=True, coords_range=30, agg='sum', coord_update=True, edge_update=False))
        #self.egnn_focal = EGNN(self.hidden_nf * 2, self.hidden_nf * 2, hidden_nf * 2, out_node_nf=self.hidden_nf * 2,
        #                      n_layers=self.n_layers_focal, act_fn=nn.SiLU(), attention=True, tanh=True, 
        #                      coords_range=30, aggregation_method='sum', sin_embedding=False, edge_update=True)
        #self.egnn_edge = EGNN(self.hidden_nf * 2, self.hidden_nf * 2, 1, out_node_nf=self.hidden_nf * 2,
        #                      n_layers=1, inv_sublayers=1, act_fn=nn.SiLU(), attention=False, tanh=True, 
        #                      coords_range=30, aggregation_method='sum', sin_embedding=False, edge_update=False)
        #self.egnn_denoise = EGNN(self.hidden_nf * 2, self.hidden_nf * 2, 1, out_node_nf=self.hidden_nf * 2,
        #                      n_layers=1, inv_sublayers=1, act_fn=nn.SiLU(), attention=False, tanh=True, 
        #                      coords_range=30, aggregation_method='sum', sin_embedding=False, edge_update=False)

        self.focal_predict = nn.Sequential(nn.Linear(self.hidden_nf +context_nf + 1, self.hidden_nf), nn.SiLU(), nn.Linear(self.hidden_nf, 1), nn.Sigmoid())
        self.edge_predict = nn.Sequential(nn.Linear(self.hidden_nf * 3 + 1 + context_nf * 2, self.hidden_nf), nn.SiLU(),  nn.Linear(self.hidden_nf, 1))
        self.node_predict = nn.Sequential(nn.Linear(self.hidden_nf + context_nf, self.hidden_nf), nn.SiLU(), nn.Linear(self.hidden_nf, out_node_nf))
        #self.node_size_predict = nn.Sequential(nn.Linear(self.hidden_nf * 2, self.hidden_nf), nn.SiLU(), nn.Dropout(0.1), nn.Linear(self.hidden_nf, 26))
        self.loss_lambda = {'focal_loss': focal_loss, 'edge_loss': edge_loss, 'node_loss': node_loss}#, 'perturb_loss': perturb_loss}
        self._edges_dict = {}

    def forward(self, batch):
        h = batch['node_feat'][0]
        #size = batch['node_size']
        if self.array_dict is not None:
            array = batch['node_array']
        bs, n_nodes = h.shape[:2]

        x = batch['node_pos']
        x_cache = x.clone()
        predict_idx = batch['predict_idx']
        edge_search = batch['edge_search_pad']
        edge_search_orig = batch['edge_search_pad_orig']
        edge_search_flat = batch['edge_search_flat']
        node_mask = batch['node_feat'][1][:, :, 0].view(bs*n_nodes, -1)
        edge_mask = batch['edge_mask'].view(bs*n_nodes*n_nodes, -1)
        focal = batch['focal']
        focal_cand = batch['focal_cand']
        real_focal = batch['real_focal']
        undiscovered = batch['undiscovered']
        label = batch['label']

        h = h.view(bs*n_nodes, -1)
        #size = size.view(bs*n_nodes)
        if self.array_dict is not None:
            array = array.view(bs*n_nodes, -1)
        x = x.view(bs*n_nodes, -1)
        h_f = self.feature_embedding(h[:, :self.in_node_nf])
        h_v = self.vocab_embedding(h[:, self.in_node_nf + self.context_nf].long())
        if self.context_nf > 0:
            h_c = h[:, self.in_node_nf:self.in_node_nf + self.context_nf]
        #h_s = self.size_embedding(size.long())
        h = torch.cat([h_f, h_v], dim=1)
        h = self.node_embedding(h)
        if self.context_nf > 0:
            h = torch.cat([h, h_c], dim=1)
        

        val = torch.sum(batch['search_adj_matrix'].view(bs*n_nodes, n_nodes), dim=-1 ,keepdim=True)
        edge_note = batch['search_adj_matrix'].view(bs*n_nodes*n_nodes, 1)
        edges_full = self.get_adj_matrix(n_nodes, bs, h.device)
        edge_attr = torch.sum((x[edges_full[0]] - x[edges_full[1]]) ** 2, dim=1, keepdim=True)
        edge_attr = torch.cat([edge_attr, edge_note], dim=1)
        edge_feat_full = self.edge_embedding(edge_attr)
        

        #full connect FP on full graph
        for i in range(0, self.n_layers_full):
            h, x, edge_feat_full = self._modules["gcl_full_%d" % i](h, edges_full, x, edge_attr=edge_feat_full, node_mask=node_mask, edge_mask=edge_mask)
        #h, x, edge_feat_full = self.egnn_full(h, x, edges_full, edge_attr=edge_feat_full, node_mask=node_mask, edge_mask=edge_mask)
        edge_feat_full = edge_feat_full.view(bs, n_nodes, n_nodes, -1)
        #h_cache, x_cache = h, x
        #predict the focal node
        #flat MP here
        max_depth = len(edge_search)
        if max_depth > 1:
            edges = edge_search_flat
            edge_feat_focal = edge_feat_full[edges[0]//n_nodes, edges[0]%n_nodes, edges[1]%n_nodes, :]
            edge_feat_focal = edge_feat_focal.view(edges[0].shape[0], -1)
            for i in range(0, self.n_layers_focal):
                h, x, edge_feat_focal = self._modules['gcl_focal_%d' % i](h, edges, x, edge_attr=edge_feat_focal, node_mask=node_mask)

            #h, x, edge_feat_focal = self.egnn_focal(h, x, edges, edge_attr=edge_feat_focal, node_mask=node_mask)

            focal_predict = self.focal_predict(torch.cat([h[focal_cand], val[focal_cand]], dim=1))
            edges_weight = self.split_edges(edge_search_flat, n_nodes, bs)
            node_weight = self.split_nodes(focal_cand, n_nodes, bs)
            edges_weight = [len(e) for e in edges_weight]
            node_weight = np.cumsum([0,] + [len(n) for n in node_weight])
            focal_loss = torch.tensor(0.).to(h.device)
            for i in range(bs):
                if edges_weight[i] != 0:
                    focal_loss += nn.BCELoss()(focal_predict[node_weight[i]: node_weight[i + 1]].squeeze(-1) , focal[node_weight[i]: node_weight[i + 1]].float())#/ edges_weight[i]
            focal_accuracy = 0
            focal_count = 0
            focal_check = self.split_nodes(focal_cand, n_nodes, bs)

            for i, f_k in enumerate(focal_check):
                if len(f_k) > 0:
                    f_k = [j + i * n_nodes for j in f_k]
                    f_k = [focal_cand.index(f) for f in f_k]
                    if focal[f_k[torch.argmax(focal_predict[f_k])]] == 1:
                        focal_accuracy += 1
                    focal_count += 1
            focal_accuracy /= (focal_count + 1e-8)
        else:
            focal_loss = torch.tensor(0.0).to(h.device)
            focal_accuracy = 0

        #predict the edge
        #h, x = h_cache, x_cache
        circle = [[i * n_nodes, i * n_nodes] for i in range(bs)]
        edge_search_orig = [circle] + edge_search_orig
        for depth in range(max_depth):
            edges = torch.tensor(edge_search_orig[depth]).to(h.device).T
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
            h, x = self._modules['gcl_edge'](h, edges, x, edge_attr=edge_attr, node_mask=node_mask)
        if max_depth > 0:
                #h, x = self.egnn_edge(h, x, edges, edge_attr=edge_attr, node_mask=node_mask)
            h_focal = h[real_focal, :].unsqueeze(1).repeat(1, n_nodes, 1)
            x_focal = x[real_focal, :].unsqueeze(1).repeat(1, n_nodes, 1)
            edge_focal = torch.stack([
                edge_feat_full[f//n_nodes, f%n_nodes, :, :] for f in real_focal
            ])
            h = h.view(bs, n_nodes, -1)
            x = x.view(bs, n_nodes, -1)
            h_attach = torch.stack([h[f//n_nodes, :, :] for f in real_focal])
            x_attach = torch.stack([x[f//n_nodes, :, :] for f in real_focal])
            edge_distance = torch.sum((x_attach - x_focal) ** 2, dim=2, keepdim=True)
            edge_predict = self.edge_predict(torch.cat([h_focal, edge_focal, h_attach, edge_distance], dim=-1))
            edge_loss = torch.tensor(0.0).to(h.device)
            focal_i = 0
            edge_count = 0
            edge_accuracy = 0
            edges_weight = self.split_edges([item for sublist in edge_search_orig for item in sublist], n_nodes, bs)
            edges_weight = [len(e) for e in edges_weight]
            for i in range(bs):
                if predict_idx[i] != 0:
                    target = torch.tensor(undiscovered[i].index((predict_idx[i])), dtype=torch.long).to(h.device).unsqueeze(0)
                    edge_predict_i = edge_predict[focal_i, undiscovered[i], :].squeeze(-1).unsqueeze(0)
                    edge_loss += nn.CrossEntropyLoss()(edge_predict_i, target) #/ edges_weight[i]
                    if torch.argmax(edge_predict_i, dim=-1) == target:
                        edge_accuracy += 1
                    edge_count += 1
                    focal_i += 1
            #edge_loss /= (edge_count + 1e-8)
            edge_accuracy /= (edge_count + 1e-8)
        else:
            edge_loss = torch.tensor(0.0).to(h.device)
            edge_accuracy = 0

        #predict the type
        #h, x = h_cache, x_cache
        h = h.view(bs*n_nodes, -1)
        x = x.view(bs*n_nodes, -1)
        edge_search = [circle] + edge_search
        if max_depth > 0:
            for depth in range(max_depth + 1):
                edges = torch.tensor(edge_search[depth]).to(h.device).T
                edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
                h, x = self._modules["gcl_denoise"](h, edges, x, edge_attr=edge_attr, node_mask=node_mask)

                #h, x = self.egnn_denoise(h, x, edges, edge_attr=edge_attr, node_mask=node_mask)
        h = h.view(bs, n_nodes, -1) 
        h_node = torch.stack([h[i, predict_idx[i], :] for i in range(bs)])
        node_predict = self.node_predict(h_node)
        #node_size_predict = self.node_size_predict(h_node)

        node_loss = torch.tensor(0.0).to(h.device)
        #node_size_loss = torch.tensor(0.0).to(h.device)
        edges_weight = self.split_edges([item for sublist in edge_search_orig for item in sublist], n_nodes, bs)
        edges_weight = [len(e) for e in edges_weight]
        #size = size.view(bs, n_nodes)
        #size = torch.stack([size[i, pr] for i, pr in enumerate(predict_idx)])
        if self.array_dict is not None:
            array = array.view(bs, n_nodes)
            array = torch.stack([array[i, pr] for i, pr in enumerate(predict_idx)])
        for i in range(bs):
            if self.array_dict is not None:
                softmax_space = self.array_dict[1][array[i].item()]
            else:
                softmax_space = [all_num for all_num in range(node_predict[i].shape[0])]
            node_predict_i = node_predict[i, softmax_space].unsqueeze(0)
            label_i = torch.tensor(softmax_space.index(label[i].item()), dtype=torch.long).to(h.device).unsqueeze(-1)
            node_loss += nn.CrossEntropyLoss()(node_predict_i, label_i)# / (edges_weight[i] + 1)
            #node_size_loss += nn.CrossEntropyLoss()(node_size_predict[i].unsqueeze(0), label_size[i].unsqueeze(-1)) / (edges_weight[i] + 1)
        node_accuracy = 0
        for i in range(bs):
            if self.array_dict is not None:
                softmax_space = self.array_dict[1][array[i].item()]
            else:
                softmax_space = [all_num for all_num in range(node_predict[i].shape[0])]
            if torch.argmax(node_predict[i, softmax_space]) == softmax_space.index(label[i].item()):
                node_accuracy += 1
        node_accuracy /= bs

        '''
        #predict the perturb coordinate
        x = x.view(bs, n_nodes, -1)
        if max_depth > 0:
            predict_attach_x = torch.stack([x[i, predict_idx[i], :] for i in range(bs)])
            real_x_focal = torch.stack([x_cache[i, predict_idx[i], :] for i in range(bs)]) - batch['perturb_x']
            perturb_loss = torch.mean((predict_attach_x - real_x_focal) ** 2)
        '''

        return {'focal_loss': focal_loss, 'focal_accuracy': torch.tensor(focal_accuracy), 
                'edge_loss': edge_loss, 'edge_accuracy': torch.tensor(edge_accuracy),
                'node_loss': node_loss, 'node_accuracy': torch.tensor(node_accuracy),
                'total_loss': self.loss_lambda['focal_loss'] * focal_loss + self.loss_lambda['edge_loss'] * edge_loss + self.loss_lambda['node_loss'] * node_loss}#'perturb_loss': perturb_loss,
    
    def sample_AR(self, batch):
        #output the new adj matrix and node types
        h = batch['node_feat'][0]
        bs, n_nodes = h.shape[:2]
        h = h.view(bs*n_nodes, -1)
        if self.array_dict is not None:
            array = torch.tensor([check_array_in_list(a[:-(2+self.context_nf)], self.array_dict[0]) for a in h])
            array = array.view(bs, n_nodes)
        

        x = batch['node_pos']
        node_mask = batch['node_feat'][1][:, :, 0]
        node_nums_batch = torch.sum(node_mask, dim=1).int()
        node_mask = node_mask.view(bs*n_nodes, -1)
        edge_mask = batch['edge_mask'].view(bs*n_nodes*n_nodes, -1)
        adj_matrix = batch['search_adj_matrix']
        val = torch.sum(adj_matrix.view(bs*n_nodes, n_nodes), dim=-1, keepdim=True)

        discovered = [i[0].item() for i in node_mask.squeeze(0).nonzero() if adj_matrix[i[0]//n_nodes, i[0]%n_nodes, :].sum() > 0]
        undiscovered = [i[0].item() for i in node_mask.squeeze(0).nonzero() if adj_matrix[i[0]//n_nodes, i[0]%n_nodes, :].sum() == 0]

        adj_matrix = torch.stack([m - torch.diag_embed(torch.diag(m)) for m in adj_matrix])


        
        x = x.view(bs*n_nodes, -1)
        #size = size.view(bs*n_nodes)
        h_f = self.feature_embedding(h[:, :self.in_node_nf])
        h_v = self.vocab_embedding(h[:, self.in_node_nf + self.context_nf].long())
        if self.context_nf > 0:
            h_c = h[:, self.in_node_nf:self.in_node_nf + self.context_nf]
        #h_s = self.size_embedding(size.long())
        h = torch.cat([h_f, h_v], dim=1)
        h = self.node_embedding(h)
        if self.context_nf > 0:
            h = torch.cat([h, h_c], dim=1)

        edges_full = self.get_adj_matrix(n_nodes, bs, h.device)
        edge_attr = torch.sum((x[edges_full[0]] - x[edges_full[1]]) ** 2, dim=1, keepdim=True)
        edge_attr = torch.cat([edge_attr, adj_matrix.view(bs*n_nodes*n_nodes, 1)], dim=1)
        edge_feat_full = self.edge_embedding(edge_attr)


        #h, x, edge_feat_full = self.egnn_full(h, x, edges_full, edge_attr=edge_feat_full, node_mask=node_mask, edge_mask=edge_mask)
        for i in range(0, self.n_layers_full):
            h, x, edge_feat_full = self._modules["gcl_full_%d" % i](h, edges_full, x, edge_attr=edge_feat_full, node_mask=node_mask, edge_mask=edge_mask)
        #h_cache, x_cache = h, x

        edge_feat_full = edge_feat_full.view(bs, n_nodes, n_nodes, -1)

        if adj_matrix.sum() > 0:
            edge_search_flat = []
            for i, adj_matrix_slice in enumerate(adj_matrix):
                edge_search_flat.append(
                    self.adj_matrix_to_edges_flat(self.strip_adj_matrix(adj_matrix_slice, node_nums_batch[i])))
            edge_search_flat = self.concat_edges(edge_search_flat, n_nodes)#only one layer
            edges = [torch.tensor([item for sublist in edge_search_flat[0] for item in sublist]).to(h.device), 
                    torch.tensor([item for sublist in edge_search_flat[1] for item in sublist]).to(h.device)]
            edge_feat_focal = edge_feat_full[edges[0]//n_nodes, edges[0]%n_nodes, edges[1]%n_nodes, :]
            edge_feat_focal = edge_feat_focal.view(edges[0].shape[0], -1)
            #h, x, edge_feat_focal = self.egnn_focal(h, x, edges, edge_attr=edge_feat_focal, node_mask=node_mask)
            for i in range(0, self.n_layers_focal):
                h, x, edge_feat_focal = self._modules['gcl_focal_%d' % i](h, edges, x, edge_attr=edge_feat_focal, node_mask=node_mask)
            h = h.view(bs, n_nodes, -1)
            focal_bins = self.split_nodes(discovered, n_nodes, bs)
            #print(f'focal-bins{focal_bins}')
            val = val.view(bs, n_nodes, -1)
            focal_predict = [focal_bins[i][torch.argmax(self.focal_predict(torch.cat([h[i, focal_bins[i], :], val[i, focal_bins[i]]], dim=-1)))] if len(focal_bins[i]) > 0 else -1 
                                for i in range(len(focal_bins))]
            focal_predict = [focal + i * n_nodes if focal >= 0 else -1 for i, focal in enumerate(focal_predict)]
        elif len(discovered) == 0:
            focal_predict = [-1 for _ in range(bs)]
        else:
            focal_predict = [0 for _ in range(bs)]

        #predict the edge
        edges_result = []
        #h, x = h_cache, x_cache
        h = h.view(bs*n_nodes, -1)
        if len(discovered) > 0:

            if adj_matrix.sum() > 0:
                edge_search_origin = []
                for i, adj_matrix_slice in enumerate(adj_matrix):
                    strip_m = self.strip_adj_matrix(adj_matrix_slice, node_nums_batch[i])
                    if focal_predict[i] >= 0:
                        edge_search_origin.append(
                            self.adj_matrix_to_edges_bfs(strip_m, blur_feature=h[i*n_nodes: i*n_nodes + strip_m.shape[0], :], end=focal_predict[i]%n_nodes))
                    else:
                        edge_search_origin.append([])

                edge_search_origin = self.concat_edges(edge_search_origin, n_nodes)
                circle = [[i * n_nodes, i * n_nodes] for i in range(bs)]
                edge_search_origin = [circle] + edge_search_origin
                max_depth = len(edge_search_origin)
                for depth in range(max_depth):
                    edges = torch.tensor(edge_search_origin[depth]).to(h.device).T
                    edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
                    #h, x = self.egnn_edge(h, x, edges, edge_attr=edge_attr, node_mask=node_mask)
                    h, x = self._modules['gcl_edge'](h, edges, x, edge_attr=edge_attr, node_mask=node_mask)

            focal_predict_remove = [f for f in focal_predict if f>=0]
            h_focal = h[focal_predict_remove, :].unsqueeze(1).repeat(1, n_nodes, 1)
            x_focal = x[focal_predict_remove, :].unsqueeze(1).repeat(1, n_nodes, 1)
            edge_focal = torch.stack([
                edge_feat_full[f//n_nodes, f%n_nodes, :, :] for f in focal_predict_remove
            ])
            h = h.view(bs, n_nodes, -1)
            x = x.view(bs, n_nodes, -1)
            h_attach = torch.stack([h[f//n_nodes, :, :] for f in focal_predict_remove])
            x_attach = torch.stack([x[f//n_nodes, :, :] for f in focal_predict_remove])
            edge_distance = torch.sum((x_attach - x_focal) ** 2, dim=2, keepdim=True)
            edge_predict = self.edge_predict(torch.cat([h_focal, edge_focal, h_attach, edge_distance], dim=-1))
            edge_predict_bins = self.split_nodes(undiscovered, n_nodes, bs)
            
            focal_i = 0
            for i in range(bs):
                #print(f'predict {focal_predict}')
                if not (0 in edge_predict_bins[i]):
                    edges_end = edge_predict_bins[i][torch.argmax(edge_predict[focal_i, edge_predict_bins[i], :])]
                    #print([focal_predict_remove[focal_i]%n_nodes, edges_end])
                    adj_matrix[i] = self.attach_to_adj_matrix(adj_matrix[i], [[focal_predict_remove[focal_i] % n_nodes, edges_end], [edges_end, focal_predict_remove[focal_i] % n_nodes]])
                    edges_result.append([focal_predict_remove[focal_i]%n_nodes, edges_end])
                    focal_i += 1
                else:
                    edges_result.append([-1, 0])
        else:
            edges_result = [[-1, 0] for _ in range(bs)]

        #predict the type
        #h, x = h_cache, x_cache
        h = h.view(bs*n_nodes, -1)
        x = x.view(bs*n_nodes, -1)
        edge_search = []
        for i, adj_matrix_slice in enumerate(adj_matrix):
            strip_m = self.strip_adj_matrix(adj_matrix_slice, node_nums_batch[i])
            if focal_predict[i] > 0:
                edge_search.append(
                    self.adj_matrix_to_edges_bfs(strip_m, blur_feature=h[i*n_nodes: i*n_nodes + strip_m.shape[0], :], end=edges_result[i][1]))
            else:
                edge_search.append([])            
        edge_search = self.concat_edges(edge_search, n_nodes)
        circle = [[i * n_nodes, i * n_nodes] for i in range(bs)]
        edge_search = [circle] + edge_search
        max_depth = len(edge_search)
        if max_depth > 0:
            for depth in range(max_depth):
                edges = torch.tensor(edge_search[depth]).to(h.device).T
                edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
                #h, x = self.egnn_denoise(h, x, edges, edge_attr=edge_attr, node_mask=node_mask)
                h, x = self._modules["gcl_denoise"](h, edges, x, edge_attr=edge_attr, node_mask=node_mask)

        h = h.view(bs, n_nodes, -1) 
        h_node = torch.stack([h[i, edges_result[i][1], :] for i in range(bs)])
        node_predict = self.node_predict(h_node)
        #size = size.view(bs, n_nodes)
        #size = torch.stack([size[i, edges_result[i][1]] for i in range(bs)])
        #node_size_predict = torch.argmax(self.node_size_predict(h_node), dim=-1)
        if self.array_dict is not None:
            array = torch.stack([array[i, edges_result[i][1]] for i in range(bs)])
            array = [self.array_dict[1][i] for i in array]
        edges_result = [e if e[0] >= 0 else [0] for e in edges_result]

        #x = x.view(bs, n_nodes, -1)
        #x_result = torch.stack([x[i, edges_result[i][1], :] if len(edges_result[i]) > 1 else x[i, edges_result[i][0], :] for i in range(bs)])

        if self.array_dict is not None:
            return edges_result, node_predict, array, adj_matrix#, x_result
        else:
            return edges_result, node_predict, adj_matrix#, x_result

        
    def pad_adj_matrix(self, adj_matrix):
        #pad bs, n_nodes, n_nodes into bs*n_nodes, bs*n_nodes
        bs, n_nodes = adj_matrix.shape[:2]
        adj_matrix_pad = torch.zeros(bs*n_nodes, bs*n_nodes).to(adj_matrix.device)
        for i in range(bs):
            adj_matrix_pad[i*n_nodes: (i+1)*n_nodes, i*n_nodes: (i+1)*n_nodes] = adj_matrix[i]
        return adj_matrix_pad
    
    def strip_adj_matrix(self, adj_matrix_pad, n_nodes):
        #n_nodes = torch.sum(adj_matrix_pad, dim=0).nonzero().max() + 1
        return adj_matrix_pad[:n_nodes, :n_nodes]
    
    def adj_matrix_to_edges_flat(self, adj_matrix):
        return adj_matrix.nonzero().T.tolist()
    
    def adj_matrix_to_edges_bfs(self, adj_matrix, blur_feature, end, priority=False):
        #adj_matrix need to remove the padding first
        if adj_matrix.sum() == 0:
            return [[]]
        else:
            edges = np.array(adj_matrix.nonzero().cpu())
            #val = np.sum(np.array(adj_matrix.cpu()), axis=-1)
            node_count = set()
            for e0, e1 in edges:
                node_count.add(e0)
                node_count.add(e1)
            num_nodes = len(node_count)
            bfs_result = get_bfs_order_new(edges, num_nodes, end)
            return bfs_result
    
    def adj_matrix_to_edges_dfs(self, adj_matrix, blur_feature, end, priority=False):
        #adj_matrix need to remove the padding first
        blur_feature = blur_feature.cpu()
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
            if priority:
                for i in range(num_nodes):
                    rank_ind = priority_rank(blur_feature[graph[i]], val[graph[i]])
                    rank_ind.reverse()
                    graph[i] = [graph[i][r] for r in rank_ind]
            dfs_result = get_dfs_order(graph, end)
            dfs_order, dfs_paths = dfs_result['order'], dfs_result['path']
            dfs_depth = max([d[1] for d in dfs_order])
            dfs_paths_reverse = []
            for e in dfs_paths[:dfs_depth]:
                dfs_paths_reverse = [[e[1], e[0]]] + dfs_paths_reverse
            return dfs_paths_reverse
        
    def attach_to_adj_matrix(self, adj_matrix, edges):
        for e in edges:
            adj_matrix[e[0], e[1]] = 1
        return adj_matrix
    
    def concat_edges(self, edges, n_nodes):
        max_depth = max([len(e) for e in edges])
        concat = [[] for _ in range(max_depth)]
        for i, e in enumerate(edges):
            e = flat_add(e, i*n_nodes)
            for l_i, l in enumerate(e):
                if len(l) > 0:
                    if isinstance(l[0], int):
                        concat[l_i].append(l)
                    else:
                        concat[l_i].extend(l)
                else:
                    concat[l_i].extend(l)
        return concat
    def split_nodes(self, node_idxs, n_nodes, bs):
        bins = [[] for _ in range(bs)]
        for i in node_idxs:
            bins[i//n_nodes].append(i%n_nodes)
        return bins
    
    def split_edges(self, edges, n_nodes, bs):
        bins = [[] for _ in range(bs)]
        for e in edges:
            bins[e[0]//n_nodes].append([e[0]%n_nodes, e[1]%n_nodes])
        return bins

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

        edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
        return edges

def flat_add(edge_list, add_num):
    for layer in edge_list:
        for edge_id, _ in enumerate(layer):
            if isinstance(layer[edge_id], list):
                layer[edge_id] = (layer[edge_id][0] + add_num, layer[edge_id][1] + add_num)
            else:
                layer[edge_id] += add_num
    return edge_list

def check_array_in_list(array, list_a):
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    all_diff = []
    for ind, ref in enumerate(list_a):
        diff = ((array - ref)**2).sum()
        all_diff.append(diff)
        if diff == 0:
            return ind
    return all_diff.index(min(all_diff))
