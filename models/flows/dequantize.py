import torch
import torch.nn.functional as F
from models.egnn.models import EGNN
from models.flows.utils import (assert_correctly_masked,
                                sample_gaussian_with_mask,
                                standard_gaussian_log_likelihood_with_mask,
                                sum_except_batch)
from torch import nn


class EGNN_output_he(nn.Module):
    def __init__(self, in_node_nf, out_node_nf, out_edge_nf, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True,
                 attention=False, agg='sum'):
        super().__init__()
        self.egnn = EGNN(in_node_nf=in_node_nf, in_edge_nf=1,
                         hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                         n_layers=n_layers, recurrent=recurrent,
                         attention=attention,
                         out_node_nf=out_node_nf, out_edge_nf=out_edge_nf, agg=agg)

        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        #self.device = device
        # self.n_dims = None
        self._edges_dict = {}

    def forward(self, x, h, edge_attr, node_mask, edge_mask):
        bs, n_nodes, dims = x.shape
        assert self.in_node_nf == h.size(2)
        self.device = x.device
        node_mask = node_mask.view(bs * n_nodes, 1)
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        x = x.view(bs*n_nodes, dims) * node_mask

        h = h.view(bs*n_nodes, self.in_node_nf) * node_mask
        edge_attr = edge_attr * edge_mask

        h_final, x_final, edge_attr_final = self.egnn(
            h, x, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)

        h_final = h_final.view(bs, n_nodes, self.out_node_nf)
        edge_attr_final = edge_attr_final.view(bs, n_nodes * n_nodes, -1)

        return h_final, edge_attr_final

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



def sigmoid(x, node_mask):
    z = torch.sigmoid(x)
    node_mask = node_mask.unsqueeze(2).repeat(1, 1, x.size(2))
    ldj = sum_except_batch(node_mask * (F.logsigmoid(x) + F.logsigmoid(-x)))
    return z, ldj


def affine(x, translation, log_scale):
    z = translation + x * log_scale.exp()
    ldj = sum_except_batch(log_scale)
    return z, ldj


def transform_to_hypercube_partition(float_feature, interval_noise):
    assert interval_noise.min().item() >= 0., interval_noise.max().item() <= 1.
    return float_feature + interval_noise


def transform_to_argmax_partition(onehot, u, node_mask):
    assert torch.allclose(
        onehot.sum(-1, keepdims=True) * node_mask,
        torch.ones_like(onehot[..., 0:1]) * node_mask)

    T = (onehot * u).sum(-1, keepdim=True)
    z = onehot * u + node_mask * (1 - onehot) * (T - F.softplus(T - u))
    ldj = (1 - onehot) * F.logsigmoid(T - u) * node_mask

    assert_correctly_masked(z, node_mask)
    assert_correctly_masked(ldj, node_mask)

    ldj = sum_except_batch(ldj)

    return z, ldj


class VariationalDequantizer(nn.Module):
    def __init__(self, node_nf, node_nf_int, device, agg='sum'):
        super().__init__()
        self.net_fn = EGNN_output_he(
            in_node_nf=node_nf, out_node_nf=node_nf_int*2, out_edge_nf=2, device=device, agg=agg
        )

    def sample_qu_xh(self, node_mask, edge_mask, x, h, edge_attr):
        h_net_out, e_net_out = self.net_fn(x, h, edge_attr, node_mask, edge_mask)
        h_mu, h_log_sigma = torch.chunk(h_net_out, chunks=2, dim=-1)
        e_mu, e_log_sigma = torch.chunk(e_net_out, chunks=2, dim=-1)

        h_eps = sample_gaussian_with_mask(h_mu.size(), h_mu.device, node_mask)
        h_log_q_eps = standard_gaussian_log_likelihood_with_mask(h_eps, node_mask)
        edge_mask = edge_mask.view(h_mu.size(0), h_mu.size(1) * h_mu.size(1))
        e_eps = sample_gaussian_with_mask(e_mu.size(), e_mu.device, edge_mask)
        e_log_q_eps = standard_gaussian_log_likelihood_with_mask(e_eps, edge_mask)


        h_u, h_ldj = affine(h_eps, h_mu, h_log_sigma)
        h_log_qu = h_log_q_eps - h_ldj
        e_u, e_ldj = affine(e_eps, e_mu, e_log_sigma)
        e_log_qu = e_log_q_eps - e_ldj

        return h_u, h_log_qu, e_u, e_log_qu

    def transform_to_partition_v(self, int_feature, edge_attr, u_int_feature, u_edge_attr, node_mask, edge_mask):
        u_int_feature, ldj_int_feature = sigmoid(u_int_feature, node_mask)
        edge_mask = edge_mask.view(u_int_feature.size(0), u_int_feature.size(1) * u_int_feature.size(1))
        edge_attr = edge_attr.view(u_int_feature.size(0), u_int_feature.size(1) * u_int_feature.size(1), -1)
        u_edge_attr, ldj_edge_attr = sigmoid(u_edge_attr, edge_mask)
        ldj = ldj_int_feature + ldj_edge_attr

        v_int_feature = transform_to_hypercube_partition(int_feature, u_int_feature)
        v_edge_attr = transform_to_hypercube_partition(edge_attr, u_edge_attr)
        return v_int_feature, v_edge_attr, ldj

    def forward(self, tensor, node_mask, edge_mask, x):
        int_feature, float_feature, edge_attr = tensor['int_feature'], tensor['float_feature'], tensor['edge_attr']

        h = torch.cat([int_feature, float_feature], dim=2)

        n_int_feature, n_float_feature = int_feature.size(2), float_feature.size(2)

        int_feature_u, log_qu_xh, edge_u, log_qu_xe = self.sample_qu_xh(node_mask, edge_mask, x, h, edge_attr)

        
        
        v_int_feature, v_edge, ldj = self.transform_to_partition_v(
            int_feature, edge_attr, int_feature_u, edge_u, node_mask, edge_mask)
        log_qv_xh = log_qu_xh + log_qu_xe - ldj

        if node_mask is not None:
            node_mask_int = node_mask.unsqueeze(2).repeat(1, 1, v_int_feature.size(2))
            v_int_feature = v_int_feature * node_mask_int
        if edge_mask is not None:
            v_edge = v_edge.view(-1).unsqueeze(-1)
            v_edge = v_edge * edge_mask

        v = {'int_feature': v_int_feature, 'float_feature': float_feature, 'edge_attr': v_edge}
        return v, log_qv_xh

    def reverse(self, tensor):
        int_feature, float_feature, edge_attr = tensor['int_feature'], tensor['edge_attr']
        int_feature = torch.floor(int_feature)
        edge_attr = torch.floor(edge_attr)
        return {'int_feature': int_feature, 'float_feature': float_feature, 'edge_attr': edge_attr}

class UniformDequantizer(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(UniformDequantizer, self).__init__()

    def forward(self, tensor, node_mask, edge_mask, x):
        int_feature, float_feature, edge_attr = tensor['int_feature'], tensor['float_feature'], tensor['edge_attr']
        zeros = torch.zeros(int_feature.size(0), device=int_feature.device)

        out_int_feature = int_feature + torch.rand_like(int_feature) - 0.5
        out_edge_attr = edge_attr + torch.rand_like(edge_attr) - 0.5

        if node_mask is not None:
            node_mask_int = node_mask.unsqueeze(2).repeat(1, 1, out_int_feature.size(2))
            out_int_feature = out_int_feature * node_mask_int
        if edge_mask is not None:
            out_edge_attr = out_edge_attr * edge_mask

        out = {'int_feature': out_int_feature, 'float_feature': float_feature, 'edge_attr': out_edge_attr}
        return out, zeros

    def reverse(self, tensor):
        int_feature, float_feature, edge_attr = tensor['int_feature'], tensor['float_feature'], tensor['edge_attr']
        int_feature = torch.round(int_feature)
        edge_attr = torch.round(edge_attr)
        return {'int_feature': int_feature, 'float_feature': float_feature, 'edge_attr': edge_attr}
