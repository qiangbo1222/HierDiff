import math

import torch
from torch import nn

# from zmq import device

class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn="silu", attention=False, edge_update=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        if act_fn == "silu":
            act_fn = nn.SiLU()
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.edge_update = edge_update

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
        
        if self.edge_update:
            self.mes_mlp = nn.Sequential(
                nn.Linear(edges_in_d + hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, edges_in_d),
                act_fn)

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, col, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg
    
    def mes_model(self, edge_attr, mes, edge_mask):
        out = torch.cat([mes, edge_attr], dim=1)

        edge_attr = self.mes_mlp(out)
        if edge_mask is not None:
            edge_attr = edge_attr * edge_mask
        return edge_attr

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if self.edge_update:
            edge_attr = self.mes_model(edge_attr, edge_feat, edge_mask)
        if node_mask is not None:
            h = h * node_mask
        if edge_mask is not None:
            edge_attr = edge_attr * edge_mask
        return h, edge_attr


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, col, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, act_fn=nn.SiLU(), n_layers=2, attention=True, edge_update=True,
                 norm_diff=True, tanh=False, coords_range=30, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        #self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.sin_embedding = sin_embedding
        self.edge_update = edge_update
        if self.sin_embedding is not None:
            distance_nf = self.sin_embedding.dim
        else:
            distance_nf = 1
        if self.edge_update:
             self.edge_embedding = nn.Linear(edge_feat_nf + distance_nf, hidden_nf)
        else:
            self.edge_embedding = nn.Linear(edge_feat_nf + distance_nf, edge_feat_nf)

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention, edge_update=edge_update, 
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        #self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        edge_attr = self.edge_embedding(edge_attr)

        for i in range(0, self.n_layers):
            if self.edge_update:
                h, edge_attr = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            else:
                h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        if edge_mask is not None:
            edge_attr = edge_attr * edge_mask
        return h, x, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, in_edge_nf=1, act_fn="silu", n_layers=3, attention=False, edge_update=True,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=30, norm_constant=0, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=1, aggregation_method='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        # self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.edge_update = edge_update

        if act_fn == "silu":
            act_fn = nn.SiLU()

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            in_edge_nf += self.sin_embedding.dim
        else:
            self.sin_embedding = None
            

        self.node_embedding = nn.Linear(in_node_nf, self.hidden_nf)
        if self.edge_update:
            self.edge_embedding = nn.Linear(in_edge_nf + 1, self.hidden_nf)
            edge_feat_nf = self.hidden_nf
        else:
            edge_feat_nf = in_edge_nf
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf,
                                                               act_fn=act_fn, n_layers=inv_sublayers, edge_update=edge_update,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=self.coords_range_layer, norm_constant=norm_constant,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        # self.to(self.device)

    def forward(self, h, x, edge_index, edge_attr, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.node_embedding(h)

        if self.edge_update:
            edge_attr = torch.cat([distances, edge_attr], dim=1)
            edge_attr = self.edge_embedding(edge_attr)
        else:
            edge_attr = distances
        for i in range(0, self.n_layers):
            if self.edge_update:
                h, x, edge_attr = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=edge_attr)
            else:
                h, x, _ = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=edge_attr)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        if edge_mask is not None:
            edge_attr = edge_attr * edge_mask
        if self.edge_update:
            return h, x, edge_attr
        else:
            return h, x


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=30., min_res=30. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        #self.frequencies = 
        self.register_buffer("frequencies", 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res)
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    import pdb

    # pdb.set_trace()  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    # print(data.device,segment_ids.device)
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
