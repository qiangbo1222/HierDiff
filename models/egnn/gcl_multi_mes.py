import torch
from torch import linalg as LA
from torch import nn

from models.egnn.utils import unsorted_segment_mean, unsorted_segment_sum

ATOM_FDIM = 35
BOND_FDIM = 5
MAX_ATOM_NUM = 54

class E_GCL_multi_mes(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, jt_mess_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), recurrent=True, attention=False, clamp=False, norm_diff=True, tanh=False, coords_range=1, agg='sum', coord_update=True, edge_update=True):
        super(E_GCL_multi_mes, self).__init__()
        input_edge = input_nf * 2
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.agg_type = agg
        self.tanh = tanh
        edge_coords_nf = 1


        self.mes_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        self.edge_update = edge_update
        if edge_update:
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_nf + edge_coords_nf + edges_in_d, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf))

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_update = coord_update
        if coord_update:
            coord_mlp = []
            coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
            coord_mlp.append(act_fn)
            coord_mlp.append(layer)
            if self.tanh:
                coord_mlp.append(nn.Tanh())
                self.coords_range = coords_range

            self.coord_mlp = nn.Sequential(*coord_mlp)
        
        self.clamp = clamp

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
            self.att_mlp_aug = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def mes_model(self, source, target, radial, edge_attr, edge_mask):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.mes_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        if edge_mask is not None:
            edge_mask = edge_mask.repeat(1, out.size(1))
            out = out * edge_mask
        return out
    
    
    def edge_model(self, edge_feat, radial, edge_attr, edge_mask):
        out = torch.cat([edge_feat, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if edge_mask is not None:
            edge_mask = edge_mask.repeat(1, out.size(1))
            out = out * edge_mask
        return out


    def node_model(self, x, edge_index, edge_feat):
        row, col = edge_index
        #agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))
        agg = unsorted_segment_sum(edge_feat, col, num_segments=x.size(0))#for directed graph
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask):
        row, col = edge_index
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
        #trans = torch.clamp(trans, min=-100, max=100)
        if edge_mask is not None:
            edge_mask = edge_mask.repeat(1, trans.shape[1])
            trans = trans * edge_mask

        if self.agg_type == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.agg_type == 'mean':
            if node_mask is not None:
                #raise Exception('This part must be debugged before use')
                agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
                M = unsorted_segment_sum(node_mask[col], row, num_segments=coord.size(0))
                agg = agg/(M-1)
            else:
                agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coordinates aggregation type")
        coord = coord + agg
        return coord
    

    def forward(self, h, edge_index, coord, mess_holder, node_holder, node_embed, jt_idx, jt_mess, edge_attr=None, node_mask=None, edge_mask=None):
        ''''''
        row, col = edge_index
        mess_collect = []
        for r, c in zip(row, col):
            cand_idx = r // MAX_ATOM_NUM
            r, c = r % MAX_ATOM_NUM, c % MAX_ATOM_NUM
            jt_r, jt_c = mess_holder[cand_idx, r, c]
            if jt_r !=0 and jt_c !=0:
                mess_collect.append(jt_mess[jt_idx[cand_idx], jt_r - 1, jt_c - 1])
        mess_collect = torch.stack(mess_collect)
        node_collect = []
        for i in range(h.shape[0]):
            cand_idx = i // MAX_ATOM_NUM
            jt_node_idx = node_holder[cand_idx, i % MAX_ATOM_NUM]
            node_collect.append(node_embed[jt_idx[cand_idx], jt_node_idx - 1])#debug here
        node_collect = torch.stack(node_collect)
        
        
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_attr = edge_attr + mess_collect
        h = h + node_collect
        edge_feat = self.mes_model(h[row], h[col], radial, edge_attr, edge_mask)
        if self.coord_update:
            coord = self.coord_model(coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask)

        h, agg = self.node_model(h, edge_index, edge_feat)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        if self.edge_update:
            edge_attr = self.edge_model(edge_feat, radial, edge_attr, edge_mask)
        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        if self.edge_update :
            if edge_mask is not None:
                edge_attr = edge_attr * edge_mask
            return h, coord, edge_attr
        return h, coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff/(norm + 1)

        return radial, coord_diff

    



