import torch
import torch.nn as nn

from models.egnn.gcl import E_GCL
from models.flows.utils import remove_mean, remove_mean_with_mask


class EGNN_dynamics(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
            act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True, attention=True,
                 condition_time=True, tanh=False, mode='egnn_dynamics', agg='sum', edge_update=True):
        super().__init__()
        self.mode = mode
        self.edge_update = edge_update
        if mode == 'egnn_dynamics':
            if edge_update:
                self.egnn = EGNN(
                    in_node_nf=in_node_nf + context_node_nf, in_edge_nf=2,
                    hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                    n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg, edge_update=True)
            else:
                self.egnn = EGNN(
                    in_node_nf=in_node_nf + context_node_nf, in_edge_nf=0,
                    hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                    n_layers=n_layers, recurrent=recurrent, attention=attention, tanh=tanh, agg=agg, edge_update=False)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0, hidden_nf=hidden_nf, out_node_nf= 3 + in_node_nf, device=device, act_fn=act_fn, n_layers=n_layers, recurrent=recurrent, attention=attention)

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, state, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, state, node_mask, edge_mask, context, edges=None):
        self.device = node_mask.device
        bs, n_nodes, dims = state.shape
        if self.edge_update:
            dims -= n_nodes
            h_dims = dims - self.n_dims
            xh = state[:, :, n_nodes:]
            attn_matrix = state[:, :, :n_nodes]
        else:
            h_dims = dims - self.n_dims
            xh = state
        if edges is None:
            edges = self.get_adj_matrix(n_nodes, bs, self.device)
        else:
            edges = edges.view(bs*n_nodes, bs*n_nodes).nonzero()
            edges = [edges[:, 0], edges[:, 1]]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        if self.edge_update:
            attn_matrix = attn_matrix.contiguous().view(bs*n_nodes*n_nodes, -1) * edge_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            h_time = torch.empty_like(h[:, 0:1]).fill_(t)
            h = torch.cat([h, h_time], dim=1)
            if self.edge_update:
                edge_time = torch.empty_like(attn_matrix[:, 0:1]).fill_(t)
                attn_matrix = torch.cat([attn_matrix, edge_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)




        if self.mode == 'egnn_dynamics':
            if self.edge_update:
                h_final, x_final, edge_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=attn_matrix)
            else:
                h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]
            if self.edge_update:
                edge_final = edge_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)
        if self.edge_update:
            edge_final = edge_final.view(bs, n_nodes, -1)
        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            if self.edge_update:
                return torch.cat([edge_final, vel, h_final], dim=2)
            else:
                return torch.cat([vel, h_final], dim=2)

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


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, edge_update=True, recurrent=True, attention=False, out_node_nf=None, out_edge_nf=None, tanh=False, coords_range=30, agg='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        if out_edge_nf is None:
            out_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        #self.device = device
        self.edge_update = edge_update
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)/self.n_layers
        if agg == 'mean':
            self.coords_range_layer = self.coords_range_layer * 19
        self.embedding_node = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_node_out = nn.Linear(self.hidden_nf, out_node_nf)
        if self.edge_update:
            self.embedding_edge = nn.Linear(in_edge_nf, self.hidden_nf)
            self.embedding_edge_out = nn.Linear(self.hidden_nf, out_edge_nf)
        for i in range(0, n_layers):
            if in_edge_nf != 0:
                self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=self.hidden_nf, act_fn=act_fn, recurrent=recurrent, attention=attention, tanh=tanh, coords_range=self.coords_range_layer, agg=agg, edge_update=edge_update))
            else:
                self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=0, act_fn=act_fn, recurrent=recurrent, attention=attention, tanh=tanh, coords_range=self.coords_range_layer, agg=agg, edge_update=edge_update))
        #self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding_node(h)
        if self.edge_update:
            edge_attr = self.embedding_edge(edge_attr)
            for i in range(0, self.n_layers):
                h, x, edge_attr = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            edge_attr = self.embedding_edge_out(edge_attr)
        else:
             for i in range(0, self.n_layers):
                h, x = self._modules["gcl_%d" % i](h, edges, x, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_node_out(h)
        
        

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        if self.edge_update and edge_mask is not None:
            edge_attr = edge_attr * edge_mask
            return h, x, edge_attr
        else:
            return h, x

