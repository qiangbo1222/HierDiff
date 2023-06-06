import torch


def masked_mean(x, node_mask, dim, keepdim=False):
    return x.sum(dim=dim, keepdim=keepdim) / node_mask.sum(dim=dim, keepdim=keepdim)


def masked_stdev(x, node_mask, dim, keepdim=False):
    mean = masked_mean(x, node_mask, dim, keepdim=True)

    diff = (x - mean) * node_mask
    diff_2 = diff.pow(2).sum(dim=dim, keepdim=keepdim)

    diff_div_N = diff_2 / node_mask.sum(dim=dim, keepdim=keepdim)
    return torch.sqrt(diff_div_N + 1e-5)


class ActNormPositionAndFeatures(torch.nn.Module):
    def __init__(self, in_node_nf, n_dims, in_edge_nf=None):
        super().__init__()
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.in_edge_nf = in_edge_nf

        self.x_log_s = torch.nn.Parameter(torch.zeros(1, 1))
        if self.in_edge_nf is not None:
            self.e_t = torch.nn.Parameter(torch.zeros(1, in_edge_nf))
            self.e_log_s = torch.nn.Parameter(torch.zeros(1, in_edge_nf))

        self.h_t = torch.nn.Parameter(torch.zeros(1, in_node_nf))
        self.h_log_s = torch.nn.Parameter(torch.zeros(1, in_node_nf))
        self.register_buffer('initialized', torch.tensor(0))

    def initialize(self, x, h, edge_attr=None, node_mask=None, edge_mask=None):
        print('initializing')
        with torch.no_grad():
            h_mean = masked_mean(h, node_mask, dim=0, keepdim=True)
            h_stdev = masked_stdev(h, node_mask, dim=0, keepdim=True)
            h_log_stdev = torch.log(h_stdev + 1e-8)

            self.h_t.data.copy_(h_mean.detach())
            self.h_log_s.data.copy_(h_log_stdev.detach())

            x_stdev = masked_stdev(x, node_mask, dim=(0, 1), keepdim=True)
            x_log_stdev = torch.log(x_stdev + 1e-8)

            self.x_log_s.data.copy_(x_log_stdev.detach())
            if self.in_edge_nf is not None:
                edge_mean = masked_mean(edge_attr, edge_mask, dim=0, keepdim=True)
                edge_stdev = masked_stdev(edge_attr, edge_mask, dim=0, keepdim=True)
                edge_log_stdev = torch.log(edge_stdev + 1e-8)

                self.e_t.data.copy_(edge_mean.detach())
                self.e_log_s.data.copy_(edge_log_stdev.detach())

            self.initialized.fill_(1)

    def forward(self, exh, node_mask, edge_mask, context=None, reverse=False):
        bs, n_nodes, dims = exh.shape
        if self.in_edge_nf is not None:
            dims -= n_nodes
            h_dims = dims - self.n_dims
            xh = exh[:, :, n_nodes:]
            edge_attr = exh[:, :, :n_nodes]
            edge_attr = edge_attr.contiguous().view(-1, 1)
        else:
            h_dims = dims - self.n_dims
            xh = exh
            edge_attr = None

        # edges = self.get_adj_matrix(n_nodes, bs, self.device)
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(xh.device)
        else:
            h = xh[:, self.n_dims:].clone()
        if self.in_edge_nf is not None:
            edge_attr = edge_attr * edge_mask

        if not self.initialized:
            self.initialize(x, h, edge_attr, node_mask, edge_mask)

        h_log_s = self.h_log_s.expand_as(h)
        h_t = self.h_t.expand_as(h)
        if self.in_edge_nf is not None:
            e_log_s = self.e_log_s.expand_as(edge_attr)
            e_t = self.e_t.expand_as(edge_attr)
            e_d_ldj = -(e_log_s * edge_mask).sum(1)
        x_log_s = self.x_log_s.expand_as(x)

        h_d_ldj = -(h_log_s * node_mask).sum(1)
        
        x_d_ldj = -(x_log_s * node_mask).sum(1)
        d_ldj = h_d_ldj + x_d_ldj
        d_ldj = d_ldj.view(bs, n_nodes).sum(1)
        if self.in_edge_nf is not None:
            d_ldj += e_d_ldj.view(bs, n_nodes*n_nodes).sum(1)

        if not reverse:
            h = (h - h_t) / torch.exp(h_log_s) * node_mask
            if self.in_edge_nf is not None:
                edge_attr = (edge_attr - e_t) / torch.exp(e_log_s) * edge_mask
            x = x / torch.exp(x_log_s) * node_mask

        else:
            h = (h * torch.exp(h_log_s) + h_t) * node_mask
            if self.in_edge_nf is not None:
                edge_attr = (edge_attr * torch.exp(e_log_s) + e_t) * edge_mask
            x = x * torch.exp(x_log_s) * node_mask

        x = x.view(bs, n_nodes, self.n_dims)
        if self.in_edge_nf is not None:
            edge_attr = edge_attr.view(bs, n_nodes, n_nodes)
        h = h.view(bs, n_nodes, h_dims)
        if self.in_edge_nf is not None:
            exh = torch.cat([edge_attr, x, h], dim=2)
        else:
            exh = torch.cat([x, h], dim=2)

        if not reverse:
            return exh, d_ldj, 0
        else:
            return exh

    def reverse(self, exh, node_mask, edge_mask, context=None):
        assert self.initialized
        return self(exh, node_mask, edge_mask, context, reverse=True)
