import torch
from models.flows.utils import (
    center_gravity_zero_gaussian_log_likelihood,
    center_gravity_zero_gaussian_log_likelihood_with_mask,
    sample_center_gravity_zero_gaussian,
    sample_center_gravity_zero_gaussian_with_mask, sample_gaussian_with_mask,
    standard_gaussian_log_likelihood_with_mask)


class PositionFeatureEdgePrior(torch.nn.Module):
    def __init__(self, n_dim, in_node_nf, in_edge_nf):
        super().__init__()
        self.n_dim = n_dim
        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf

    def forward(self, z_x, z_h, z_edge, node_mask=None, edge_mask=None):
        assert len(z_x.size()) == 3
        assert node_mask.size()[:2] == z_x.size()[:2] 
        node_mask_x = node_mask.unsqueeze(2).repeat(1, 1, z_x.size(2))
        node_mask_h = node_mask.unsqueeze(2).repeat(1, 1, z_h.size(2))
        edge_mask = edge_mask.view(node_mask_x.size(0), node_mask_x.size(1), node_mask_x.size(1))
        assert (z_x * (1 - node_mask_x)).sum() < 1e-8 and \
               (z_h * (1 - node_mask_h)).sum() < 1e-8 and \
                (z_edge * (1 - edge_mask)).sum() < 1e-8, \
               'These variables should be properly masked.'

        log_pz_x = center_gravity_zero_gaussian_log_likelihood_with_mask(
            z_x, node_mask
        )

        log_pz_h = standard_gaussian_log_likelihood_with_mask(
            z_h, node_mask
        )
        
        log_pz_edge = standard_gaussian_log_likelihood_with_mask(
            z_edge, edge_mask
        )
        log_pz = log_pz_x + log_pz_h + log_pz_edge
        return log_pz, (log_pz_x, log_pz_h, log_pz_edge)

    def sample(self, n_samples, n_nodes, node_mask, edge_mask):
        node_mask_x = node_mask.unsqueeze(-1)
        edge_mask = edge_mask.view(n_samples, n_nodes * n_nodes)
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dim), device=node_mask.device,
            node_mask=node_mask_x)
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z_edge = sample_gaussian_with_mask(
            size=(n_samples, n_nodes * n_nodes, self.in_edge_nf), device=node_mask.device,
            node_mask=edge_mask)
        z_edge = z_edge.view(n_samples, n_nodes, -1)

        return z_x, z_h, z_edge

class PositionFeaturePrior(torch.nn.Module):
    def __init__(self, n_dim, in_node_nf, in_edge_nf):
        super().__init__()
        self.n_dim = n_dim
        self.in_node_nf = in_node_nf

    def forward(self, z_x, z_h, node_mask=None):
        assert len(z_x.size()) == 3
        #assert len(node_mask.size()) == 3
        assert node_mask.size()[:2] == z_x.size()[:2] 
        node_mask_x = node_mask.unsqueeze(2).repeat(1, 1, z_x.size(2))
        node_mask_h = node_mask.unsqueeze(2).repeat(1, 1, z_h.size(2))
        assert (z_x * (1 - node_mask_x)).sum() < 1e-8 and \
               (z_h * (1 - node_mask_h)).sum() < 1e-8, \
               'These variables should be properly masked.'

        log_pz_x = center_gravity_zero_gaussian_log_likelihood_with_mask(
            z_x, node_mask
        )

        log_pz_h = standard_gaussian_log_likelihood_with_mask(
            z_h, node_mask
        )

        log_pz = log_pz_x + log_pz_h
        return log_pz, (log_pz_x, log_pz_h)

    def sample(self, n_samples, n_nodes, node_mask):
        node_mask_x = node_mask.repeat(1, 1, self.n_dim)
        node_mask_h = node_mask.squeeze(-1)
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dim), device=node_mask.device,
            node_mask=node_mask_x)
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask_h)

        return z_x, z_h


class PositionPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return center_gravity_zero_gaussian_log_likelihood(x)

    def sample(self, size, device):
        samples = sample_center_gravity_zero_gaussian(size, device)
        return samples
