import numpy as np
import torch
from equivariant_diffusion.utils import (
    center_gravity_zero_gaussian_log_likelihood,
    center_gravity_zero_gaussian_log_likelihood_with_mask,
    sample_center_gravity_zero_gaussian,
    sample_center_gravity_zero_gaussian_with_mask, sample_gaussian_with_mask,
    standard_gaussian_log_likelihood_with_mask)
from torch.distributions.categorical import Categorical


class PositionFeaturePrior(torch.nn.Module):
    def __init__(self, n_dim, in_node_nf):
        super().__init__()
        self.n_dim = n_dim
        self.in_node_nf = in_node_nf

    def forward(self, z_x, z_h, node_mask=None):
        assert len(z_x.size()) == 3
        assert len(node_mask.size()) == 3
        assert node_mask.size()[:2] == z_x.size()[:2]

        assert (z_x * (~node_mask)).sum() < 1e-8 and \
               (z_h * (~node_mask)).sum() < 1e-8, \
               'These variables should be properly masked.'

        log_pz_x = center_gravity_zero_gaussian_log_likelihood_with_mask(
            z_x, node_mask
        )

        log_pz_h = standard_gaussian_log_likelihood_with_mask(
            z_h, node_mask
        )

        log_pz = log_pz_x + log_pz_h
        return log_pz

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask):
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dim), device=node_mask.device,
            node_mask=node_mask)
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)

        return z_x, z_h


class PositionPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return center_gravity_zero_gaussian_log_likelihood(x)

    @torch.no_grad()
    def sample(self, size, device):
        samples = sample_center_gravity_zero_gaussian(size, device)
        return samples

class DistributionNodes(torch.nn.Module):
    def __init__(self, histogram):
        super().__init__()

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])

        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        #print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    @torch.no_grad()
    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return [self.n_nodes[i] for i in idx.tolist()]
    
    # def _slow_forward(self, *input, **kwargs):
    #     return super()._slow_forward(*input, **kwargs)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p

        log_probs = log_p[batch_n_nodes]

        return log_probs

