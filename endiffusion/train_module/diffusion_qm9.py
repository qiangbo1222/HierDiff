import logging
# from dataset.qm9.utils import prepare_context
import math
from distutils.debug import DEBUG
from typing import Dict

import numpy as np
import pytorch_lightning as pl
# from endiffusion.equivariant_diffusion.utils import remove_mean_with_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml
from dataset.datasets_statistics import get_dataset_info
from hydra.utils import instantiate
from loss.criterion import gaussian_KL, gaussian_KL_for_dimension
from models.distributions import DistributionNodes
from models.module.en_dynamics import EGNN_dynamics_QM9
from models.noise_model import GammaNetwork, PredefinedNoiseSchedule
from models.layers.egnn_new import EGNN
from models.utils import (assert_correctly_masked, assert_mean_zero_with_mask,
                          cdf_standard_gaussian, check_mask_correct, expm1,
                          remove_mean_with_mask,
                          sample_center_gravity_zero_gaussian_with_mask,
                          sample_gaussian_with_mask, softplus,
                          sum_except_batch)
from numpy import histogram
from yaml import Dumper, Loader

logger = logging.getLogger(__name__)
RESIDUE_LIST = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

class DiffusionQM9(pl.LightningModule):
    # TODO change the args and the params here to the hydra args and the hydra params
    # def __init__(self, dynamics: models.module.EGNN_dynamics_QM9, in_node_nf: int, n_dims: int,
    #         timesteps: int = 1000, parametrization='eps', noise_schedule='learned',
    #         noise_precision=1e-4, loss_type='vlb', norm_values=(1., 1., 1.),
    #         norm_biases=(None, 0., 0.), include_charges=True):
    def __init__(self,cfg:Dict[str,any]):        
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters() # type: dict
        self.pocket = cfg.pocket
        
        self.node_coarse_type = cfg.dynamics["node_coarse_type"]
        if self.node_coarse_type == "prop":
            self.in_node_nf = 8
            cfg.dynamics["in_node_nf"] = self.in_node_nf
        elif self.node_coarse_type == "elem":
            self.in_node_nf = 3
            cfg.dynamics["in_node_nf"] = self.in_node_nf
        else:
            raise NotImplementedError("node_coarse_type should be prop or elem")
        

        if self.pocket:
            self.pocket_embed = nn.Embedding(21, self.in_node_nf)
        #loss function here need to be disentangled

        #dequantizer
        
        assert cfg.loss_type in {'vlb', 'l2'}
        self.loss_type = cfg.loss_type
        self.include_charges = cfg.include_charges
        if cfg.noise_schedule == 'learned':
            assert cfg.loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert cfg.parametrization == 'eps'

        if cfg.noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(**cfg.pre_noise)
        
        if cfg.dataset == "qm9":
            self.remove_h = True
        else:
            self.remove_h = False
        
        self.hcontinous = cfg.hcontinous


        #self.dataset_info = get_dataset_info("qm9",remove_h=self.remove_h)
        # self.in_node_nf =



        if cfg.dynamics.condition_time:
            self.in_node_nf += 1
        else:
            print('Warning: dynamics model is _not_ conditioned on time.')
            self.in_node_nf =  self.in_node_nf
        # The network that will predict the denoising.
        self.dynamics = EGNN_dynamics_QM9(**cfg.dynamics)
        #print("dynamics done")
        self.n_dims = cfg.dynamics.n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = cfg.timesteps
        self.parametrization = cfg.parametrization

        self.norm_values = cfg.norm_values
        self.norm_biases = cfg.norm_biases
        #print("we here")
        self.register_buffer('buffer', torch.zeros(1))

        if cfg.noise_schedule != 'learned':
            self.check_issues_norm_values()
        
        self.data_augmentation = cfg.data_augmentation 
        

        histogram = yaml.load(open(cfg.analyze), Loader=Loader)
        self.nodes_dist = DistributionNodes(histogram=histogram)

        # print("here")
    
    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')
    
    
    def phi(self, x, t, node_mask, edge_mask, context, mol_shape=None):
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context, mol_shape)

        return net_out
    
    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h = (h - self.norm_biases[1]) / self.norm_values[1] * node_mask

        return x, h, delta_log_px

    def unnormalize(self, x, h, node_mask):
        x = x * self.norm_values[0]
        h = h * self.norm_values[1] + self.norm_biases[1]
        h = h * node_mask

        return x, h

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error

    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)./ which will not be used in dequantizer data"""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))
    
    def log_constants_p_h_given_z0(self, h, node_mask):
        """Computes p(h|z0)./ which will be only used for blur generation"""
        batch_size = h.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = n_nodes * self.in_node_nf

        zeros = torch.zeros((h.size(0), 1), device=h.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_h = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_h - 0.5 * np.log(2 * np.pi))
    
    

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, :self.n_dims]

        h = z0[:, :, self.n_dims:]
        x, h = self.unnormalize(x, h, node_mask)
        return x, h
    
    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False, mol_shape=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt[:, :mol_shape])

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context, mol_shape)
        eps_t = eps_t[:, :mol_shape]
        zt = zt[:, :mol_shape]
        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask[:, :mol_shape])
        #assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask[:, :mol_shape])
        eps_t[:, :, :self.n_dims] = remove_mean_with_mask(eps_t[:, :, :self.n_dims], node_mask[:, :mol_shape])
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask[:, :mol_shape], fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask[:, :mol_shape]),
             zs[:, :, self.n_dims:]], dim=2
        )
        return torch.cat([zs, zt[:, mol_shape:]], dim=1)
    
    @torch.no_grad()
    def sample(self, num_samples, device, context=None, pocket_cond=None):
        sample_n = self.nodes_dist.sample(num_samples)
        node_mask = torch.zeros([num_samples, max(sample_n), 1])
        if context is not None:
            context = torch.zeros([num_samples, max(sample_n), 1]).to(device) + context#only for global context
        edge_mask = torch.zeros(num_samples, max(sample_n), max(sample_n))

        for i in range(len(sample_n)):
            node_mask[i, :sample_n[i]] = 1
            edge_mask[i, :sample_n[i], :sample_n[i]] = 1 - torch.eye(sample_n[i])
        node_mask = node_mask.to(device).bool()
        edge_mask = edge_mask.to(device).bool()

        z = self.sample_combined_position_feature_noise(num_samples, max(sample_n), node_mask).to(device)
        if pocket_cond is not None:
            pocket_feat = self.pocket_embed(pocket_cond[0].type_as(z).long())
            pocket_pos = pocket_cond[1].type_as(z)
            pocket_node_mask = pocket_cond[2].type_as(z).bool()
            pocket_edge_mask = pocket_cond[3].type_as(z).bool()

            node_mask_concat = torch.cat([node_mask, pocket_node_mask], dim=1)
            edge_mask_concat = torch.zeros(num_samples, max(sample_n) + pocket_pos.size(1), max(sample_n) + pocket_pos.size(1)).type_as(edge_mask)
            edge_mask_concat[:, :max(sample_n), :max(sample_n)] = edge_mask
            edge_mask_concat[:, max(sample_n):, max(sample_n):] = pocket_edge_mask

        mol_shape = max(sample_n)

        for s in reversed(range(0, self.T)):
            s_array = torch.full((num_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array, t_array, torch.cat([z, torch.cat([pocket_pos, pocket_feat], dim=-1)], dim=1), node_mask_concat, edge_mask_concat, context, mol_shape=mol_shape)
        
        z = z[:, :mol_shape]
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)
        x = [x[i, :sample_n[i]].cpu() for i in range(len(sample_n))]
        h = [h[i, :sample_n[i]].cpu() for i in range(len(sample_n))]
        if context is not None:
            context_out = [context[i, :sample_n[i]].cpu() for i in range(len(sample_n))]
            results = [{'x': x[i], 'h': h[i], 'context': context_out[i]} for i in range(len(sample_n))]
        else:
            results = [{'x': x[i], 'h': h[i]} for i in range(len(sample_n))]
        return results
    
    def sample_batches(self, batch_size, num_batches, device, context_range=None, protein_data_all=None):
        """Samples batches of graphs."""
        if protein_data_all is not None:
            print("sample batches with protein data")
            batches_protein_data = []
            for protein_data in protein_data_all:
                protein_feat = protein_data["residue_type"]
                protein_pos = np.array(protein_data["coord"])
                protein_feat = torch.tensor([RESIDUE_LIST.index(x) + 1 for x in protein_feat])
                protein_pos = torch.tensor(protein_pos)
                batches_protein_data.append((protein_feat, protein_pos))
            max_protein_len = max([x[0].shape[0] for x in batches_protein_data])
            protein_feat_tensor = torch.zeros([len(batches_protein_data), max_protein_len], dtype=torch.long)
            protein_pos_tensor = torch.zeros([len(batches_protein_data), max_protein_len, 3])
            protein_feat_mask = torch.zeros([protein_feat_tensor.shape[0],protein_feat_tensor.shape[1],1]).bool()
            protein_edge_mask = torch.zeros([protein_feat_tensor.shape[0],protein_feat_tensor.shape[1],protein_feat_tensor.shape[1]]).bool()
            for i, (protein_feat, protein_pos) in enumerate(batches_protein_data):
                protein_feat_tensor[i, :protein_feat.shape[0]] = protein_feat
                protein_pos_tensor[i, :protein_pos.shape[0]] = protein_pos
                protein_feat_mask[i, :protein_feat.shape[0], 0] = 1
                protein_edge_mask[i, :protein_feat.shape[0], :protein_feat.shape[0]] = 1 - torch.eye(protein_feat.shape[0])
            protein_cond_all = [protein_feat_tensor, protein_pos_tensor, protein_feat_mask, protein_edge_mask]
        else:
            protein_cond_all = None

        results = []
        test_names = []
        for i in tqdm.tqdm(range(num_batches)):
            split_idx = [i * batch_size, (i + 1) * batch_size]
            if protein_cond_all is not None:
                protein_cond = [x[split_idx[0] %len(protein_cond_all[0]): (split_idx[1]-1)%len(protein_cond_all[0])+1] for x in protein_cond_all]
                pocket_name = [protein_data_all[i]['pocket_name'] + '/' + protein_data_all[i]['ligand_name'] for i in range(split_idx[0] %len(protein_cond_all), split_idx[1]%len(protein_cond_all))]
                results.extend(self.sample(batch_size, device, context=None, pocket_cond=protein_cond))
                test_names.extend(pocket_name)
            elif context_range is not None:
                results.extend(self.sample(batch_size, device, context=context_range[i%len(context_range)], pocket_cond=None))
            else:
                results.extend(self.sample(batch_size, device, context=None, pocket_cond=None))
            
        return results, test_names

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps


    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z



    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        if self.node_coarse_type == 'prop':
            int_nf = 5
            cont_nf = 3
        elif self.node_coarse_type == 'elem':
            int_nf = 3
            cont_nf = 0

        z_h_int = z_t[:, :, self.n_dims: self.n_dims + int_nf]

        # Take only part over x.
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        #the part over continues h
        eps_h = eps[:, :, self.n_dims + int_nf: self.n_dims + int_nf + cont_nf]
        net_h = net_out[:, :, :self.n_dims + int_nf: self.n_dims + int_nf + cont_nf]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)
        log_p_h_given_z_without_constants = -0.5 * self.compute_error(net_h, gamma_0, eps_h)

        
        # Compute delta indicator masks.
        h_integer = torch.round(h[:, :, :int_nf] * self.norm_values[2] + self.norm_biases[2]).long()#h all interger features
        

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = torch.log(
        cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
        - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
        + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)
        
        '''
        #code for catgorical h---results: loss do not converge
        h_cat = h[:, :, :] * self.norm_values[2] + self.norm_biases[2] - 1

        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((h_cat + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_cat - 0.5) / sigma_0_int)
            + epsilon)
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * (h_cat + 1) * node_mask)
        '''
        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z_without_constants + log_ph_integer
        #log_p_xh_given_z = log_p_x_given_z_without_constants + log_ph_cat

        return log_p_xh_given_z
        #log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_out, gamma_0, eps)
        #return log_p_x_given_z_without_constants


    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always, mol_shape=None):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0
        
        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        if mol_shape is None:
            mol_shape = x.size(1)
        x, x_fix = x[:, :mol_shape], x[:, mol_shape:]
        h, h_fix = h[:, :mol_shape], h[:, mol_shape:]
        node_mask, node_mask_fix = node_mask[:, :mol_shape], node_mask[:, mol_shape:]
        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        #doing dequantization: using uniform dequantization
        # h, log_qh_x = dequantizer(h,node_mask,edge_mask,x)


        xh = torch.cat([x, h], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        # print("xh size",xh.size())
        # print("eps size",eps.size())
        # print("in_node_f",self.in_node_nf)

        assert_mean_zero_with_mask(x, node_mask)
        z_t = alpha_t * xh + sigma_t * eps
        
        xh_fix = torch.cat([x_fix, h_fix], dim=2)
        assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        z_t = torch.cat([z_t, xh_fix], dim=1)
        node_mask = torch.cat([node_mask, node_mask_fix], dim=1)


        # Neural net prediction.
        net_out = self.phi(z_t, t, node_mask, edge_mask, context, mol_shape=mol_shape)

        # Compute the error.
        net_out = net_out[:, :mol_shape]
        error = self.compute_error(net_out, gamma_t, eps)

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask[:, :mol_shape])
        neg_log_constants += -self.log_constants_p_h_given_z0(h, node_mask[:, :mol_shape])

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask[:, :mol_shape])

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask[:, :mol_shape])
            z_0 = alpha_0 * xh + sigma_0 * eps_0
            z_0 = torch.cat([z_0, xh_fix], dim=1)

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context, mol_shape=mol_shape)
            net_out = net_out[:, :mol_shape]
            node_mask = node_mask[:, :mol_shape]

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0[:, :mol_shape], gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t[:, :mol_shape], gamma_t, eps, net_out, node_mask[:, :mol_shape])

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}

    def nll(self, x, h, node_mask=None, edge_mask=None, context=None, mol_shape=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)
        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False, mol_shape=mol_shape)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True, mol_shape=mol_shape)

        neg_log_pxh = loss

        # Correct for normalization on x.
        # print("neg size",neg_log_pxh.size() )
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh
    
    def forward(self, batch):
        x = batch['positions']
        mol_shape = None
        if self.pocket:
            mol_shape = x.shape[1]
            x = torch.cat([x, batch["protein_pos"]], dim=1)

        
        node_mask = batch['atom_mask']
        if self.pocket:
            node_mask = torch.cat([node_mask, batch["protein_feat_mask"]], dim=1)
            # print(batch['atom_mask'])
        edge_mask = batch['edge_mask']
        if self.pocket:
            edge_mask_shape = edge_mask.shape[1] + batch["protein_edge_mask"].shape[1]
            edge_mask_concat = torch.zeros(edge_mask.shape[0], edge_mask_shape, edge_mask_shape).type_as(edge_mask)
            edge_mask_concat[:, :mol_shape, :mol_shape] = edge_mask
            edge_mask_concat[:, mol_shape:, mol_shape:] = batch["protein_edge_mask"]
            edge_mask = edge_mask_concat
       
        h = batch["node_feature"]
        if self.pocket:
            protein_feat = self.pocket_embed(batch["protein_feat"])
            h = torch.cat([h, protein_feat], dim=1)

        x = remove_mean_with_mask(x,node_mask, fix_size=mol_shape)
        #assert_mean_zero_with_mask(x, node_mask)

        
        #global conditioning here
        if self.cfg.dynamics.context_node_nf > 0:
            context = batch['context']
        else:
            context = None

        
        bs, n_nodes, n_dims = x.size()
        #loss function here, and we fix the training loop.
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
        assert_correctly_masked(x, node_mask)
        neg_log_pxh = self.nll(x, h, node_mask, edge_mask, context=context, mol_shape=mol_shape)
        
        
        nll = neg_log_pxh

        # Average over batch.
        nll = nll.mean(0)

    

        return {"loss":nll}
    
    def _gather_result(self, result):
        # collect steps
        result = {
            key: torch.cat([x[key] for x in result])
            if len(result[0][key].shape) > 0
            else torch.tensor([x[key] for x in result]).to(result[0][key])
            for key in result[0].keys()
        }
        # collect machines
        result = {
            key: torch.cat(list(self.all_gather(result[key])))
            for key in result.keys()
        }
        return result
    
    def _compute_metrics(self,result):
        #a function used for get on epoch end metrics.
        out_loss = result['loss']
        mean_loss = out_loss.mean()
        return {'loss': mean_loss}
    
    def training_step(self, batch, batch_idx):
        re = self.forward(batch)
        self.log("train_loss",re["loss"],on_epoch=True,prog_bar=True)
        return re["loss"]
    
    def validation_step(self, batch, batch_idx):
        re = self.forward(batch)
        return re
    
    def test_step(self, batch, batch_idx):
        re = self.forward(batch)
        return re

    def validation_epoch_end(self, result):
        #return super().validation_epoch_end(outputs)
        metrics = self._compute_metrics(self._gather_result(result))
        self.log(
            "val_loss",
            metrics["loss"],
            on_epoch=True,
            prog_bar=True,
        )
    
    def test_epoch_end(self, result):
        metrics = self._compute_metrics(self._gather_result(result))
        if self.global_rank == 0:
            self.log("test/ppl", metrics["loss"], on_epoch=True)
    
    #for learning rate scheduler

    def _set_num_training_steps(self, scheduler_cfg):
        if "num_training_steps" in scheduler_cfg:
            scheduler_cfg = dict(scheduler_cfg)
            if self.global_rank == 0:
                logger.info("Computing number of training steps...")
                num_training_steps = [self.num_training_steps()]
            else:
                num_training_steps = [0]
            torch.distributed.broadcast_object_list(
                num_training_steps,
                0,
                group=torch.distributed.group.WORLD,
            )
            scheduler_cfg["num_training_steps"] = num_training_steps[0]
            logger.info(
                f"Training steps: {scheduler_cfg['num_training_steps']}"
            )
        return scheduler_cfg


    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.num_training_batches != float("inf"):
            dataset_size = self.trainer.num_training_batches
        else:
            dataset_size = len(
                self.trainer._data_connector._train_dataloader_source.dataloader()
            )

        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches > 0
        ):
            dataset_size = min(dataset_size, self.trainer.limit_train_batches)
        else:
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)

        accelerator_connector = self.trainer._accelerator_connector
        if accelerator_connector.use_ddp2 or accelerator_connector.use_dp:
            effective_gpus = 1
        else:
            effective_gpus = self.trainer.devices
            if effective_gpus < 0:
                effective_gpus = torch.cuda.device_count()

        effective_devices = effective_gpus * self.trainer.num_nodes
        effective_batch_size = (
            self.trainer.accumulate_grad_batches * effective_devices
        )
        max_estimated_steps = (
            math.ceil(dataset_size // effective_batch_size)
            * self.trainer.max_epochs
        )
        logger.info(
            f"{max_estimated_steps} = {dataset_size} // "
            f"({effective_gpus} * {self.trainer.num_nodes} * "
            f"{self.trainer.accumulate_grad_batches}) "
            f"* {self.trainer.max_epochs}"
        )

        max_estimated_steps = (
            min(max_estimated_steps, self.trainer.max_steps)
            if self.trainer.max_steps and self.trainer.max_steps > 0
            else max_estimated_steps
        )
        return max_estimated_steps

    def configure_optimizers(self):
        #use the cfg to represent the nested structure.
        optimizer = instantiate(self.cfg.optim, self.parameters())

        scheduler = instantiate(
            self._set_num_training_steps(self.cfg.scheduler), optimizer
        )
        #scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    # def configure_optimizers(self):
    #     optimizer = instantiate(self.cfg.optim,self.parameters())
    #     return super().configure_optimizers()
