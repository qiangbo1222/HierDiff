from typing import Dict, Any
import torch

class VLBloss:
    def __init__(self, cfg: Dict[str,Any]):
        self.cfg = cfg
        
    # def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
    #     """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

    #     # This part is about whether to include loss term 0 always.
    #     if t0_always:
    #         # loss_term_0 will be computed separately.
    #         # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
    #         lowest_t = 1
    #     else:
    #         # estimator = loss_t,           where t ~ U({0, ..., T})
    #         lowest_t = 0

    #     # Sample a timestep t.
    #     t_int = torch.randint(
    #         lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
    #     s_int = t_int - 1
    #     t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

    #     # Normalize t to [0, 1]. Note that the negative
    #     # step of s will never be used, since then p(x | z0) is computed.
    #     s = s_int / self.T
    #     t = t_int / self.T

    #     # Compute gamma_s and gamma_t via the network.
    #     gamma_s = self.inflate_batch_array(self.gamma(s), x)
    #     gamma_t = self.inflate_batch_array(self.gamma(t), x)

    #     # Compute alpha_t and sigma_t from gamma.
    #     alpha_t = self.alpha(gamma_t, x)
    #     sigma_t = self.sigma(gamma_t, x)

    #     # Sample zt ~ Normal(alpha_t x, sigma_t)
    #     eps = self.sample_combined_position_feature_noise(
    #         n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

    #     # Concatenate x, h[integer] and h[categorical].
    #     xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
    #     # Sample z_t given x, h for timestep t, from q(z_t | x, h)
    #     z_t = alpha_t * xh + sigma_t * eps

    #     assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)

    #     # Neural net prediction.
    #     net_out = self.phi(z_t, t, node_mask, edge_mask, context)

    #     # Compute the error.
    #     error = self.compute_error(net_out, gamma_t, eps)

    #     if self.training and self.loss_type == 'l2':
    #         SNR_weight = torch.ones_like(error)
    #     else:
    #         # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
    #         SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
    #     assert error.size() == SNR_weight.size()
    #     loss_t_larger_than_zero = 0.5 * SNR_weight * error

    #     # The _constants_ depending on sigma_0 from the
    #     # cross entropy term E_q(z0 | x) [log p(x | z0)].
    #     neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

    #     # Reset constants during training with l2 loss.
    #     if self.training and self.loss_type == 'l2':
    #         neg_log_constants = torch.zeros_like(neg_log_constants)

    #     # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
    #     kl_prior = self.kl_prior(xh, node_mask)

    #     # Combining the terms
    #     if t0_always:
    #         loss_t = loss_t_larger_than_zero
    #         num_terms = self.T  # Since t=0 is not included here.
    #         estimator_loss_terms = num_terms * loss_t

    #         # Compute noise values for t = 0.
    #         t_zeros = torch.zeros_like(s)
    #         gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
    #         alpha_0 = self.alpha(gamma_0, x)
    #         sigma_0 = self.sigma(gamma_0, x)

    #         # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
    #         eps_0 = self.sample_combined_position_feature_noise(
    #             n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
    #         z_0 = alpha_0 * xh + sigma_0 * eps_0

    #         net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

    #         loss_term_0 = -self.log_pxh_given_z0_without_constants(
    #             x, h, z_0, gamma_0, eps_0, net_out, node_mask)

    #         assert kl_prior.size() == estimator_loss_terms.size()
    #         assert kl_prior.size() == neg_log_constants.size()
    #         assert kl_prior.size() == loss_term_0.size()

    #         loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

    #     else:
    #         # Computes the L_0 term (even if gamma_t is not actually gamma_0)
    #         # and this will later be selected via masking.
    #         loss_term_0 = -self.log_pxh_given_z0_without_constants(
    #             x, h, z_t, gamma_t, eps, net_out, node_mask)

    #         t_is_not_zero = 1 - t_is_zero

    #         loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

    #         # Only upweigh estimator if using the vlb objective.
    #         if self.training and self.loss_type == 'l2':
    #             estimator_loss_terms = loss_t
    #         else:
    #             num_terms = self.T + 1  # Includes t = 0.
    #             estimator_loss_terms = num_terms * loss_t

    #         assert kl_prior.size() == estimator_loss_terms.size()
    #         assert kl_prior.size() == neg_log_constants.size()

    #         loss = kl_prior + estimator_loss_terms + neg_log_constants

    #     assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

    #     return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
    #                   'error': error.squeeze()}

   