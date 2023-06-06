# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods. based on songyang's model"""

import torch
import numpy as np
from scipy import integrate
from models import utils as mutils


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""
  #dequantizer, model, x, h, node_mask, edge_mask, context
  def div_fn(xh, t, eps,node_mask):
    with torch.enable_grad():
      xh.requires_grad_(True)
      fn_eps = torch.sum(fn(xh, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, xh)[0]
    xh.detach()
    #TODO: check whether there is some bug
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(xh.shape))))

  return div_fn


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

  def drift_fn(model, x, t,node_mask,edge_mask,context):
    """The drift function of the reverse-time SDE."""
    #our score model is a sde
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t,node_mask,edge_mask,context)[0]

  def div_fn(model, xh, t, noise,node_mask, edge_mask, context):
    return get_div_fn(lambda xx, tt,nn,ee,cc: drift_fn(model, xx, tt,nn,ee,cc))(xh, t, noise,node_mask, edge_mask, context)

  def likelihood_fn(model, x, h, dequantizer, prior, nodes_dist, node_mask, edge_mask):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A score model.
      data: A PyTorch tensor. == xh

    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    bs, n_nodes, n_dims = x.size()

    h, log_qh_x = dequantizer(h, node_mask, edge_mask, x)

    h = torch.cat([h['categorical'], h['integer']], dim=2)

    data = torch.cat([x, h], dim=2)


    with torch.no_grad():
      shape = data.shape
      if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(data)
      elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

      def ode_func(t, x):
        sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))
        logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
        return np.concatenate([drift, logp_grad], axis=0)

      init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      zp = solution.y[:, -1]
      z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
      """TODO check here"""
      delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
      #modify here
      z_x, z_h = z[:, :, 0:n_dims].clone(), z[:, :, n_dims:].clone()

      mutils.assert_correctly_masked(z_x, node_mask)
      mutils.assert_correctly_masked(z_h, node_mask)

      N = node_mask.squeeze(2).sum(1).long()

      log_pN = nodes_dist.log_prob(N)

      log_pz = prior(z_x, z_h, node_mask)

      assert log_pz.size() == delta_logp.size()

      log_px = (log_pz + delta_logp - log_qh_x + log_pN).mean()  # Average over batch.

      nll = -log_px
      
      mean_abs_z = torch.mean(torch.abs(z)).item()

      return nll, mean_abs_z, nfe


      #prior_logp = sde.prior_logp(z)
    #   bpd = -(prior_logp + delta_logp) / np.log(2)
    #   N = np.prod(shape[1:])
    #   bpd = bpd / N
    #   # A hack to convert log-likelihoods to bits/dim
    #   offset = 7. - inverse_scaler(-1.)
    #   bpd = bpd + offset
    #   return bpd, z, nfe

  return likelihood_fn




def get_likelihood_offset_fn(sde, score_fn, eps=1e-5):
  """Create a function to compute the unbiased log-likelihood bound of a given data point.
  """

  def likelihood_offset_fn(xh, node_mask, edge_mask, context):
    """TODO check whether the zero of center gravity should be involved here
    """
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      prng: An array of random states. The list dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      data: A JAX array of shape [#devices, batch size, ...].

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      N: same as input
    """
    # rng, step_rng = jax.random.split(prng)
    shape = xh.shape

    eps_vec = torch.full((shape[0],), eps).to(xh.device)

    p_mean, p_std = sde.marginal_prob(xh, eps_vec)
    # rng, step_rng = jax.random.split(rng)
    z = torch.rand_like(xh)
    
    noisy_data = p_mean + p_std[:,None,None] * z

    score = score_fn(noisy_data, eps_vec,node_mask, edge_mask, context)
    #here the score has been normalized

    alpha, beta = sde.marginal_prob(torch.ones_like(xh), eps_vec)
    q_mean = noisy_data / alpha + (beta ** 2)[:,None,None] * (score / alpha)
    q_std = beta / torch.mean(alpha, dim=(1, 2))

    n_dim = np.prod(xh.shape[1:])
    p_entropy = n_dim / 2. * (np.log(2 * np.pi) + 2 * torch.log(p_std) + 1.)
    q_recon = n_dim / 2. * (np.log(2 * np.pi) + 2 * torch.log(q_std)) + 0.5 / (q_std ** 2)[:,None,None] *  torch.square(xh - q_mean).sum(dim=(1, 2))
    offset = q_recon - p_entropy
    return offset

  return likelihood_offset_fn
