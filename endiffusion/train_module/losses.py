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

"""All functions related to loss computation and optimization.
"""

from turtle import shape
import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from models.utils import assert_mean_zero_with_mask
import train_module.sde_lib as sde_lib
from train_module.sde_lib import VESDE, VPSDE
from train_module.likelihood import get_div_fn,get_likelihood_offset_fn

def sample_combined_position_feature_noise(n_samples, n_nodes, node_mask, n_dims, in_node_nf):
      """
      Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
      """
      z_x = mutils.sample_center_gravity_zero_gaussian_with_mask(
          size=(n_samples, n_nodes, n_dims), device=node_mask.device,
          node_mask=node_mask)
      z_h = mutils.sample_gaussian_with_mask(
          size=(n_samples, n_nodes, in_node_nf), device=node_mask.device,
          node_mask=node_mask)
      z = torch.cat([z_x, z_h], dim=2)

      return z

def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn



def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, t,node_mask,edge_mask,context):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, t,node_mask,edge_mask,context)
    else:
      model.train()
      return model(x, t,node_mask,edge_mask,context)

  return model_fn


def get_score_fn(sde, model_fn, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  # model_fn = get_model_fn(model, train=train)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t,node_mask,edge_mask,context):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, t,node_mask,edge_mask,context)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, t,node_mask,edge_mask,context)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None]
      return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t,node_mask,edge_mask,context):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, t,node_mask,edge_mask,context)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn



#This is implemented according to the original continous ddpm/ SDE
def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True,
 importance__weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.
  equivalent to get_deq_fn in original code
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(dequantizer, model, x, h, node_mask, edge_mask, context):
    """Compute the loss function.

    Args:
      dequantizer: a dequantization model
      model: scoremodel / should be an 
      prior: the prior module
      nodes_dist: the histogram of node distribution
      x,h: A mini-batch of training data.
      node_mask/edge_mask/context:

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    bs, n_nodes, n_dims = x.size()

    h, log_qh_x = dequantizer(h, node_mask, edge_mask, x)

    h = torch.cat([h['categorical'], h['integer']], dim=2)

    xh = torch.cat([x, h], dim=2)

    mutils.assert_correctly_masked(xh,node_mask)


    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)

    #here we use the whole sde loss for training the model
    if likelihood_weighting and importance__weighting:
      t = sde.sample_importance_weighted_time_for_likelihood((xh.shape[0],), eps=eps)
      #Z = sde.likelihood_importance_cum_weight(sde.T, eps=eps)
    else:
      t = torch.rand(xh.shape[0], device=xh.device) * (sde.T - eps) + eps

    #z = torch.randn_like(xh)

    z = sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask,n_dims=x.size(2),in_node_nf=h.size(2))

    mean, std = sde.marginal_prob(xh, t)
    perturbed_data = mean + std[:, None, None] * z

    assert_mean_zero_with_mask(perturbed_data[:,:,:n_dims],node_mask)

    score = score_fn(perturbed_data, t, node_mask, edge_mask, context)
    #score have been normalized according to the zero center of gravity


    if likelihood_weighting:
      if importance__weighting:
        losses = torch.square(score * std[:, None, None] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
      else:
        g2 = sde.sde(torch.zeros_like(xh), t)[1] ** 2
        losses = torch.square(score + z / std[:, None, None])
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
    else:
      losses = torch.square(score * std[:, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    # if not likelihood_weighting:
    #   losses = torch.square(score * std[:, None, None] + z)
    #   losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    # else:
    #   g2 = sde.sde(torch.zeros_like(xh), t)[1] ** 2
    #   losses = torch.square(score + z / std[:, None, None])
    #   losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
    loss = torch.mean(losses)
    #here the weight of the dequantizer is not quite aware.
    losses = loss
    return loss

  return loss_fn



#This is implemented according to the mle training of SDE. Which is p_sde objective/ will add the score flow part to calculated  
def get_deq_loss_fn(sde_, reduce_mean=True, importance_weighting=True, eps=1e-5,eps_offset=False):
  """Create a loss function for training with arbirary SDEs.
  equivalent to get_deq_fn in original code
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def div_drift_fn(xh, t, eps, node_mask):
    div_fn = get_div_fn(lambda xh, t: sde_.sde(xh, t)[0])
    #div_fn(xh, t, eps,node_mask, edge_mask, context)
    #TODO: check whether here we need the node_mask stuffs.
    return div_fn(xh, t, eps, node_mask)

  def loss_fn(dequantizerfn, scorefn, x, h, node_mask, edge_mask, context):
    """Compute the loss function.

    Args:
      dequantizer: a dequantization model
      model: scoremodel / should be an 
      prior: the prior module
      nodes_dist: the histogram of node distribution
      x: A mini-batch of training data with 3d dim.
      h: A mini-batch of the 
      node_mask/edge_mask/context:

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    # sde_ = VESDE
    bs, n_nodes, n_dims = x.size()
    # print("x",x.size())
    # print("h",h)
    h, log_qh_x = dequantizerfn(x, h, node_mask, edge_mask)

    # h_dims = h.size(2) # get h_dims


    #h = torch.cat([h['categorical'], h['integer']], dim=2)

    xh = torch.cat([x, h], dim=2)

    mutils.assert_correctly_masked(xh,node_mask)

    mean, std = sde_.marginal_prob(xh, torch.ones((xh.size(0),),device=x.device) * sde_.T)

    z = sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, n_dims=x.size(2),in_node_nf=h.size(2))

    # print(mean.device, std.device, z.device)
    perturb_data = mean + std[:, None, None] * z
    p_x = perturb_data[:,:,:n_dims]
    p_h = perturb_data[:,:,n_dims:] #h part



    neg_prior_logp = -sde_.prior_logp_h(p_h,node_mask) -sde_.prior_logp_x(p_x,node_mask)

    #here we use the whole sde loss for training the model
    if importance_weighting:
      t = sde_.sample_importance_weighted_time_for_likelihood((xh.shape[0],), eps=eps).to(xh.device)
      Z = sde_.likelihood_importance_cum_weight(sde_.T, eps=eps)
    else:
      t = torch.rand(xh.shape[0], device=xh.device) * (sde_.T - eps) + eps

    z = sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, n_dims=x.size(2),in_node_nf=h.size(2))

    mean, std = sde_.marginal_prob(xh, t)

    perturbed_data = mean + std[:, None, None] * z
    score = scorefn(perturbed_data, t,node_mask, edge_mask, context)

    #how to make sure the output is zero-centered of gravity.
    if importance_weighting:
        losses = torch.square(score * std[:, None, None] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        grad_norm = torch.sum(torch.square(z).reshape((z.shape[0],-1)),dim=-1)
        #losses = (losses-grad_norm) * Z
        losses = losses * Z
    else:
        g2 = sde_.sde(torch.zeros_like(xh), t)[1] ** 2
        losses = torch.square(score + z / std[:, None, None])
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        grad_norm = torch.sum(torch.square(z).reshape((z.shape[0],-1)),dim=-1)
        grad_norm = grad_norm * g2 / (std ** 2)
        losses = losses-grad_norm
      
    z = sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask, n_dims=x.size(2),in_node_nf=h.size(2))
    t = torch.rand(xh.shape[0], device=xh.device) * (sde_.T - eps) + eps
    mean, std = sde_.marginal_prob(xh, t)
    noisy_data = mean + std[:, None, None] * z

    def rademacher(shape, dtype=torch.float32, device="GPU"):
      """Sample from Rademacher distribution."""
      rand = ((torch.rand(shape) < 0.5)) * 2 - 1
      return rand.to(dtype).to(device)

    epsilon = rademacher(xh.shape,device=xh.device)
    #TODO: check drift_div the node mask things
    drift_div = div_drift_fn(noisy_data, t, epsilon, node_mask)
    #xh, t, eps, node_mask, edge_mask, context



    losses = neg_prior_logp + 0.5 * (losses - 2 * drift_div) - log_qh_x

    if eps_offset:
      offset_fn = get_likelihood_offset_fn(sde_, scorefn, eps)
      # rng, step_rng = random.split(rng)
      losses = losses + offset_fn(xh,node_mask, edge_mask, context)   

    losses = losses 
    loss = torch.mean(losses)

    return loss, torch.mean(neg_prior_logp), torch.mean(-drift_div),  torch.mean(-grad_norm)

  
  return loss_fn



def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(dequantizer, model, x, h, node_mask, edge_mask, context):
    model_fn = mutils.get_model_fn(model, train=train)
    h, log_qh_x = dequantizer(h, node_mask, edge_mask, x)

    h = torch.cat([h['categorical'], h['integer']], dim=2)

    xh = torch.cat([x, h], dim=2)

    labels = torch.randint(0, vesde.N, (xh.shape[0],), device=xh.device)
    sigmas = smld_sigma_array.to(xh.device)[labels]
    noise = torch.randn_like(xh) * sigmas[:, None, None, None]
    perturbed_data = noise + xh
    score = model_fn(perturbed_data, labels, node_mask, edge_mask, context)
    target = -noise / (sigmas ** 2)[:, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(dequantizer, model, x, h, node_mask, edge_mask, context):
    model_fn = mutils.get_model_fn(model, train=train)

    h, log_qh_x = dequantizer(h, node_mask, edge_mask, x)

    h = torch.cat([h['categorical'], h['integer']], dim=2)

    xh = torch.cat([x, h], dim=2)

    labels = torch.randint(0, vpsde.N, (xh.shape[0],), device=xh.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(xh.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(xh.device)
    noise = torch.randn_like(xh)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None] * xh + \
                     sqrt_1m_alphas_cumprod[labels, None, None] * noise
    score = model_fn(perturbed_data, labels,node_mask, edge_mask, context)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
