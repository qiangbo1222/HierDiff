import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn.models import CNF
from torchdyn.nn import Augmenter, DataControl, DepthCat
from torchdyn.utils import *


def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
    frob = sum_except_batch(jvp.pow(2))
    trJ = sum_except_batch(jvp * noise)
    return trJ, frob

def exact_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    frob = 0.
    for i in range(x_in.shape[1]):
        trj_row = torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
        trj += trj_row
        frob += trj_row.pow(2)
    return trJ, frob

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)

def find_dims(flat_dims, node_dims):
    #print((-node_dims + (node_dims * node_dims + 4 * flat_dims)** 0.5) / 2)
    return int((-node_dims + (node_dims * node_dims + 4 * flat_dims)** 0.5) / 2)


class CNF_reg(nn.Module):
    def __init__(self, dynamics, node_dims, trace_estimator='hutch', regulization_weight=0, noise_dist=None):
        super(CNF_reg, self).__init__()
        self.dynamics = dynamics
        self.trace_estimator = trace_estimator
        self.regulization_weight = regulization_weight
        self.node_dims = node_dims
        self.noise_dist = noise_dist
    
    def reset_trace_estimator(self, trace_estimator):
        self.trace_estimator = trace_estimator
    
    def get_noise(self, tensor):
        if self.trace_estimator == 'hutch':
            self.noise_ = torch.randn_like(tensor)
        
    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)
            if self.regulization_weight > 0:
                x, trJ, reg_term = x[:, :-2], x[:, -2], x[:, -1]
            else:
                x, trJ = x[:, :-1], x[:, -1]

            node_num = find_dims(int(x.shape[1]), int(self.node_dims))
            x = x.view([x.size(0), node_num, -1]).contiguous()

            dx = self.dynamics(float(t), x)

        if self.regulization_weight > 0:
            dx2 = sum_except_batch(dx.pow(2))

        if self.trace_estimator == 'hutch':
            self.get_noise(dx)
            trJ, frob = hutch_trace(dx, x, self.noise_)

        elif self.trace_estimator == 'exact':
            trJ, frob = exact_trace(dx, x)

        else:
            raise ValueError('Unknown trace estimator')

        dx = dx.view([dx.size(0), -1])

        if self.regulization_weight > 0:
            reg_term = frob + dx2
            return torch.cat([dx, trJ.unsqueeze(-1), reg_term.unsqueeze(-1)], dim=-1)
        else:
            return torch.cat([dx, trJ.unsqueeze(-1)], dim=-1)

class FFJORD_dyn(nn.Module):
    def __init__(self, dynamics, node_dims, trace_method='hutch', ode_regularization=0, solver='dopri5', tol=1e-4):
        super(FFJORD_dyn, self).__init__()
        self.trace_method = trace_method
        self.ode_regularization = ode_regularization
        self.CNF = CNF_reg(dynamics, node_dims, trace_method, ode_regularization)
        self.tol = tol
        self.tol_test = tol * 1e-3
        self.solver = solver

    @property
    def atol(self):
        return self.tol if self.training else self.tol_test

    @property
    def rtol(self):
        return self.tol if self.training else self.tol_test

    def forward(self, x, node_mask=None, edge_mask=None, context=None):
        trJ = x.new_zeros(x.shape[0])
        reg_term = x.new_zeros(x.shape[0])
        x = torch.cat([x.view([x.shape[0], -1]), trJ.unsqueeze(-1), reg_term.unsqueeze(-1)], dim=-1)

        if node_mask is not None or edge_mask is not None or context is not None:
            self.CNF.dynamics.forward = self.CNF.dynamics.wrap_forward(
                node_mask, edge_mask, context)
        
        self.ODE = NeuralODE(self.CNF, solver=self.solver, 
                                sensitivity='adjoint', 
                                atol=self.atol, rtol=self.rtol,
                                solver_adjoint=self.solver,
                                atol_adjoint=self.atol, rtol_adjoint=self.rtol)
        x = self.ODE(x, torch.linspace(0, 1, 2))
        if self.ode_regularization > 0:
            x, trJ, reg_term = x[:, :-2], x[:, -2], x[:, -1]
            return x, trJ, reg_term
        else:
            x, trJ = x[:, :-1], x[:, -1]
            return x, trJ, torch.zeros_like(trJ)
    
    def reverse_fn(self, z, node_mask=None, edge_mask=None, context=None):
        if node_mask is not None or edge_mask is not None or context is not None:
            self.CNF.dynamics.forward = self.CNF.dynamics.wrap_forward(
                node_mask, edge_mask, context)
        with torch.no_grad():
            self.ODE = NeuralODE(self.CNF, solver=self.solver, 
                                sensitivity='adjoint', 
                                atol=self.atol, rtol=self.rtol,
                                solver_adjoint=self.solver,
                                atol_adjoint=self.atol, rtol_adjoint=self.rtol)
            z = self.ODE(z, torch.linspace(1, 0, 2))
        return z
        
    def reverse(self, z, node_mask=None, edge_mask=None, context=None):
        xt = self.reverse_fn(z, node_mask, edge_mask, context)
        return xt
