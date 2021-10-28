"Encoder and Decoder modules for NHA and ODE baselines"

import torch
from torch.distributions import Independent, Normal, Categorical
import torch.nn as nn
from src.odeint import jagged_fixed_odeint
from time import time
from copy import deepcopy


################ MODEL WRAPPERS ##################
class AugmentedNeuralODE(nn.Module):
    "Augmented Neural ODE with dynamics on zero-augmented states"
    def __init__(self, dec, solver, dims_to_augment=3):
        super().__init__()
        self.enc, self.dec = nn.Linear(1, 1), dec
        self.solver = solver
        self.dims_to_augment = dims_to_augment

    def forward(self, x_ivps, t_segments):
        t0 = time()
        z0_parallel = torch.cat([x_ivps, torch.zeros(x_ivps.shape[0], 1,
                                                     self.dims_to_augment).to(x_ivps[0])], -1).to(x_ivps[0])
        sol = jagged_fixed_odeint(self.dec, z0_parallel, t_segments, solver=self.solver)
        forward_t = time() - t0
        return sol, forward_t


class DCNeuralODE(nn.Module):
    "Augmented Neural ODE with augmentation done via data-control term"
    def __init__(self, enc, dec, solver, n_pts_enc):
        super().__init__()
        self.enc, self.dec = enc, dec
        self.solver = solver
        self.n_pts_enc = n_pts_enc

    def forward(self, x_ivps, x_feats, t_segments):
        t0 = time()
        q_parallel = self.encode(x_ivps, x_feats)
        z0_parallel = torch.cat([x_ivps, q_parallel[:, None]], -1).to(x_ivps[0])
        sol = jagged_fixed_odeint(self.dec, z0_parallel, t_segments, solver=self.solver)
        forward_t = time() - t0
        return sol, forward_t

    def encode(self, x_ivps, x_feats):
        return self.enc(x_feats)


class LatentODE(nn.Module):
    def __init__(self, enc, dec, solver, n_pts_enc):
        super().__init__()
        self.enc, self.dec, self.solver = enc, dec, solver
        self.n_pts_enc = n_pts_enc

    def forward(self, x_ivps, x_feats, t_segments):
        t0 = time()
        q_parallel = self.encode(x_ivps, x_feats)
        z0_parallel = torch.cat([x_ivps, q_parallel[:, None]], -1).to(x_ivps[0])
        sol = jagged_fixed_odeint(self.dec, z0_parallel, t_segments, solver=self.solver)
        forward_t = time() - t0
        return sol, forward_t

    def encode(self, x_ivps, x_feats):
        return self.enc(x_feats)


class NHA(nn.Module):
    def __init__(self, enc, dec, solver, n_pts_enc):
        super().__init__()
        self.enc, self.dec, self.solver = enc, dec, solver
        self.n_pts_enc = n_pts_enc

    def forward(self, x_ivps, x_feats, t_segments):
        t0 = time()
        q_parallel = self.encode(x_ivps, x_feats)
        z0_parallel = torch.cat([x_ivps, q_parallel], -1).to(x_ivps[0])
        sol = jagged_fixed_odeint(self.dec, z0_parallel, t_segments, solver=self.solver)
        forward_t = time() - t0
        return sol, forward_t

    def encode(self, x_ivps, x_feats, keep_one_hot=False):
        return self.enc(x_ivps, x_feats, keep_one_hot)


############ ENCODERS #####################
def one_hot(idx, n_classes=3):
    return torch.zeros(len(idx), n_classes).to(idx).scatter_(1, idx.unsqueeze(1), 1.)


class CDEFunc(nn.Module):
    def __init__(self, net, hidden_dim):
        super().__init__()
        self.f = net
        self.X = None
        self.hidden_dim = hidden_dim
    def forward(self, t, x):
        f = self.f(x).reshape(x.shape[0], x.shape[1], self.hidden_dim)
        return (f*self.X.derivative(t).unsqueeze(-1)).sum(-1)


class LatentEncoder(nn.Module):
    def __init__(self, net, n_modes=3):
        super().__init__()
        self.net = net
        self.sp = nn.Softplus()
        self.n_modes = n_modes

    def forward(self, x):
        x = self.net(x)
        mu, sigma = x[...,:self.n_modes], self.sp(x[...,self.n_modes:])
        dist = Independent(Normal(mu, sigma), reinterpreted_batch_ndims=0)
        return dist.rsample((1,))[0]


class NHADiscreteModeEncoder(nn.Module):
    def __init__(self, net, categorical_type, device):
        super().__init__()
        self.net = net
        self.categorical_type = categorical_type
        self.softmax = nn.Softmax(-1)
        self.sparsemax = Sparsemax(dim=-1, device=device)

    def forward(self, x_ivps, x_feats, keep_one_hot=False):
        y = self.net(x_feats)
        if self.categorical_type == "STE":
            y = self.softmax(y)
            dist = Categorical(y)
            sample = dist.sample((1,))
            sample = one_hot(sample[0], n_classes=y.shape[1])
            y = sample + y - y.detach()
        elif self.categorical_type == "sparsemax":
            y = self.sparsemax(y)
        else:
            y = self.softmax(y)
        if keep_one_hot: return y.unsqueeze(1)
        else: return y.unsqueeze(1)



############### DECODERS #################
class RegularDecoder(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, t, z):
        return self.f(z)


class AugmentedDecoder(nn.Module):
    def __init__(self, f, aug_dims=3):
        super().__init__()
        self.f, self.aug_dims = f, aug_dims

    def forward(self, t, z):
        dx = self.f(z)
        return torch.cat([dx, torch.zeros_like(z[..., -self.aug_dims:])], -1)


class ConditionedNHADecoder(nn.Module):
    def __init__(self, f, n_modes, config):
        super().__init__()

        self.nets = []
        for k in range(n_modes):
            if config.nonlinear: f_ = nn.Sequential(nn.Linear(1, 32),
                                                    nn.Tanh(),
                                                    nn.Linear(32, 2))
            else: f_ = nn.Sequential(nn.Linear(1, 2))
            self.nets.append(f_)
            self.add_module(f'vf_mode_{k}', f_)
            p1, p2 = f_[-1].weight, f_[-1].bias
            torch.nn.init.zeros_(p1)
            torch.nn.init.zeros_(p2)

        self.n_modes = n_modes
        self.state_dim = f_[-1].out_features


    def forward(self, t, z):
        x, modes = z[..., :self.state_dim], z[..., self.state_dim:]
        #print(x.shape)
        fx = torch.cat([f(x[...,:1]).unsqueeze(-1) for f in self.nets], -1)  # batch, dim, n_modes
        dx = (fx * modes.unsqueeze(1)).sum(-1)
        return torch.cat([dx, torch.zeros_like(modes)], -1)


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, device, dim=None):
        super().__init__()

        self.dim = -1 if dim is None else dim
        self.device = device

    def forward(self, input):
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=self.device, dtype=input.dtype).view(
            1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input