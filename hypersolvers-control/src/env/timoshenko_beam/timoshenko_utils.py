from torchdyn.numerics import odeint, Euler, HyperEuler
import torch
import torch.nn as nn
from dicts import *
import matplotlib.pyplot as plt


class TimoshenkoBeam(nn.Module):
    """
    Linear discretization of a high-dimensional Timoshenko Beam
    x_dot = Ax + Bu
    """
    def __init__(self, A, B, u):
        super().__init__()
        self.A = A
        self.B = B
        self.u = u # controller (nn.Module)
        self.nfe = 0 # number of function evaluations
        self.cur_f = None # current function evaluation
        self.cur_u = None # current controller evaluation 
        
    def forward(self, t, x):
        self.nfe += 1
        self.cur_u = self.u(t, x)
        self.cur_f = torch.einsum('ij, ...j->...i', self.A, x) +\
                     torch.einsum('ij, ...j->...i', self.B, self.cur_u)
        return self.cur_f
    
    
# Error analysis
def smape(yhat, y):
    return torch.abs(yhat - y) / (torch.abs(yhat) + torch.abs(y)) / 2


def MAE_on_dim(x, nominal, dim=1):
    return torch.abs(x - nominal).mean(dim)


# Divide variables for plotting
def divide_timoshenko_variables(x):
    v_t = x[..., dofs_dict['v_t']]
    v_r = x[..., dofs_dict['v_r']]
    sig_t = x[..., dofs_dict['sig_t']]
    sig_r = x[..., dofs_dict['sig_r']]
    return v_t, v_r, sig_t, sig_r


def plot_initial_final_states(x, lim_low=-1, lim_high=1):
    v_t, v_r, sig_t, sig_r = divide_timoshenko_variables(x)
    x_v_t = x_dict['v_t'].cpu()
    x_v_r = x_dict['v_r'].cpu()
    x_sig_t = x_dict['sig_t'].cpu()
    x_sig_r = x_dict['sig_r'].cpu()

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Initial and final states of the Timoshenko Beam')
    axs[0,0].scatter(x_v_t, v_t[-1])
    axs[0,1].scatter(x_v_r, v_r[-1])
    axs[1,0].scatter(x_sig_t, sig_t[-1])
    axs[1,1].scatter(x_sig_r, sig_r[-1])
    axs[0,0].scatter(x_v_t, v_t[0])
    axs[0,1].scatter(x_v_r, v_r[0])
    axs[1,0].scatter(x_sig_t, sig_t[0])
    axs[1,1].scatter(x_sig_r, sig_r[0])
    axs[1,0].set_ylim([lim_low, lim_high])
    axs[1,1].set_ylim([lim_low, lim_high])

    
def plot_test(f, x0, t, hypersolver, device='cpu'):
    x_eu = odeint(f.to(device), x0.to(device), t, solver='euler')[1].detach().cpu()
    x_he = odeint(f.to(device), x0.to(device), t, solver=hypersolver)[1].detach().cpu()
    x_mp = odeint(f.to(device), x0.to(device), t, solver='midpoint')[1].detach().cpu()
    x_rk4 = odeint(f.to(device), x0.to(device), t, solver='rk4')[1].detach().cpu()
    xT = odeint(f.to(device), x0.to(device), t, solver='tsit5', atol=1e-5, rtol=1e-5)[1].detach().cpu()
    uT = f.u(0, xT.to(device)).repeat(xT.shape[0], 1, 1)

    v_t_nom, v_r_nom, sig_t_nom, sig_r_nom = divide_timoshenko_variables(xT[:,0,:])
    trajs = [x_eu, x_he, x_mp, x_rk4, xT]
    titles = ['Euler', 'HyperEuler', 'Midpoint', 'RK4', 'Tsit5']
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'black']
    nums = [0, 2, 4, 6, 8]

    fig, axs = plt.subplots(4, 9, figsize=(20, 8))
    for traj, title, color, i in zip(trajs, titles, colors, nums):
        v_t, v_r, sig_t, sig_r = divide_timoshenko_variables(traj[:,0,:])\

        # Trajectories
        axs[0,i].plot(t.cpu(), v_t, color=color);
        axs[1,i].plot(t.cpu(), v_r, color=color);
        axs[2,i].plot(t.cpu(), sig_t, color=color);
        axs[3,i].plot(t.cpu(), sig_r, color=color);

        # Error propagation
        if title == 'Tsit5': continue
        err_v_t = MAE_on_dim(v_t, v_t_nom)
        err_v_r = MAE_on_dim(v_r, v_r_nom)
        err_sig_t = MAE_on_dim(sig_t, sig_t_nom)
        err_sig_r = MAE_on_dim(sig_r, sig_r_nom)

        j = i+1
        axs[0,j].plot(t.cpu(), err_v_t, color=color);
        axs[1,j].plot(t.cpu(), err_v_r, color=color);
        axs[2,j].plot(t.cpu(), err_sig_t, color=color);
        axs[3,j].plot(t.cpu(), err_sig_r, color=color);
        axs[0,j].set_title(r'$v_t$')
        axs[1,j].set_title(r'$v_r$')
        axs[2,j].set_title(r'$\sigma_t$')
        axs[3,j].set_title(r'$\sigma_r$')