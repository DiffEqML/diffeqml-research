import torch
import torchdyn; from torchdyn.models import *;

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

def grid_apply(X, Y, F):
    z = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)
    f = F(z).detach().cpu() ; U, V = f[:,0].reshape(X.shape), f[:, 1].reshape(X.shape)
    return U, V

def plot(model, prior, target, step, save=True, show=False):
    model = model.cpu()
    sample = prior.sample(torch.Size([1 << 10])).cpu()
    traj = model[1].trajectory(Augmenter(1, 1)(sample), s_span=torch.linspace(0, 1, 100)).detach().cpu() ; 
    traj = traj[:,:,1:]
    xx = torch.linspace(-2, 2, 100) ; X, Y = torch.meshgrid(xx, xx) 
    U, V = grid_apply(X, Y, model[1].defunc.m.net)
    #
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(sample[:,0], sample[:,1], s=10, alpha=0.3, c='black')
    for i in range(1<<10):
        ax[0].plot(traj[:,i,0], traj[:,i,1], alpha=0.2, c='olive')
    ax[0].scatter(target[:,0].cpu(), target[:,1].cpu(), s=4, alpha=.8, c='orange')
    ax[0].scatter(traj[-1,:,0], traj[-1,:,1], s=4, alpha=.8, c='blue')
    ax[1].contourf(X, Y, torch.sqrt(U**2+V**2), 100, cmap='inferno')
    ax[1].streamplot(xx.numpy(), xx.numpy(), U.T, V.T, color='w', density=2)
    ax[0].set_xlim([-2, 2]) ; ax[0].set_ylim([-2, 2]) ; ax[1].set_xlim([-2, 2]) ; ax[1].set_ylim([-2, 2])
    ax[0].set_title('trajectories / reconstructed density') ; ax[1].set_title('learned vector field')
    if save: plt.savefig('train_plots/ffjord_train_%.3d' % step, dpi=100)
    if not show: plt.close(fig)
        
def density_scatter(x, y, ax=None, bins=20, **kwargs):
    x, y = x.numpy(), y.numpy()
    data , x_e, y_e = np.histogram2d(x, y, bins = bins, density = True )
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x,y]).T , method="splinef2d", bounds_error=False)
    z[np.where(np.isnan(z))] = 0.0
    cmap = cm.inferno
    ax.scatter( x, y, c=z, **kwargs , s=1, cmap=cmap, rasterized=True)
    ax.set_facecolor(cmap(0.))    
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    return ax
        
def plot_ffjord(model, hsolver, prior, step, save=True, show=False):
    model = model.cpu() ; hsolver = hsolver.cpu()
    x = prior.sample(torch.Size([1 << 10]))
    logpx = prior.log_prob(x)[:,None].to(x)
    xtrJ = torch.cat([logpx, x], 1).cpu()
    s_span = torch.linspace(0, 1, 2)
    dopri_traj = model[1].trajectory(xtrJ, s_span).detach()
    solver_traj = hsolver.trajectory(xtrJ.detach(), s_span).detach()
    heun_traj = hsolver.trajectory(xtrJ.detach(), s_span, nn=0).detach()
    # plot
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4))
    density_scatter(dopri_traj[-1,:, 1], dopri_traj[-1,:,2], ax[0])
    density_scatter(solver_traj[-1,:,1], solver_traj[-1,:,2], ax[1])
    density_scatter(heun_traj[-1,:,1], heun_traj[-1,:,2], ax[2])
    ax[0].set_xlim([-2, 2]) ; ax[0].set_ylim([-2, 2]) ; ax[1].set_xlim([-2, 2]) ; ax[1].set_ylim([-2, 2]) ; ax[2].set_xlim([-2, 2]) ; ax[2].set_ylim([-2, 2])
    ax[0].set_title('dopri5') ; ax[1].set_title('HyperHeun') ; ax[2].set_title('Heun')
    if save: plt.savefig('train_plots/hsolver_ffjord_%.3d' % step, dpi=100)
    if not show: plt.close(fig)