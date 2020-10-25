import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch ; import torchdyn; from torchdyn.models import *; from torchdyn.datasets import *

def sample_annuli(n_samples=1<<10, device=torch.device('cpu')):
    X, y = ToyDataset().generate(n_samples, 'spheres', dim=2, noise=.05)
    return 2*X.to(device), y.long().to(device)

def plot_scatter(ax, X, y):
	colors = ['blue', 'orange']
	ax.scatter(X[:,0], X[:,1], c=[colors[int(yi)] for yi in y], alpha=0.2, s=10.)
    
def dec_bound(model, x):
    P = [p for p in model[-1].parameters()]
    w1, w2, b = P[0][0][0].cpu().detach(), P[0][0][1].cpu().detach(), P[1][0].cpu().detach().item()
    return (-w1*x - b + .5)/w2

def plot_traj(model,  device=torch.device("cpu")):
    x0, ys = sample_annuli(n_samples=200) ; s = torch.linspace(0, 1, 20)
    model = model.cpu() ; xS = model[0].trajectory(x0, s).detach() ; model = model.to(device)
    r = 1.05*torch.linspace(xS[:,:,-2].min(), xS[:,:,-2].max(), 2)
    pS = torch.cat([model[-1](xS[:,i,-2:].to(device))[None,:,:] for i in range(len(x0))])

    fig, ax = plt.subplots(1, 1, figsize=(5,5), sharex=True, sharey=True)
    for i in range(len(x0)):
        x, y, p = xS[:,i,-2].numpy(), xS[:,i,-1].numpy(), model[-1](xS[:,i,-2:].to(device)).view(-1).detach().cpu().numpy()
        points = np.array([x, y]).T.reshape(-1, 1, 2) ; segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(pS.min(), pS.max())
        lc = LineCollection(segments, cmap='inferno', norm=norm, alpha=.3)
        lc.set_array(p) ; lc.set_linewidth(2) ; line = ax.add_collection(lc)
    pS_ = model[-1](xS[-1,:,-2:].to(device)).view(-1).detach().cpu().numpy()
    ax.scatter(xS[-1,:,-2], xS[-1,:,-1], c='lime', edgecolor='none', s=30)
    ax.scatter(xS[0,:,-2], xS[0,:,-1], c='black', alpha=.5, s=30)
    ax.plot(r, dec_bound(model, r), '--k')
    ax.set_xlim(xS[:,:,-2].min(), xS[:,:,-2].max()) ; ax.set_ylim(xS[:,:,-1].min(), xS[:,:,-1].max())
    return model

def plot_vector_field(model, n_grid=200, n_points=512, device=torch.device("cpu")):
    S = torch.Tensor([0., .25, 0.5, .75, 1.]) ; x0, ys = sample_annuli(n_samples=n_points)
    model = model.cpu() ; xS = model[0].trajectory(x0, S).detach() ; model = model.to(device)
    xx, yy = torch.linspace(xS[:,:,-2].min(), xS[:,:,-2].max(), n_grid), torch.linspace(xS[:,:,-1].min(), xS[:,:,-1].max(), n_grid)
    X, Y = torch.meshgrid(xx, yy) ; z = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], 1).to(device) ;
    fig, ax = plt.subplots(1, len(S), figsize=(20, 4))
    skip=(slice(None,None,4),slice(None,None,4))
    for i, s in enumerate(S):
        F = model[0].defunc(s, z).detach().cpu() ; fx, fy = F[:, 0].reshape(n_grid, n_grid), F[:, 1].reshape(n_grid, n_grid)
        ax[i].contourf(X, Y, torch.sqrt(fx**2 + fy**2), cmap='inferno')
        ax[i].streamplot(X[skip].T.numpy(), Y[skip].T.numpy(), fx[skip].T.numpy(), fy[skip].T.numpy(), color='w')
        ax[i].scatter(xS[i,:,0], xS[i,:,1], color='lime', alpha=.5, s=20)
        ax[i].set_xlim(xS[:,:,-2].min(), xS[:,:,-2].max()) ; ax[i].set_ylim(xS[:,:,-1].min(), xS[:,:,-1].max())
    return model

def plot_decision_boundary(model, xlim=[-2, 2], ylim=[-2, 2],  n_grid=200, n_points=512, device=torch.device("cpu")):
    x0, _ = sample_annuli(n_samples=n_points)
    model = model.to(device)
    xx, yy = torch.linspace(xlim[0], xlim[1], n_grid), torch.linspace(ylim[0], ylim[1], n_grid)
    X, Y = torch.meshgrid(xx, yy) ; z = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], 1).to(device)
    D = model(z).cpu().detach().numpy().reshape(n_grid, n_grid)
    D[D > 1], D[D < 0] = 1, 0
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.contourf(X, Y, D, 100, cmap="inferno_r")
    ax.scatter(x0[:, 0], x0[:,1], color='lime', alpha=.5, s=20)
    ax.set_xlim(X.min(), X.max()) ; ax.set_ylim(Y.min(), Y.max())
    return model
    