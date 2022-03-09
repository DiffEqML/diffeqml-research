import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from torchdiffeq import odeint

def smape(yhat, y):
    '''Add the mean'''
    return torch.abs(yhat - y) / (torch.abs(yhat) + torch.abs(y)) / 2

def mape(yhat, y):
    '''Add the mean'''
    return torch.abs((yhat - y)/yhat)

def get_macs(net:nn.Module, iters=1):
    params = []
    for p in net.parameters(): params.append(p.shape)
    with torch.cuda.device(0):
        macs, _ = get_model_complexity_info(net, (iters, params[0][1]), as_strings=False)
    return int(macs)

def get_flops(net:nn.Module, iters=1):
    # Relationship between macs and flops:
    # https://github.com/sovrasov/flops-counter.pytorch/issues/16
    params = []
    for p in net.parameters(): params.append(p.shape)
    with torch.cuda.device(0):
        macs, _ = get_model_complexity_info(net, (iters, params[0][1]), as_strings=False)
    return int(2*macs)


def compute_residual(t, Δt, x, u, hs, method='dopri5'):
    """
    x: u_batch, batch, dim
    u: u_batch, batch, dim
    """
    t_span = torch.tensor([t, t + Δt]).to(x)
    sys.u.u0 = u
    f = sys._dynamics(t, x)
    if x.shape == u.shape:
        xfu = torch.cat([x, f, u], -1)
    else:
        xfu = torch.cat([x, f, u.repeat(1, x.shape[1], 1)], -1)

    g = hs(xfu) # computer residual adjustment
    zd = odeint(sys._dynamics, x, t_span, method=method)[-1] # compute solution with high-precision solver
    z  = odeint(sys._dynamics, x, t_span, method='euler')[-1] # computer euler solution
    R = (zd - z)/(Δt**2) # computer actual residual
    L = torch.norm(R - g, dim=-1, p=2)
    return L

def generate_mesh(dynamics, system, hs, u_lower=-10, u_upper=10, 
                  res=1, s_span=torch.linspace(0, 1, 5)):
    """Generate mesh of models, hypersolvers and controls
    for choosing training parameters more wisely
    
    Args:
        dynamics: e.g. matrix 'A' in a linear system
        system: controlled system (nn.Module)
        hs: hypersolver `g`_\omega
        u_lower: lower control bound
        u_upper: upper control bound
        res: resolution of the meshgrid
        s_span: integration steps
        
    Returns:
        models: mesh of the models
        hypersolvers: mesh of the hypersolvers
        mesh_controls: mesh of the controls
    """
    device = dynamics.device
    n_grid = int((u_upper - u_lower)/res + 1)
    u = torch.linspace(u_lower, u_upper, n_grid).to(device)
    u1, u2 = torch.meshgrid(u, u)
    mesh_controls = torch.cat((u1.reshape(-1, 1), u2.reshape(-1, 1)), 1)

    # Generate dynamics (we initialize with the constant controllers)
    models = []
    hypersolvers = []
#     for i in range(n_grid): # Bug?
    for i in range(mesh_controls.shape[0]):
        m = NeuralDE(system(
                    dynamics, mesh_controls[i]), 
                sensitivity='autograd', 
                s_span=s_span, 
                solver='dopri5').to(device) # Initialize with dopri5 so to get a better solution
        models.append(m)
        hypersolvers.append(HyperEuler(m.defunc, hs).to(device))
        
    return models, hypersolvers, mesh_controls


def calculate_hypersolver_loss(x0, model, hs, t_span=torch.linspace(0, 1, 5)):
    """Calculate loss of the hypersolver compared to the high precision solver
    
    Args:
        model: neuralDE model
        hs: hypersolver
        t_span: integration steps"""
    dt = t_span[1] - t_span[0]
    base_traj  = model.trajectory(x0.detach(), t_span.detach())
    residuals = hs.base_residuals(base_traj.detach(), t_span.detach()).detach()
    corrections = hs.hypersolver_residuals(base_traj.detach(), t_span.detach())
    # Maybe the loss should be adjusted?
#     loss = torch.norm(corrections - residuals.detach()).mean() * dt**2
    loss = torch.norm(corrections - residuals.detach(), p=2, dim=(1)).mean() * dt**2
    return loss