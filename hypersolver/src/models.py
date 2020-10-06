import torch ; import torch.nn as nn
import torchdyn; from torchdyn.models import *;

def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]  
    return trJ

class CNF(nn.Module):
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace;
        self.noise_dist, self.noise = noise_dist, None
        if self.trace_estimator in REQUIRES_NOISE:
            assert self.noise_dist is not None, 'This type of trace estimator requires specification of a noise distribution'
            
    def forward(self, x):   
        with torch.set_grad_enabled(True):
            x_in = torch.autograd.Variable(x[:,1:], requires_grad=True).to(x) # first dimension reserved to divergence propagation          
            # the neural network will handle the data-dynamics here
            x_out = self.net(x_in)
                
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return torch.cat([-trJ[:, None], x_out], 1) + 0*x # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph

# special hypersolver version for `HyperHeun`. Can be adapted to the template with 
# the appropriate tableau

class HyperHeun(nn.Module):
    def __init__(self, f, solvnet):
        super().__init__()  
        self.m = solvnet
        self.norm = 1e-3
        self.f = f
        self.controlled = True
        
    def forward(self, ds, dx, dx_, x, x0):
        ds = ds*torch.ones(x.shape[0],1).to(x)
        if self.controlled:
            xout = torch.cat([x,dx,dx_,x0,ds],1)
        else:
            xout = torch.cat([x,dx,dx_,ds],1)
        xout = self.m(xout)       
        return xout                   
                   
    def trajectory(self, x0, s_span, nn=1):
        traj = []
        ds = s_span[1] - s_span[0]
        x = x0
        for i, s in enumerate(s_span):            
            dx = self.f(s, x).detach()
            dx_ = self.f(s+ds, x + ds*dx).detach()
            traj.append(x[None])
            x = x + .5*ds*(dx + dx_) + nn*(ds**3)*self(ds, dx, dx_, x, x0) 
        return torch.cat(traj)