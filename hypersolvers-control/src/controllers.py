import torch
from torch import mm
from torch import nn
from warnings import warn
tanh = nn.Tanh() 

class BoxConstrainedController(nn.Module):
    """Simple controller  based on a Neural Network with
    bounded control inputs

    Args:
        in_dim: input dimension
        out_dim: output dimension
        hid_dim: hidden dimension
        zero_init: initialize last layer to zeros
    """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 h_dim=64, 
                 num_layers=2, 
                 zero_init=True,
                 input_scaling=None, 
                 output_scaling=None,
                 constrained=False):
        
        super().__init__()
        # Create Neural Network
        layers = []
        layers.append(nn.Linear(in_dim, h_dim))
        for i in range(num_layers):
            if i < num_layers-1:
                layers.append(nn.Softplus())
            else:
                # last layer has tanh as activation function
                # which acts as a regulator
                layers.append(nn.Tanh())
                break
            layers.append(nn.Linear(h_dim, h_dim))
        layers.append(nn.Linear(h_dim, out_dim))
        self.layers = nn.Sequential(*layers)
        
        # Initialize controller with zeros in the last layer
        if zero_init: self._init_zeros()
        self.zero_init = zero_init
        
        # Scaling
        if constrained is False and output_scaling is not None:
            warn("Output scaling has no effect without the `constrained` variable set to true")
        if input_scaling is None:
            input_scaling = torch.ones(in_dim)
        if output_scaling is None:
            # scaling[:, 0] -> min value
            # scaling[:, 1] -> max value
            output_scaling = torch.cat([-torch.ones(out_dim)[:,None],
                                         torch.ones(out_dim)[:,None]], -1)
        self.in_scaling = input_scaling
        self.out_scaling = output_scaling
        self.constrained = constrained
        
    def forward(self, t, x):
        x = self.layers(self.in_scaling.to(x)*x)
        if self.constrained:
            # we consider the constraints between -1 and 1
            # and then we rescale them
            x = tanh(x)
            # x = torch.clamp(x, -1, 1) # not working in some applications # TODO: fix the tanh to clamp
            x = self._rescale(x)
        return x
    
    def _rescale(self, x):
        s = self.out_scaling.to(x)
        return 0.5*(x + 1)*(s[...,1]-s[...,0]) + s[...,0]
    
    def _reset(self):
        '''Reinitialize layers'''
        for p in self.layers.children():
            if hasattr(p, 'reset_parameters'):
                p.reset_parameters()
        if self.zero_init: self._init_zeros()

    def _init_zeros(self):
        '''Reinitialize last layer with zeros'''
        for p in self.layers[-1].parameters(): 
            nn.init.zeros_(p)
            

class RandConstController(nn.Module):
    """Constant controller
    We can use this for residual propagation and MPC steps (forward propagation)"""
    def __init__(self, shape=(1,1), u_min=-1, u_max=1):
        super().__init__()
        self.u0 = torch.Tensor(*shape).uniform_(u_min, u_max)
        
    def forward(self, t, x):
        return self.u0


## Scipy solvers
from scipy.linalg import solve_continuous_are, solve_discrete_are # LQR

# LQR solvers in PyTorch. Original implementation:
# https://github.com/markwmuller/controlpy

def continuous_lqr(A, B, Q, R, device="cpu"):
    """Solve the continuous time LQR controller for a continuous time system.

    A and B are system matrices, describing the systems dynamics:
     dx/dt = A x + B u

    The controller minimizes the infinite horizon quadratic cost function:
     cost = integral (x.T*Q*x + u.T*R*u) dt

    where Q is a positive semidefinite matrix, and R is positive definite matrix.

    Returns K, X, eigVals:
    Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
    The optimal input is then computed as:
     input: u = -K*x
    """
    # Ref Bertsekas, p.151

    # First, try to solve the continuous Riccati equation
    # NOTE: PyTorch currently not supported by scipy, hence the transfer
    # Need to rework the solver in PyTorch to obtain a speedup
    X = torch.Tensor(solve_continuous_are(
        A.cpu().numpy(), B.cpu().numpy(), Q.cpu().numpy(), R.cpu().numpy())).to(device)

    # Compute the LQR gain
    K = mm(torch.inverse(R), (mm(B.T, X)))

    eigenvalues = torch.eig(A - B * K)

    return K, X, eigenvalues

def discrete_lqr(A, B, Q, R, device="cpu"):
    """Solve the discrete time LQR controller for a discrete time system.

    A and B are system matrices, describing the systems dynamics:
     x[k+1] = A x[k] + B u[k]

    The controller minimizes the infinite horizon quadratic cost function:
     cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]

    where Q is a positive semidefinite matrix, and R is positive definite matrix.

    Returns K, X, eigVals:
    Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
    The optimal input is then computed as:
     input: u = -K*x
    """
    #ref Bertsekas, p.151

    # First, try to solve the discrete Riccati equation
    # NOTE: PyTorch currently not supported by scipy, hence the transfer
    # Need to rework the solver in PyTorch to obtain a speedup
    X = torch.Tensor(solve_discrete_are(
        A.cpu().numpy(), B.cpu().numpy(), Q.cpu().numpy(), R.cpu().numpy())).to(device)

    # Compute the LQR gain
    K = mm(torch.inverse(mm(mm(B.T, X),B)+R), (mm(mm(B.T, X), A)))

    eigenvalues = torch.eig(A - B * K)

    return K, X, eigenvalues