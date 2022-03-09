import torch
import torch.nn as nn

class IntegralCost(nn.Module):
    '''Integral cost function
    Args:
        x_star: torch.tensor, target position
        P: float, terminal cost weights
        Q: float, state weights
        R: float, controller regulator weights
    '''
    def __init__(self, x_star, P=0, Q=1, R=0):
        super().__init__()
        self.x_star = x_star
        self.P, self.Q, self.R, = P, Q, R
        
    def forward(self, x, u=torch.Tensor([0.])):
        cost = (self.P * (x[-1] - self.x_star)**2).sum(-1).mean()
        cost += (self.Q * (x - self.x_star)**2).sum(-1).mean()
        cost += (self.R * (u - 0)**2).sum(-1).mean()
        return cost

    
def circle_loss(z, a=1):
    """Make the system follow a circle with radius a"""
    x, y = z[...,:1], z[...,1:]
    loss = torch.abs(x**2 + y**2 - a)
    return loss.mean()

def circus_loss(z, a=1., k=2.1):
    """Make the system follow an elongated circus-like shape with
    curve a and length k"""
    x, y = z[...,:1], z[...,1:]
    
    a1 = torch.sqrt((x + a)**2 + y**2)
    a2 = torch.sqrt((x - a)**2 + y**2)
    return torch.abs(a1*a2 - k).mean()
