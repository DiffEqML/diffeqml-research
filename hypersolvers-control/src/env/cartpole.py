import torch
from torch import cos, sin, sign
from .template import ControlledSystemTemplate


class CartPoleGymVersion(ControlledSystemTemplate):
    '''Continuous version of the OpenAI Gym cartpole
    Inspired by: https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        
    def _dynamics(self, t, x_):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x_) # controller
        
        # States
        x   = x_[..., 0:1]
        dx  = x_[..., 1:2]
        θ   = x_[..., 2:3]
        dθ  = x_[..., 3:4]
        
        # Auxiliary variables
        cosθ, sinθ = cos(θ), sin(θ)
        temp = (u + self.polemass_length * dθ**2 * sinθ) / self.total_mass
        
        # Differential Equations
        ddθ = (self.gravity * sinθ - cosθ * temp) / \
                (self.length * (4.0/3.0 - self.masspole * cosθ**2 / self.total_mass))
        ddx = temp - self.polemass_length * ddθ * cosθ / self.total_mass
        self.cur_f = torch.cat([dx, ddx, dθ, ddθ], -1)
        return self.cur_f

    def render(self):
        raise NotImplementedError("TODO: add the rendering from OpenAI Gym")


class CartPole(ControlledSystemTemplate):
    """
    Realistic, continuous version of a cart and pole system. This version considers friction for the cart and the pole. 
    We do not consider the case in which the normal force can be negative: reasonably, the cart should not try to "jump off" the track. 
    This also allows us not needing to consider the previous step's sign.
    References: 
        - http://coneural.org/florian/papers/05_cart_pole.pdf
        - https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf
        - https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8
        - https://github.com/AadityaPatanjali/gym-cartpolemod
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.frictioncart = 0 # 5e-4
        self.frictionpole = 0 # 2e-6

    def _dynamics(self, t, x_):
        self.nfe += 1 # increment number of function evaluations
        u = self._evaluate_controller(t, x_) # controller
        
        # States
        x, dx, θ, dθ = self._divide_states(x_)
        
        # Auxiliary variables
        cosθ, sinθ = cos(θ), sin(θ)
        temp = (u + self.polemass_length * dθ**2 * sinθ) / self.total_mass 
        signed_μc = self.frictioncart * sign(dx)
        μp = self.frictionpole

        # Differential Equations
        nom_ddθ = self.gravity * sinθ - (μp * dθ) / (self.masspole * self.length) - \
                         cosθ * (temp + (self.masspole * self.length * dθ**2 * signed_μc * cosθ) / self.total_mass - signed_μc * self.gravity) # nominator ddθ
        den_ddθ = self.length * (4/3 - self.masspole * cosθ * (cosθ - signed_μc) / self.total_mass) # denominator ddθ
        ddθ = nom_ddθ / den_ddθ # angular acceleration of the pole
        nc = (self.masscart + self.masspole) * self.gravity - self.masspole * self.length * (ddθ * sinθ + dθ**2 * cosθ) # normal force cart
        ddx = temp + (- self.polemass_length * ddθ * cosθ - signed_μc * nc) / self.total_mass # acceleration of the track
        self.cur_f = torch.cat([dx, ddx, dθ, ddθ], -1)
        return self.cur_f

    def _divide_states(self, x_):
        x   = x_[..., 0:1]
        dx  = x_[..., 1:2]
        θ   = x_[..., 2:3]
        dθ  = x_[..., 3:4]
        return x, dx, θ, dθ

    def kinetic_energy(self, x_):
        x, dx, θ, dθ = self._divide_states(x_) 
        return 1/2 * (self.masscart + self.masspole) * dx**2 + self.masspole * dx * dθ * self.length * cos(θ) + 1/2 * self.masspole * self.length**2 * dθ**2

    def potential_energy(self, x_):
        x, _, θ, _ = self._divide_states(x_) 
        return self.masspole * self.gravity * self.length * cos(θ)

    def render(self):
        raise NotImplementedError("TODO: add the rendering from OpenAI Gym")