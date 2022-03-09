import sys; sys.path.append(2*'../') # go n dirs back to import
from src import *

import torch
import torch.nn as nn
from math import pi as π
from torchdyn.datasets import *
from torchdyn.numerics import odeint, Euler, HyperEuler
import argparse
from box import Box

parser = argparse.ArgumentParser()
parser.add_argument('--number', default=0)
parser.add_argument('--experiment', default='MultistageHypersolver')
args = parser.parse_args()

config = Box.from_yaml(filename='config/{}.yaml'.format(args.experiment))

# Change device according to your configuration
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device = config.device

# Cost function declaration
x_star = torch.Tensor(config.cost.target).to(device)
def transform_to_weight_matrix(w):
    return torch.Tensor(w).to(device)
P, Q, R = transform_to_weight_matrix(config.cost.P), transform_to_weight_matrix(config.cost.Q), transform_to_weight_matrix(config.cost.R)
cost_func = IntegralCost(x_star, P=P, Q=Q, R=R)

# Time span
dt = config.horizon.dt
t0, tf = config.horizon.t0, config.horizon.tf # initial and final time for controlling the system
steps = int((tf - t0)/dt) + 1
t_span = torch.linspace(t0, tf, steps).to(device)

# Initial condition
# We make it vary according to "args.number"
x0 = torch.Tensor(config.horizon.init_position).to(device)
limits = 0.1 # 0.2 meters of difference
variation = (int(args.number) - 4.5)/4.5 * limits
x0[0] += variation

# Real system
const_u = RandConstController([1, 1], -1, 1).to(device) # dummy constant controller for simulation
real_system = CartPole(const_u, solver=config.dynamics.real.solver)
real_system.frictioncart = config.dynamics.real.friction_cart
real_system.frictionpole = config.dynamics.real.friction_pole

# Controller and optimizer
u = BoxConstrainedController(config.controller.input_dim, config.controller.output_dim, num_layers=config.controller.num_layers,
                             constrained=config.controller.constrained, output_scaling=torch.Tensor([config.controller.u_min, config.controller.u_max])).to(device)
opt = torch.optim.Adam(u.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay) # optimizer

# System to be corrected
system = CartPole(u, solver='euler', retain_u=True).to(device) 
system.frictioncart = config.dynamics.known.friction_cart
system.frictionpole = config.dynamics.known.friction_pole
solver = config.dynamics.known.solver

if solver=='multistagehs':
    # Instantiate hypernetwork
    class HyperNetwork(nn.Module):
        """Simple hypernetwork using as input the current state, vector field and controller"""
        def __init__(self, net):
            super().__init__()
            self.net = net
        
        def forward(self, t, x):
            # we access the global variable to avoid maximum recursion depth error
            # https://stackoverflow.com/questions/6809402/python-maximum-recursion-depth-exceeded-while-calling-a-python-object
            xfu = torch.cat([x, system.cur_f, system.cur_u], -1)
            return self.net(xfu)
    # Loading
    multistagehs = torch.load(config.dynamics.known.solver_path)
    inner_stage_net = multistagehs.inner_stage.net 
    outer_stage_net = multistagehs.outer_stage.net 
    inner_stage_hypernet = HyperNetwork(inner_stage_net)
    outer_stage_hypernet = HyperNetwork(outer_stage_net)
    multistagehs.inner_stage = inner_stage_hypernet
    multistagehs.outer_stage = outer_stage_hypernet
    solver = multistagehs

# Instantiate known system solver
system.solver = solver

# Solver MPC
mpc = TorchMPC(system, cost_func, t_span, opt, eps_accept=config.mpc.eps_accept, max_g_iters=config.mpc.max_g_iters,
            lookahead_steps=config.mpc.lookahead_steps, lower_bounds=None,
            upper_bounds=None, penalties=None).to(device)
loss_mpc = mpc.forward_simulation(real_system, x0, t_span, reset=False)

# Save
torch.save(mpc.trajectory_nominal.cpu(), 'data/{}_traj_{}.pt'.format(args.experiment, args.number))
torch.save(mpc.control_inputs.cpu(), 'data/{}_controls_{}.pt'.format(args.experiment, args.number))
torch.save(loss_mpc, 'data/{}_loss_{}.pt'.format(args.experiment, args.number))