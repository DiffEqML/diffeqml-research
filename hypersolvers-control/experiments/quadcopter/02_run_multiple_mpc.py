import torch
import torch.nn as nn
from torchdiffeq import odeint
import sys; sys.path.append(2*'../')
from src import *
import matplotlib.pyplot as plt
from torch import cos, sin, sign, norm
import torchdiffeq
from tqdm import tqdm
import matplotlib.pyplot as plt

# device = torch.device('cuda:0')
device=torch.device('cpu') # feel free to change

# Create initial distribution
x0, y0, z0 = 0., 0., 0.
phi0, theta0, psi0 = 0., 0., 0.
wdot_hover = 0.              # Hovering motor acc

init = torch.tensor([x0,
                    y0,
                    z0,
                    phi0,
                    theta0,
                    psi0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0], dtype=torch.float).to(device)

low = -torch.ones(12).to(device)
high = torch.ones(12).to(device)
low[0:3]*=5
high[0:3]*=5
low[3:6]*=50
high[3:6]*=50
low[6:9]*=50
high[6:9]*=50
low[9:12]*=100
high[9:12]*=100

in_scal = torch.cat([1/high, 1/high])
u_scal = 1/20000*torch.ones(4).to(device)
in_scal = torch.cat([in_scal, u_scal])

out_scal = high

# Zero out positions contribution term
out_scal[0:6] = 0 
print('Input scaling:\n', in_scal)
print('Output scaling:\n', out_scal)


# Load hypersolver model
hdim = 64

class Hypersolver(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(28, hdim),
            nn.Softplus(),
            nn.Linear(hdim, hdim),
            nn.Softplus(),
            nn.Linear(hdim, 12)]
        self.layers = nn.Sequential(*layers)
        
    def forward(self, t, x):
        x = x*in_scal
        x = self.layers(x)
        x = x*out_scal
        return x
    
hs = torch.load('saved_models/hypersolver_0.02_new_quadcopter.pt')

# Desired final condition
# if the final condition close to the initial point, euler may perform better than hypereuler
# and even midpoint / rk: indeed, error on these trajectories will be very low since the system is slow 
# (small velocity, small turn rate etc. ) hence a low order solver will perform well.
# Here we are interested in fast maneuvers and "difficult" control objectives (i.e., regions with higher residuals)
target = torch.Tensor([8, 8, 8])

class PositioningCost(nn.Module):
    def __init__(self, target, Q=1, R=0, P=0):
        super().__init__()
        self.target = target
        self.Q, self.R, self.P = Q, R, P
        
    def forward(self, traj, u=None, mesh_p=None):
        cost =  .1*torch.norm(traj[...,-1, :3] - self.target) +            1*torch.norm(traj[..., :3] - self.target) +            0.01*torch.norm(traj[..., 3:6]) +            0.01*torch.norm(traj[..., 6:9]) +            0.01*torch.norm(traj[..., 9:12])
        return cost
cost_function = PositioningCost(target)

# MPC simulation variables
# Choose the same dt as the hypersolver training for best results
# increasing the time span indefinetely can result in numerical errors especially for the euler solver
dt = 0.02
tf = 3
t_span = torch.linspace(0, tf, int(tf/dt)+1)
lr = 1e-2
weight_decay = 1e-4

steps_nom = 1 # Nominal steps to do between each MPC step (we use dopri5 anyways)
max_iters = 20
eps_accept = 1e-3
lookahead_steps = 25 #50


experiments = ['hypereuler', 'euler', 'midpoint', 'rk4']
num = 30

def run_experiment(x0, e, n):
    print('='*50)
    print('Experiment: ', e, '\nNum: ', n)
    name = e + '_' + str(n)
    if e == 'hypereuler': drone = QuadcopterGym(None, hypersolve=hs, _use_xfu=True)
    else: drone = QuadcopterGym(None, solver=e, retain_u=True) # remember to put this for retaining control input
    out_scal_u = torch.Tensor([0, drone.MAX_RPM])
    controller = BoxConstrainedController(12, 4, output_scaling=out_scal_u, constrained=True)
    opt = torch.optim.Adam(controller.parameters(), lr=lr, weight_decay=weight_decay)
    drone.u = controller

    mpc = TorchMPC(drone, cost_function, t_span, opt, eps_accept=eps_accept, max_g_iters=max_iters,
                lookahead_steps=lookahead_steps, lower_bounds=None,
                upper_bounds=None, penalties=None).to(device)

    real_system = QuadcopterGym(RandConstController((1,1),1,1), use_torchdyn=False, solver='dopri5')
    # Remember to reset the controller! This avoid numerical instabilities
    mpc.forward_simulation(real_system, x0, t_span, reset=True, reinit_zeros=False)
    torch.save(mpc.trajectory_nominal, 'data/trajectory_'+name+'.pt')
    torch.save(mpc.control_inputs, 'data/controls_'+name+'.pt')
    print('Training ended for:' + name)



# Run in parallel
# Each experiment only requires ~1 core since there is no
# batching, so we can run them together to save time
# may take time on other machines (using and AMD Threadripper)
import multiprocessing
jobs = []
for n in range(num):
    # Change initial position and keep it the same for the experiments
    var = 0.5 # possible variation from initial position in meters
    init_state = init + torch.zeros(12).to(device)
    perturbation = torch.Tensor(3).uniform_(-var, var).to(device)
    init_state[:3] += perturbation
    for e in experiments:
        p = multiprocessing.Process(target=run_experiment, args =(init_state, e, n))
        jobs.append(p)
        p.start()

## Execute in parallel
print('Starting parallel execution...')
for proc in tqdm(jobs):
    proc.join()     
    

