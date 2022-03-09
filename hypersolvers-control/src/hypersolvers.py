from .activations import *
from torchdyn.numerics.solvers import SolverTemplate, Midpoint
from warnings import warn

class HyperNetwork(nn.Module):
    """Simple hypernetwork using as input the current state, vector field and controller"""
    def __init__(self, net, sys):
        super().__init__()
        self.net = net
        self.sys = sys
        warn("Setting system as internal parameter may cause maximum recursion error. You may access it as a global variable instead " \
            "https://stackoverflow.com/questions/6809402/python-maximum-recursion-depth-exceeded-while-calling-a-python-object")

    def forward(self, t, x):
        xfu = torch.cat([x, self.sys.cur_f, self.sys.cur_u], -1)
        return self.net(xfu)
        
    
class MultiStageHypersolver(SolverTemplate):
    """
    Explicit multistage ODE stepper: inner stage is a vector field corrector
    while the outer stage is a residual approximator of the ODE solver
    """
    def __init__(self, inner_stage: nn.Module, outer_stage: nn.Module,
                       base_solver=Midpoint, dtype=torch.float32):
        super().__init__(order=base_solver().order)
        self.dtype = dtype
        self.stepping_class = 'fixed'
        self.base_solver = base_solver
        self.inner_stage = inner_stage
        self.outer_stage = outer_stage

    def step(self, f, x, t, dt, k1=None):
        # Correct vector field with inner stage and propagate
        self.vector_field = f
        _, _x_sol, _ = self.base_solver().step(self.corrected_vector_field, x, t, dt, k1=k1)
        # Residual correction with outer stage
        x_sol = _x_sol + dt**self.base_solver().order * self.outer_stage(t, f(t, x))
        return _, x_sol, _ 

    def corrected_vector_field(self, t, x):
        return self.vector_field(t, x) + self.inner_stage(t, x)


class TanhHyperSolver(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        # Initialize activations
        self.a1 = nn.Tanh()
        self.a2 = nn.Tanh()
    
    def forward(self, x):
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.fc3(x)
        return x


class ReLUHyperSolver(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        # Initialize activations
        self.a1 = nn.ReLU()
        self.a2 = nn.ReLU()

    def forward(self, x):
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.fc3(x)
        return x


class SnakeHyperSolver(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        # Initialize activations
        self.a1 = Snake(hidden_dim)
        self.a2 = Snake(hidden_dim)

    def forward(self, x):
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.fc3(x)
        return x



