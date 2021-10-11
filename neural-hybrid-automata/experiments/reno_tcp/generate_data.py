"Script to generate trajectories of the internal states of a Reno TCP"
from absl import app, flags
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np
import sys; sys.path.append('../../')

from torchdyn.numerics import *
from torchdyn.numerics.odeint import odeint_hybrid
from torchdyn.numerics.utils import EventCallback, StochasticEventCallback
from src.odeint import odeint_hybrid

FLAGS = flags.FLAGS
flags.DEFINE_string("system", "TCP", "Hybrid system to simulate")
flags.DEFINE_integer("seed", 123456, "RNG seed")
flags.DEFINE_integer("num_simulations", 40, "Number of simulations to run and save")


###### TCP event classes and simulator
class TCPEvent1(StochasticEventCallback):
    def check_event(self, t, x):
        w, s, q, λ = x[..., :1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
        ev1 = (λ - self.s >= 0).bool()
        ev2 = (q == 0).bool()
        return torch.logical_and(ev1, ev2)

    def jump_map(self, t, x):
        self.s = self.expdist.sample(x.shape[:1])
        w, s, q, λ = x[..., :1], x[..., 1:2], x[..., 2:3], x[..., 3:]
        wj, sj = 0.693 + torch.zeros_like(w), torch.zeros_like(s)
        qj, λj = 1 + torch.zeros_like(q), torch.zeros_like(λ)
        return torch.cat([wj, sj, qj, λj], 1)


class TCPEvent2(StochasticEventCallback):
    def check_event(self, t, x):
        w, s, q, λ = x[..., :1], x[..., 1:2], x[..., 2:3], x[..., 4:5]
        ev1 = (λ - self.s >= 0).bool()
        ev2 = (q == 1).bool()
        return torch.logical_and(ev1, ev2)

    def jump_map(self, t, x):
        self.s = self.expdist.sample(x.shape[:1])
        w, s, q, λ = x[..., :1], x[..., 1:2], x[..., 2:3], x[..., 3:]
        wj, sj = torch.zeros_like(w), torch.zeros_like(s)
        qj, λj = torch.zeros_like(q), torch.zeros_like(λ)
        return torch.cat([wj, sj, qj, λj], 1)


class TCPEvent3(StochasticEventCallback):
    def check_event(self, t, x):
        w, s, q, λ = x[..., :1], x[..., 1:2], x[..., 2:3], x[..., 4:5]
        ev1 = (λ - self.s >= 0).bool()
        ev2 = (q == 1).bool()
        return torch.logical_and(ev1, ev2)

    def jump_map(self, t, x):
        self.s = self.expdist.sample(x.shape[:1])
        w, s, q, λ = x[..., :1], x[..., 1:2], x[..., 2:3], x[..., 3:]
        wj, sj = w / 2, s
        qj, λj = 2 + torch.zeros_like(q), torch.zeros_like(λ)
        return torch.cat([wj, sj, qj, λj], 1)


class TCPEvent4(StochasticEventCallback):
    def check_event(self, t, x):
        w, s, q, λ = x[..., :1], x[..., 1:2], x[..., 2:3], x[..., 4:5]
        ev1 = (λ - self.s >= 0).bool()
        ev2 = (q == 2).bool()
        return torch.logical_and(ev1, ev2)

    def jump_map(self, t, x):
        self.s = self.expdist.sample(x.shape[:1])
        w, s, q, λ = x[..., :1], x[..., 1:2], x[..., 2:3], x[..., 3:]
        wj, sj = w / 2, s
        qj, λj = 2 + torch.zeros_like(q), torch.zeros_like(λ)
        λj[:, -1:] = λ[:, -1:]
        return torch.cat([wj, sj, qj, λj], 1)


class TCPEvent5(StochasticEventCallback):
    def check_event(self, t, x):
        w, s, q, λ = x[..., :1], x[..., 1:2], x[..., 2:3], x[..., 4:5]
        ev1 = (λ - self.s >= 0).bool()
        ev2 = (q == 2).bool()
        return torch.logical_and(ev1, ev2)

    def jump_map(self, t, x):
        self.s = self.expdist.sample(x.shape[:1])
        w, s, q, λ = x[..., :1], x[..., 1:2], x[..., 2:3], x[..., 3:]
        wj, sj = torch.zeros_like(w), torch.zeros_like(s)
        qj, λj = torch.zeros_like(q), torch.zeros_like(λ)
        return torch.cat([wj, sj, qj, λj], 1)


class TCPSimulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.w0e1 = 0.693
        self.τoff = 3
        self.RTT = 1
        self.nack = 2
        self.pdrop = 0.05
        self.l2 = 0.693
        self.k = 20

    def forward(self, t, x):
        w, s, q = x[..., :1], x[..., 1:2], x[..., 2]
        nel_q0, nel_q1, nel_q2 = len(x[q == 0]), len(x[q == 1]), len([q == 2])

        sol = []
        if nel_q0 > 0:
            q0_dx = torch.zeros_like(x[q == 0][..., :2])
            sol.append(q0_dx)

        if nel_q1 > 0:
            q1_dw = (self.l2 * w[q == 1]) / (self.nack * self.RTT)
            q1_ds = w[q == 1] / self.RTT
            q1_dx = torch.cat([q1_dw, q1_ds], 1)
            sol.append(q1_dx)

        if nel_q2 > 0:
            q2_dw = torch.ones_like(w[q == 2]) / (self.nack * self.RTT)
            q2_ds = w[q == 2] / self.RTT
            q2_dx = torch.cat([q2_dw, q2_ds], 1)
            sol.append(q2_dx)
        dx = torch.cat(sol, 0)

        # lambda dynamics (for stochastic event simulation)
        λ = x[..., 3:]
        dλ1 = 1 / self.τoff * torch.ones_like(λ[..., :1])  # ev1
        dλ2 = w / (self.k * self.RTT)
        dλ3 = self.pdrop * w / self.RTT
        dλ4 = self.pdrop * w / self.RTT
        dλ5 = w / (self.k * self.RTT)
        dλ = torch.cat([dλ1, dλ2, dλ3, dλ4, dλ5], 1)
        return torch.cat([dx, torch.zeros_like(x[..., -1:]), dλ], 1)


def generate_trajectories(argv):
    # set all seeds and use deterministic backend
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.use_deterministic_algorithms(True)

    solver = DormandPrince45()

    path = Path('../../data')
    if not path.exists(): path.mkdir()

    if FLAGS.system == "TCP":
        path = Path('../../data/tcp')

        if not path.exists(): path.mkdir()

        print("Testing simulation of TCP...")
        f = TCPSimulator()
        x0_ = torch.tensor([[0., 0., 0]])
        # append zeroed initial intensity states, one for each event
        callbacks = [TCPEvent1(), TCPEvent2(), TCPEvent3(), TCPEvent4(), TCPEvent5()]
        x0 = torch.cat([x0_, torch.zeros(x0_.shape[0], len(callbacks))], 1)
        for cb in callbacks: cb.initialize(x0)
        # t_eval, sol = odeint_hybrid(f, x0, t_span=torch.tensor([0., 100.]), j_span=10, callbacks=callbacks,
        #                             t_eval=torch.linspace(0, 100, 2000)[1:-1], event_tol=1e-4, solver=solver, atol=1e-6,
        #                             rtol=1e-6)
        t_eval, sol = odeint_hybrid(f, x0, t_span=torch.linspace(0, 100, 2000), j_span=10, callbacks=callbacks, 
                                    event_tol=1e-4, solver=solver, atol=1e-6, rtol=1e-6)
        print("Generating data...")
        for i in tqdm(range(FLAGS.num_simulations)):
            x0_ = torch.tensor([[0., 0., 0]])
            # append zeroed initial intensity states, one for each event
            callbacks = [TCPEvent1(), TCPEvent2(), TCPEvent3(), TCPEvent4(), TCPEvent5()]
            x0 = torch.cat([x0_, torch.zeros(x0_.shape[0], len(callbacks))], 1)
            for cb in callbacks: cb.initialize(x0)
            t_eval, sol = odeint_hybrid(f, x0, t_span=torch.linspace(0, 100, 2000), j_span=10, callbacks=callbacks, 
                                        event_tol=1e-4, solver=solver, atol=1e-6, rtol=1e-6)

            torch.save(t_eval, f'../../data/tcp/raw_t_eval_{i}')
            torch.save(sol, f'../../data/tcp/raw_sol_{i}')


    else:
        print(f"{FLAGS.system} simulator not available in current data generation pipeline")

if __name__ == '__main__':
    app.run(generate_trajectories)