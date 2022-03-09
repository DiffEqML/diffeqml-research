import torch
import torch.nn as nn

class ControlledSystemTemplate(nn.Module):
    """Template Model compatible with hypersolvers
    The hypersolver is defined inside of the dynamics module
    """
    def __init__(self, u, 
                 solver='euler', 
                 hypersolve=None, 
                 retain_u=False, 
                 use_torchdyn=True,
                 _use_xfu=False):
        super().__init__()
        self.u = u
        self.solver = solver
        self.hypersolve = hypersolve
        self.retain_u = retain_u # use for retaining control input (e.g. MPC simulation)
        self.nfe = 0 # count number of function evaluations of the vector field
        self.cur_f = None # current dynamics evaluation
        self.cur_u = None # current controller value
        self._retain_flag = False # temporary flag for evaluating the controller only the first time
        if use_torchdyn: from torchdyn.numerics.odeint import odeint
        else: from torchdiffeq import odeint
        self.odeint = odeint
        self.use_torchdyn = use_torchdyn
        self._use_xfu = _use_xfu

    def forward(self, x0, t_span):
        x = [x0[None]]
        xt = x0
        if self.hypersolve:
            # Use the hypersolver to carry forward the system simulation
            for i in range(len(t_span)-1):
                '''HyperEuler step
                x(t+1) = x(t) + Δt*f + Δt^2*g(x,f,u)'''
                Δt = t_span[i+1] - t_span[i]
                f = self._dynamics(t_span[i], xt)
                if self._use_xfu:
                    xfu = torch.cat([xt, self.cur_f, self.cur_u], -1)
                    g = self.hypersolve(0., xfu)
                else:
                    g = self.hypersolve(0., xt)
                self._retain_flag = False
                xt = xt + Δt*f + (Δt**2)*g
                x.append(xt[None])
            traj = torch.cat(x)
        elif self.retain_u:
            '''Iterate over the t_span: evaluate the controller the first time only and then retain it'''
            for i in range(len(t_span)-1):
                self._retain_flag = False
                diff_span = torch.linspace(t_span[i], t_span[i+1], 2)
                if self.use_torchdyn: xt = self.odeint(self._dynamics, xt, diff_span, solver=self.solver)[1][-1]
                else: xt = self.odeint(self._dynamics, xt, diff_span, method=self.solver)[-1]
                x.append(xt[None])
            traj = torch.cat(x)
        else:
            '''Compute trajectory with odeint and base solvers'''
            if self.use_torchdyn: traj = self.odeint(self._dynamics, xt, t_span, solver=self.solver)[1][None]
            else: traj = self.odeint(self._dynamics, xt, t_span, method=self.solver)[None]
            # traj = odeint(self._dynamics, xt, t_span, method=self.solver)[None]
        return traj

    def reset_nfe(self):
        """Return number of function evaluation and reset"""
        cur_nfe = self.nfe; self.nfe = 0
        return cur_nfe

    def _evaluate_controller(self, t, x):
        '''
        If we wish not to re-evaluate the control input, we set the retain
        flag to True so we do not re-evaluate next time
        '''
        if self.retain_u:
            if not self._retain_flag:
                self.cur_u = self.u(t, x)
                self._retain_flag = True
            else: 
                pass # We do not re-evaluate the control input
        else:
            self.cur_u = self.u(t, x)
        return self.cur_u
    
        
    def _dynamics(self, t, x):
        '''
        Model dynamics in the form xdot = f(t, x, u)
        '''
        raise NotImplementedError