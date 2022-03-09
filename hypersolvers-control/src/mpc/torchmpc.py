import torch
import torch.nn as nn

class TorchMPC(nn.Module):
    def __init__(self,
                 system,
                 cost_function,
                 t_span,
                 opt,
                 max_g_iters=100,
                 eps_accept=0.01,
                 lookahead_steps=100,
                 lower_bounds=None,
                 upper_bounds=None,
                 penalties=None,
                 penalty_function=nn.Softplus(),
                 scheduler=None,
                 regulator=1e-4,
                 verbose=True):
        '''
        Gradient-based nMPC compatible with continuous-time task
        Controller, cost and system modules are defined separately
        Constrained optimization:
        1) For control inputs:
            - the controller module is already defined with constraints
        2) For states:
            - we add a penalty function for constraint violation i.e.
            ReLU; we could also use Lagrangian methods such as
            https://arxiv.org/abs/2102.12894

        Args:
            system: controlled system module to be controlled
            cost_function: cost function module
            t_span: tensor containing the time span
            opt: optimizer module such as Adam or LBFGS
            max_g_iters (int, optional): maximum number of gradient iterations
            eps_accept (float, optional): cost function value under which optimization is stopped
            lookahead_steps (int, optional): number of receding horizon steps
            lower_bounds (list, optional): lower bounds corresponding to each state variable. Default: None
            upper_bounds (list, optional): upper bounds corresponding to each state variable. Default: None
            penalties (tensor, optional): penalty weights for each state. Default: None
            penalty_function (module, optional): function for penalizing constraint violation. Default: nn.Softplus()
            scheduler (optimizer, optional): learning rate or other kind of scheduler. Default: None
            regulator (float, optional): penalize high control inputs. Default: 1e-4
            verbose (bool, optional): print out debug information. Default: True
        '''

        super().__init__()
        self.sys, self.t_span = system, t_span
        self.opt = opt
        self.eps_accept, self.max_g_iters = eps_accept, max_g_iters
        self.lookahead_steps = lookahead_steps
        self.cost_function = cost_function
        self.loss, self.control_loss, self.reg_loss = 0, 0, 0
        self.trajectory = None
        self.trajectory_nominal = None
        self.controls_inputs = None
        self.verbose = verbose
        self.scheduler = scheduler
        self.reg = regulator
        # Constraints
        self.lower_c = lower_bounds
        self.upper_c = upper_bounds
        if lower_bounds is not None or upper_bounds is not None:
            self._check_bounds()
            if penalties is None:
                raise ValueError("Penalty weights were not defined")
        self.λ = penalties
        self.penalty_func = penalty_function

    def forward(self, x):
        '''
        Module forward loop: solve the optimization problem in the given time span from position x
        '''

        # update receding horizon
        remaining_span = self.t_span[:self.lookahead_steps]
        # optimize receding horizon
        self._solve_subproblem(x, remaining_span)
        return self.trajectory

    def forward_simulation(self, real_sys, x0, t_span, steps_nom=10, x_up_to=None, reset=False, reinit_zeros=False):
        '''
        Simulate MPC by propagating the system forward with a high precision solver:
        the optimization problem is repeated until the end of the time span

        Args:
            real_sys: controlled system module describing the nominal system evolution
            x0: initial position
            t_span: time span in which the system is simulated
            steps_nom (int, optional): number of nominal steps per each MPC step. Default: 10
            x_up_to (int, optional): consider only first states for the controllers. Default: None
            reset (bool, optional): reset all the controller parameters after each nominal system propagation. Default: False
            reinit_zeros (bool, optional): reset the last layer of controller parameters. Default: False

        Returns:
            val_loss: validation loss of the computed trajectory
        '''
        # Obtain time spans
        t0, tf = t_span[0].item(), t_span[-1].item()
        steps = len(t_span)
        Δt = (tf - t0) / (steps - 1)
        # Variables initialization for simulation
        t_0 = t0; x_0 = x0
        traj = []; controls = []

        if self.verbose: print('Starting simulation... Timestamp: {:.4f}'.format(t_0))
        # Inner loop: simulate the MPC by keping the control input constant between sampling times
        for j in range(steps - 1):
            # Updates
            self.t_span = torch.linspace(t_0, tf + Δt * self.lookahead_steps,
                                         int((tf - t_0 + Δt * self.lookahead_steps) / Δt) + 1).to(x0)
            # t span to use in the system forward simulation
            Δt_span = torch.linspace(t_0, t_0 + Δt, steps_nom + 1).to(x0)
            # We reset every time the controller
            if reset: self._reset()
            if reinit_zeros: self.sys.u._init_zeros()
            # Train the MPC
            self(x_0)
            # Update constant controller with current MPC input to retain
            # We may want to use a part of the state for the controller, as in this case
            if x_up_to: real_sys.u.u0 = self.sys.u(t_0, x_0[..., 0:x_up_to]).to(x0)
            else: real_sys.u.u0 = self.sys.u(t_0, x_0).to(x0)
            controls.append(real_sys.u.u0[None])
            # Propagate system forward
            part_traj = real_sys(x_0, Δt_span).squeeze(0).detach()
            if j == 0:
                traj.append(part_traj)
            # we do not append the solution 0 since it was already
            # calculated
            else:
                traj.append(part_traj[1:])
            t_0 = t_0 + Δt
            x_0 = part_traj[-1]
            if self.verbose: print(' | Timestamp: {:.4f} s'.format(t_0))

        if self.verbose: print('The simulation has ended!')
        # Cost function evaluation via nominal trajectory
        self.trajectory_nominal = torch.cat(traj, 0).detach()
        self.control_inputs = torch.cat(controls, 0).detach()
        val_loss = self.cost_function(self.trajectory_nominal).cpu().detach()
        if self.lower_c is not None or self.upper_c is not None:
            val_loss += self._penalize_constraints(self.trajectory_nominal)  # constraint loss
        return val_loss

    def _solve_subproblem(self, x, remaining_span):
        '''
        Solve optimization sub-problem for the remaining time span
        '''
        opt, i = self.opt, 0
        while i <= self.max_g_iters:
            # Calculate loss via closure()
            # This function is required by LBFGS and can support
            # other optimizers e.g. Adam or SGD
            def closure():
                traj = self.sys(x, remaining_span)
                # apply cost function, the module is defined externally
                control_loss = self.cost_function(traj)
                reg_loss = (self.reg * self.sys.cur_u).abs().mean()  # add regulator loss
                loss = control_loss + reg_loss  # disable reg loss for now
                if self.lower_c is not None or self.upper_c is not None:
                    loss += self._penalize_constraints(traj)  # constraint loss
                loss.backward()  # run gradient engine
                # Saving metrics
                self.loss = loss.detach()
                self.control_loss = control_loss.detach()
                self.reg_loss = reg_loss.detach()
                self.trajectory = traj
                return loss

            # Optimization step
            opt.step(closure)
            if self.scheduler: self.scheduler.step()
            opt.zero_grad(); i += 1
            
            # Check for errors due i.e. to stiff system giving inf values
            if torch.isnan(self.loss):
                self._force_stop_simulation("""Loss function yielded a nan value. \
                    This may be due to a stiff system whose ODE solver integrated to +- inf. \
                    Try lowering step size or use another solver, i.e. and adaptive one""")

            if self.loss <= self.eps_accept:
                if self.verbose:
                    print(f'\rInner-loop converged, last cost: {self.loss.item():.3f}, iterations: {i}', end='')
                return
        if self.verbose:
            print(f'\rInner-loop did not converge, last cost: {self.loss.item():.3f}', end='')
        return

    def _penalize_constraints(self, x):
        '''Calculate penalty for constraints violation'''
        P = 0
        # Lower Constraints
        for c_low, i in zip(self.lower_c, range(len(self.lower_c))):
            if c_low is None:
                pass
            else:
                P += (self.λ[i] * (self.penalty_func(-x[..., i] + c_low))).abs().mean()

        # Upper Constraints
        for c_up, i in zip(self.upper_c, range(len(self.upper_c))):
            if c_up is None:
                pass
            else:
                P += (self.λ[i] * (self.penalty_func(x[..., i] - c_up))).abs().mean()
        return P

    def _reset(self):
        '''
        Reinitialize controller parameter under task changes
        Reset functon is defined inside of the controller module
        '''
        self.sys.u._reset()
        

    def _check_bounds(self):
        '''Check constraints validity'''
        if self.lower_c is not None and self.upper_c is not None:
            if len(self.lower_c) != len(self.upper_c):
                raise ValueError("Constraints should be of the same "
                                 "dimension; use None for unconstrained variables. "
                                 "Got dimensions {} and {}".format(
                    len(self.lower_c), len(self.upper_c)))

            for i in range(len(self.lower_c)):
                if self.lower_c[i] is not None and self.upper_c[i] is not None:
                    if self.lower_c > self.upper_c:
                        raise ValueError("At least one lower constraint is "
                                         "greater than its upper constraint")

    def _force_stop_simulation(self, message):
        '''Simulation stop handler for i.e. nan cost function'''
        raise RuntimeError(r"The simulation has been forcefully stopped. Reason: {}".format(message))
