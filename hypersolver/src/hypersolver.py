# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from .custom_fixed_explicit import ButcherTableau, GenericExplicitButcher

class HyperSolverTemplate(nn.Module):
    def __init__(self, f, g):
        super().__init__()  
        self.g = g
        self.f = f  
    
    def forward(self, ds, dz, z):
        "Computes one-step residuals"
        raise NotImplementedError

    def base_residuals(self, base_traj, s_span): 
        "Computes residuals of `base_method` on `base_traj`"
        raise NotImplementedError
    
    def odeint(self, x, s_span, use_residual=1):
        raise NotImplementedError
        
               
class HyperSolver(HyperSolverTemplate):
    def __init__(self, f, g, base_solver):
        super().__init__(f, g)  
        self.base_solver = base_solver
        self.p = self.base_solver.order
    
    def forward(self, ds, dz, z):
        ds = ds*torch.ones([*z.shape[:1], 1 , *z.shape[2:]]).to(z)
        z = torch.cat([z, dz, ds], 1)
        z = self.g(z)       
        return z
    
    def base_residuals(self, base_traj, s_span): 
        "Computes residuals of `base_method` on `base_traj`"
        ds = s_span[1] - s_span[0]
        fi = torch.cat(
            [self.base_solver(s, base_traj[i], self.f, ds)[None,:,:] for i, s in enumerate(s_span[:-1])])
        return (base_traj[1:] - base_traj[:-1] - ds*fi)/ds**(self.p + 1)
        return residuals
    
    def hypersolver_residuals(self, base_traj, s_span): 
        "Computes residuals of the hypersolver on `base_traj`"
        ds = (s_span[1] - s_span[0]).expand(*base_traj[:-1].shape)
        dz = torch.cat([self.f(s, base_traj[i])[None,:,:] for i, s in enumerate(s_span[:-1])])
        residuals = torch.cat([self(ds_[0,0], dz_, z_)[None] for ds_, dz_, z_ in zip(ds, dz, base_traj[:-1])], 0)
        return residuals
    
    def odeint(self, z, s_span, use_residual=True):
        traj = torch.zeros(len(s_span), *z.shape); traj[0] = z        
        for i, s in enumerate(s_span[:-1]):
            ds = s_span[i+1] - s_span[i]
            dz = self.f(s, z)
            if use_residual: z = z + ds*self.base_solver(s, z, self.f, ds) + ds**(self.p + 1) * self(ds, dz, z)
            else: z = z + ds*self.base_solver(s, z, self.f, ds)   
            traj[i+1] = z
        return traj


class HyperEuler(HyperSolver):
    def __init__(self, f, g):
        tableau = ButcherTableau([[0]], [1], [0], [])
        base_solver = GenericExplicitButcher(tableau)
        super().__init__(f, g, base_solver)  
    
class HyperMidpoint(HyperSolver):
    def __init__(self, f, g):
        tableau = ButcherTableau([[0, 0], [0.5, 0]], [0, 1], [0, 0.5], [])
        base_solver = GenericExplicitButcher(tableau)
        super().__init__(f, g, base_solver)  
    
class Hyper2Alpha(HyperSolver):
    def __init__(self, f, g, alpha=1):
        tableau = ButcherTableau([[0, 0], [alpha, 0]], [1-1/(2*alpha), 1/(2*alpha), [0, alpha]], [])
        base_solver = GenericExplicitButcher(tableau)
        super().__init__(f, g, base_solver)   