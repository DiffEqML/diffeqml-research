import torch
import torch.nn as nn
from torch import cos, sin, sign, norm
import numpy as np
from .template import ControlledSystemTemplate

sqrt2 = np.sqrt(2)

class QuadcopterGym(ControlledSystemTemplate):
    '''
    Quadcopter state space model compatible with batch inputs and hypersolvers
    Appropriately modified version to run efficiently in Pytorch
    References and thanks:
    Learning to Fly—a Gym Environment with PyBullet Physics for
Reinforcement Learning of Multi-agent Quadcopter Control
    https://arxiv.org/pdf/2103.02142 
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parameters
        self.G = 9.81
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.M = 0.027
        self.L = 0.0397
        self.THRUST2WEIGHT_RATIO = 2.25
        self.J = torch.diag(torch.Tensor([1.4e-5, 1.4e-5, 2.17e-5]))
        self.J_INV = torch.linalg.inv(self.J)
        self.KF = 3.16e-10
        self.KM = 7.94e-12
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        # DroneModel.CF2X:
        self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/sqrt2
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)

        
    def _dynamics(self, t, x):
        self.nfe += 1 # increment number of function evaluations

        # Control input evaluation
        rpm = self._evaluate_controller(t, x)
        
        pos       = x[...,  0:3]
        rpy       = x[...,  3:6]
        vel       = x[...,  6:9]
        rpy_rates = x[..., 9:12]

        # Compute forces and torques
        forces = rpm**2 * self.KF
        thrust_z = torch.sum(forces, dim=-1)
        thrust = torch.zeros(pos.shape)
        thrust[..., 2] = thrust_z
        
        rotation = euler_matrix(rpy[...,0], rpy[...,1], rpy[...,2])
        thrust_world_frame =  torch.einsum('...ij, ...j-> ...i', rotation, thrust)
        force_world_frame = thrust_world_frame - torch.Tensor([0, 0, self.GRAVITY])
        z_torques = rpm**2 *self.KM
        z_torque = (-z_torques[...,0] + z_torques[...,1] - z_torques[...,2] + z_torques[...,3])
        
        # DroneModel.CF2X:
        x_torque = (forces[...,0] + forces[...,1] - forces[...,2] - forces[...,3]) * (self.L/sqrt2)
        y_torque = (- forces[...,0] + forces[...,1] + forces[...,2] - forces[...,3]) * (self.L/sqrt2)
        
        torques = torch.cat([x_torque[...,None], y_torque[...,None], z_torque[...,None]], -1)
        torques = torques - torch.cross(rpy_rates, torch.einsum('ij,...i->...j',self.J, rpy_rates))
        rpy_rates_deriv = torch.einsum('ij,...i->...j', self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M
        self.cur_f = torch.cat([vel,
                                rpy_rates,
                                no_pybullet_dyn_accs,
                                rpy_rates_deriv], -1)
        
        return self.cur_f


def euler_matrix(ai, aj, ak, repetition=True):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    Readapted for Pytorch: some tricks going on
    """
    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    
    i = 0; j = 1; k = 2 # indexing
    
    # Tricks to create batched matrix [...,3,3] 
    # any suggestion to make code more readable is welcome!
    M = torch.cat(3*[torch.cat(3*[torch.zeros(ai.shape)[..., None, None]], -1)], -2) 
    if repetition:
        M[..., i, i] = cj
        M[..., i, j] = sj * si
        M[..., i, k] = sj * ci
        M[..., j, i] = sj * sk
        M[..., j, j] = -cj * ss + cc
        M[..., j, k] = -cj * cs - sc
        M[..., k, i] = -sj * ck
        M[..., k, j] = cj * sc + cs
        M[..., k, k] = cj * cc - ss
    else:
        M[..., i, i] = cj * ck
        M[..., i, j] = sj * sc - cs
        M[..., i, k] = sj * cc + ss
        M[..., j, i] = cj * sk
        M[..., j, j] = sj * ss + cc
        M[..., j, k] = sj * cs - sc
        M[..., k, i] = -sj
        M[..., k, j] = cj * si
        M[..., k, k] = cj * ci
    return M

# Torch utilities
def euler_to_quaterion_torch(r1, r2, r3):
    '''
    Transform angles from Euler to Quaternions
    For ZYX, Yaw-Pitch-Roll
    psi   = RPY[0] = r1
    theta = RPY[1] = r2
    phi   = RPY[2] = r3
    '''
    cr1 = torch.cos(0.5*r1)
    cr2 = torch.cos(0.5*r2)
    cr3 = torch.cos(0.5*r3)
    sr1 = torch.sin(0.5*r1)
    sr2 = torch.sin(0.5*r2)
    sr3 = torch.sin(0.5*r3)

    q0 = cr1*cr2*cr3 + sr1*sr2*sr3
    q1 = cr1*cr2*sr3 - sr1*sr2*cr3
    q2 = cr1*sr2*cr3 + sr1*cr2*sr3
    q3 = sr1*cr2*cr3 - cr1*sr2*sr3

    # e0,e1,e2,e3 = qw,qx,qy,qz
    q = torch.Tensor([q0,q1,q2,q3])
    # q = q*np.sign(e0)
    q = q/torch.linalg.norm(q)  
    return q

def quaternion_to_cosine_matrix_torch(q):
    '''
    Transform quaternions into
    rotational matrix
    Torch version 
    All the [...,] are ellipses supporting batch training
    '''
    # batched matrix of zeros with q dims 
    dcm = q[...,None] * 0 
    dcm[...,0,0] = q[...,0]**2 + q[...,1]**2 - q[...,2]**2 - q[...,3]**2
    dcm[...,0,1] = 2.0*(q[...,1]*q[...,2] - q[...,0]*q[...,3])
    dcm[...,0,2] = 2.0*(q[...,1]*q[...,3] + q[...,0]*q[...,2])
    dcm[...,1,0] = 2.0*(q[...,1]*q[...,2] + q[...,0]*q[...,3])
    dcm[...,1,1] = q[...,0]**2 - q[...,1]**2 + q[...,2]**2 - q[...,3]**2
    dcm[...,1,2] = 2.0*(q[...,2]*q[...,3] - q[0]*q[...,1])
    dcm[...,2,0] = 2.0*(q[...,1]*q[...,3] - q[0]*q[...,2])
    dcm[...,2,1] = 2.0*(q[...,2]*q[...,3] + q[...,0]*q[...,1])
    dcm[...,2,2] = q[...,0]**2 - q[...,1]**2 - q[...,2]**2 + q[...,3]**2
    return dcm


# Utilities
def euler_to_quaterion(r1, r2, r3):
    '''
    Transform angles from Euler to Quaternions
    For ZYX, Yaw-Pitch-Roll
    psi   = RPY[0] = r1
    theta = RPY[1] = r2
    phi   = RPY[2] = r3
    '''
    cr1 = np.cos(0.5*r1)
    cr2 = np.cos(0.5*r2)
    cr3 = np.cos(0.5*r3)
    sr1 = np.sin(0.5*r1)
    sr2 = np.sin(0.5*r2)
    sr3 = np.sin(0.5*r3)

    q0 = cr1*cr2*cr3 + sr1*sr2*sr3
    q1 = cr1*cr2*sr3 - sr1*sr2*cr3
    q2 = cr1*sr2*cr3 + sr1*cr2*sr3
    q3 = sr1*cr2*cr3 - cr1*sr2*sr3

    # e0,e1,e2,e3 = qw,qx,qy,qz
    q = np.array([q0,q1,q2,q3])
    # q = q*np.sign(e0)
    q = q/np.linalg.norm(q)  
    return q


def quaternion_to_cosine_matrix(q):
    '''
    Transform quaternions into
    rotational matrix
    '''
    dcm = np.zeros([3,3])
    dcm[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    dcm[0,1] = 2.0*(q[1]*q[2] - q[0]*q[3])
    dcm[0,2] = 2.0*(q[1]*q[3] + q[0]*q[2])
    dcm[1,0] = 2.0*(q[1]*q[2] + q[0]*q[3])
    dcm[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    dcm[1,2] = 2.0*(q[2]*q[3] - q[0]*q[1])
    dcm[2,0] = 2.0*(q[1]*q[3] - q[0]*q[2])
    dcm[2,1] = 2.0*(q[2]*q[3] + q[0]*q[1])
    dcm[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    return dcm
