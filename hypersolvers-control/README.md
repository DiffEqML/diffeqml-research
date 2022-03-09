# Neural Solvers for Fast and Accurate Numerical Optimal Control
    
<p align="center">
<img src="https://github.com/Juju-botu/diffeqml-research/blob/master/hypersolvers-control/hypersolvers_control_scheme.jpg">
</p>

<div align="center">
      
![NeurIPS](https://img.shields.io/badge/ICLR-2022-red.svg?)
![License](https://img.shields.io/badge/License-Apache-black.svg?)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2106.04165-purple.svg?)](https://arxiv.org/abs/2106.04165) -->
<!-- TODO: add links! -->
 
</div>

This folder contains code of experiments for "Neural Solvers for Fast and Accurate Numerical Optimal Control" to appear in the Tenth International Conference of Learning Representations (ICLR 2022).


> Synthesizing optimal controllers for dynamical systems often involves solving optimization problems with hard real--time constraints. These constraints determine the class of numerical methods that can be applied: computationally expensive but accurate numerical routines are replaced by fast and inaccurate methods, trading inference time for solution accuracy. This paper provides techniques to improve the quality of optimized control policies given a fixed computational budget. We achieve the above via a hypersolvers approach, which hybridizes a differential equation solver and a neural network. The performance is evaluated in direct and receding--horizon optimal control tasks in both low and high dimensions, where the proposed approach shows consistent Pareto improvements in solution accuracy and control performance.

paper: [OpenReview link](https://openreview.net/forum?id=m8bypnj7Yl5)

-------------
## Running the code
All the code is made to be self-contained and to  be run in order as per the name of the notebooks and scripts in each experimental section.

### Dependencies

The entire codebase is in PyTorch. Run `pip install -r requirements.txt` to install the basic required packages.

### Experiment details
We divide the folder `experiments/` into different dynamical systems. The numbers denote the corresponding notebook(s) and/or script(s).

### Cart-Pole
Cart and pole system with control task to swing up the pole. We apply Multi-stage Hypersolvers to correct the partial dynamics (`01*`) and control the system with Model Predictive Control (MPC) (`02*`).

### Pendulum
Hamiltonian model of an inverted pendulum with torsional spring to be stabilized. We first compare a purely data-driven model with Hypersolvers (`00`), then compare Hypersolvers at different timesteps (`01`). Notebooks (`02*`) compare different architectures and generalization study. Then we pretrain the Hypersolver and apply direct control to the system in (`03*`).

### Quadcopter
We consider a 3D drone model to be driven to a target position. Notebooks (`01`), (`02`) are for training the Hypersolver while (`02`), (`03`) run MPC and plot the results.

### Spring-Mass
Linear spring-mass system. We compare in notebooks (`00*`) stochastic and active minimization and different step comparison (`01`).

### Timoshenko Beam
Modeled discretization of the Timoshenko beam PDE system to be stabilized. (`01`) trains the Hypersolver and `(02*)` run the control policy optimization.


----------------------------

If you find our work or code useful, please consider citing:

```
@article{berto2022neural,
  title={Neural Solvers for Fast and Accurate Numerical Optimal Control},
  author={Berto, Federico and Massaroli, Stefano and Poli, Michael and Park, Jinkyoo},
  journal={International Conference on Learning Representations},
  year={2022}
}
```
