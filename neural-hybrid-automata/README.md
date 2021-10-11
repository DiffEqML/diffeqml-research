<div align="center">
Neural Hybrid Automata: 
Learning Dynamics with Multiple Modes and Stochastic Transitions
      
</div>
      
<p align="center">
<img src="https://github.com/DiffEqML/diffeqml-media/blob/main/images/nha/nha_fig1.jpg">
</p>

<div align="center">
      
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2021-red.svg?)](https://papers.nips.cc/paper/2021/hash/f1686b4badcf28d33ed632036c7ab0b8-Abstract.html)
[![License](https://img.shields.io/badge/License-Apache-black.svg?)](https://papers.nips.cc/paper/2020/hash/f1686b4badcf28d33ed632036c7ab0b8-Abstract.html)
[![arXiv](https://img.shields.io/badge/arXiv-2106.04165-purple.svg?)](https://arxiv.org/abs/2106.04165)

</div>

This folder contains code of experiments for "Neural Hybrid Automata: Learning Dynamics with Multiple Modes and Stochastic Transitions" to appear in the Thirty-fourth Conference on Neural Information Processing Systems (NeurIPS 2021).


> Effective control and prediction of dynamical systems often require appropriate handling of continuous-time and discrete, event-triggered processes. Stochastic hybrid systems (SHSs), common across engineering domains, provide a formalism for dynamical systems subject to discrete, possibly stochastic, state jumps and multi-modal continuous-time flows. Despite the versatility and importance of SHSs across applications, a general procedure for the explicit learning of both discrete events and multi-mode continuous dynamics remains an open problem. This work introduces Neural Hybrid Automata (NHAs), a recipe for learning SHS dynamics without a priori knowledge on the number of modes and inter-modal transition dynamics. NHAs provide a systematic inference method based on normalizing flows, neural differential equations and self-supervision. We showcase NHAs on several tasks, including mode recovery and flow learning in systems with stochastic transitions, and end-to-end learning of hierarchical robot controllers.

paper: [arXiv link](https://arxiv.org/abs/2106.04165)

## TL;DR

Stochastic hybrid systems (SHSs) are a general class of system that formalizes elements common in real-life dynamics: (potentially stochastic) mode transitions and hybrid dynamics. At a high-level, some intuition for SHS terminology (see the paper for formal definitions):

* **Mode**: hidden latent variable conditioning the dynamics. Can change discontinuously. 
* **Event times**: events determine mode shifts and/or jumps in the state. The events can be determined as deterministic functions of the state, or they can be random variables.

We provide a multi-stage recipe to learn without knowing how many modes the underlying data generating process has. The procedure is shown below:


<p align="center">
<img src="https://github.com/DiffEqML/diffeqml-media/blob/main/images/nha/nha_fig2.jpg">
</p>

Note that we do assume the existence of a segmentation algorithm to separate subtrajectories due to the phenomenon discussed at the end of Section 4 and in Appendix A.


-------------
## Running the code

### Dependencies

The entire codebase is in PyTorch.

```
torchdyn
torchdiffeq
ml_collections
sklearn
```

### Experiment details

* ***Reno TCP***: we carry out a quantitative evaluation on quality of learned flows (mean squared
error) and quality of mode clusters recovered during self–supervision (v–measure).
* ***Mode mixing in switching systems***: we highlight and varify robustness against mode mixing, a phenomenon occurring during learning of multi–mode systems.
* **Behavioral control of wheeled robots**: NHAs enable task–based behavioral control. We investigate a point–to–point navigation task where a higher level reinforcement learning (RL) planner
determines mode switching for a lower–level optimal controller.


### Reno Transmission Control Protocol Dynamics

`python generate_data.py` to generate TCP trajectories seen in the paper. `python train_mode_recovery.py` to perform self-supervised mode recovery and trajectory reconstruction.

### Mode Mixing in Switching Systems

Seeded notebooks for `softmax` and `categorical` encoders are under `experiments/lss`. TL;DR: `categorical` encoders are better for mode recovery as no mode mixing can occur + it is possible to prune redundant modes since they are guaranteed to be "similar" across the state space (see Appendix B.3).


----------------------------

If you find our work or code useful, please consider citing:

```
@article{poli2021neural,
  title={Neural Hybrid Automata: Learning Dynamics with Multiple Modes and Stochastic Transitions},
  author={Poli, Michael and Massaroli, Stefano and Scimeca, Luca and Oh, Seong Joon and Chun, Sanghyuk and Yamashita, Atsushi and Asama, Hajime and Park, Jinkyoo and Garg, Animesh},
  journal={arXiv preprint arXiv:2106.04165},
  year={2021}
}
```
