# Supplementary code
Code for the paper: "Neural Solvers for Fast and Accurate Numerical Optimal Control"

## Instructions:
1. Run `pip install -r requirements.txt` to install necessary dependencies
2. The notebooks, coupled with `src` code, should be self-contained and runnable in order, top to bottom

## Contents
The folders are organized with the name of the respective systems. 
You may also find `saved_models` folders in which pre-trained hypersolver models are stored. `media` directories contain images that were generated running the codes. You may find experiments used in the paper to be replicated by running the corresponding notebooks as following:

### Spring-mass 
Experiments under `experiments/spring_mass` folder:
- `00a_pretrain_stochastic_active`: pretrain the hypersolvers with stochastic exploration and active error minimization
- `00b_plot_stochastic_vs_active`: plot results for the above
- `01_different_timesteps_comparison`: plot the comparison of hypersolver residuals at different time-steps

### Inverted Pendulum
Experiments under the `experiments/pendulum` folder:
- `00_data_driven_only_vs_hypersolver`: (extra) comparing fully learned vector fields with the hypersolver approach
- `01_timesteps_comparison`: comparison of hypersolver residuals at different time steps
- `02a_architectures_training`: train different hypersolver architectures
 - `02b_architectures_plotting_generalization` plot generalization results
- `03a_pre_training`: hypersolver pre-training on single step size
- `03b_direct_control`: direct optimal control
- `03c_plot`: plot control performance

### Cart-Pole
Experiments under `experiments/cartpole` folder:
- `00_multistage_pretrain `: pre-train multi-stage hypereuler
- `01_multistage_plot_stages`: plot results with ablation study on second stage
- `02a_multistage_mpc`: run MPC with different solvers
- `02b_plot_trajectories`: plot controlled trajectories 

### Quadcopter
Experiments under `experiments/quadcopter` folder:
- `00_training_hs`: pre-train hypersolver
- `01_plot_training_results`: residual plots
- `02_run_multiple_mpc`: run MPC with different solvers and initial conditions (script)
- `03_plot_results`: plot controlled trajectories and performance analysis


## Notes

1. Due to ODE overheads (and experimental nature of the tasks), iterating over model variants may take a while; i.e., there are several hyperparameters to tune for `TorchMPC`
2. Consider running on `cuda` for boosting inference time in hypersolver models
3. You may comment out the `tikzplotlib` imports and figure saving, since they require time and need extra installation of styles. Also, you may comment out the `plt.rcParams.update()` which are used for LaTeX style images but also require extra downloads
4. Other comments and instructions are available in the codes

## Feedback
Don't hesitate to contact us for any problem about reproducibility issues, bugs, suggestion etc. Your feedback is precious for us! :)