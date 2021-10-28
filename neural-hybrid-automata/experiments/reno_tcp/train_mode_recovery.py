import sys
sys.path.append('../../')
from absl import app, flags
from ml_collections.config_flags import config_flags

import torch
import torch.nn as nn
from torchdyn.numerics import *
from src.nn import ConditionedNHADecoder, NHADiscreteModeEncoder, NHA, \
                AugmentedNeuralODE, RegularDecoder, AugmentedDecoder, LatentEncoder, \
                DCNeuralODE, LatentODE

import numpy as np
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt
import pathlib
import wandb
from script_config import train_flows_config

from eval_utils import *
from dataset_utils import *


FLAGS = flags.FLAGS
config_flags.DEFINE_config_dict("config", config=train_flows_config)
flags.DEFINE_string("model", "NHA", "Model to evaluate [NODE, DCNODE, LatentNODE, NHA, NHANCDE]")
flags.DEFINE_string("system", "tcp", "System to train on")
flags.DEFINE_integer("seed", 1234, "Seed for reproducibility")
flags.DEFINE_integer("n_train_trajs", 5, "Number of training trajectories (single fold)")
flags.DEFINE_integer("n_val_folds", 5, "Number of folds for cross-validation")
flags.DEFINE_integer("n_test_trajs", 15, "Number of test trajectories")
flags.DEFINE_integer("n_modes", 10, "Number of NHA latent modes (or augmentation dims for baselines)")
flags.DEFINE_integer("add_emb_dims", 0, "Additional augmented dims for all models (non-mode related)")
flags.DEFINE_bool("test_result", True, "Whether to test final model. Turned on only after tuning on cross-val.")


def build_model(solver, n_modes, device):
    if FLAGS.model == 'NHA':
        f = nn.Sequential(nn.Linear(2 + n_modes, 2))
        dec = ConditionedNHADecoder(None, n_modes, FLAGS.config)
        if FLAGS.config.dropout:
            enc = nn.Sequential(nn.Linear(10, 64), nn.Dropout(0.3), nn.ReLU(),
                                     nn.Linear(64, 64), nn.Dropout(0.3), nn.ReLU(), nn.Linear(64, 64),
                                     nn.Dropout(0.3), nn.Tanh(), nn.Linear(64, n_modes, bias=False))
        else:
            enc = nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                                     nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32),
                                     nn.Tanh(), nn.Linear(32, n_modes, bias=False))
        model = NHA(NHADiscreteModeEncoder(enc, categorical_type=FLAGS.config.categorical_type,
                                           device=FLAGS.config.device), dec, solver, n_pts_enc=n_modes).to(device)


    elif FLAGS.model == 'NODE':
        f = nn.Sequential(nn.Linear(2+n_modes, 32), nn.SELU(), nn.Linear(32, 32), nn.SELU(), nn.Linear(32, 32),
                          nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2+n_modes))
        dec = RegularDecoder(f)
        model = AugmentedNeuralODE(dec, solver, dims_to_augment=n_modes).to(device)

    elif FLAGS.model == 'DCNODE':
        f = nn.Sequential(nn.Linear(2+n_modes, 2))
        dec = AugmentedDecoder(f, aug_dims=n_modes)
        if FLAGS.config.dropout:
            enc = nn.Sequential(nn.Linear(10, 64), nn.Dropout(0.3), nn.ReLU(),
                                     nn.Linear(64, 64), nn.Dropout(0.3), nn.ReLU(), nn.Linear(64, 64),
                                     nn.Dropout(0.3), nn.Tanh(), nn.Linear(64, n_modes, bias=False))
        else:
            enc = nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                                     nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32),
                                     nn.Tanh(), nn.Linear(32, n_modes, bias=False))
        model = DCNeuralODE(enc, dec, solver, n_pts_enc=3).to(device)

    elif FLAGS.model == 'LatentNODE':
        f = nn.Sequential(nn.Linear(2 + n_modes, 2))
        dec = AugmentedDecoder(f, aug_dims=n_modes)
        if FLAGS.config.dropout:
            enc = nn.Sequential(nn.Linear(10, 64), nn.Dropout(0.3), nn.ReLU(),
                                     nn.Linear(64, 64), nn.Dropout(0.3), nn.ReLU(), nn.Linear(64, 64),
                                     nn.Dropout(0.3), nn.Tanh(), nn.Linear(64, 2*n_modes, bias=False))
        else:
            enc = nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                                     nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32),
                                     nn.Tanh(), nn.Linear(32, 2*n_modes, bias=False))
        model = LatentODE(LatentEncoder(enc, n_modes=n_modes), dec, solver, n_pts_enc=3).to(device)

    p = f[-1].weight
    torch.nn.init.zeros_(p)
    p = f[-1].bias
    torch.nn.init.zeros_(p)
    return model



def main(argv):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    import os
    os.environ['WANDB_SILENT'] = "true"

    # save model folder
    model_path, plot_path = pathlib.Path('./checkpoints/tcp/'), pathlib.Path('./figures/tcp/')
    if not model_path.exists(): model_path.mkdir(parents=True)
    if not plot_path.exists(): plot_path.mkdir(parents=True)

    device = FLAGS.config.device

    data_sol_segments, data_t_segments, n_segments = \
        preprocess_data(FLAGS.n_train_trajs, FLAGS.n_val_folds, FLAGS.n_test_trajs)

    solver = RungeKutta4().to(device)

    # train and evaluate on each fold
    for k in range(FLAGS.n_val_folds):

        # model definition
        model = build_model(solver, FLAGS.n_modes, device)

        # encoder takes first three points
        x_ivps = torch.cat([traj_seg[:1, ..., :2] for traj_seg in data_sol_segments[k]], 0).to(device)

        # compute encoder feats
        x_feats = get_enc_feats(data_t_segments[k], data_sol_segments[k], FLAGS.model, device)
        gt_modes = torch.cat([traj_seg[:1, ..., 2:3] for traj_seg in data_sol_segments[k]], 0)[:,0,0].numpy()
        t_traj_segments = [t_.to(device) for t_ in data_t_segments[k]]
        solver.sync_device_dtype(x_feats, t_traj_segments[0])

        # initialize logging
        wandb.init(project="NHA_ready", name=f"""{FLAGS.model}_
                    modes:{FLAGS.n_modes}""", 
                    config=FLAGS.config
                )


        opt_enc = FLAGS.config["enc_optim"](list(model.enc.parameters()), lr=FLAGS.config["enc_lr"])
        opt_dec = FLAGS.config["dec_optim"](list(model.dec.parameters()), lr=FLAGS.config["dec_lr"])

        criterion = nn.MSELoss(reduction="none")

        vscore = 0.
        for epoch in range(FLAGS.config["epochs"]):
            x_perturbed_ivps = x_ivps
            if FLAGS.model in ["LatentNODE", "DCNODE", "NHA", "NHANCDE"]:
                sol, forward_t = model(x_perturbed_ivps, x_feats, t_traj_segments)
            else:
                sol, forward_t = model(x_perturbed_ivps, t_traj_segments)

            loss, smape = 0., 0.
            for sol_, target_ in zip(sol, data_sol_segments[k]):
                l = len(sol_)
                loss += criterion(sol_[:l, 0, :2], target_[:l, 0, :2].to(device).detach()).sum(1).mean()
                # compute sMAPE for logging purposes only
                with torch.no_grad():
                    target_ = target_[:l, 0, :2].to(device)
                    smape += ((sol_[:l, 0, :2] - target_).abs() / (sol_[:l, 0, :2].abs() +
                                                                   target_.abs() + 1e-2) / 2).sum(1).mean(0) * 100

            loss /= len(sol)
            smape /= len(sol)
            loss.backward(); opt_enc.step(); opt_dec.step()
            opt_enc.zero_grad(); opt_dec.zero_grad()


            if epoch % 10 == 0:
                with torch.no_grad():

                    fname = plot_path.name + f'train_it_{epoch}_fold_{k}.jpg'
                    if FLAGS.model in ["NHA", "NHANCDE"]:
                        q_parallel = model.encode(x_ivps, x_feats, keep_one_hot=True)
                        pred_modes = q_parallel[:,0,:].argmax(1).detach().cpu().numpy()
                        vscore = v_measure_score(gt_modes, pred_modes)
                    else:
                        q_parallel = torch.zeros(1)

                    fig = plot_predictions(fname, FLAGS.model, q_parallel, sol,
                                           data_sol_segments[k], data_t_segments[k])
                    plt.close()
                    img = wandb.Image(fig)
                    wandb.log({"vscore": vscore, "predictions": img}, commit=False)

                    print(
                        f"""Model: {FLAGS.model}, Tr. Fold: {k}, Epoch: {epoch}, Loss (MSE): {loss:.3f}, Loss (sMAPE): {smape:.3f}, V-Score: {vscore:.3f}, Forward simulation time: {forward_t:.3f}""")
                    wandb.log({"Training loss (MSE)": loss, "Training loss (sMAPE)": smape,
                               "Forward simulation time": forward_t})

        # validate on each fold
        folds_to_validate = list(range(FLAGS.n_val_folds))
        folds_to_validate.remove(k)

        val_losses = []
        for fold in folds_to_validate:
            print(f"Validating fold: {fold}")
            val_loss, smape, vscore = validate(model, FLAGS.model, criterion,
                                               data_sol_segments[fold], data_t_segments[fold], device)
            val_losses.append(val_loss)
            wandb.log({"Validation Loss (MSE)": val_loss, "Validation Loss (sMAPE)": smape, "Val V-Score:": vscore})
            print(f"""Model: {FLAGS.model}, Fold: {fold}, Epoch: {epoch}, Loss (MSE): {loss:.3f}""")

        torch.save(model.state_dict(), model_path.name + f'{FLAGS.model}_{FLAGS.n_modes}_{FLAGS.config.categorical_type}_{FLAGS.config.nonlinear}_{k}')


    ############### END OF PER-FOLD TRAINING #########################
    k = torch.stack(val_losses).argmin().item()
    model.load_state_dict(torch.load(model_path.name + f'{FLAGS.model}_{FLAGS.n_modes}_{FLAGS.config.categorical_type}_{FLAGS.config.nonlinear}_{k}'))
    model.eval()

    if FLAGS.test_result:
        test_loss, smape, vscore = validate(model, FLAGS.model,
                                            criterion, data_sol_segments[-1], data_t_segments[-1], device)
        wandb.log({"Test Loss (MSE)": test_loss, "Test Loss (sMAPE)": smape, "Test V-Score": vscore})


if __name__ == '__main__':
    app.run(main)