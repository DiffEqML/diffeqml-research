" Plotting and evaluation functions "
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import v_measure_score
from dataset_utils import get_enc_feats


def plot_predictions(fname, model_type, latent_mode_parallel, model_sol,
                     data_sol_segments, data_t_segments, n_plot=80):
    sol_plot = model_sol[:n_plot]
    t_traj_segments_plot = data_t_segments[:n_plot]
    sol_traj_segments_plot = data_sol_segments[:n_plot]


    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(311)
    for t_, s_ in zip(t_traj_segments_plot, sol_plot):
        ax.plot(t_.cpu(), s_[:, 0, 0].cpu().detach(), c='b')

    for t_, s_ in zip(t_traj_segments_plot, sol_traj_segments_plot):
        ax.plot(t_.cpu(), s_[:, 0, 0].cpu().detach(), c='r')
        ax.vlines(t_[-1].cpu(), 0, 5, color='black', linestyle='--', alpha=0.2)
    ax.set_ylabel('$w(t)$')

    ax = fig.add_subplot(312)
    for t_, s_ in zip(t_traj_segments_plot, sol_plot):
        ax.plot(t_.cpu(), s_[:, 0, 1].cpu().detach(), c='b')

    for t_, s_ in zip(t_traj_segments_plot, sol_traj_segments_plot):
        ax.plot(t_.cpu(), s_[:, 0, 1].cpu().detach(), c='r')
        ax.vlines(t_[-1].cpu(), 0, 5, color='black', linestyle='--', alpha=0.2)
    ax.set_ylabel('$s(t)$')

    if model_type in ["NHA", "NHANCDE"]:
        mode0_parallel_plot = latent_mode_parallel[:n_plot]

        mode0_parallel_plot = [mode0_.argmax() * torch.ones_like(sol_) \
                               for mode0_, sol_ in zip(mode0_parallel_plot, sol_plot)]

        ax = fig.add_subplot(313)
        for t_, s_ in zip(t_traj_segments_plot, mode0_parallel_plot):
            ax.plot(t_.cpu().detach(), s_[:, 0].cpu().detach(), c='black')
            ax.vlines(t_[-1].cpu(), 0, 13, color='black', linestyle='--', alpha=0.2)
        #ax.set_ylim(-0.1, 6)
        ax.legend(["Latent mode"])

    #plt.savefig(fname)
    return fig


def validate(model, model_type, criterion, data_sol, data_t, device):
    model = model.eval()
    x_ivps = torch.cat([traj_seg[:1, ..., :2] for traj_seg in data_sol], 0).to(device)
    gt_modes = torch.cat([traj_seg[:1, ..., 2:3] for traj_seg in data_sol], 0)[:, 0, 0].numpy()
    t_traj_segments = [t_.to(device) for t_ in data_t]
    x_feats = get_enc_feats(data_t, data_sol, model_type, device)

    if model_type in ["NHA", "NHACDE"]:
        q_parallel = model.encode(x_ivps, x_feats, keep_one_hot=True)
        pred_modes = q_parallel[:,0,:].argmax(1).detach().cpu().numpy()
        vscore = v_measure_score(gt_modes, pred_modes)
    else: vscore = 0.

    if model_type in ["LatentNODE", "DCNODE", "NHA", "NHANCDE"]:
        sol, forward_t = model(x_ivps, x_feats, t_traj_segments)
    else:
        sol, forward_t = model(x_ivps, t_traj_segments)

    loss, smape = 0., 0.
    for sol_, target_ in zip(sol, data_sol):
        l = len(sol_)
        loss += criterion(sol_[:l, 0, :2].to(device), target_[:l, 0, :2].to(device).detach()).sum(1).mean()
        target_ = target_[:l, 0, :2].to(device)
        smape += ((sol_[:l, 0, :2] - target_).abs() / (sol_[:l, 0, :2].abs() + target_.abs() + 1e-2) / 2).sum(1).mean(
            0) * 100
    loss /= len(sol)
    smape /= len(sol)
    model = model.train()
    return loss, smape, vscore
