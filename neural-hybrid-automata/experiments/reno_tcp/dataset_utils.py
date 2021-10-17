" Preprocessing utilities, including event detection (segment subplitting) and stacking "

import torch
import pyro.distributions as dist
import glob


def preprocess_data(n_train_trajs, n_val_folds, n_test_trajs, norm_trajs=False,
                    noisy_segmentation=False, corruption_probability=0.5, corruption_intensity=20):

    n_total_trajs = n_train_trajs * n_val_folds + n_test_trajs

    # regular sorting vs alphanumeric sorting does not matter as long as the sorting strategy is the same
    # across all models
    sol_files, t_eval_files = sorted(glob.glob('../../data/tcp/raw_sol*')), \
                              sorted(glob.glob('../../data/tcp/raw_t_eval*'))


    assert len(sol_files) >= n_total_trajs, "Need more simulations for given training / validation split"

    sol_folds_files = [sol_files[k*n_train_trajs:(k+1)*n_train_trajs] for k in range(n_val_folds)]
    t_folds_files = [t_eval_files[k*n_train_trajs:(k+1)*n_train_trajs] for k in range(n_val_folds)]
    sol_test_f = sol_files[-n_test_trajs:]
    t_test_f = t_eval_files[-n_test_trajs:]

    # process folds, then test
    n_segments = []
    data_sol_segments, data_t_segments = [], []

    for k in range(n_val_folds):
        loaded_trajs  = [torch.load(fsol) for fsol in sol_folds_files[k]]
        loaded_t_trajs = [torch.load(fsol) for fsol in t_folds_files[k]]

        if norm_trajs:
            # compute normalization statistics
            max_trajs = torch.max(torch.cat([torch.max(traj[..., :2], 0)[0] for traj in loaded_trajs]), 0)[0]
            min_trajs = torch.min(torch.cat([torch.min(traj[..., :2], 0)[0] for traj in loaded_trajs]), 0)[0]
            loaded_trajs = [torch.cat([(traj[..., :2] - min_trajs) / (max_trajs - min_trajs), traj[..., 2:]], -1)
                                 for traj in loaded_trajs]

        sol, t_eval = [], []
        for sol_true, t_eval_true in zip(loaded_trajs, loaded_t_trajs):
            sol.append(sol_true)
            if len(t_eval) == 0: t_eval.append(t_eval_true)
            # ensure time increases between simulations: this step makes subsequent processing operations
            # to run Neural CDE decoders in batch more convenient, see `get_enc_feats`
            else: t_eval.append(t_eval_true + t_eval[-1][-1])
        sol, t_eval = torch.cat(sol), torch.cat(t_eval)
        if noisy_segmentation:
            sol_segments, t_segments = noisy_segment_flows(sol, t_eval, corruption_probability, corruption_intensity)
        else:
            sol_segments, t_segments = segment_flows(sol, t_eval)
        data_sol_segments.append(sol_segments)
        n_segments.append(len(sol_segments))
        data_t_segments.append(t_segments)

    sol, t_eval = [], []
    for file_sol, file_eval in zip(sol_test_f, t_test_f):
        t_eval_true, sol_true = torch.load(file_eval), torch.load(file_sol)
        sol.append(sol_true)
        if len(t_eval) == 0: t_eval.append(t_eval_true)
        else: t_eval.append(t_eval_true + t_eval[-1][-1])  # ensure time increases between simulations
    sol, t_eval = torch.cat(sol), torch.cat(t_eval)
    sol_segments, t_segments = segment_flows(sol, t_eval)
    data_sol_segments.append(sol_segments)
    n_segments.append(len(sol_segments))
    data_t_segments.append(t_segments)
    return data_sol_segments, data_t_segments, n_segments


def segment_flows(sol, t_eval):
    q_traj, w_traj = sol[:, 0, 2], sol[:, 0, 0]
    switch_idxs = []
    for i in range(len(q_traj) - 1):
        if torch.logical_or(q_traj[i] != q_traj[i - 1],
                            torch.logical_and((w_traj[i] - w_traj[i - 1]).abs() >= w_traj[i - 1] / 2,
                                              w_traj[i - 1] > 0)):
            switch_idxs.append(i)

    if switch_idxs[0] > 0:
        sol_segments = [sol[:switch_idxs[0]]]
        t_segments = [t_eval[:switch_idxs[0]]]
    else:
        sol_segments, t_segments = [], []


    for i, switch_idx in enumerate(switch_idxs[:-1]):
        sol_segments.append(sol[switch_idx:switch_idxs[i + 1]])
        t_segments.append(t_eval[switch_idx:switch_idxs[i + 1]])
    return sol_segments, t_segments


def noisy_segment_flows(sol, t_eval, corruption_prob=0.5, corruption_intensity=20):

    bernoulli = dist.Bernoulli(corruption_prob)
    coin_flip = dist.Bernoulli(0.5)
    q_traj, w_traj = sol[:, 0, 2], sol[:, 0, 0]
    switch_idxs = []
    prev_idx = torch.tensor([0.])
    next_idx = len(q_traj) - 1
    for i in range(len(q_traj) - 1):
        if torch.logical_or(q_traj[i] != q_traj[i - 1],
                            torch.logical_and((w_traj[i] - w_traj[i - 1]).abs() >= w_traj[i - 1] / 2,
                                              w_traj[i - 1] > 0)):
            if bernoulli.sample_n(1) == 1:
                # introduce noise
                shift_noise = torch.randint(0, corruption_intensity, (1,))
                sign = coin_flip.sample_n(1)

                # clip for first vals
                neg_shift = torch.clamp(i - shift_noise, prev_idx.item(), i)
                pos_shift = torch.clamp(i + shift_noise, i, next_idx)
                if sign == 1:
                    switch_idxs.append(neg_shift.data)
                else:
                    switch_idxs.append(pos_shift.data)
            else:
                switch_idxs.append(torch.tensor([i]))

            prev_idx = switch_idxs[-1]

    switch_idxs = torch.cat(switch_idxs)
    if switch_idxs[0] > 0:
        sol_segments = [sol[:switch_idxs[0]]]
        t_segments = [t_eval[:switch_idxs[0]]]
    else:
        sol_segments, t_segments = [], []

    for i, switch_idx in enumerate(switch_idxs[:-1]):
        sol_traj = sol[switch_idx:switch_idxs[i + 1]]
        if len(sol_traj) > 0:
            sol_segments.append(sol[switch_idx:switch_idxs[i + 1]])
            t_segments.append(t_eval[switch_idx:switch_idxs[i + 1]])
    return sol_segments, t_segments


def get_unique_and_idxs(arr):
    unique, inverse = torch.unique(arr, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    idxs = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, idxs


def get_enc_feats(data_t_segments, data_sol_segments, model_type, device):
    padded_feats = torch.nn.utils.rnn.pad_sequence(data_sol_segments).to(device)[:, :, 0, :2]
    x_feats = padded_feats[:5, :, :2].permute(1, 0, 2).reshape(-1, 10)
    return x_feats