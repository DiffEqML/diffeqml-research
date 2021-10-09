import torch
import ml_collections

def get_train_flows_config():
  config = ml_collections.ConfigDict()

  # training
  config.enc_optim = torch.optim.Adam
  config.enc_lr = 5e-4
  config.dec_optim = torch.optim.Adam
  config.dec_lr = 1e-2
  config.epochs = 4000
  config.norm_trajectories = False

  # model setup
  config.categorical_type = "STE" # whether to use Softmax, Sparsemax or Categorical STE ['softmax', 'STE', 'sparsemax']
  config.nonlinear = False # linear or nonlinear decoder
  config.dropout = True


  config.device = torch.device("cuda:0")

  return config


def get_train_flows_noise_config():
  config = ml_collections.ConfigDict()

  # training
  config.enc_optim = torch.optim.Adam
  config.enc_lr = 5e-4
  config.dec_optim = torch.optim.Adam
  config.dec_lr = 1e-2
  config.epochs = 4000
  config.norm_trajectories = False
  config.corruption_probability = 0.1
  config.corruption_intensity = 10

  # model setup
  config.categorical_type = "STE" # whether to use Softmax, Sparsemax or Categorical STE ['softmax', 'STE', 'sparsemax']
  config.nonlinear = False # linear or nonlinear decoder
  config.dropout = True

  config.device = torch.device("cuda:1")

  return config


train_flows_config = get_train_flows_config()
train_flows_noise_config = get_train_flows_noise_config()