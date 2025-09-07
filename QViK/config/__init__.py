import numpy as np; import torch as th
from config.data import BLOBS, MNIST, CIFAR, make_dataset
from config.kernels import QViK, make_kernel

BASE = lambda seed: {'seed': seed, 'verbose': False}

# Colors = 
green = '#2E9961'
blue = '#1795CC'
darkblue = '#173ACC'
red = '#CC173A'
yellow = '#B2CC17'
orange = '#CC9017'
pltargs = {'marker': 'o', 'linestyle': '-'}    

class Config:
  def __init__(self, config:dict):
    assert isinstance(config, dict)
    for key, val in config.items():
      if isinstance(val, (list, tuple)): setattr(self, key, [x for x in val])
      else: setattr(self, key, val)
  def __setitem__(self, key, value): setattr(self, key, value)


def setup(config):
  """Setup the provided config. Returns:
  D=(X_train, X_test, y_train, y_test) and kernel"""
  config = Config(config)
  np.random.seed(config.seed); th.manual_seed(config.seed)
  return make_dataset(config), make_kernel(config), config
