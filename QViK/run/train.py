import os 
import csv
import time

import numpy as np
import torch as th

from sklearn.svm import SVC

from config import *
from metrics import *
from kernels import *



def smooth(data, window=40):
  padded_data = np.pad(data, (window//2, window-1-window//2), mode='edge')
  smoothed_data = np.convolve(padded_data, np.ones(window)/window, mode='valid')
  return smoothed_data.tolist()


def fit(kernel, D, config, add_loss=False):
  X_train, X_test, y_train, y_test = D
  with th.no_grad():
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    if add_loss: metrics['Train/Loss'] = - target_alignment(kernel, X_train, y_train)
  return metrics


def train(kernel, D, config, log_interval=100):
  X_train, X_test, y_train, y_test = D
  metrics = {k: [v] for k,v in fit(kernel, D, config).items()}
  loss = lambda K,X,y: -target_alignment(K, X, y)
  callback = lambda K, l, s: [metrics[k].extend(log_interval*(v,)) for k,v in fit(K, D, config).items()]
  if kernel.aggregation == "lin" or kernel.aggregation == "k2a":
    losses = kernel.train(X_train, y_train, loss, callback=callback, log_interval=log_interval, **config.train_kwargs)
    metrics = { 'Train/Loss': [l.detach() for l in losses],
      **{k: smooth(v, 2*log_interval) for k,v in metrics.items()} }
  return metrics


def log_time(config, seeds, time):
  if not os.path.exists('logs/runs.csv'): 
    with open(r'logs/runs.csv', 'w') as f: csv.writer(f).writerow(['config', 'seeds', 'time'])
  with open(r'logs/runs.csv', 'a') as f: csv.writer(f).writerow([config, seeds, time])  


def execute(config:dict, seeds:list[int], training:callable, **kwargs):
  metrics = []; start = time.process_time()
  for seed in seeds:
    D, kernel, cfg = setup({**BASE(seed), **config})
    metrics.append(training(kernel, D, cfg, **kwargs))
  metrics = {k: np.array([m[k] for m in metrics]) for k in metrics[0]}
  log_time(config, seeds, start - time.process_time())
  return metrics, kernel, D, cfg
