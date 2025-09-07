import numpy as np
import torch as th
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import scipy


def target_alignment(kernel, X, Y):
    """Kernel-target alignment between kernel and labels."""
    K = kernel(X)
    y = Y.view(-1, 1)
    T = (y == y.T).float()
    inner_product = th.sum(K * T)
    norm = th.sqrt(th.sum(K * K) * th.sum(T * T))
    return inner_product / norm

evaluate = lambda model, X, y: { f'Test/{key}': metric(X, y) 
  for key, metric in {
    'Accuracy': lambda X,y: accuracy_score(model.predict(X), y),
    'L1': lambda X,y: mean_absolute_error(model.predict(X), y),
    'L2': lambda X,y: mean_squared_error(model.predict(X), y),
    **({'KTA': lambda X,y: target_alignment(model.kernel, X, y)} if callable(model.kernel) else {}),
  }.items()
}


def CI(data, confidence=0.95):
  return range(data.shape[1]), *np.clip(scipy.stats.t.interval(
     confidence=confidence, 
     df=data.shape[0], 
     loc=np.mean(data, axis=0), 
     scale=scipy.stats.sem(data, axis=0)
  ),0,1)
