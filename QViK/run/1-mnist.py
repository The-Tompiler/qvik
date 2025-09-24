from config import *
from run.train import execute, train, fit
from run.plot import *

# data = MNIST(2); seeds = list(range(42,50)); steps = 10
data = CIFAR(10); seeds = list(range(4)); steps = 100

for label, (alg, run, args, cfg) in {
  'QViK (ours)': (QViK(4, 'sum', epochs=steps), train, {'log_interval': 20 }, (blue, True, False)),
  'RBF       ': ({'kernel': 'rbf'}, fit, {}, (yellow, False, True)),
  'Linear    ': ({'kernel': 'linear'}, fit, {}, (orange, False, True)),
}.items(): write(label, *cfg, data, steps, execute({**data, **alg}, seeds, run, **args)[0]) 
