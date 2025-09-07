from sklearn.svm._classes import BaseLibSVM

def make_kernel(config):
  """Make a kernel based on the config"""
  if config.kernel in BaseLibSVM._sparse_kernels: return config.kernel
  import kernels
  if config.kernel not in dir(kernels): raise ValueError(f"Unknown kernel: {config.kernel}")
  return getattr(kernels, config.kernel)(**config.kernel_kwargs, verbose=config.verbose)
  

QViK = lambda eta, aggregation='sum', projection=0, epochs=1000, mode='fidelity': {
  'kernel': 'QViK',
  'kernel_kwargs': {
    'eta': eta,
    'aggregation': aggregation,
    'projection': projection,
    'mode': mode,
  },
  'train_kwargs': {
    'epochs': epochs, 
    'batch_size': None
  }
}
