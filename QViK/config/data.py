import torch as th; import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

size = { 'n_samples': 200,  'test_size': 0.1 } # 1000

BLOBS = lambda n_classes=10: { 'dataset': 'blobs', 'data_kwargs': { **size, 'n_features': 3, 'centers': n_classes}}
MNIST = lambda n_classes=10: { 'dataset': 'mnist', 'data_kwargs': { **size, 'n_classes': n_classes}}
CIFAR = lambda n_classes=10: { 'dataset': 'cifar', 'data_kwargs': { **size, 'n_classes': n_classes}}


def fetch(dataset, n_samples=10, n_classes=2):
    X, y = fetch_openml(dataset, version=1, return_X_y=True, as_frame=False)
    idxs = [np.argwhere(y == str(i))[:int(n_samples/n_classes)] for i in range(n_classes)]
    X = np.concatenate([X[idx].squeeze() for idx in idxs], axis=0) / 255.0  # normalize
    y = np.concatenate([y[idx].squeeze() for idx in idxs], axis=0).astype(int)
    return X, y


def _make(dataset, seed, test_size=0.2, **kwargs):
    if dataset in ['blobs']: kwargs['random_state'] = seed
    dcls = { 'blobs': make_blobs, 'mnist': 'mnist_784', 'cifar': 'CIFAR_10'}[dataset]
    f, kwargs = (dcls, kwargs) if callable(dcls) else (fetch, {**kwargs, 'dataset': dcls})
    return [th.tensor(d) for d in train_test_split(*f(**kwargs), test_size=test_size)]


def make_dataset(config):
  if config.dataset == 'mnist' and config.kernel == 'QViK': 
    config.kernel_kwargs['grayscale'] = True
  if config.kernel == 'HEE': D = pca(config.kernel_kwargs['eta'], *D)
  if config.kernel == "rbf" or config.kernel == "linear":
     config.kernel_kwargs = {}
  D = _make(config.dataset, config.seed, **config.data_kwargs)
  config.kernel_kwargs['inputs'] = D[0].size(1)
  return D


# TODO: check if needed
def pca(dataset_dim, x_train, x_test, y_train, y_test):
    from sklearn.decomposition import PCA
    feature_mean = th.mean(x_train, axis=0)
    scikit_pca = PCA(n_components=dataset_dim)
    x_train = scikit_pca.fit_transform(x_train - feature_mean)
    x_test = scikit_pca.transform(x_test - feature_mean)
    y_test, y_train = (y_test + 1) / 2, (y_train + 1) / 2
    return *[th.tensor(d) for d in [x_train, x_test]], y_train.int(), y_test.int()

