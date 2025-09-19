import numpy as np
import matplotlib.pyplot as plt
from metrics import CI



# Uncomment for md table overview
# print(f"| Dataset | Model      | Test Accuracy | Kernel Target Alignment |")
# print(f"| ------- | ---------- | ------------- | ----------------------- |")
# prnt = lambda mean, ci: f" | {mean[-1]:.3f} ± {(ci[2][-1]-ci[1][-1])/2:.3f}"
prnt = lambda mean, ci: f" & ${mean[-1]:.2f} \pm {(ci[2][-1]-ci[1][-1])/2:.2f}$"

def plot(axes, label, color, kta, const, data, steps, results, X=None):
  # summary = f"| {data['dataset']:7s} | {label}"
  fill = lambda data: np.repeat(np.expand_dims(data, axis=1), 2, axis=1) if const and not X else data
  metrics = [(lambda x: fill(x), 'Test/Accuracy'), *([(lambda x: fill(-x), 'Train/Loss')] if kta else [])]
  for i, (f, key) in enumerate(metrics):
    mean, ci = f(results[key]).mean(axis=0), CI(f(results[key])); #summary += prnt(mean, ci)
    if X is None: 
      if 'Accuracy' in key: print(prnt(mean, ci),end=" ")
      if const:
        axes[i].plot((0,steps), mean, label=label, color=color, linestyle='--')
        axes[i].fill_between((0,steps), *ci[1:], color=color, alpha=.1)
      else: axes[i].plot(mean, label=label, color=color); axes[i].fill_between(*ci, color=color, alpha=.1)
    else: 
      # TODO sort by X
      # Step 1: Get the sorted indices from the list
      idx = np.argsort(X); mean, ci, X = mean[idx], (ci[0], ci[1][idx], ci[2][idx]), [X[i] for i in idx]

      # Step 2: Apply to all arrays/lists
      # a_sorted = a[sorted_indices]
      # b_sorted = b[sorted_indices]
      # c_sorted = [c[i] for i in sorted_indices]
      axes[i].errorbar( 
        X, mean, yerr=[mean - ci[1], ci[2] - mean], label=label, color=color, 
        fmt='o', linestyle='--' if const else '-', capsize=3)
      # TODO: error bars instead of fill_between
      # axes[i].fill_between(X, *ci[1:], color=color, alpha=.1)
  # print(summary+' |')



# def write(label, color, kta, const, data, steps, results):
def write(label, *args, suffix=''):
  import json, os
  # import pandas as pd
  data = dict(zip(('color', 'kta', 'const', 'data', 'steps', 'results'), args))
  data['results'] = {k: v.tolist() for k,v in data['results'].items()}
  path = f'logs/{data["data"]["dataset"]}/'
  if 'blobs' in data['data']['dataset']: 
    path = f"{path[:-1]}-{data['data']['data_kwargs']['n_features']}_{data['data']['data_kwargs']['centers']}/"
  if not os.path.exists(path): os.makedirs(path)
  # pd.DataFrame(data).to_json(f'{path}/{label}.json')
  with open(f'{path}/{label}{suffix}.json', 'w') as f: json.dump(data, f)


def make_fig(X_label='Epochs'):
  fig, axes = plt.subplots(1, 2, figsize=(8, 3))
  axes[0].set_ylabel('Test Accuracy')
  axes[1].set_ylabel('Kernel Target Alignment')
  axes[0].set_xlabel(X_label); axes[1].set_xlabel(X_label)
  return fig, axes

def make_histogram(data, bins = 100):
  plt.hist(x=data, bins= bins)
  plt.xlabel("Binned Matrixentry Values")
  plt.ylabel("Number of Entries")
  plt.show()

def save_fig(fig, axes, path):
  axes[0].legend(loc="lower right")
  # axes[0].set_ylim([0, 1])
  fig.tight_layout()
  fig.savefig(path)


def fetch(dataset):
  p = f"logs/{dataset}/"
  # load = lambda path: pd.read_json(open(path, 'r'))
  load = lambda path: json.load(open(path, 'r'))
  data = {f[:-5]: dict(load(p+f)) for f in os.listdir(p) if f.endswith(".json")}
  data = {f: { **d, 'results': {k: np.array(v) for k,v in d['results'].items()}} for f, d in data.items()}
  data = dict(sorted(data.items(), key=lambda item: next((i for i, o in enumerate(
    ['QViK (ours)', 'QEK', 'HEE Linear', 'QViK Static', 'HEE PCA', 'HEE', 'RBF', 'Linear']
  ) if o.strip() in item[0]), 7)))
  return data


def _stack(a, b, X): 
    """Helper function to merge two dictionaries recursively"""
    merged = {}; 
    # fn = lambda v: v[:,-1] if len(v.shape)==2 else v
    # TODO return merged keys / merged values when mergin lists
    for k, v in a.items():
      # if k in keys: merged[k] = np.stack((fn(v), fn(b[k]))).T
      if k == 'random_state': continue
      if isinstance(v, dict): merged[k] = _stack(v, b[k], X)
      elif isinstance(v, np.ndarray): 
        if a[k].shape == b[k].shape: merged[k] = np.stack((a[k][:,-1], b[k][:,-1]) if len(a[k].shape) == 2 else (a[k],b[k])).T
        elif a[k].shape[0] == b[k].shape[0]:
          merged[k] = np.hstack((a[k][:,-1][:,np.newaxis], b[k]) if len(a[k].shape) == 2 else (a[k][:,np.newaxis], b[k]))
      elif isinstance(b[k], list): assert k in X; merged[k] = [v, *b[k]]; X[k] = merged[k] 
      elif v != b[k]: merged[k] = [v, b[k]]; X[k] = merged[k] 
      else: merged[k] = v
    return merged


def merge(dataset, key):
  datasets = [d for d in os.listdir('logs/') if os.path.isdir(f'logs/{d}') and dataset + '-' in d]

  # Find unique entries for non-viz keys, filter into multiple subsets and save separately
  
  raw_data = [fetch(d) for d in datasets]; all_keys = {}
  filtered = lambda k,v: [d for d in raw_data if next(iter(d.values()))['data']['data_kwargs'][k] == v]
  reduce(lambda x, y: _stack(y, x, all_keys), raw_data)
  # print(data)
  collection = {}
  print(all_keys)

  if len(all_keys) == 1 and key in all_keys:
    label = f"{dataset}-{key}"; X = {}
    data = reduce(lambda x, y: _stack(y, x, X), raw_data)
    if len(X): collection[label] = (X[key], data)
  
  for k, val in all_keys.items():
    if k == key: continue
    for v in set(val):
      label = f"{dataset}-{key}-{v}_{k}"; X = {}
      data = reduce(lambda x, y: _stack(y, x, X), filtered(k,v))
      if len(X): collection[label] = (X[key], data)

  return collection


# Load data and generate plots 
if __name__ == "__main__":
  import os; import json; import sys
  from functools import reduce
  
  dataset = sys.argv[1] if len(sys.argv) > 1 else 'circles'  # Default value if no argument is provided
  if len(sys.argv) == 3: 
    for label, (X, data) in merge(dataset, sys.argv[2]).items():
      print(f"Plotting {label} ({X})")
      fig, axes = make_fig(X_label=sys.argv[2])
      [plot(axes, f, **d, X=X) for f, d in data.items()]
      save_fig(fig, axes, f"logs/{label}.pdf")
  else:

    data = fetch(dataset)

    fig, axes = make_fig()
    print(f" & {'& '.join(data.keys())}\\{dataset}", end=" ")
    # print(dataset, end=" ")
    [plot(axes, f, **d) for f, d in data.items()]
    save_fig(fig, axes, f"logs/{dataset}.pdf")
    print("\\\\")
