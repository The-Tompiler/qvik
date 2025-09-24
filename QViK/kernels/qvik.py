import torch as th; import numpy as np
from torch.optim import Adam, SGD

import torchquantum as tq
import torchquantum.functional as tqf

from math import sqrt 
from scipy.special import factorial

from kernels.base import BQK
from kernels.k2a import K2A

from run.plot import make_histogram


class QViK(BQK):
  def __init__(self, eta:int, inputs:int, grayscale=False, projection:int=0, 
               aggregation='mean', optimizer='Adam', lr=0.1, 
               mode='fidelity', fast_exp:bool=True, verbose:bool=False, ) -> None:
    """Quantum Vision Kernel Args:  
        eta (int): number of qubits
        inputs (int): input dimension
        grayscale (bool): if True, input is grayscale (1 channel), otherwise RGB (3 channels)
        projection (int): generator stride when mapping inputs to generators, defaults to 0 (no stride)
        aggregation (str): mean | sum | max | lin | K2A (aggregation method)
        optimizer (str): Adam | SGD (optimizer for K2A or linear aggregation)
        lr (float): learning rate for the optimizer, defaults to 0.1
        mode (str): projector | fidelity | simulation (mode of kernel computation)
          fidelity supports batched execution and is executed using einsum
          projector is executed using bmm and might be slower for batched inputs
        verbose (bool): print training progress, defaults to True
        fast_exp (bool):  use fast matrix exponential computation (i.e., e^∑-i*h_i*p_i over ∏e^-i*h_i*p_i) (default: True)"""
    super().__init__(eta, inputs, mode=mode, intial_state=0.5, verbose=verbose)
    self.fast_exp = fast_exp
    self.comp_method = 'bmm' if mode == 'projector' else 'einsum'

    # Determine dimension of the Hilbert space for eta qubits and number of generators
    self.eta = eta; _H = 2 ** self.eta; self.generators = _H ** 2 - 1

    colors = 1 if grayscale else 3 # color depth
    s = sqrt(self.inputs / colors) # linear patch size
    assert s%1 == 0, 'Input is not square'    

    p = int(sqrt(self.generators / colors))
    while (s % p) != 0: p -= 1                                      # is this really the best option 
    self._p = (p, p, colors); 
    self._i = (int(s), int(s), colors)
    n = int(inputs / np.prod(self._p))

    if verbose: print(f"Fitted patch dimensions: {n}x{p}x{p} ({np.prod(self._p[:2])}, Overlap: {s%p}) [{self.generators - np.prod(self._p)} / {self.generators} idle]")
    self.H = self.build_generators(projection)
    self.optimizer = None
    self.aggregation = aggregation
    
    if aggregation == 'mean': self.aggregate = lambda x: th.mean(x, dim=0)
    elif aggregation == 'sum': self.aggregate = lambda x: th.sum(x, dim=0)
    elif aggregation == 'max': self.aggregate = lambda x: th.max(x, dim=0).values
    elif aggregation == 'min': self.aggregate = lambda x: th.min(x, dim=0).values
    elif aggregation == 'singlepatch': self.aggregate = lambda x: x[3]
    elif aggregation == 'K2A': 
      self.aggregate = K2A(embed_dim=n, num_heads=int(sqrt(n)), symmetrize=False, verbose=verbose)
      self.optimizer = eval(optimizer)(self.aggregate.parameters(), lr=lr)
    elif aggregation == 'lin': 
      self.weights = th.nn.Parameter(th.randn(n, dtype=th.float32))
      self.aggregate = lambda x: th.einsum('pij,p->ij', x, self.weights)
      self.optimizer = eval(optimizer)(self.parameters(), lr=lr)
    else: assert False, f'Aggregation {aggregation} not supported'

    test_generatorset = self.build_structured_generators(4,20)


  def patch(self, data):
    # Calculate number of patches along each spatial dimension
    data = data.view(data.shape[0], *self._i)

    # Rearrange data into patches
    data = data.unfold(1, self._p[0], self._p[0]).unfold(2, self._p[1], self._p[1])

    # Shape (n_samples, n_patches_w, n_patches_h, w_patch, h_patch, c)
    data = data.permute(0, 1, 2, 4, 5, 3).contiguous()

    # Linearize batch and patch dimensions and append H dimensions
    n = np.prod(data.shape[1:3]) # number of patches
    data = data.view(-1, np.prod(self._p), 1, 1) # (n_samples x n_patches, flat_patch, 1, 1)
    return data, n


  def project(self, a, b):
    A, B = self.patch(a[None,:])[0], self.patch(b[None,:])[0]
    K = th.zeros(A.shape[0])
    for p in range(A.shape[0]): K[p] = super().project(A[p], B[p])
    return K


  def process(self, data, inverse=False):
    self.phi = data[None,:]
    self.execute(self.device, range(self.eta), inverse=inverse)
    if inverse: [self.device.h(i) for i in range(self.eta)]


  def evolve(self, data):
    """Init device and evolve the data, supporting batched execution in fidelity mode"""
    self.phi, p = self.patch(data)
    qc = tq.QuantumDevice(self.eta, bsz=data.shape[0] * p)
    [tqf.h(qc, i) for i in range(self.eta)]
    self.execute(qc, range(self.eta))
    states = qc.get_states_1d()
    states = states.view(p, data.shape[0], states.shape[-1])
    return states
  

  def execute(self, qdev, wires, inverse=False):
    """Forward the OpHamilExp module. Args:
      qdev: The QuantumDevice.
      wires: The wires. """
    # TODO: rm else case?
    if self.fast_exp: 
      tqf.qubitunitaryfast(qdev, wires, params=self.U.to(qdev.device), 
                          inverse=inverse, comp_method=self.comp_method)
    else: 
      [tqf.qubitunitaryfast(qdev, wires, params=U.to(qdev.device), 
                            inverse=inverse, comp_method=self.comp_method) 
        for U in (self.U.flip(dims=(0,)) if inverse else self.U)]
 

  def kernel(self, A:th.tensor, B:th.tensor):
    """Kernel extension to handle patches and aggregation."""
    result = self.aggregate(super().kernel(A, B))
    #make_histogram(data= np.array(result).flatten())
    return result


  def build_generators(self, projection:int):
    """Build the VGGs from the generators. """
    H = th.zeros(1, np.prod(self._p), 2 ** self.eta, 2 ** self.eta, dtype=th.complex128) 
    
    # Construct the set of generators for the VGGs
    norm = lambda i: sqrt(factorial(i)); _H = 2 ** self.eta
    G = [
      *[[((i,j),(j,i)), (1+0j, 1+0j)] for i in range(_H - 1) for j in range(i + 1, _H)],
      *[[((i,j),(j,i)), (0+1j, 0-1j)] for i in range(_H - 1) for j in range(i + 1, _H)],
      *[[2*((*(range(i+1)),),), (*(i)*(1/norm(i),), *[-i/norm(i)])] for i in range(1, _H)]
    ]

    w = 2**projection
    idx = [g % self.generators for g in range(0, self.generators*w,w)]
    
    for g in range(np.prod(self._p)):
      H[0][g][G[idx[g]][0]] = th.tensor(G[idx[g]][1], dtype=th.complex128)

    assert all([th.allclose(h, th.conj_physical(h.T)) for h in H[0]])  
    return H


  def build_structured_generators(self, number_of_patches, inputsize):
    
    entries_per_patch = inputsize//number_of_patches
    assert inputsize/number_of_patches%1.0 ==0
    # current problem is we created matrices which are too large
    self.patch_eta = 0
    while (2**self.patch_eta)**2-1 < entries_per_patch:
      self.patch_eta += 1
    combinations = self.patch_eta*(self.patch_eta-1)
    self.total_eta = number_of_patches*self.patch_eta
    first_gens = th.zeros(number_of_patches, entries_per_patch, 2 ** self.total_eta, 2 ** self.total_eta, dtype=th.complex128) 
    list_of_onequbit_generators = [th.tensor([[1,0],[0,1]]), th.tensor([[0,1],[1,0]]),th.tensor([[0,1j],[-1j,0]]),th.tensor([[1,0],[0,-1]])]

    #in case the results of this approach are bad, use only the reauli pauli matrices not the identity in the list
    for singlepatch in range(number_of_patches):
      patchqubits = [i for i in range(singlepatch*self.patch_eta, (singlepatch+1)*self.patch_eta)];
      for entry in range(entries_per_patch):
        entry_counter = entry 
        for current_eta in range(1, self.total_eta):
          if current_eta in patchqubits:
            index = patchqubits.index(current_eta)
            if 0 in patchqubits:
              current_generator = list_of_onequbit_generators[entry_counter//(4**(len(patchqubits)-index-1))]
            else:
              current_generator= th.kron(current_generator, list_of_onequbit_generators[entry_counter//(4**(len(patchqubits)-index-1))])
            entry_counter -= (entry_counter//(4**(len(patchqubits)-index-1))) * (4**(len(patchqubits)-index-1))
          else:
            if 0 in patchqubits:
              current_generator = list_of_onequbit_generators[0]
            else:
              current_generator= th.kron(current_generator, list_of_onequbit_generators[0])
        first_gens[singlepatch,entry] =current_generator

    return first_gens
  
  @property
  def exponent_matrix(self): 
    matrix = self.H * -1j * self.phi
    if self.fast_exp: return matrix.sum(dim=1)
    return matrix
  

  @property 
  def U(self): return th.matrix_exp(self.exponent_matrix)
