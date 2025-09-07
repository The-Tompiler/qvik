from typing import Optional
import torch as th
import numpy as np
import torchquantum as tq

class BQK(tq.QuantumModule):

  def __init__(self, eta:int, inputs:int, mode='fidelity', intial_state:float=0.0, verbose:bool=True) -> None:
    """BaseQuantumKernel Class Args:  
      eta (int): number of qubits
      mode (str): projector | fidelity | simulation (mode of kernel computation)
        fidelity supports batched execution and is executed using einsum
        projector is executed using bmm and might be slower for batched inputs
      inputs (int): input dimension (sigma), defaults to 1/2 of the available groups
      initial_state (th.tensor): initial state of the quantum device, defaults to ground state
      verbose (bool): print training progress, defaults to True"""
    super().__init__()
    self.eta = eta
    self.device = tq.QuantumDevice(n_wires=eta)
    self.inital_state = th.tensor([[intial_state]* 2 ** self.eta], dtype=th.complex64)
    self.verbose = verbose
    self.mode = mode
    self.inputs = inputs
    self.calls = 0
    



  def process(self, data, inverse=False):
    """Process the supplied data using self.device when using project mode"""
    raise NotImplementedError("Please Implement this method")
  

  def evolve(self, data):
    """Init device and evolve the data, supporting batched execution in fidelity mode"""
    raise NotImplementedError("Please Implement this method")
  

  def project(self, a, b):
    self.device.reset_op_history(); 
    self.device.set_states(self.inital_state)
    self.process(a); self.process(b, inverse=True)
    state = self.device.get_states_1d().view(-1)
    return th.abs(state[0]**2)
  

  def kernel(self, A:th.tensor, B:th.tensor):
    """Compute the kernel of batch inputs A and B with dim [batchsize, inputs]"""
    if self.mode == 'projector': 
      return th.cat([self.project(a,b) for b in B for a in A], dim=0).view(-1, A.shape[0], B.shape[0]) / 2
      k = [[self.project(a,b) for b in B] for a in A]
      if A.shape[0] + B.shape[0] == 2: return k[0][0][:, None, None]
      print(A.shape, B.shape, len(k), len(k[0]), len(k[0][0]))
      print(k[0][0].shape)
      return th.stack(k)
    elif self.mode == 'fidelity': 
      a, b = self.evolve(A), self.evolve(B)
      b = b.conj().permute(0, 2, 1) if len(b.shape) == 3 else b.conj().transpose(0, 1)
      return th.abs(th.matmul(a, b)) ** 2
    elif self.mode == 'simulation': raise NotImplementedError("Noise simulation currently not supported")
    else: assert False, f'{self.mode} not supported'


  def forward(self, A:th.tensor, B:Optional[th.tensor]=None): 
    self.calls += 1
    return self.kernel(A, A if B is None else B)
  

  def train(self, X:th.tensor, Y:th.tensor, loss:callable,  epochs:int=1000, 
            batch_size:Optional[int]=None,  callback:Optional[callable]=None, 
            log_interval:int=100):
    """ Train the kernel using the supplied data and loss function. Args:
      X, Y (th.tensor): input and  data
      loss (callable): loss function
      epochs (int): number of training epochs
      batch_size (int): batch size for training
      callback (callable): callback function for training
      log_interval (int): log interval for training progress"""
    
    assert self.optimizer, "Optimizer not set"; losses = []
    for ep in range(epochs):
      self.optimizer.zero_grad()
      if batch_size is not None: 
        batch_idx = np.random.choice(list(range(X.size(0))), batch_size)
        X, Y = X[batch_idx], Y[batch_idx]
      losses.append(loss(self, X, Y))
      losses[-1].backward()
      self.optimizer.step()
      if ep%log_interval==0: 
        if callback is not None: callback(self, losses[-1], ep)
        if self.verbose: print(f"KTA {-losses[-1]:.3f} @Step {ep}")
    return losses
