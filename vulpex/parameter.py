import numpy as np

class Parameter:
  def __init__(self, data):
    self.data = np.asarray(data)
    self.grad = np.zeros_like(self.data)

  def zero_grad(self):
    self.grad.fill(0.0)