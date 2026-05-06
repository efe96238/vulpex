import numpy as np
from .base import Layer
from ..utils import get_rng

# Maybe add spatial Dropout later (1D, 2D, 3D)

class Dropout(Layer):
  def __init__(self, p=0.5):
    if not 0 <= p < 1:
      raise ValueError(f"Dropout probability must be in [0, 1), got {p}.")
    self.p = p
  
  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if self.training:
      self.mask = (get_rng().random(x.shape) > self.p)
      return x * self.mask / (1 - self.p)
    
    else:
      return x
  
  def backward(self, grad):
    return grad * self.mask / (1 - self.p)
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "Dropout",
      "p": self.p
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(p={self.p})"