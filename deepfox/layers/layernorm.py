import numpy as np
from .base import Layer
from ..parameter import Parameter

class LayerNorm(Layer):
  def __init__(self, normalized_shape, eps=1e-5):
    self.normalized_shape = normalized_shape
    self.eps = eps

    if isinstance(normalized_shape, int):
      self.normalized_shape = (normalized_shape,)

    self.gamma = Parameter(np.ones(self.normalized_shape))
    self.beta = Parameter(np.zeros(self.normalized_shape))

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    axes = tuple(range(-len(self.normalized_shape), 0))

    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)

    self.x_centered = x - mean
    self.std_inv = 1.0 / np.sqrt(var + self.eps)
    self.x_norm = self.x_centered * self.std_inv

    return self.gamma.data * self.x_norm + self.beta.data
  
  def backward(self, grad):
    grad = np.asarray(grad)
    
    axes = tuple(range(-len(self.normalized_shape), 0))
    batch_axes = tuple(range(len(self.x.shape) - len(self.normalized_shape)))
    
    self.beta.grad = np.sum(grad, axis=batch_axes)
    self.gamma.grad = np.sum(grad * self.x_norm, axis=batch_axes)

    N = np.prod(self.normalized_shape)

    dx_norm = grad * self.gamma.data
    dx = (1.0 / N) * self.std_inv * (N * dx_norm - np.sum(dx_norm, axis=axes, keepdims=True) - self.x_norm * np.sum(dx_norm * self.x_norm, axis=axes, keepdims=True))

    return dx
  
  def parameters(self):
    return [self.gamma, self.beta]
  
  def get_config(self):
    return {
      "type": "LayerNorm",
      "normalized_shape": self.normalized_shape,
      "eps": self.eps
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(normalized_shape={self.normalized_shape}, eps={self.eps})"