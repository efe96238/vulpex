import numpy as np
from ..parameter import Parameter
from .base import Layer
from ..utils import get_rng

class Linear(Layer):
  def __init__(self, in_features, out_features, bias=True):
    self.in_features = in_features
    self.out_features = out_features

    limit = np.sqrt(2.0 / in_features)

    self.weights = Parameter(
      get_rng().standard_normal((in_features, out_features)) * limit
    )

    if bias:
      self.bias = Parameter(np.zeros((1, out_features)))
    else:
      self.bias = None

  def forward(self, x):
    self.x = np.asarray(x)

    if self.x.ndim < 2:
      raise ValueError(f"Linear expects at least 2D input (batch_size, features), got {self.x.ndim}D.")

    if self.x.shape[-1] != self.in_features:
      raise ValueError(f"Expected input features {self.in_features}, got {self.x.shape[-1]}.")

    out = self.x @ self.weights.data
    if self.bias is not None:
      out += self.bias.data
    return out

  def backward(self, grad):
    grad = np.asarray(grad)

    self.weights.grad = self.x.T @ grad
    if self.bias is not None:
      self.bias.grad = np.sum(grad, axis=0, keepdims=True)

    return grad @ self.weights.data.T

  def parameters(self):
    if self.bias is not None:
      return [self.weights, self.bias]
    return [self.weights]

  def get_config(self):
    return {
      "type": "Linear",
      "in_features": self.in_features,
      "out_features": self.out_features,
      "bias": self.bias is not None
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"