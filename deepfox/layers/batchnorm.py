import numpy as np
from .base import Layer
from ..parameter import Parameter

class BatchNorm1D(Layer):
  def __init__(self, num_features, momentum=0.1, eps=1e-5):
    self.num_features = num_features
    self.momentum = momentum
    self.eps = eps

    self.gamma = Parameter(np.ones((1, num_features)))
    self.beta = Parameter(np.zeros((1, num_features)))
    self.running_mean = np.zeros((1, num_features))
    self.running_var = np.ones((1, num_features))

  def forward(self, x):
    x = np.asarray(x)
    self.input_shape = x.shape

    if x.ndim not in (2, 3):
      raise ValueError(f"BatchNorm1D expects 2D (batch, features) or 3D (batch, channels, length) input, got {x.ndim}D.")

    features = x.shape[1]
    if features != self.num_features:
      raise ValueError(f"Expected {self.num_features} features/channels, got {features}.")

    if x.ndim == 3:
      batch, channels, length = x.shape
      x = x.transpose(0, 2, 1).reshape(-1, channels)

    if self.training:
      mean = np.mean(x, axis=0, keepdims=True)
      var = np.var(x, axis=0, keepdims=True)
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
    else:
      mean = self.running_mean
      var = self.running_var

    self.x_centered = x - mean
    self.std_inv = 1.0 / np.sqrt(var + self.eps)
    self.x_norm = self.x_centered * self.std_inv

    out = self.gamma.data * self.x_norm + self.beta.data

    if len(self.input_shape) == 3:
      batch, channels, length = self.input_shape
      out = out.reshape(batch, length, channels).transpose(0, 2, 1)

    return out
  
  def backward(self, grad):
    grad = np.asarray(grad)
    N = self.x_norm.shape[0]

    if len(self.input_shape) == 3:
      batch, channels, length = self.input_shape
      grad = grad.transpose(0, 2, 1).reshape(-1, channels)

    self.beta.grad = np.sum(grad, axis=0, keepdims=True)
    self.gamma.grad = np.sum(grad * self.x_norm, axis=0, keepdims=True)

    dx_norm = grad * self.gamma.data
    dx = (1.0 / N) * self.std_inv * (N * dx_norm - np.sum(dx_norm, axis=0, keepdims=True) - self.x_norm * np.sum(dx_norm * self.x_norm, axis=0, keepdims=True))

    if len(self.input_shape) == 3:
      batch, channels, length = self.input_shape
      dx = dx.reshape(batch, length, channels).transpose(0, 2, 1)
    
    return dx
  
  def parameters(self):
    return [self.gamma, self.beta]
  
  def get_config(self):
    return {
      "type": "BatchNorm1D",
      "num_features": self.num_features,
      "momentum": self.momentum,
      "eps": self.eps
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(num_features={self.num_features}, momentum={self.momentum}, eps={self.eps})"
  
class BatchNorm2D(Layer):
  def __init__(self, num_features, momentum=0.1, eps=1e-5):
    self.num_features = num_features
    self.momentum = momentum
    self.eps = eps

    self.gamma = Parameter(np.ones((1, num_features)))
    self.beta = Parameter(np.zeros((1, num_features)))
    self.running_mean = np.zeros((1, num_features))
    self.running_var = np.ones((1, num_features))

  def forward(self, x):
    x = np.asarray(x)
    self.input_shape = x.shape

    if x.ndim != 4:
      raise ValueError(f"BatchNorm2D expects 4D input (batch, channels, height, width), got {x.ndim}D.")

    channels = x.shape[1]
    if channels != self.num_features:
      raise ValueError(f"Expected {self.num_features} channels, got {channels}.")

    if x.ndim == 4:
      batch, channels, height, width = x.shape
      x = x.transpose(0, 2, 3, 1).reshape(-1, channels)

    if self.training:
      mean = np.mean(x, axis=0, keepdims=True)
      var = np.var(x, axis=0, keepdims=True)
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
    else:
      mean = self.running_mean
      var = self.running_var

    self.x_centered = x - mean
    self.std_inv = 1.0 / np.sqrt(var + self.eps)
    self.x_norm = self.x_centered * self.std_inv

    out = self.gamma.data * self.x_norm + self.beta.data

    if len(self.input_shape) == 4:
      batch, channels, height, width = self.input_shape
      out = out.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)

    return out
  
  def backward(self, grad):
    grad = np.asarray(grad)
    N = self.x_norm.shape[0]

    if len(self.input_shape) == 4:
      batch, channels, height, width = self.input_shape
      grad = grad.transpose(0, 2, 3, 1).reshape(-1, channels)

    self.beta.grad = np.sum(grad, axis=0, keepdims=True)
    self.gamma.grad = np.sum(grad * self.x_norm, axis=0, keepdims=True)

    dx_norm = grad * self.gamma.data
    dx = (1.0 / N) * self.std_inv * (N * dx_norm - np.sum(dx_norm, axis=0, keepdims=True) - self.x_norm * np.sum(dx_norm * self.x_norm, axis=0, keepdims=True))

    if len(self.input_shape) == 4:
      batch, channels, height, width = self.input_shape
      dx = dx.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)
    
    return dx
  
  def parameters(self):
    return [self.gamma, self.beta]
  
  def get_config(self):
    return {
      "type": "BatchNorm2D",
      "num_features": self.num_features,
      "momentum": self.momentum,
      "eps": self.eps
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(num_features={self.num_features}, momentum={self.momentum}, eps={self.eps})"
  
class BatchNorm3D(Layer):
  def __init__(self, num_features, momentum=0.1, eps=1e-5):
    self.num_features = num_features
    self.momentum = momentum
    self.eps = eps

    self.gamma = Parameter(np.ones((1, num_features)))
    self.beta = Parameter(np.zeros((1, num_features)))
    self.running_mean = np.zeros((1, num_features))
    self.running_var = np.ones((1, num_features))

  def forward(self, x):
    x = np.asarray(x)
    self.input_shape = x.shape

    if x.ndim != 5:
      raise ValueError(f"BatchNorm3D expects 5D input (batch, channels, depth, height, width), got {x.ndim}D.")

    channels = x.shape[1]
    if channels != self.num_features:
      raise ValueError(f"Expected {self.num_features} channels, got {channels}.")

    if x.ndim == 5:
      batch, channels, depth, height, width = x.shape
      x = x.transpose(0, 2, 3, 4, 1).reshape(-1, channels)

    if self.training:
      mean = np.mean(x, axis=0, keepdims=True)
      var = np.var(x, axis=0, keepdims=True)
      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
    else:
      mean = self.running_mean
      var = self.running_var

    self.x_centered = x - mean
    self.std_inv = 1.0 / np.sqrt(var + self.eps)
    self.x_norm = self.x_centered * self.std_inv

    out = self.gamma.data * self.x_norm + self.beta.data

    if len(self.input_shape) == 5:
      batch, channels, depth, height, width = self.input_shape
      out = out.reshape(batch, depth, height, width, channels).transpose(0, 4, 1, 2, 3)

    return out
  
  def backward(self, grad):
    grad = np.asarray(grad)
    N = self.x_norm.shape[0]

    if len(self.input_shape) == 5:
      batch, channels, depth, height, width = self.input_shape
      grad = grad.transpose(0, 2, 3, 4, 1).reshape(-1, channels)

    self.beta.grad = np.sum(grad, axis=0, keepdims=True)
    self.gamma.grad = np.sum(grad * self.x_norm, axis=0, keepdims=True)

    dx_norm = grad * self.gamma.data
    dx = (1.0 / N) * self.std_inv * (N * dx_norm - np.sum(dx_norm, axis=0, keepdims=True) - self.x_norm * np.sum(dx_norm * self.x_norm, axis=0, keepdims=True))

    if len(self.input_shape) == 5:
      batch, channels, depth, height, width = self.input_shape
      dx = dx.reshape(batch, depth, height, width, channels).transpose(0, 4, 1, 2, 3)
    
    return dx
  
  def parameters(self):
    return [self.gamma, self.beta]
  
  def get_config(self):
    return {
      "type": "BatchNorm3D",
      "num_features": self.num_features,
      "momentum": self.momentum,
      "eps": self.eps
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(num_features={self.num_features}, momentum={self.momentum}, eps={self.eps})"