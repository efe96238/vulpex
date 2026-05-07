import numpy as np
from .layers import Layer
from .parameter import Parameter

class Sigmoid(Layer):
  def forward(self, x):
    x = np.asarray(x)
    self.out = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return self.out

  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * self.out * (1 - self.out)

  def parameters(self):
    return []

class Tanh(Layer):
  def forward(self, x):
    x = np.asarray(x)
    self.out = np.tanh(x)
    return self.out

  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * (1 - self.out ** 2)

  def parameters(self):
    return []

class Softmax(Layer):
  def forward(self, x):
    x = np.asarray(x)

    if x.ndim == 1:
      x = x.reshape(1, -1)

    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return self.out

  def backward(self, grad):
    grad = np.asarray(grad)

    if grad.ndim == 1:
      grad = grad.reshape(1, -1)

    dot = np.sum(grad * self.out, axis=1, keepdims=True)
    return self.out * (grad - dot)

  def parameters(self):
    return []
  
class LogSoftmax(Layer):
  def forward(self, x):
    x = np.asarray(x)

    if x.ndim == 1:
      x = x.reshape(1, -1)

    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    self.softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    self.out = shifted - np.log(np.sum(exp_x, axis=1, keepdims=True))
    return self.out
  
  def backward(self, grad):
    grad = np.asarray(grad)

    if grad.ndim == 1:
      grad = grad.reshape(1, -1)

    return grad - self.softmax * np.sum(grad, axis=1, keepdims=True)
  
  def parameters(self):
    return []
  
class ReLU(Layer):
  def forward(self, x):
    x = np.asarray(x)
    self.x = x
    return np.maximum(0, x)

  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * (self.x > 0)

  def parameters(self):
    return []
  
class LeakyReLU(Layer):
  def __init__(self, alpha=0.01):
    self.alpha = alpha

  def forward(self, x):
    x = np.asarray(x)
    self.x = x
    return np.where(x > 0, x, self.alpha * x)

  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * np.where(self.x > 0, 1, self.alpha)
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "LeakyReLU",
      "alpha": self.alpha
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(alpha={self.alpha})"
  
class GeLU(Layer):
  def forward(self, x):
    x = np.asarray(x)
    self.x = x
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
  
  def backward(self, grad):
    grad = np.asarray(grad)
    a = np.sqrt(2 / np.pi) * (self.x + 0.044715 * self.x ** 3)
    tanh_a = np.tanh(a)
    da = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.x ** 2)
    dgelu = 0.5 * (1 + tanh_a) + 0.5 * self.x * (1 - tanh_a ** 2) * da
    return grad * dgelu
  
  def parameters(self):
    return []
  
class SiLU(Layer):
  def forward(self, x):
    x = np.asarray(x)
    self.x = x
    self.sigmoid = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return x * self.sigmoid
  
  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * (self.sigmoid * (1 + self.x * (1 - self.sigmoid)))
  
  def parameters(self):
    return []
  
class ELU(Layer):
  def __init__(self, alpha=1.0):
    self.alpha = alpha

  def forward(self, x):
    x = np.asarray(x)
    self.x = x
    return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
  
  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * np.where(self.x > 0, 1, self.alpha * np.exp(self.x))
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "ELU",
      "alpha": self.alpha
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(alpha={self.alpha})"
  
class PReLU(Layer):
  def __init__(self, init_alpha=0.25):
    self.alpha = Parameter(np.array([[init_alpha]]))

  def forward(self, x):
    x = np.asarray(x)
    self.x = x
    return np.where(x > 0, x, self.alpha.data * x)
  
  def backward(self, grad):
    grad = np.asarray(grad)
    self.alpha.grad = np.sum(grad * np.where(self.x > 0, 0, self.x), keepdims=True)
    return grad * np.where(self.x > 0, 1, self.alpha.data)
  
  def parameters(self):
    return [self.alpha]
  
  def get_config(self):
    return {
      "type": "PReLU",
      "init_alpha": float(self.alpha.data[0, 0])
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(alpha={float(self.alpha.data[0, 0]):.4f})"
  
class SELU(Layer):
  def __init__(self):
    self.lmbda = 1.0507 # lambda is a reserved keyword
    self.alpha = 1.6733

  def forward(self, x):
    x = np.asarray(x)
    self.x = x
    return np.where(x > 0, self.lmbda * x, self.lmbda * self.alpha * (np.exp(x) - 1))
  
  def backward(self, grad):
    grad = np.asarray(grad)
    return grad * np.where(self.x > 0, self.lmbda, self.lmbda * self.alpha * np.exp(self.x))
  
  def parameters(self):
    return []