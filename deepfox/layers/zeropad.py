import numpy as np
from .base import Layer

class ZeroPad1D(Layer):
  def __init__(self, padding):
    self.padding = padding

    if isinstance(padding, int):
      self.padding = (padding, padding)

  def forward(self, x):
    x = np.asarray(x)

    return np.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[1])))
  
  def backward(self, grad):
    grad = np.asarray(grad)
    end = -self.padding[1] if self.padding[1] > 0 else None
    return grad[:, :, self.padding[0]:end]
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "ZeroPad1D",
      "padding": self.padding
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(padding={self.padding})"
  
class ZeroPad2D(Layer):
  def __init__(self, padding):
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding))
    elif isinstance(padding, tuple) and len(padding) == 2:
      if isinstance(padding[0], int):
        self.padding = ((padding[0], padding[0]), (padding[1], padding[1]))
      else:
        self.padding = padding
    else:
      raise ValueError("padding must be an int, tuple of 2 ints, or tuple of 2 tuples.")

  def forward(self, x):
    x = np.asarray(x)
    return np.pad(x, ((0, 0), (0, 0), self.padding[0], self.padding[1]))

  def backward(self, grad):
    grad = np.asarray(grad)
    h_end = -self.padding[0][1] if self.padding[0][1] > 0 else None
    w_end = -self.padding[1][1] if self.padding[1][1] > 0 else None
    return grad[:, :, self.padding[0][0]:h_end, self.padding[1][0]:w_end]

  def parameters(self):
    return []

  def get_config(self):
    return {
      "type": "ZeroPad2D",
      "padding": self.padding
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(padding={self.padding})"

class ZeroPad3D(Layer):
  def __init__(self, padding):
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding), (padding, padding))
    elif isinstance(padding, tuple) and len(padding) == 3:
      if isinstance(padding[0], int):
        self.padding = ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]))
      else:
        self.padding = padding
    else:
      raise ValueError("padding must be an int, tuple of 3 ints, or tuple of 3 tuples.")

  def forward(self, x):
    x = np.asarray(x)
    return np.pad(x, ((0, 0), (0, 0), self.padding[0], self.padding[1], self.padding[2]))

  def backward(self, grad):
    grad = np.asarray(grad)
    d_end = -self.padding[0][1] if self.padding[0][1] > 0 else None
    h_end = -self.padding[1][1] if self.padding[1][1] > 0 else None
    w_end = -self.padding[2][1] if self.padding[2][1] > 0 else None
    return grad[:, :, self.padding[0][0]:d_end, self.padding[1][0]:h_end, self.padding[2][0]:w_end]

  def parameters(self):
    return []

  def get_config(self):
    return {
      "type": "ZeroPad3D",
      "padding": self.padding
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(padding={self.padding})"