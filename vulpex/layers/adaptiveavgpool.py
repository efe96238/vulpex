import numpy as np
from .base import Layer

class AdaptiveAvgPool1D(Layer):
  def __init__(self, output_size):
    self.output_size = output_size

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if x.ndim != 3:
      raise ValueError(f"AdaptiveAvgPool1D expects 3D input (batch, channels, length), got {x.ndim}D.")

    batch_size, channels, length = x.shape

    if length < self.output_size:
      raise ValueError(f"Input length ({length}) must be >= output_size ({self.output_size}).")

    out = np.zeros((batch_size, channels, self.output_size))

    for b in range(batch_size):
      for c in range(channels):
        for i in range(self.output_size):
          start = i * length // self.output_size
          end = (i + 1) * length // self.output_size

          window = x[b, c, start:end]
          out[b, c, i] = np.mean(window)

    self.input_length = length
    return out

  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, channels, _ = grad.shape

    dx = np.zeros_like(self.x)

    for b in range(batch_size):
      for c in range(channels):
        for i in range(self.output_size):
          start = i * self.input_length // self.output_size
          end = (i + 1) * self.input_length // self.output_size

          window_size = end - start
          dx[b, c, start:end] += grad[b, c, i] / window_size

    return dx

  def parameters(self):
    return []

  def get_config(self):
    return {
      "type": "AdaptiveAvgPool1D",
      "output_size": self.output_size
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(output_size={self.output_size})"

class AdaptiveAvgPool2D(Layer):
  def __init__(self, output_size):
    self.output_size = output_size

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if x.ndim != 4:
      raise ValueError(f"AdaptiveAvgPool2D expects 4D input (batch, channels, height, width), got {x.ndim}D.")

    batch_size, channels, height, width = x.shape

    if height < self.output_size or width < self.output_size:
      raise ValueError(f"Input spatial dims ({height}x{width}) must be >= output_size ({self.output_size}).")

    out = np.zeros((batch_size, channels, self.output_size, self.output_size))

    for b in range(batch_size):
      for c in range(channels):
        for i in range(self.output_size):
          for j in range(self.output_size):
            h_start = i * height // self.output_size
            h_end = (i + 1) * height // self.output_size

            w_start = j * width // self.output_size
            w_end = (j + 1) * width // self.output_size

            window = x[b, c, h_start:h_end, w_start:w_end]
            out[b, c, i, j] = np.mean(window)

    self.input_height = height
    self.input_width = width
    return out

  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, channels, _, _ = grad.shape

    dx = np.zeros_like(self.x)

    for b in range(batch_size):
      for c in range(channels):
        for i in range(self.output_size):
          for j in range(self.output_size):
            h_start = i * self.input_height // self.output_size
            h_end = (i + 1) * self.input_height // self.output_size

            w_start = j * self.input_width // self.output_size
            w_end = (j + 1) * self.input_width // self.output_size

            window_size = (h_end - h_start) * (w_end - w_start)
            dx[b, c, h_start:h_end, w_start:w_end] += grad[b, c, i, j] / window_size

    return dx

  def parameters(self):
    return []

  def get_config(self):
    return {
      "type": "AdaptiveAvgPool2D",
      "output_size": self.output_size
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(output_size={self.output_size})"
  
class AdaptiveAvgPool3D(Layer):
  def __init__(self, output_size):
    self.output_size = output_size

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if x.ndim != 5:
      raise ValueError(f"AdaptiveAvgPool3D expects 5D input (batch, channels, depth, height, width), got {x.ndim}D.")

    batch_size, channels, depth, height, width = x.shape

    if depth < self.output_size or height < self.output_size or width < self.output_size:
      raise ValueError(f"Input spatial dims ({depth}x{height}x{width}) must be >= output_size ({self.output_size}).")

    out = np.zeros((batch_size, channels, self.output_size, self.output_size, self.output_size))

    for b in range(batch_size):
      for c in range(channels):
        for d in range(self.output_size):
          for i in range(self.output_size):
            for j in range(self.output_size):
              d_start = d * depth // self.output_size
              d_end = (d + 1) * depth // self.output_size

              h_start = i * height // self.output_size
              h_end = (i + 1) * height // self.output_size

              w_start = j * width // self.output_size
              w_end = (j + 1) * width // self.output_size

              window = x[b, c, d_start:d_end, h_start:h_end, w_start:w_end]
              out[b, c, d, i, j] = np.mean(window)

    self.input_depth = depth
    self.input_height = height
    self.input_width = width
    return out

  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, channels, _, _, _ = grad.shape

    dx = np.zeros_like(self.x)

    for b in range(batch_size):
      for c in range(channels):
        for d in range(self.output_size):
          for i in range(self.output_size):
            for j in range(self.output_size):
              d_start = d * self.input_depth // self.output_size
              d_end = (d + 1) * self.input_depth // self.output_size

              h_start = i * self.input_height // self.output_size
              h_end = (i + 1) * self.input_height // self.output_size

              w_start = j * self.input_width // self.output_size
              w_end = (j + 1) * self.input_width // self.output_size

              window_size = (d_end - d_start) * (h_end - h_start) * (w_end - w_start)
              dx[b, c, d_start:d_end, h_start:h_end, w_start:w_end] += grad[b, c, d, i, j] / window_size

    return dx

  def parameters(self):
    return []

  def get_config(self):
    return {
      "type": "AdaptiveAvgPool3D",
      "output_size": self.output_size
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(output_size={self.output_size})"