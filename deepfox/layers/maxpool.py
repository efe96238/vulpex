import numpy as np
from .base import Layer

class MaxPool1D(Layer):
  def __init__(self, kernel_size, stride=None, padding=0):
    self.kernel_size = kernel_size
    self.stride = stride if stride is not None else kernel_size
    self.padding = padding

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if x.ndim != 3:
      raise ValueError(f"MaxPool1D expects 3D input (batch, channels, length), got {x.ndim}D.")

    batch_size, channels, input_len = x.shape

    if input_len + 2 * self.padding < self.kernel_size:
      raise ValueError(f"Input length ({input_len}) + padding ({self.padding}) is smaller than kernel_size ({self.kernel_size}).")

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)), constant_values=-np.inf)
    else:
      x_padded = x

    self.x_padded = x_padded

    out_len = (input_len + 2 * self.padding - self.kernel_size) // self.stride + 1

    out = np.zeros((batch_size, channels, out_len))
    self.max_indices = np.zeros((batch_size, channels, out_len), dtype=int)

    for b in range(batch_size):
      for c in range(channels):
        for i in range(out_len):
          start = i * self.stride
          end = start + self.kernel_size

          window = x_padded[b, c, start:end]

          out[b, c, i] = np.max(window)
          self.max_indices[b, c, i] = start + np.argmax(window)

    return out

  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, channels, out_len = grad.shape

    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for c in range(channels):
        for i in range(out_len):
          dx_padded[b, c, self.max_indices[b, c, i]] += grad[b, c, i]

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding]
    else:
      dx = dx_padded

    return dx
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "MaxPool1D",
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
  
class MaxPool2D(Layer):
  def __init__(self, kernel_size, stride=None, padding=0):
    self.kernel_size = kernel_size
    self.stride = stride if stride is not None else kernel_size
    self.padding = padding

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if x.ndim != 4:
      raise ValueError(f"MaxPool2D expects 4D input (batch, channels, height, width), got {x.ndim}D.")

    batch_size, channels, height, width = x.shape

    if height + 2 * self.padding < self.kernel_size or width + 2 * self.padding < self.kernel_size:
      raise ValueError(f"Input spatial dims ({height}x{width}) + padding ({self.padding}) are smaller than kernel_size ({self.kernel_size}).")

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), constant_values=-np.inf)
    else:
      x_padded = x

    self.x_padded = x_padded

    out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

    out = np.zeros((batch_size, channels, out_height, out_width))
    self.max_h_indices = np.zeros((batch_size, channels, out_height, out_width), dtype=int)
    self.max_w_indices = np.zeros((batch_size, channels, out_height, out_width), dtype=int)

    for b in range(batch_size):
      for c in range(channels):
        for i in range(out_height):
          for j in range(out_width):
            h_start = i * self.stride
            h_end = h_start + self.kernel_size

            w_start = j * self.stride
            w_end = w_start + self.kernel_size

            window = x_padded[b, c, h_start:h_end, w_start:w_end]

            out[b, c, i, j] = np.max(window)

            h_offset, w_offset = divmod(np.argmax(window), self.kernel_size)

            self.max_h_indices[b, c, i, j] = h_start + h_offset
            self.max_w_indices[b, c, i, j] = w_start + w_offset

    return out
  
  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, channels, out_height, out_width = grad.shape

    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for c in range(channels):
        for i in range(out_height):
          for j in range(out_width):
            dx_padded[b, c, self.max_h_indices[b, c, i, j], self.max_w_indices[b, c, i, j]] += grad[b, c, i, j]

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    else:
      dx = dx_padded

    return dx
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "MaxPool2D",
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
  
class MaxPool3D(Layer):
  def __init__(self, kernel_size, stride=None, padding=0):
    self.kernel_size = kernel_size
    self.stride = stride if stride is not None else kernel_size
    self.padding = padding

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if x.ndim != 5:
      raise ValueError(f"MaxPool3D expects 5D input (batch, channels, depth, height, width), got {x.ndim}D.")

    batch_size, channels, depth, height, width = x.shape

    if depth + 2 * self.padding < self.kernel_size or height + 2 * self.padding < self.kernel_size or width + 2 * self.padding < self.kernel_size:
      raise ValueError(f"Input spatial dims ({depth}x{height}x{width}) + padding ({self.padding}) are smaller than kernel_size ({self.kernel_size}).")

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)), constant_values=-np.inf)
    else:
      x_padded = x

    self.x_padded = x_padded

    out_depth = (depth + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

    out = np.zeros((batch_size, channels, out_depth, out_height, out_width))
    self.max_d_indices = np.zeros((batch_size, channels, out_depth, out_height, out_width), dtype=int)
    self.max_h_indices = np.zeros((batch_size, channels, out_depth, out_height, out_width), dtype=int)
    self.max_w_indices = np.zeros((batch_size, channels, out_depth, out_height, out_width), dtype=int)

    for b in range(batch_size):
      for c in range(channels):
        for d in range(out_depth):
          for i in range(out_height):
            for j in range(out_width):
              d_start = d * self.stride
              d_end = d_start + self.kernel_size

              h_start = i * self.stride
              h_end = h_start + self.kernel_size

              w_start = j * self.stride
              w_end = w_start + self.kernel_size

              window = x_padded[b, c, d_start:d_end, h_start:h_end, w_start:w_end]

              out[b, c, d, i, j] = np.max(window)

              flat_idx = np.argmax(window)
              d_offset = flat_idx // (self.kernel_size * self.kernel_size)
              remainder = flat_idx % (self.kernel_size * self.kernel_size)
              h_offset, w_offset = divmod(remainder, self.kernel_size)

              self.max_d_indices[b, c, d, i, j] = d_start + d_offset
              self.max_h_indices[b, c, d, i, j] = h_start + h_offset
              self.max_w_indices[b, c, d, i, j] = w_start + w_offset

    return out
  
  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, channels, out_depth, out_height, out_width = grad.shape

    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for c in range(channels):
        for d in range(out_depth):
          for i in range(out_height):
            for j in range(out_width):
              dx_padded[b, c, self.max_d_indices[b, c, d, i, j], self.max_h_indices[b, c, d, i, j], self.max_w_indices[b, c, d, i, j]] += grad[b, c, d, i, j]

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding]
    else:
      dx = dx_padded

    return dx
  
  def parameters(self):
    return []
  
  def get_config(self):
    return {
      "type": "MaxPool3D",
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"