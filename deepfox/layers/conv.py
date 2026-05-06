import numpy as np
from .base import Layer
from ..parameter import Parameter
from ..utils import get_rng

class Conv1D(Layer):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    fan_in = in_channels * kernel_size
    limit = np.sqrt(2.0 / fan_in)

    self.weights = Parameter(
      get_rng().standard_normal((out_channels, in_channels, kernel_size)) * limit
    )

    if bias:
      self.bias = Parameter(np.zeros((1, out_channels)))
    else:
      self.bias = None

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if x.ndim != 3:
      raise ValueError(f"Conv1D expects 3D input (batch_size, channels, length), got {x.ndim}D.")

    batch_size, in_channels, input_len = x.shape

    if in_channels != self.in_channels:
      raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}.")

    effective_len = input_len + 2 * self.padding
    if effective_len < self.kernel_size:
      raise ValueError(f"Input length ({input_len}) + padding ({self.padding}) is smaller than kernel_size ({self.kernel_size}).")

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)))
    else:
      x_padded = x

    self.x_padded = x_padded

    out_len = (input_len - self.kernel_size + 2 * self.padding) // self.stride + 1

    out = np.zeros((batch_size, self.out_channels, out_len))

    for b in range(batch_size):
      for oc in range(self.out_channels):
        for i in range(out_len):
          start = i * self.stride
          end = start + self.kernel_size

          window = x_padded[b, :, start:end]

          kernel = self.weights.data[oc]
          value = np.sum(window * kernel)
          if self.bias is not None:
            value += self.bias.data[0, oc]

          out[b, oc, i] = value

    return out

  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, _, out_len = grad.shape

    dweights = np.zeros_like(self.weights.data)
    if self.bias is not None:
      dbias = np.zeros_like(self.bias.data)
    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for oc in range(self.out_channels):
        for i in range(out_len):
          start = i * self.stride
          end = start + self.kernel_size

          window = self.x_padded[b, :, start:end]

          dweights[oc] += grad[b, oc, i] * window
          if self.bias is not None:
            dbias[0, oc] += grad[b, oc, i]
          dx_padded[b, :, start:end] += grad[b, oc, i] * self.weights.data[oc]

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding]
    else:
      dx = dx_padded

    self.weights.grad = dweights
    if self.bias is not None:
      self.bias.grad = dbias

    return dx

  def parameters(self):
    if self.bias is not None:
      return [self.weights, self.bias]
    return [self.weights]

  def get_config(self):
    return {
      "type": "Conv1D",
      "in_channels": self.in_channels,
      "out_channels": self.out_channels,
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding,
      "bias": self.bias is not None
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None})"

class Conv2D(Layer):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    fan_in = in_channels * kernel_size * kernel_size
    limit = np.sqrt(2.0 / fan_in)

    self.weights = Parameter(
      get_rng().standard_normal((out_channels, in_channels, kernel_size, kernel_size)) * limit
    )

    if bias:
      self.bias = Parameter(np.zeros((1, out_channels)))
    else:
      self.bias = None

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if x.ndim != 4:
      raise ValueError(f"Conv2D expects 4D input (batch_size, channels, height, width), got {x.ndim}D.")

    batch_size, in_channels, height, width = x.shape

    if in_channels != self.in_channels:
      raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}.")

    effective_h = height + 2 * self.padding
    effective_w = width + 2 * self.padding
    if effective_h < self.kernel_size or effective_w < self.kernel_size:
      raise ValueError(f"Input spatial dims ({height}x{width}) + padding ({self.padding}) are smaller than kernel_size ({self.kernel_size}).")

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
    else:
      x_padded = x

    self.x_padded = x_padded

    out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
    out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

    out = np.zeros((batch_size, self.out_channels, out_height, out_width))

    for b in range(batch_size):
      for oc in range(self.out_channels):
        for i in range(out_height):
          for j in range(out_width):
            h_start = i * self.stride
            h_end = h_start + self.kernel_size

            w_start = j * self.stride
            w_end = w_start + self.kernel_size

            window = x_padded[b, :, h_start:h_end, w_start:w_end]

            kernel = self.weights.data[oc]
            value = np.sum(window * kernel)
            if self.bias is not None:
              value += self.bias.data[0, oc]

            out[b, oc, i, j] = value

    return out

  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, _, out_height, out_width = grad.shape

    dweights = np.zeros_like(self.weights.data)
    if self.bias is not None:
      dbias = np.zeros_like(self.bias.data)
    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for oc in range(self.out_channels):
        for i in range(out_height):
          for j in range(out_width):
            h_start = i * self.stride
            h_end = h_start + self.kernel_size

            w_start = j * self.stride
            w_end = w_start + self.kernel_size

            window = self.x_padded[b, :, h_start:h_end, w_start:w_end]

            dweights[oc] += grad[b, oc, i, j] * window
            if self.bias is not None:
              dbias[0, oc] += grad[b, oc, i, j]
            dx_padded[b, :, h_start:h_end, w_start:w_end] += grad[b, oc, i, j] * self.weights.data[oc]

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    else:
      dx = dx_padded

    self.weights.grad = dweights
    if self.bias is not None:
      self.bias.grad = dbias

    return dx

  def parameters(self):
    if self.bias is not None:
      return [self.weights, self.bias]
    return [self.weights]

  def get_config(self):
    return {
      "type": "Conv2D",
      "in_channels": self.in_channels,
      "out_channels": self.out_channels,
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding,
      "bias": self.bias is not None
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None})"

class Conv3D(Layer):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    fan_in = in_channels * kernel_size * kernel_size * kernel_size
    limit = np.sqrt(2.0 / fan_in)

    self.weights = Parameter(
      get_rng().standard_normal((out_channels, in_channels, kernel_size, kernel_size, kernel_size)) * limit
    )

    if bias:
      self.bias = Parameter(np.zeros((1, out_channels)))
    else:
      self.bias = None

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if x.ndim != 5:
      raise ValueError(f"Conv3D expects 5D input (batch_size, channels, depth, height, width), got {x.ndim}D.")

    batch_size, in_channels, depth, height, width = x.shape

    if in_channels != self.in_channels:
      raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}.")

    effective_d = depth + 2 * self.padding
    effective_h = height + 2 * self.padding
    effective_w = width + 2 * self.padding
    if effective_d < self.kernel_size or effective_h < self.kernel_size or effective_w < self.kernel_size:
      raise ValueError(f"Input spatial dims ({depth}x{height}x{width}) + padding ({self.padding}) are smaller than kernel_size ({self.kernel_size}).")

    if self.padding > 0:
      x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding)))
    else:
      x_padded = x

    self.x_padded = x_padded

    out_depth = (depth - self.kernel_size + 2 * self.padding) // self.stride + 1
    out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
    out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

    out = np.zeros((batch_size, self.out_channels, out_depth, out_height, out_width))

    for b in range(batch_size):
      for oc in range(self.out_channels):
        for d in range(out_depth):
          for i in range(out_height):
            for j in range(out_width):
              h_start = i * self.stride
              h_end = h_start + self.kernel_size

              w_start = j * self.stride
              w_end = w_start + self.kernel_size

              d_start = d * self.stride
              d_end = d_start + self.kernel_size

              window = x_padded[b, :, d_start:d_end, h_start:h_end, w_start:w_end]

              kernel = self.weights.data[oc]
              value = np.sum(window * kernel)
              if self.bias is not None:
                value += self.bias.data[0, oc]

              out[b, oc, d, i, j] = value

    return out

  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size, _, out_depth, out_height, out_width = grad.shape

    dweights = np.zeros_like(self.weights.data)
    if self.bias is not None:
      dbias = np.zeros_like(self.bias.data)
    dx_padded = np.zeros_like(self.x_padded)

    for b in range(batch_size):
      for oc in range(self.out_channels):
        for d in range(out_depth):
          for i in range(out_height):
            for j in range(out_width):
              h_start = i * self.stride
              h_end = h_start + self.kernel_size

              w_start = j * self.stride
              w_end = w_start + self.kernel_size

              d_start = d * self.stride
              d_end = d_start + self.kernel_size

              window = self.x_padded[b, :, d_start:d_end, h_start:h_end, w_start:w_end]

              dweights[oc] += grad[b, oc, d, i, j] * window
              if self.bias is not None:
                dbias[0, oc] += grad[b, oc, d, i, j]
              dx_padded[b, :, d_start:d_end, h_start:h_end, w_start:w_end] += grad[b, oc, d, i, j] * self.weights.data[oc]

    if self.padding > 0:
      dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding]
    else:
      dx = dx_padded

    self.weights.grad = dweights
    if self.bias is not None:
      self.bias.grad = dbias

    return dx

  def parameters(self):
    if self.bias is not None:
      return [self.weights, self.bias]
    return [self.weights]

  def get_config(self):
    return {
      "type": "Conv3D",
      "in_channels": self.in_channels,
      "out_channels": self.out_channels,
      "kernel_size": self.kernel_size,
      "stride": self.stride,
      "padding": self.padding,
      "bias": self.bias is not None
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None})"