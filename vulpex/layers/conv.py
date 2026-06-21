import numpy as np
from .base import Layer
from ..parameter import Parameter
from ..utils import get_rng

def im2col_1d(x, kernel_size, stride):
  batch, in_channels, length = x.shape
  out_len = (length - kernel_size) // stride + 1

  starts = np.arange(out_len) * stride
  offsets = np.arange(kernel_size)
  indices = starts[:, None] + offsets[None, :]

  cols = x[:, :, indices]
  cols = cols.transpose(0, 2, 1, 3).reshape(batch, out_len, -1)

  return cols

def col2im_1d(cols, x_shape, kernel_size, stride):
  batch, in_channels, length = x_shape
  out_len = (length - kernel_size) // stride + 1

  dx = np.zeros((batch, in_channels, length))

  cols = cols.reshape(batch, out_len, in_channels, kernel_size)
  cols = cols.transpose(0, 2, 1, 3)

  starts = np.arange(out_len) * stride
  offsets = np.arange(kernel_size)
  indices = starts[:, None] + offsets[None, :]

  np.add.at(dx, (slice(None), slice(None), indices), cols)

  return dx

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

    self.cols = im2col_1d(x_padded, self.kernel_size, self.stride)

    W = self.weights.data.reshape(self.out_channels, -1).T

    out = self.cols @ W

    if self.bias is not None:
      out += self.bias.data

    out = out.transpose(0, 2, 1)

    return out

  def backward(self, grad):
    grad = np.asarray(grad)

    grad_t = grad.transpose(0, 2, 1)

    batch, out_len, _ = grad_t.shape
    cols_flat = self.cols.reshape(batch * out_len, -1)
    grad_flat = grad_t.reshape(batch * out_len, -1)
    dW = cols_flat.T @ grad_flat
    self.weights.grad = dW.T.reshape(self.weights.data.shape)

    if self.bias is not None:
      self.bias.grad = np.sum(grad_t, axis=(0, 1), keepdims=False)

    W = self.weights.data.reshape(self.out_channels, -1).T
    dcols = grad_t @ W.T
    dx = col2im_1d(dcols, self.x_padded.shape, self.kernel_size, self.stride)

    if self.padding > 0:
      dx = dx[:, :, self.padding:-self.padding]

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

def im2col_2d(x, kernel_size, stride):
  batch, in_channels, height, width = x.shape
  out_height = (height - kernel_size) // stride + 1
  out_width = (width - kernel_size) // stride + 1

  h_starts = np.arange(out_height) * stride
  h_offsets = np.arange(kernel_size)
  h_indices = h_starts[:, None] + h_offsets[None, :]

  w_starts = np.arange(out_width) * stride
  w_offsets = np.arange(kernel_size)
  w_indices = w_starts[:, None] + w_offsets[None, :]

  cols = x[:, :, h_indices[:, :, None, None], w_indices[None, None, :, :]]
  cols = cols.transpose(0, 2, 4, 1, 3, 5)
  cols = cols.reshape(batch, out_height * out_width, -1)

  return cols

def col2im_2d(cols, x_shape, kernel_size, stride):
  batch, in_channels, height, width = x_shape
  out_height = (height - kernel_size) // stride + 1
  out_width = (width - kernel_size) // stride + 1

  dx = np.zeros((batch, in_channels, height, width))

  cols = cols.reshape(batch, out_height, out_width, in_channels, kernel_size, kernel_size)

  cols = cols.transpose(0, 3, 1, 4, 2, 5)

  h_starts = np.arange(out_height) * stride
  h_offsets = np.arange(kernel_size)
  h_indices = h_starts[:, None] + h_offsets[None, :]

  w_starts = np.arange(out_width) * stride
  w_offsets = np.arange(kernel_size)
  w_indices = w_starts[:, None] + w_offsets[None, :]

  np.add.at(dx, (slice(None), slice(None), h_indices[:, :, None, None], w_indices[None, None, :, :]), cols)

  return dx

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

    self.cols = im2col_2d(x_padded, self.kernel_size, self.stride)

    W = self.weights.data.reshape(self.out_channels, -1).T

    out = self.cols @ W

    if self.bias is not None:
      out += self.bias.data

    out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

    out = out.transpose(0, 2, 1).reshape(batch_size, self.out_channels, out_height, out_width)

    return out

  def backward(self, grad):
    grad = np.asarray(grad)

    batch_size = grad.shape[0]

    grad_t = grad.reshape(batch_size, self.out_channels, -1).transpose(0, 2, 1)

    cols_flat = self.cols.reshape(-1, self.cols.shape[-1])
    grad_flat = grad_t.reshape(-1, self.out_channels)
    dW = cols_flat.T @ grad_flat
    self.weights.grad = dW.T.reshape(self.weights.data.shape)

    if self.bias is not None:
      self.bias.grad = np.sum(grad_t, axis=(0, 1), keepdims=False).reshape(1, -1)

    W = self.weights.data.reshape(self.out_channels, -1).T
    dcols = grad_t @ W.T
    dx = col2im_2d(dcols, self.x_padded.shape, self.kernel_size, self.stride)

    if self.padding > 0:
      dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]

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