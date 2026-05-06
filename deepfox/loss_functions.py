import numpy as np

class MSE:
  def __init__(self, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.reduction = reduction

  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    if y.shape != y_pred.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs y_pred {y_pred.shape}.")

    self.y = y
    self.y_pred = y_pred

    losses = (y - y_pred) ** 2

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    grad = 2 * (self.y_pred - self.y)

    if self.reduction == 'mean':
      return grad / self.y.size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad

class MAE:
  def __init__(self, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.reduction = reduction

  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    if y.shape != y_pred.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs y_pred {y_pred.shape}.")

    self.y = y
    self.y_pred = y_pred

    losses = np.abs(y - y_pred)

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    grad = np.sign(self.y_pred - self.y)

    if self.reduction == 'mean':
      return grad / self.y.size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad

class BCE:
  def __init__(self, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.reduction = reduction

  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    if y.shape != y_pred.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs y_pred {y_pred.shape}.")

    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    self.y = y
    self.y_pred = y_pred

    losses = -(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred))

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    grad = -(self.y / self.y_pred) + (1 - self.y) / (1 - self.y_pred)

    if self.reduction == 'mean':
      return grad / self.y.size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad

class BCEWithLogits:
  def __init__(self, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.reduction = reduction

  def forward(self, y, logits):
    y = np.asarray(y)
    logits = np.asarray(logits)

    if y.shape != logits.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs logits {logits.shape}.")

    self.y = y

    self.sigmoid = np.where(logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits)))

    losses = np.maximum(logits, 0) + (-logits * y) + np.log(1 + np.exp(-np.abs(logits)))

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    grad = self.sigmoid - self.y

    if self.reduction == 'mean':
      return grad / self.y.size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad

class CrossEntropy:
  def __init__(self, weight=None, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.weight = np.asarray(weight, dtype=np.float64) if weight is not None else None
    self.reduction = reduction

  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    if y.shape != y_pred.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs y_pred {y_pred.shape}.")

    if y.ndim != 2:
      raise ValueError(f"CrossEntropy expects 2D input (batch_size, num_classes), got {y.ndim}D.")

    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    self.y = y
    self.y_pred = y_pred

    losses = -np.sum(y * np.log(y_pred), axis=1)

    if self.weight is not None:
      sample_weights = np.sum(y * self.weight, axis=1)
      losses = losses * sample_weights

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    batch_size = self.y.shape[0]
    grad = -(self.y / self.y_pred)

    if self.weight is not None:
      sample_weights = np.sum(self.y * self.weight, axis=1, keepdims=True)
      grad = grad * sample_weights

    if self.reduction == 'mean':
      return grad / batch_size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad

class CrossEntropyWithLogits:
  def __init__(self, weight=None, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.weight = np.asarray(weight, dtype=np.float64) if weight is not None else None
    self.reduction = reduction

  def forward(self, y, logits):
    y = np.asarray(y)
    logits = np.asarray(logits)

    if y.shape != logits.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs logits {logits.shape}.")

    if y.ndim != 2:
      raise ValueError(f"CrossEntropyWithLogits expects 2D input (batch_size, num_classes), got {y.ndim}D.")

    self.y = y

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    log_sum_exp = np.log(np.sum(exp_shifted, axis=1, keepdims=True))
    log_softmax = shifted - log_sum_exp

    self.softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    losses = -np.sum(y * log_softmax, axis=1)

    if self.weight is not None:
      sample_weights = np.sum(y * self.weight, axis=1)
      losses = losses * sample_weights

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    batch_size = self.y.shape[0]
    grad = self.softmax - self.y

    if self.weight is not None:
      sample_weights = np.sum(self.y * self.weight, axis=1, keepdims=True)
      grad = grad * sample_weights

    if self.reduction == 'mean':
      return grad / batch_size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad

class HuberLoss:
  def __init__(self, delta=1.0, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.delta = delta
    self.reduction = reduction

  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    if y.shape != y_pred.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs y_pred {y_pred.shape}.")

    self.y = y
    self.y_pred = y_pred

    diff = np.abs(y - y_pred)
    losses = np.where(diff <= self.delta, 0.5 * (y - y_pred) ** 2, self.delta * (diff - 0.5 * self.delta))

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    diff = np.abs(self.y - self.y_pred)
    grad = np.where(diff <= self.delta, self.y_pred - self.y, self.delta * np.sign(self.y_pred - self.y))

    if self.reduction == 'mean':
      return grad / self.y.size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad

class NLLLoss:
  def __init__(self, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.reduction = reduction

  def forward(self, y, log_probs):
    y = np.asarray(y)
    log_probs = np.asarray(log_probs)

    if y.shape != log_probs.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs log_probs {log_probs.shape}.")

    if y.ndim != 2:
      raise ValueError(f"NLLLoss expects 2D input (batch_size, num_classes), got {y.ndim}D.")

    self.y = y

    losses = -np.sum(y * log_probs, axis=1)

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    batch_size = self.y.shape[0]
    grad = -self.y

    if self.reduction == 'mean':
      return grad / batch_size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad

class HingeLoss:
  def __init__(self, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.reduction = reduction

  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    if y.shape != y_pred.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs y_pred {y_pred.shape}.")

    self.y = y
    self.y_pred = y_pred

    losses = np.maximum(0, 1 - y * y_pred)

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    grad = np.where(1 - self.y * self.y_pred > 0, -self.y, 0.0)

    if self.reduction == 'mean':
      return grad / self.y.size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad

class KLDivergence:
  def __init__(self, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.reduction = reduction

  def forward(self, y, log_probs):
    y = np.asarray(y)
    log_probs = np.asarray(log_probs)

    if y.shape != log_probs.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs log_probs {log_probs.shape}.")

    if y.ndim != 2:
      raise ValueError(f"KLDivergence expects 2D input (batch_size, num_classes), got {y.ndim}D.")

    eps = 1e-15
    y = np.clip(y, eps, 1.0 - eps)

    self.y = y

    losses = np.sum(y * (np.log(y) - log_probs), axis=1)

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    batch_size = self.y.shape[0]
    grad = -self.y

    if self.reduction == 'mean':
      return grad / batch_size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad

class CosineEmbeddingLoss:
  def __init__(self, margin=0.0, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.margin = margin
    self.reduction = reduction

  def forward(self, y, x1, x2):
    y = np.asarray(y, dtype=np.float64)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    if x1.shape != x2.shape:
      raise ValueError(f"Shape mismatch: x1 {x1.shape} vs x2 {x2.shape}.")

    if x1.ndim != 2:
      raise ValueError(f"CosineEmbeddingLoss expects 2D inputs (batch_size, features), got {x1.ndim}D.")

    self.y = y
    self.x1 = x1
    self.x2 = x2

    eps = 1e-8

    self.dot = np.sum(x1 * x2, axis=1, keepdims=True)
    self.norm1 = np.sqrt(np.sum(x1 ** 2, axis=1, keepdims=True)) + eps
    self.norm2 = np.sqrt(np.sum(x2 ** 2, axis=1, keepdims=True)) + eps

    self.cos = self.dot / (self.norm1 * self.norm2)

    losses = np.where(y == 1, 1 - self.cos, np.maximum(0, self.cos - self.margin))

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    batch_size = self.y.shape[0]

    dcos_dx1 = (self.x2 / (self.norm1 * self.norm2)) - self.cos * (self.x1 / (self.norm1 ** 2))
    dcos_dx2 = (self.x1 / (self.norm1 * self.norm2)) - self.cos * (self.x2 / (self.norm2 ** 2))

    mask_pos = (self.y == 1)
    mask_neg = (self.y == -1) & (self.cos - self.margin > 0)

    scale = np.where(mask_pos, -1.0, np.where(mask_neg, 1.0, 0.0))

    dx1 = scale * dcos_dx1
    dx2 = scale * dcos_dx2

    if self.reduction == 'mean':
      return dx1 / batch_size, dx2 / batch_size
    elif self.reduction == 'sum':
      return dx1, dx2
    elif self.reduction == 'none':
      return dx1, dx2

class SmoothL1Loss:
  def __init__(self, beta=1.0, reduction='mean'):
    if reduction not in ('mean', 'sum', 'none'):
      raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
    self.beta = beta
    self.reduction = reduction

  def forward(self, y, y_pred):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    if y.shape != y_pred.shape:
      raise ValueError(f"Shape mismatch: y {y.shape} vs y_pred {y_pred.shape}.")

    self.y = y
    self.y_pred = y_pred

    diff = np.abs(y - y_pred)
    losses = np.where(diff <= self.beta, 0.5 * (y - y_pred) ** 2 / self.beta, diff - 0.5 * self.beta)

    if self.reduction == 'mean':
      return np.mean(losses)
    elif self.reduction == 'sum':
      return np.sum(losses)
    elif self.reduction == 'none':
      return losses

  def backward(self):
    diff = np.abs(self.y - self.y_pred)
    grad = np.where(diff <= self.beta, (self.y_pred - self.y) / self.beta, np.sign(self.y_pred - self.y))

    if self.reduction == 'mean':
      return grad / self.y.size
    elif self.reduction == 'sum':
      return grad
    elif self.reduction == 'none':
      return grad