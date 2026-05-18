import numpy as np
from .utils import get_rng

class DataLoader:
  def __init__(self, X, y=None, batch_size=32, shuffle=True, drop_last=False, seed=None):
    self.X = np.asarray(X)
    self.y = np.asarray(y) if y is not None else None
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.drop_last = drop_last
    self.seed = seed
    self.rng = np.random.default_rng(seed) if seed is not None else get_rng()

    if not isinstance(batch_size, int) or batch_size < 1:
      raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

    if self.y is not None and self.X.shape[0] != self.y.shape[0]:
      raise ValueError("X and y must have the same number of samples.")

    self.n_samples = self.X.shape[0]

  def __iter__(self):
    if self.shuffle:
      self.indices = self.rng.permutation(self.n_samples)
    else:
      self.indices = np.arange(self.n_samples)
    self.cursor = 0
    return self

  def __next__(self):
    if self.cursor >= self.n_samples:
      raise StopIteration

    start = self.cursor
    end = start + self.batch_size
    batch_indices = self.indices[start:end]

    if len(batch_indices) < self.batch_size and self.drop_last:
      raise StopIteration

    self.cursor = end

    X_batch = self.X[batch_indices]

    if self.y is not None:
      return X_batch, self.y[batch_indices]
    return (X_batch,)
  
  def __len__(self):
    if self.drop_last:
      return self.n_samples // self.batch_size
    return (self.n_samples + self.batch_size - 1) // self.batch_size
  
  def __repr__(self):
    return f"{self.__class__.__name__}(samples={self.n_samples}, batch_size={self.batch_size}, shuffle={self.shuffle}, drop_last={self.drop_last}, seed={self.seed})"