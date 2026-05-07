import numpy as np
from .base import Layer
from ..parameter import Parameter
from ..utils import get_rng

class Embedding(Layer):
  def __init__(self, num_embeddings, embedding_dim):
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim

    self.weights = Parameter(
      get_rng().standard_normal((num_embeddings, embedding_dim))
    )

  def forward(self, x):
    x = np.asarray(x)
    self.x = x

    if not np.issubdtype(x.dtype, np.integer):
      raise ValueError(f"Embedding expects integer indices, got dtype {x.dtype}.")

    if np.any(x < 0) or np.any(x >= self.num_embeddings):
      raise ValueError(f"Index out of range. Expected indices in [0, {self.num_embeddings}), got min={x.min()}, max={x.max()}.")

    return self.weights.data[x]
  
  def backward(self, grad):
    dweights = np.zeros_like(self.weights.data)
    np.add.at(dweights, self.x, grad)
    self.weights.grad = dweights
    return None

  def parameters(self):
    return [self.weights]
  
  def get_config(self):
    return {
      "type": "Embedding",
      "num_embeddings": self.num_embeddings,
      "embedding_dim": self.embedding_dim
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"