import numpy as np
from .utils import get_rng

def xavier_uniform(shape):
  fan_in = shape[0]
  fan_out = shape[1]
  limit = np.sqrt(6.0 / (fan_in + fan_out))
  return get_rng().uniform(-limit, limit, size=shape)

def xavier_normal(shape):
  fan_in = shape[0]
  fan_out = shape[1]
  std = np.sqrt(2.0 / (fan_in + fan_out))
  return get_rng().normal(0, std, size=shape)

def he_uniform(shape):
  fan_in = shape[0]
  limit = np.sqrt(6.0 / fan_in)
  return get_rng().uniform(-limit, limit, size=shape)

def he_normal(shape):
  fan_in = shape[0]
  std = np.sqrt(2.0 / fan_in)
  return get_rng().normal(0, std, size=shape)

def zeros(shape):
  return np.zeros(shape)

def ones(shape):
  return np.ones(shape)