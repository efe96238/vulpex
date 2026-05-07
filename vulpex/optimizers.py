import numpy as np

class Adam:
  def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps

    self.m = {}
    self.v = {}
    self.t = 0

  def step(self, params):
    self.t += 1

    for i, p in enumerate(params):
      g = p.grad

      if i not in self.m:
        self.m[i] = np.zeros_like(p.data)
        self.v[i] = np.zeros_like(p.data)

      self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
      self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

      m_hat = self.m[i] / (1 - self.beta1 ** self.t)
      v_hat = self.v[i] / (1 - self.beta2 ** self.t)

      p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class AdamW:
  def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.weight_decay = weight_decay

    self.m = {}
    self.v = {}
    self.t = 0

  def step(self, params):
    self.t += 1

    for i, p in enumerate(params):
      g = p.grad

      if i not in self.m:
        self.m[i] = np.zeros_like(p.data)
        self.v[i] = np.zeros_like(p.data)

      self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
      self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

      m_hat = self.m[i] / (1 - self.beta1 ** self.t)
      v_hat = self.v[i] / (1 - self.beta2 ** self.t)

      adam_update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
      decay_update = self.lr * self.weight_decay * p.data

      p.data -= adam_update + decay_update

class SGD:
  def __init__(self, lr=0.001):
    self.lr = lr

  def step(self, params):
    for p in params:
      p.data -= self.lr * p.grad

class MomentumSGD:
  def __init__(self, lr=0.001, beta=0.9):
    self.lr = lr
    self.beta = beta

    self.v = {}

  def step(self, params):
    for i, p in enumerate(params):
      g = p.grad

      if i not in self.v:
        self.v[i] = np.zeros_like(p.data)

      self.v[i] = self.beta * self.v[i] - self.lr * g
      p.data += self.v[i]

class RMSProp:
  def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
    self.lr = lr
    self.beta = beta
    self.eps = eps

    self.s = {}

  def step(self, params):
    for i, p in enumerate(params):
      g = p.grad

      if i not in self.s:
        self.s[i] = np.zeros_like(p.data)

      self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (g * g)
      p.data -= self.lr * g / (np.sqrt(self.s[i]) + self.eps)