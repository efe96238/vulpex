
class StepLR:
  def __init__(self, optimizer, step_size=10, gamma=0.1):
    self.optimizer = optimizer
    self.step_size = step_size
    self.gamma = gamma

    self.current_epoch = 0

  def step(self, metric=None):
    self.current_epoch += 1
    if self.current_epoch % self.step_size == 0:
      self.optimizer.lr *= self.gamma

  def get_config(self):
    return {
      "type": "StepLR",
      "optimizer": self.optimizer.__class__.__name__,
      "step_size": self.step_size,
      "gamma": self.gamma
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(optimizer={self.optimizer}, step_size={self.step_size}, gamma={self.gamma})"
  
class ReduceOnPlateau:
  def __init__(self, optimizer, patience=10, factor=0.1, min_lr=0):
    self.optimizer = optimizer
    self.patience = patience
    self.factor = factor
    self.min_lr = min_lr

    self.best = float('inf')
    self.counter = 0

  def step(self, metric):
    if metric < self.best:
      self.best = metric
      self.counter = 0
    else:
      self.counter += 1
      if self.counter >= self.patience:
        self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
        self.counter = 0

  def get_config(self):
    return {
      "type": "ReduceOnPlateau",
      "optimizer": self.optimizer.__class__.__name__,
      "patience": self.patience,
      "factor": self.factor,
      "min_lr": self.min_lr
    }
  
  def __repr__(self):
    return f"{self.__class__.__name__}(optimizer={self.optimizer}, patience={self.patience}, factor={self.factor}, min_lr={self.min_lr})"