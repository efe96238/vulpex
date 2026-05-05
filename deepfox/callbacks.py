import copy

class EarlyStopping:
  def __init__(self, patience=5, min_delta=0.0):
    self.patience = patience
    self.min_delta = min_delta

    self.best = float('inf')
    self.counter = 0
    self.best_weights = None
    self.stopped_epoch = None

  def step(self, metric, model):
    if metric < self.best - self.min_delta:
      self.best = metric
      self.counter = 0
      self.best_weights = copy.deepcopy(model.parameters())
      return False
    else:
      self.counter += 1
      if self.counter >= self.patience:
        return True
      return False

  def restore(self, model):
    if self.best_weights is not None:
      for p, best_p in zip(model.parameters(), self.best_weights):
        p.data[...] = best_p.data

  def get_config(self):
    return {
      "type": "EarlyStopping",
      "patience": self.patience,
      "min_delta": self.min_delta
    }

  def __repr__(self):
    return f"{self.__class__.__name__}(patience={self.patience}, min_delta={self.min_delta})"