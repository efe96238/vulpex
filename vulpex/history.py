
class History:
  def __init__(self):
    self.data = {}

  def record(self, metric, value):
    if metric not in self.data:
      self.data[metric] = []
    self.data[metric].append(value)

  def __getitem__(self, metric):
    return self.data[metric]

  def __repr__(self):
    metrics = ", ".join(f"{k}: {len(v)} epochs" for k, v in self.data.items())
    return f"History({metrics})"