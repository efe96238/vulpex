
class Layer:
  training = True

  def forward(self, x):
    raise NotImplementedError

  def backward(self, grad):
    raise NotImplementedError

  def parameters(self):
    return []

  def zero_grad(self):
    for p in self.parameters():
      p.zero_grad()

  def train(self):
    self.training = True
    return self
  
  def eval(self):
    self.training = False
    return self

  def get_config(self):
    return {
      "type": self.__class__.__name__
    }
  
  def __repr__(self):
    return self.__class__.__name__
  
class Sequential(Layer):
  def __init__(self, *layers):
    self.layers = list(layers)

  def forward(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def backward(self, grad):
    for layer in reversed(self.layers):
      grad = layer.backward(grad)
    return grad
  
  def train(self):
    self.training = True
    for layer in self.layers:
      layer.train()
    return self

  def eval(self):
    self.training = False
    for layer in self.layers:
      layer.eval()
    return self
  
  def parameters(self):
    params = []
    for layer in self.layers:
      params.extend(layer.parameters())
    return params
  
  def get_config(self):
    return {
      "type": "Sequential",
      "layers": [layer.get_config() for layer in self.layers]
    }
  
  def __repr__(self):
    items = [
      "\n".join("  " + line for line in repr(layer).split("\n"))
      for layer in self.layers
    ]
    joined = ",\n".join(items)
    return f"{self.__class__.__name__}(\n{joined}\n)"