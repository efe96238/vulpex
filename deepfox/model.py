import json
import zipfile
import numpy as np

class Model:
  def __init__(self, *blocks):
    self.blocks = list(blocks)
    self.training = True

  def add(self, block):
    self.blocks.append(block)

  def forward(self, x):
    for block in self.blocks:
      x = block.forward(x)
    return x

  def backward(self, grad):
    for block in reversed(self.blocks):
      grad = block.backward(grad)
    return grad
  
  def train(self):
    self.training = True
    for block in self.blocks:
      block.train()
    return self
  
  def eval(self):
    self.training = False
    for block in self.blocks:
      block.eval()
    return self

  def parameters(self):
    params = []
    for block in self.blocks:
      params.extend(block.parameters())
    return params

  def zero_grad(self):
    for p in self.parameters():
      p.zero_grad()

  def predict(self, X, batch_size=None):
    X = np.asarray(X)
    was_training = self.training
    self.eval()

    if batch_size is None:
      out = self.forward(X)
    else:
      from .dataloader import DataLoader
      loader = DataLoader(X, batch_size=batch_size, shuffle=False)
      outputs = []
      for (X_batch,) in loader:
        outputs.append(self.forward(X_batch))
      out = np.concatenate(outputs, axis=0)

    if was_training:
      self.train()

    return out

  def evaluate(self, X, y, loss, batch_size=None):
    y = np.asarray(y)
    preds = self.predict(X, batch_size)
    return loss.forward(y, preds)

  def _build_manifest(self):
    params = self.parameters()

    manifest = {
      "format": "DeepFox",
      "version": 1,
      "blocks": [block.get_config() for block in self.blocks],
      "parameters": []
    }

    for i, p in enumerate(params):
      manifest["parameters"].append({
        "name": f"param_{i}",
        "shape": list(p.data.shape),
        "dtype": str(p.data.dtype)
      })

    return manifest

  def save(self, path):
    if not path.endswith(".dpx"):
      path += ".dpx"

    manifest = self._build_manifest()
    params = self.parameters()

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
      archive.writestr("manifest.json", json.dumps(manifest, indent=2))

      for i, p in enumerate(params):
        with archive.open(f"param_{i}.npy", "w") as f:
          np.save(f, p.data)

  def load(self, path):
    with zipfile.ZipFile(path, "r") as archive:
      manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
      params = self.parameters()

      if manifest.get("format") != "DeepFox":
        raise ValueError("Invalid .dpx file format.")

      saved_blocks = manifest.get("blocks", [])
      current_blocks = [block.get_config() for block in self.blocks]

      if saved_blocks != current_blocks:
        raise ValueError("Model architecture does not match the .dpx file.")

      saved_params = manifest.get("parameters", [])

      if len(saved_params) != len(params):
        raise ValueError("Parameter count does not match the .dpx file.")

      for i, p in enumerate(params):
        with archive.open(f"param_{i}.npy") as f:
          loaded = np.load(f)

        expected_shape = tuple(saved_params[i]["shape"])

        if loaded.shape != expected_shape:
          raise ValueError(f"Shape mismatch for param_{i}: expected {expected_shape}, got {loaded.shape}")

        p.data[...] = loaded
        p.grad = np.zeros_like(p.data)

  def __call__(self, x):
    return self.forward(x)
  
  def __repr__(self):
    items = [
      "\n".join("  " + line for line in repr(block).split("\n"))
      for block in self.blocks
    ]
    joined = ",\n".join(items)
    return f"{self.__class__.__name__}(\n{joined}\n)"