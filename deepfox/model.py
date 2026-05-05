import json
import zipfile
import numpy as np
import time
from .dataloader import DataLoader
from .history import History

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
  
  def fit(self, X, y, epochs=10, batch_size=32, optimizer=None, loss=None, validation_data=None, scheduler=None, early_stopping=None, verbose=True):
    if optimizer is None:
      raise ValueError("optimizer cannot be None.")
    if loss is None:
      raise ValueError("loss cannot be None.")

    if batch_size is not None:
      loader = DataLoader(X, y, batch_size=batch_size, shuffle=True)
    else:
      X, y = np.asarray(X), np.asarray(y)
      loader = [(X, y)]
    history = History()

    for epoch in range(epochs):
      start = time.time()
      self.train()
      batch_losses = []

      for X_batch, y_batch in loader:
        self.zero_grad()
        pred = self.forward(X_batch)
        batch_loss = loss.forward(y_batch, pred)
        grad = loss.backward()
        self.backward(grad)
        optimizer.step(self.parameters())
        batch_losses.append(batch_loss)

      epoch_loss = np.mean(batch_losses)
      history.record("loss", float(epoch_loss))

      if validation_data is not None:
        X_val, y_val = validation_data
        val_loss = self.evaluate(X_val, y_val, loss)
        history.record("val_loss", float(val_loss))

      if scheduler is not None:
        scheduler.step(epoch_loss)

      if early_stopping is not None:
        monitor = val_loss if validation_data is not None else epoch_loss
        if early_stopping.step(monitor, self):
          early_stopping.stopped_epoch = epoch + 1
          early_stopping.restore(self)
          if verbose:
            print(f"Early stopping at epoch {epoch + 1}. Restoring best weights.")
          break
      
      elapsed = time.time() - start

      if verbose:
        msg = f"Epoch {epoch + 1}/{epochs} | Training Loss: {epoch_loss:.4f}"
        if validation_data is not None:
          msg += f" | Validation Loss: {val_loss:.4f}"
        msg += f" | Duration: {elapsed:.2f}s"
        print(msg)

    return history

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