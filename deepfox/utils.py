import numpy as np

_rng_container = {
  "rng": np.random.default_rng()
}

def seed(value):
  _rng_container["rng"] = np.random.default_rng(value)

def get_rng():
  return _rng_container["rng"]

def argmax(x, axis=None):
  x = np.asarray(x)
  if axis is None:
    flat = x.ravel()
    max_idx = 0
    for i in range(1, flat.size):
      if flat[i] > flat[max_idx]:
        max_idx = i
    return np.unravel_index(max_idx, x.shape) if x.ndim > 1 else max_idx

  x = np.moveaxis(x, axis, -1)
  out_shape = x.shape[:-1]
  x_flat = x.reshape(-1, x.shape[-1])

  result = np.empty(x_flat.shape[0], dtype=int)
  for i in range(x_flat.shape[0]):
    max_idx = 0
    for j in range(1, x_flat.shape[1]):
      if x_flat[i, j] > x_flat[i, max_idx]:
        max_idx = j
    result[i] = max_idx

  return result.reshape(out_shape)

def train_test_val_split(X, y, test_size=0.2, val_size=None, shuffle=True, seed=None, stratify=False):
  X = np.asarray(X)
  y = np.asarray(y)

  if X.shape[0] != y.shape[0]:
    raise ValueError("X and y shapes must match.")

  if not 0 <= test_size <= 1:
    raise ValueError("test_size must be a number between 0 and 1.")

  if val_size is not None:
    if not 0 <= val_size <= 1:
      raise ValueError("val_size must be a number between 0 and 1.")
    if test_size + val_size >= 1.0:
      raise ValueError("test_size + val_size must be less than 1.0.")

  n_samples = X.shape[0]
  rng = np.random.default_rng(seed)

  if stratify:
    labels = np.argmax(y, axis=1) if y.ndim > 1 else y
    classes = np.unique(labels)

    train_idx, test_idx, val_idx = [], [], []

    for cls in classes:
      cls_indices = np.where(labels == cls)[0]
      if shuffle:
        cls_indices = rng.permutation(cls_indices)

      n_cls = len(cls_indices)
      n_test_cls = max(1, int(n_cls * test_size))
      n_val_cls = max(1, int(n_cls * val_size)) if val_size is not None else 0
      n_train_cls = n_cls - n_test_cls - n_val_cls

      if n_train_cls < 1:
        raise ValueError(f"Not enough samples in class {cls} for the requested split sizes.")

      train_idx.extend(cls_indices[:n_train_cls])
      test_idx.extend(cls_indices[n_train_cls:n_train_cls + n_test_cls])
      if val_size is not None:
        val_idx.extend(cls_indices[n_train_cls + n_test_cls:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    if shuffle:
      train_idx = rng.permutation(train_idx)
      test_idx = rng.permutation(test_idx)

    if val_size is not None:
      val_idx = np.array(val_idx)
      if shuffle:
        val_idx = rng.permutation(val_idx)
      return X[train_idx], X[test_idx], X[val_idx], y[train_idx], y[test_idx], y[val_idx]
    else:
      return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

  else:
    indices = rng.permutation(n_samples) if shuffle else np.arange(n_samples)

    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size) if val_size is not None else 0
    n_train = n_samples - n_test - n_val

    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]

    if val_size is not None:
      val_idx = indices[n_train + n_test:]
      return X[train_idx], X[test_idx], X[val_idx], y[train_idx], y[test_idx], y[val_idx]
    else:
      return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    
def clip_grad_norm(parameters, max_norm):
  total_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in parameters))
  if total_norm > max_norm:
    scale = max_norm / total_norm
    for p in parameters:
      p.grad *= scale

def clip_grad_value(parameters, clip_value):
  for p in parameters:
    p.grad = np.clip(p.grad, -clip_value, clip_value)