import numpy as np
from .utils import argmax

def _to_labels(y):
  y = np.asarray(y)
  if y.ndim == 2:
    return argmax(y, axis=1)
  if y.ndim == 1:
    return y
  raise ValueError(f"Expected 1D or 2D input, got {y.ndim}D.")
  
def _confusion_counts(y_true, y_pred):
  classes = np.unique(np.concatenate([y_true, y_pred]))
  tp = np.zeros(len(classes))
  fp = np.zeros(len(classes))
  fn = np.zeros(len(classes))

  for i, c in enumerate(classes):
    tp[i] = np.sum((y_pred == c) & (y_true == c))
    fp[i] = np.sum((y_pred == c) & (y_true != c))
    fn[i] = np.sum((y_pred != c) & (y_true == c))

  return tp, fp, fn
  
def accuracy(y_true, y_pred):
  y_true, y_pred = _to_labels(y_true), _to_labels(y_pred)
  acc = np.mean(y_pred == y_true)
  return acc

def precision(y_true, y_pred, average='macro'):
  y_true, y_pred = _to_labels(y_true), _to_labels(y_pred)
  tp, fp, fn = _confusion_counts(y_true, y_pred)
  per_class = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
  if average == 'macro':
    return np.mean(per_class)
  elif average == 'micro':
    return np.sum(tp) / (np.sum(tp) + np.sum(fp))
  elif average == 'weighted':
    support = tp + fn
    return np.sum(per_class * support) / np.sum(support)
  else:
    raise ValueError("average must be macro, micro or weighted.")
  
def recall(y_true, y_pred, average='macro'):
  y_true, y_pred = _to_labels(y_true), _to_labels(y_pred)
  tp, _, fn = _confusion_counts(y_true, y_pred)
  per_class = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
  if average == 'macro':
    return np.mean(per_class)
  elif average == 'micro':
    return np.sum(tp) / (np.sum(tp) + np.sum(fn))
  elif average == 'weighted':
    support = tp + fn
    return np.sum(per_class * support) / np.sum(support)
  else:
    raise ValueError("average must be macro, micro or weighted.")
  
def f1_score(y_true, y_pred, average='macro'):
  y_true, y_pred = _to_labels(y_true), _to_labels(y_pred)
  tp, fp, fn = _confusion_counts(y_true, y_pred)
  per_class_p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
  per_class_r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
  per_class = np.where(per_class_p + per_class_r > 0, 2 * per_class_p * per_class_r / (per_class_p + per_class_r), 0.0)
  if average == 'macro':
    return np.mean(per_class)
  elif average == 'micro':
    micro_p = np.sum(tp) / (np.sum(tp) + np.sum(fp))
    micro_r = np.sum(tp) / (np.sum(tp) + np.sum(fn))
    return float(np.where(micro_p + micro_r > 0, 2 * micro_p * micro_r / (micro_p + micro_r), 0.0))
  elif average == 'weighted':
    support = tp + fn
    return np.sum(per_class * support) / np.sum(support)
  else:
    raise ValueError("average must be macro, micro or weighted.")