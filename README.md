# DeepFox

A lightweight deep learning framework built on NumPy. As low-level as PyTorch or as high-level as scikit-learn — your choice.

## Quick Start

Install DeepFox and build your first model in minutes.

```python
import numpy as np
import deepfox as dfx

# Create a simple classifier
model = dfx.Model(
  dfx.Sequential(
    dfx.Linear(784, 128),
    dfx.ReLU(),
    dfx.Linear(128, 10),
    dfx.Softmax()
  )
)

# Generate some data
X_train = np.random.randn(1000, 784)
y_train = np.eye(10)[np.random.randint(0, 10, 1000)]

# Train — one line, sklearn-style
history = model.fit(
  X_train, y_train,
  epochs=20,
  batch_size=32,
  optimizer=dfx.Adam(lr=0.001),
  loss=dfx.CrossEntropy()
)

# Predict
predictions = model.predict(X_train[:5])
```

## Two Ways to Train

### High-Level (sklearn-style)

For when you want simplicity:

```python
model = dfx.Model(
  dfx.Sequential(
    dfx.Linear(4, 16), dfx.ReLU(),
    dfx.Linear(16, 3), dfx.Softmax()
  )
)

history = model.fit(
  X_train, y_train,
  epochs=50,
  batch_size=32,
  optimizer=dfx.Adam(lr=0.001),
  loss=dfx.CrossEntropy(),
  validation_data=(X_val, y_val),
  scheduler=dfx.ReduceOnPlateau(optimizer, patience=5),
  early_stopping=dfx.EarlyStopping(patience=10)
)
```

### Low-Level (PyTorch-style)

For when you want full control:

```python
model = dfx.Model(
  dfx.Sequential(
    dfx.Linear(4, 16), dfx.ReLU(),
    dfx.Linear(16, 3), dfx.Softmax()
  )
)

optimizer = dfx.Adam(lr=0.001)
loss_fn = dfx.CrossEntropy()
loader = dfx.DataLoader(X_train, y_train, batch_size=32, shuffle=True)

for epoch in range(50):
  for X_batch, y_batch in loader:
    model.zero_grad()
    pred = model.forward(X_batch)
    loss = loss_fn.forward(y_batch, pred)
    grad = loss_fn.backward()
    model.backward(grad)
    optimizer.step(model.parameters())
```

## CNN Example

```python
model = dfx.Model(
  dfx.Sequential(
    dfx.Conv2D(1, 16, kernel_size=3, padding=1),
    dfx.BatchNorm2D(16),
    dfx.ReLU(),
    dfx.MaxPool2D(2),

    dfx.Conv2D(16, 32, kernel_size=3, padding=1),
    dfx.BatchNorm2D(32),
    dfx.ReLU(),
    dfx.MaxPool2D(2),

    dfx.Flatten(),
    dfx.Linear(32 * 7 * 7, 128),
    dfx.ReLU(),
    dfx.Dropout(0.5),
    dfx.Linear(128, 10),
    dfx.Softmax()
  )
)

model.summary((1, 28, 28))
```

## Features

### Layers

`Linear`, `Conv1D`, `Conv2D`, `Conv3D`, `MaxPool1D`, `MaxPool2D`, `MaxPool3D`, `AvgPool1D`, `AvgPool2D`, `AvgPool3D`, `AdaptiveAvgPool1D`, `AdaptiveAvgPool2D`, `AdaptiveAvgPool3D`, `BatchNorm1D`, `BatchNorm2D`, `BatchNorm3D`, `LayerNorm`, `Dropout`, `Flatten`, `Embedding`, `ZeroPad1D`, `ZeroPad2D`, `ZeroPad3D`, `Sequential`

### Activations

`ReLU`, `LeakyReLU`, `PReLU`, `ELU`, `SELU`, `GeLU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`

### Loss Functions

`MSE`, `MAE`, `BCE`, `BCEWithLogits`, `CrossEntropy`, `CrossEntropyWithLogits`, `NLLLoss`, `HuberLoss`, `HingeLoss`, `KLDivergence`, `CosineEmbeddingLoss`, `SmoothL1Loss`

All loss functions support `reduction='mean'`, `'sum'`, or `'none'`. `CrossEntropy` and `CrossEntropyWithLogits` support class weights for imbalanced datasets.

### Optimizers

`Adam`, `AdamW`, `SGD`, `MomentumSGD`, `RMSProp`

### Training Utilities

`DataLoader`, `train_test_val_split`, `StepLR`, `ReduceOnPlateau`, `EarlyStopping`, `clip_grad_norm`, `clip_grad_value`

### Weight Initialization

`xavier_uniform`, `xavier_normal`, `he_uniform`, `he_normal`, `zeros`, `ones`

### Model Tools

```python
# Architecture overview
model.summary((784,))

# Total trainable parameters
model.count_params()

# Save and load
model.save("my_model.dpx")
model.load("my_model.dpx")

# Train and evaluate
history = model.fit(X_train, y_train, epochs=20, batch_size=32, optimizer=optimizer, loss=loss_fn)
loss = model.evaluate(X_test, y_test, loss_fn)
preds = model.predict(X_test)
```

## Data Utilities

```python
# Split with optional validation set and stratified splitting
X_train, X_test, y_train, y_test = dfx.train_test_val_split(X, y, test_size=0.2, seed=42)

X_train, X_test, X_val, y_train, y_test, y_val = dfx.train_test_val_split(
  X, y, test_size=0.2, val_size=0.1, stratify=True, seed=42
)

# DataLoader for batching and shuffling
loader = dfx.DataLoader(X_train, y_train, batch_size=32, shuffle=True, drop_last=True, seed=42)
```

## Training with Callbacks

```python
optimizer = dfx.Adam(lr=0.01)

history = model.fit(
  X_train, y_train,
  epochs=100,
  batch_size=32,
  optimizer=optimizer,
  loss=dfx.CrossEntropy(),
  validation_data=(X_val, y_val),
  scheduler=dfx.StepLR(optimizer, step_size=20, gamma=0.1),
  early_stopping=dfx.EarlyStopping(patience=10, min_delta=0.001)
)

# Access training history
print(history["loss"])      # training loss per epoch
print(history["val_loss"])  # validation loss per epoch
```

## Gradient Clipping

```python
for X_batch, y_batch in loader:
  model.zero_grad()
  pred = model.forward(X_batch)
  loss = loss_fn.forward(y_batch, pred)
  grad = loss_fn.backward()
  model.backward(grad)

  dfx.clip_grad_norm(model.parameters(), max_norm=1.0)
  # or
  dfx.clip_grad_value(model.parameters(), clip_value=0.5)

  optimizer.step(model.parameters())
```

## Custom Weight Initialization

```python
layer = dfx.Linear(784, 256)
layer.weights.data = dfx.xavier_uniform((784, 256))
```

## Requirements

- Python 3.8+
- NumPy

## License

MIT
