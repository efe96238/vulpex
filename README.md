# Vulpex

A lightweight deep learning framework built on NumPy. Low-level customizability and high-level usability in one.

## Quick Start

Install Vulpex and build your first model in minutes.

```python
import numpy as np
import vulpex as vpx

# Create a simple classifier
model = vpx.Model(
  vpx.Sequential(
    vpx.Linear(784, 128),
    vpx.ReLU(),
    vpx.Linear(128, 10),
    vpx.Softmax()
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
  optimizer=vpx.Adam(lr=0.001),
  loss=vpx.CrossEntropy()
)

# Predict
predictions = model.predict(X_train[:5])
```

## Two Ways to Train

### High-Level usage

For when you want simplicity:

```python
model = vpx.Model(
  vpx.Sequential(
    vpx.Linear(4, 16), vpx.ReLU(),
    vpx.Linear(16, 3), vpx.Softmax()
  )
)

history = model.fit(
  X_train, y_train,
  epochs=50,
  batch_size=32,
  optimizer=vpx.Adam(lr=0.001),
  loss=vpx.CrossEntropy(),
  validation_data=(X_val, y_val),
  scheduler=vpx.ReduceOnPlateau(optimizer, patience=5),
  early_stopping=vpx.EarlyStopping(patience=10)
)
```

### Low-Level usage

For when you want full control:

```python
model = vpx.Model(
  vpx.Sequential(
    vpx.Linear(4, 16), vpx.ReLU(),
    vpx.Linear(16, 3), vpx.Softmax()
  )
)

optimizer = vpx.Adam(lr=0.001)
loss_fn = vpx.CrossEntropy()
loader = vpx.DataLoader(X_train, y_train, batch_size=32, shuffle=True)

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
model = vpx.Model(
  vpx.Sequential(
    vpx.Conv2D(1, 16, kernel_size=3, padding=1),
    vpx.BatchNorm2D(16),
    vpx.ReLU(),
    vpx.MaxPool2D(2),

    vpx.Conv2D(16, 32, kernel_size=3, padding=1),
    vpx.BatchNorm2D(32),
    vpx.ReLU(),
    vpx.MaxPool2D(2),

    vpx.Flatten(),
    vpx.Linear(32 * 7 * 7, 128),
    vpx.ReLU(),
    vpx.Dropout(0.5),
    vpx.Linear(128, 10),
    vpx.Softmax()
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
model.save("my_model.vpx")
model.load("my_model.vpx")

# Train and evaluate
history = model.fit(X_train, y_train, epochs=20, batch_size=32, optimizer=optimizer, loss=loss_fn)
loss = model.evaluate(X_test, y_test, loss_fn)
preds = model.predict(X_test)
```

## Data Utilities

```python
# Split with optional validation set and stratified splitting
X_train, X_test, y_train, y_test = vpx.train_test_val_split(X, y, test_size=0.2, seed=42)

X_train, X_test, X_val, y_train, y_test, y_val = vpx.train_test_val_split(
  X, y, test_size=0.2, val_size=0.1, stratify=True, seed=42
)

# DataLoader for batching and shuffling
loader = vpx.DataLoader(X_train, y_train, batch_size=32, shuffle=True, drop_last=True, seed=42)
```

## Training with Callbacks

```python
optimizer = vpx.Adam(lr=0.01)

history = model.fit(
  X_train, y_train,
  epochs=100,
  batch_size=32,
  optimizer=optimizer,
  loss=vpx.CrossEntropy(),
  validation_data=(X_val, y_val),
  scheduler=vpx.StepLR(optimizer, step_size=20, gamma=0.1),
  early_stopping=vpx.EarlyStopping(patience=10, min_delta=0.001)
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

  vpx.clip_grad_norm(model.parameters(), max_norm=1.0)
  # or
  vpx.clip_grad_value(model.parameters(), clip_value=0.5)

  optimizer.step(model.parameters())
```

## Custom Weight Initialization

```python
layer = vpx.Linear(784, 256)
layer.weights.data = vpx.xavier_uniform((784, 256))
```

## Requirements

- Python 3.8+
- NumPy

## License

BSD-3-Clause