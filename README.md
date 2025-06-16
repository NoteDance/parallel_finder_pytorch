# ParallelFinder

## Overview

`ParallelFinder` enables concurrent training of multiple PyTorch models in separate processes, tracking each model’s final-epoch loss and training duration. It uses `multiprocessing.Manager().dict()` and a `Lock` to collect and synchronize metrics across processes.

## Features

* Launch training for a list of model constructors in parallel processes.
* Compute and record final-epoch average loss and epoch duration for each model.
* Identify which model achieves the lowest loss and which completes fastest.
* Thread-safe logging via a shared dictionary and lock.
* Assign a specific device per model by providing a list of device strings.

## API Reference

### Class: `ParallelFinder`

```python
ParallelFinder(model_list)
```

* **model_list**: A list of zero-argument callables that return a new `torch.nn.Module` when called—this can be the model class itself if its `__init__` has no required arguments (e.g., `[MyModelA, MyModelB]`) or a top-level factory function/`functools.partial` for models needing parameters.

#### Attributes

* `logs`: A shared `multiprocessing.Manager().dict()` containing:

  * `'best_loss'`: lowest final-epoch loss observed.
  * `'best_loss_model_idx'`: index of the model with lowest loss.
  * `'time_for_best_loss'`: epoch time for that model.
  * `'best_time'`: shortest epoch duration observed.
  * `'best_time_model_idx'`: index of the fastest model.
  * Per-model entries: `'model_{idx}_loss'` and `'model_{idx}_time'`.
* `lock`: A `multiprocessing.Lock` ensuring safe concurrent updates to `logs`.

#### Method: `find`

```python
find(train_data, train_labels,
     epochs=1, batch_size=32,
     criterion=None, optimizer=None, optimizer_params=None,
     device_str='cuda')
```

* **train\_data**: Tensor or array-like inputs.
* **train\_labels**: Tensor or array-like targets.
* **epochs** (int): Number of epochs to run for each model.
* **batch\_size** (int): Batch size for the `DataLoader`.
* **criterion**: Loss class or factory, instantiated via `criterion()`. Same type for all models.
* **optimizer**: List of optimizer classes/factories, one per model (e.g. `[torch.optim.SGD, torch.optim.Adam]`).
* **optimizer\_params**: List of dicts of keyword args for each optimizer; length must match `model_fn_list`.
* **device\_str**: Sequence (e.g. list or tuple) of device identifier strings (`'cuda:0'`, `'cpu'`, etc.), with length ≥ number of models. For index `idx`, the code uses `torch.device(device_str[idx])`.

  * Example: `device_list = ['cuda:0', 'cuda:1']` for two models, or `['cpu', 'cpu']` if using CPU.
* **Returns**: None. After completion, inspect `finder.logs` for metrics.

## Usage Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model constructors
def make_model_a():
    return nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))

def make_model_b():
    return nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 1))

model_fns = [make_model_a, make_model_b]
finder = ParallelFinder(model_fns)

# Dummy dataset
train_X = torch.randn(1000, 10)
train_y = torch.randn(1000, 1)

# Loss and optimizers
criterion_fn = nn.MSELoss
optimizers = [optim.SGD, optim.Adam]
opt_params = [{'lr': 0.01}, {'lr': 0.001}]

# Devices per model
device_list = ['cpu', 'cpu']  # or ['cuda:0', 'cuda:1'] if GPUs available

finder.find(
    train_X, train_y,
    epochs=5,
    batch_size=64,
    criterion=criterion_fn,
    optimizer=optimizers,
    optimizer_params=opt_params,
    device_str=device_list
)

print(finder.logs)
# Keys: 'best_loss', 'best_loss_model_idx', 'time_for_best_loss',
# 'best_time', 'best_time_model_idx', 'loss_for_best_time',
# 'model_0_loss', 'model_0_time', 'model_1_loss', 'model_1_time', etc.
```

## Notes & Caveats

* **Device list requirement**: Since `device_str[idx]` is used, always supply `device_str` as a sequence of length at least the number of models.
* **GPU contention**: Multiple processes targeting the same GPU can exhaust memory or slow training. Use distinct devices if available, or run sequentially.
* **DataLoader shuffle**: The code omits `shuffle=True`. If random shuffling is desired, shuffle inputs beforehand or modify the `DataLoader` instantiation.
* **Serialization overhead**: Large datasets passed to subprocesses incur pickling costs. For very large data, consider shared memory techniques or loading data inside each process.
* **Process overhead**: Spawning many processes has overhead; for many small models or brief epochs, parallelism gains may be limited.
* **Reproducibility**: Random initializations and data ordering are independent per process. To fix seeds, set them at the start of `_train_single`.
* **Extensibility**: You may extend `_train_single` to include validation, checkpointing, learning-rate schedulers, or custom logging hooks.
