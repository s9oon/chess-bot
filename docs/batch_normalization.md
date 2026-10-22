# Batch normalization

```python
self.bn1 = nnx.BatchNorm(32, rngs=rngs)   # one per output feature map

x = nnx.relu(self.bn1(self.conv1(x)))     # conv, then batchnorm, then ReLU
```

## What it does

Batch normalization (**batchnorm**) keeps the numbers flowing through the
network in a healthy range. After a conv layer, it takes each **output feature
map** across the whole batch of boards and:

1. **Normalizes** it: subtracts the average and divides by the spread
   (standard deviation), so the values sit around 0 with a spread of about 1.
2. **Rescales** it with two learned numbers: a **scale** (multiply) and a
   **bias** (add), so the network can still pick whatever range works best.

```
normalized = (x - mean) / std
output     = scale * normalized + bias
```

The shape doesn't change: 32 output feature maps go in, 32 come out.

## One per feature map

`nnx.BatchNorm(32)` has one scale and one bias **per output feature map**:

```
bn1.scale   shape (32,)
bn1.bias    shape (32,)
```

That's why the `32` must match the conv layer's output channels.

## Why we need it

Without it, the numbers coming out of each layer can drift very large or very
small as the weights change during training. Every layer then has to keep
adjusting to the previous layer's shifting outputs, and training becomes slow
and unstable.

Batchnorm resets each layer's output to a steady range, so:

- training is **faster** and more **stable**,
- deeper networks become practical.

## Where it goes

After the conv, **before** ReLU:

```
conv → batchnorm → ReLU
```

ReLU then sees values centred around 0, so roughly half get switched off
(negative → 0) and half pass through.

## Train mode vs eval mode

Batchnorm behaves differently depending on the mode:

| Mode | Uses | When |
|---|---|---|
| `model.train()` | the mean and spread of the **current batch** | while training |
| `model.eval()` | **running averages** collected during training | when playing / evaluating |

In eval mode the score for a board doesn't depend on which other boards happen
to be in the batch. So always call `model.eval()` before using the network to
play.

## Further reading

- Ioffe & Szegedy (2015), *Batch Normalization: Accelerating Deep Network
  Training by Reducing Internal Covariate Shift*: the original paper.
