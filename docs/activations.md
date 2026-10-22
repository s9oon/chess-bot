# Activation functions

```python
def __call__(self, x):
    x = nnx.relu(self.conv1(x))
    x = nnx.relu(self.conv2(x))
    x = nnx.relu(self.conv3(x))
    return x
```

## ReLU

**ReLU is an activation function applied to all output feature maps.** After
each conv layer, it goes over every number in every output feature map and
turns the negatives into 0:

```
relu(x) = max(0, x)

feature map before ReLU:   [ 2.0, -1.3,  0.7, -0.4, ... ]
feature map after ReLU:    [ 2.0,  0.0,  0.7,  0.0, ... ]
```

The shape doesn't change: 32 output feature maps of 8x8 go in, 32 come out.
Then they become the next conv layer's input feature maps:

```
conv1 -> 32 output feature maps -> ReLU -> into conv2
conv2 -> 32 output feature maps -> ReLU -> into conv3
conv3 -> 32 output feature maps -> ReLU -> returned
```

## Why we need it

Each conv layer is an affine transformation (`W · x + b`). Stacked on their
own, affine transformations collapse into a single one:

```
W2 · (W1 · x + b1) + b2  =  W · x + b
```

So without ReLU, three conv layers would be no better than one.

ReLU is **non-linear**, which stops the collapse. Each layer can then build
on the one before: conv1 spots simple patterns, conv2 combines them into
bigger ones, and conv3 into bigger ones still.

It also acts like an on/off switch: a negative value means "this pattern isn't
here", so it becomes 0.

## Other activation functions

| Function | Range | Used for |
|---|---|---|
| **ReLU** | 0 to ∞ | between layers (ours) |
| **tanh** | -1 to +1 | a final score, e.g. our evaluation later |
| **sigmoid** | 0 to 1 | a final probability |

ReLU is the standard choice between layers because it trains faster in deep
networks than tanh or sigmoid.

## Further reading

- Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep
  Convolutional Neural Networks*, section 3.1: the paper that made ReLU
  mainstream.
- Glorot, Bordes & Bengio (2011), *Deep Sparse Rectifier Neural Networks*:
  why ReLU works well in deep networks.
