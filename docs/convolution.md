# Convolution layers

```python
self.conv1 = nnx.Conv(12, 32, (3, 3), padding="SAME", rngs=rngs)
self.conv2 = nnx.Conv(32, 32, (3, 3), padding="SAME", rngs=rngs)
self.conv3 = nnx.Conv(32, 32, (3, 3), padding="SAME", rngs=rngs)
```

Terms and figures follow *A guide to convolution arithmetic for deep
learning* ([Dumoulin & Visin, 2016][paper]).

## What a convolution is

A **kernel** is a matrix of weights that is applied to each flattened
**patch** vector on the input feature map. Applying it and adding a bias is
an **affine transformation**:

```
output value = W · x + b
               │   │   └─ bias
               │   └───── the flattened patch
               └───────── the kernel's weights
```

Doing that at every position, with the **same** kernel, builds one **output
feature map**. That whole process is the convolution.

![A 3x3 kernel sliding over a padded input to build the output](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/same_padding_no_strides.gif)

*Blue: input feature map. Shaded: the patch under the kernel. Green: output
feature map. Figure from [vdumoulin/conv_arithmetic][repo] (MIT licence).*

Two things make it a convolution rather than a normal (fully connected)
layer:

- **Local:** each output only looks at a small patch, not the whole input.
- **Shared weights:** the same kernel is used at every position.

## Our settings, in the paper's terms

| Code | Paper's term | Ours |
|---|---|---|
| first number | input channels (input feature maps) | 12 for conv1, 32 after |
| second number | output channels, one per kernel | 32 |
| `(3, 3)` | kernel size *k* | 3 |
| `strides` (default) | unit strides, *s* = 1 | moves 1 square at a time |
| `padding="SAME"` | half (same) padding, *p* = 1 | one ring of zeros around the board |

The paper's figure above is exactly our case: **half padding, unit strides**,
where the output is the same size as the input.

## Stride

The **stride** is how far the kernel jumps between patches, across and down.

- **Stride 1 (ours):** the kernel visits every position, like the figure
  above. Our 8x8 board gives an 8x8 output.
- **Stride 2:** the kernel skips every other position, so the output shrinks.
  Our 8x8 board would give a 4x4 output.

![A 3x3 kernel moving with stride 2 over a padded input](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/padding_strides.gif)

*Stride 2: the kernel jumps two squares at a time, so the output (green) is
smaller. Figure from [vdumoulin/conv_arithmetic][repo] (MIT licence).*

We use stride 1 because in chess every square matters, so we want an output
for each one.

## Padding

**Padding** adds a ring of zeros around the input, so the kernel can still sit
on the edge squares.

- **`padding="SAME"` (ours):** with a 3x3 kernel this adds one ring of zeros
  (*p* = 1). The kernel fits on every square, including the edges, and the
  board stays 8x8. This is the paper's **half (same) padding**, shown in the
  first figure.
- **`padding="VALID"`:** no zeros. The kernel can only sit where it fits
  completely, so the output shrinks: our 8x8 board would give a 6x6 output.

![A 3x3 kernel over an unpadded input: the output is smaller](https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_no_strides.gif)

*No padding: the kernel can't hang over the edge, so the output (green) is
smaller than the input. Figure from [vdumoulin/conv_arithmetic][repo] (MIT
licence).*

We use `"SAME"` because edge squares matter in chess (rooks start in the
corners, kings castle to the side), and so the board stays 8x8 through every
layer.

## Output size

The paper's formula:

```
o = (i + 2p - k) / s + 1
```

For our board (*i* = 8, *p* = 1, *k* = 3, *s* = 1):

```
o = (8 + 2 - 3) / 1 + 1 = 8
```

So every conv layer keeps the board at **8x8**.

## Many channels

The paper's figures show **one** input feature map and **one** kernel. Ours
has more of both:

- **Each kernel spans all input channels.** conv1's kernels are 3 x 3 x 12 =
  **108** weights; conv2's and conv3's are 3 x 3 x 32 = **288**.
- **Every kernel is applied to every patch.** 32 kernels give 32 output
  feature maps.

As one matrix multiply, for conv1:

```
patches (64, 108)  @  kernels (108, 32)  =  output (64, 32)  ->  (8, 8, 32)
   │                     │
   │                     └─ one column per kernel
   └─ one row per position (8 x 8), each a flattened 3x3x12 patch
```

## Shapes through the layers

```
board   (8, 8, 12)
  ↓ conv1   12 -> 32
        (8, 8, 32)
  ↓ conv2   32 -> 32
        (8, 8, 32)
  ↓ conv3   32 -> 32
        (8, 8, 32)
```

Each layer's input channels must match the previous layer's output channels.

[paper]: https://arxiv.org/abs/1603.07285
[repo]: https://github.com/vdumoulin/conv_arithmetic
