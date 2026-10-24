# The neural network

How the pieces fit together, from board to score. Details on each part:
[encoding](encoding.md), [convolution](convolution.md),
[batch normalization](batch_normalization.md), [activations](activations.md),
[loss function](loss_function.md), [backpropagation](back_propagation.md).

## The layers

```
input (8, 8, 12)                  the encoded board
  → conv → batchnorm → ReLU   ┐
  → conv → batchnorm → ReLU   ├ hidden layers (conv)
  → conv → batchnorm → ReLU   ┘   (8, 8, 32) output feature maps
  → flatten                       (2048,) in one row
  → linear        ← the head: turns the features into the score
  → tanh                          the score, -1 (black winning) to +1 (white winning)
```

In a CNN like ours, the **hidden layers are conv layers** and **linear comes at
the end**. That end part is called the **head**.

The head can be **one** linear layer, or **two** with ReLU between them (e.g.
2048 → 64 → 1). In the two-layer version the middle linear layer is also a
hidden layer, one that sits in the head.

## Conv and linear are both layers

Both do the same affine transformation, `W · x + b`. They differ in what `x`
is:

| | Conv (`nnx.Conv`) | Linear (`nnx.Linear`) |
|---|---|---|
| `x` is | one small **patch**, repeated at every position | the whole **flattened** input, done once |
| weights | **shared**: the same kernel everywhere | every input has **its own** weight |
| sees | a local area | the whole board |
| job | find patterns | judge them into a score |

A linear layer is like a conv whose kernel covers the entire input, with no
sliding.

## The head

After the conv layers we **flatten** the output feature maps and apply a
**linear layer**: a weight matrix (plus bias) that turns them into the score.

```
score = W · x + b
        │   │   └─ bias
        │   └───── the flattened output feature maps (2048 numbers)
        └───────── the linear layer's weight matrix
```

The **weights** live inside `W`: learned and fixed. What comes out is a
**value**: the score for this board.

Finally, **tanh** takes that score and squishes it into a number between
**-1 and +1**: -1 means black is winning, 0 is even, +1 means white is winning.

### Why a linear head?

Conv layers are good at **spotting patterns**, but not at **evaluating how
important they are**. That's why we stack linear layers on top:

```
conv layers  →  detect patterns    (what's going on, and where)
linear head  →  evaluate them      (so who's winning?)
```

Conv layers have two blind spots:

1. **They only see locally.** After three conv layers each number only knows
   about a 7x7 area, but an evaluation is about the whole board.
2. **They treat every square the same.** A kernel uses the same weights
   everywhere, so it can't tell that a pawn on the 7th rank matters more than
   one on the 2nd.

The linear layer connects every input to the output and gives each one its own
weight, so it sees the whole board and can learn that *where* something is
matters.

## CNN vs FNN

| | Layers |
|---|---|
| **FNN** (feed-forward / fully connected network) | **only** linear layers: every layer, hidden ones included |
| **CNN** | conv layers to find patterns, usually **plus** a small linear head |

## Status in `main.py`

- ✅ conv layers with batchnorm and ReLU
- ✅ flatten, linear head (two layers: 2048 → 64 → ReLU → 1), tanh
- ⬜ training (the scores are random until then)
