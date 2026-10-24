# Backpropagation and gradient descent

```python
loss, grads = nnx.value_and_grad(loss_fn)(model, boards, targets)
```

Training repeats two steps: **backpropagation** works out how each weight
should change, and **gradient descent** changes it.

## Backpropagation

The [loss](loss_function.md) is one number: how far our scores are from
Stockfish's. Backpropagation works out, for **every weight**, how the loss would
change if that weight were nudged a tiny bit. That's the weight's **gradient**.

- **Positive gradient:** increasing the weight makes the loss bigger, so it
  should go down.
- **Negative gradient:** increasing the weight makes the loss smaller, so it
  should go up.
- **Big gradient:** this weight has a big effect on the error.

It does this with **differentiation**, using the **chain rule**. The network is
a chain of layers, each feeding the next:

```
board → conv1 → conv2 → conv3 → linear1 → linear2 → tanh → score → loss
```

The chain rule says the effect of a weight on the loss is the effect it has on
its own layer's output, multiplied by the effect of that output on the next
layer, and so on all the way to the loss. Backpropagation starts at the loss and
works **backwards** through the layers, which is where the name comes from.

We never write this by hand: `nnx.value_and_grad` does the differentiation
automatically and returns the loss plus a gradient for every weight.

## Gradient descent

Once every weight has a gradient, nudge each one a small step **against** it,
downhill on the loss:

```
new weight = weight - learning_rate × gradient
```

The **learning rate** (e.g. `0.001`) is how big each step is:

- too big, and the weights overshoot and the loss jumps around;
- too small, and training takes forever.

Repeat over many batches and the loss keeps going down, so the network's scores
get closer and closer to Stockfish's.

## A one-weight example

Say the network were just `score = w × x`, with `w = 0.5`, `x = 1.0` and
Stockfish's answer `0.8`:

```
score    = 0.5 × 1.0      = 0.5
loss     = (0.5 - 0.8)²   = 0.09
gradient = 2 × (0.5 - 0.8) × 1.0 = -0.6    (negative: w should go up)
new w    = 0.5 - 0.1 × (-0.6)    = 0.56    (learning rate 0.1)
```

With `w = 0.56` the score is `0.56`, closer to `0.8`, and the loss drops to
`0.058`. The real network does exactly this, for all its weights at once.
