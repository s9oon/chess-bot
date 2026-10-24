# Loss function

```python
def loss_fn(model, boards, targets):
    scores = model(boards)
    return jnp.mean((scores - targets) ** 2)
```

The loss function is the arithmetic used to define the difference between the
correct answer (in this case Stockfish's evaluation) and our answer (the
network's score). Training tries to make this number as small as possible.

## Mean squared error (MSE)

For each board: take the error, **square** it, then **average** over the batch.

```
ours    Stockfish   error   squared
 0.20     0.60      -0.40     0.16
-0.10    -0.50       0.40     0.16
 0.70     0.80      -0.10     0.01
                             ─────
                      loss = 0.11
```

- **Squaring** makes every error positive, so a too-high and a too-low guess
  can't cancel out, and big mistakes cost much more than small ones.
- **Averaging** turns the whole batch into one number.

## Targets must match our range

The network outputs -1 .. +1 (tanh), so Stockfish's centipawn scores are
squashed into the same range first, e.g. `tanh(cp / 400)`. Mate scores become
-1 or +1.

## Why a batch of positions

Backpropagation works from this one loss number: it traces back how much each
weight contributed to it. One position on its own gives a noisy, unreliable
direction. Averaging over a batch of many positions, each with its own
Stockfish score, gives a much more reliable direction to nudge the weights.
