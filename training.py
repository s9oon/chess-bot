import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx

from data import load_positions, unpack_boards
from model import ChessCNN, save_model

LEARNING_RATE = 1e-3        # how big each weight nudge is
NUM_POSITIONS = 200_000     # how many positions to load from data/
BATCH_SIZE = 256            # boards per training step
MAX_EPOCHS = 1000           # passes over the training set (the time limit usually stops it first)
TRAIN_MINUTES = 60          # stop after this long
SAVE_EVERY_MINUTES = 5      # how often to save the latest weights

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
LATEST = CHECKPOINT_DIR / "latest.npz"  # saved every 5 minutes and at the end
BEST = CHECKPOINT_DIR / "best.npz"      # saved whenever validation loss improves


def loss_fn(model, boards, targets):
    scores = model(boards)
    return jnp.mean((scores - targets) ** 2)


def create_optimizer(model: ChessCNN) -> nnx.Optimizer:
    """Adam: the most popular version of gradient descent. Only the weights (nnx.Param) get nudged."""
    return nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(model: ChessCNN, optimizer: nnx.Optimizer, boards, targets):
    # Backpropagation: the loss, plus which way to nudge every weight (the gradients).
    loss, grads = nnx.value_and_grad(loss_fn)(model, boards, targets)
    # Gradient descent: nudge every weight a small step against its gradient (in GPU memory).
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: ChessCNN, boards, targets):
    return loss_fn(model, boards, targets)


def batches(boards: np.ndarray, scores: np.ndarray, shuffle: bool):
    """Yield (boards, scores) batches, unpacked and ready for the GPU.

    Drops the last partial batch so every batch has the same shape (jit only compiles once).
    """
    order = np.random.permutation(len(boards)) if shuffle else np.arange(len(boards))
    for start in range(0, len(boards) - BATCH_SIZE + 1, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        yield unpack_boards(boards[idx]), scores[idx]


def train(num_positions: int = NUM_POSITIONS, train_minutes: float = TRAIN_MINUTES,
          save_every_minutes: float = SAVE_EVERY_MINUTES) -> None:
    print(f"loading {num_positions:,} positions...")
    boards, scores = load_positions(num_positions)

    # Hold back 5% the network never trains on, to check it isn't just memorising.
    split = int(len(boards) * 0.95)
    train_boards, train_scores = boards[:split], scores[:split]
    val_boards, val_scores = boards[split:], scores[split:]

    # Baseline: the loss from always guessing the average score. Training only works if we beat it.
    baseline = float(np.mean((val_scores - train_scores.mean()) ** 2))
    print(f"train: {len(train_boards):,}  val: {len(val_boards):,}  baseline loss: {baseline:.4f}")

    model = ChessCNN(rngs=nnx.Rngs(0))
    optimizer = create_optimizer(model)

    start = time.time()
    last_save = start
    best_val = float("inf")
    out_of_time = False

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        # Train on every batch, saving the latest weights every few minutes.
        model.train()  # batchnorm uses each batch's statistics
        train_losses = []
        for b, s in batches(train_boards, train_scores, shuffle=True):
            train_losses.append(train_step(model, optimizer, b, s))

            now = time.time()
            if now - last_save >= save_every_minutes * 60:
                save_model(model, LATEST)
                last_save = now
                print(f"  [{(now - start) / 60:.1f} min] saved {LATEST.name}")
            if now - start >= train_minutes * 60:
                out_of_time = True
                break

        # Check on positions it never trained on.
        model.eval()   # batchnorm uses its running averages
        val_loss = float(np.mean([eval_step(model, b, s) for b, s in batches(val_boards, val_scores, False)]))
        print(f"epoch {epoch}: train loss {np.mean(train_losses):.4f}  val loss {val_loss:.4f}  "
              f"({time.time() - epoch_start:.0f}s)")

        if val_loss < best_val:
            best_val = val_loss
            save_model(model, BEST)
            print(f"  new best, saved {BEST.name}")

        if out_of_time:
            print(f"time limit of {train_minutes} min reached")
            break

    save_model(model, LATEST)
    print(f"done: saved {LATEST.name} (final) and {BEST.name} (val loss {best_val:.4f})")


if __name__ == "__main__":
    train()
