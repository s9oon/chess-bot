from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from config import CHANNELS, HIDDEN
from encoding import NUM_PLANES


class ChessCNN(nnx.Module):
    """A simple convolutional neural network for chess board evaluation."""

    def __init__(self, rngs: nnx.Rngs):
        # conv1 reads every input feature map from encoding.py (18: pieces, turn, castling, en passant).
        self.conv1 = nnx.Conv(NUM_PLANES, CHANNELS, (3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(CHANNELS, CHANNELS, (3, 3), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(CHANNELS, CHANNELS, (3, 3), padding="SAME", rngs=rngs)

        # Batch normalization after each conv: one per output feature map.
        # Use model.train() while training and model.eval() when playing.
        self.bn1 = nnx.BatchNorm(CHANNELS, rngs=rngs)
        self.bn2 = nnx.BatchNorm(CHANNELS, rngs=rngs)
        self.bn3 = nnx.BatchNorm(CHANNELS, rngs=rngs)

        # Head: linear layers that evaluate the patterns the convs found.
        # linear1 is a hidden layer: 8 x 8 x CHANNELS flattened values -> HIDDEN combinations.
        # linear2 is the output layer: HIDDEN combinations -> 1 score.
        self.linear1 = nnx.Linear(8 * 8 * CHANNELS, HIDDEN, rngs=rngs)
        self.linear2 = nnx.Linear(HIDDEN, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))  # conv1, then batchnorm, then ReLU
        x = nnx.relu(self.bn2(self.conv2(x)))
        x = nnx.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.shape[0], -1)  # flatten: (batch, 8, 8, CHANNELS) -> (batch, 8 * 8 * CHANNELS)
        x = nnx.relu(self.linear1(x))  # (batch, HIDDEN): ReLU switches each combination on/off
        x = self.linear2(x)            # (batch, 1): the score
        x = jnp.tanh(x)               # squish the score into -1 (black winning) .. +1 (white winning)
        return x[:, 0]                 # (batch,): one score per board


def save_model(model: ChessCNN, path: Path) -> None:
    """Save every weight (and batchnorm's running averages) to a .npz file."""
    leaves = jax.tree.leaves(nnx.state(model))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, *[np.asarray(leaf) for leaf in leaves])


def load_model(path: Path) -> ChessCNN:
    """Build a fresh ChessCNN, then overwrite its random weights with the saved ones."""
    model = ChessCNN(rngs=nnx.Rngs(0))
    saved = np.load(path)
    leaves = [saved[f"arr_{i}"] for i in range(len(saved.files))]
    _, structure = jax.tree.flatten(nnx.state(model))
    nnx.update(model, jax.tree.unflatten(structure, leaves))
    return model


if __name__ == "__main__":
    print("JAX", jax.__version__, "devices:", jax.devices())
