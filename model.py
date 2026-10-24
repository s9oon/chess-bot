import jax
import jax.numpy as jnp
from flax import nnx


class ChessCNN(nnx.Module):
    """A simple convolutional neural network for chess board evaluation."""

    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(12, 32, (3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(32, 32, (3, 3), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(32, 32, (3, 3), padding="SAME", rngs=rngs)

        # Batch normalization after each conv: one per output feature map (32).
        # Use model.train() while training and model.eval() when playing.
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        self.bn2 = nnx.BatchNorm(32, rngs=rngs)
        self.bn3 = nnx.BatchNorm(32, rngs=rngs)

        # Head: linear layers that evaluate the patterns the convs found.
        # linear1 is a hidden layer: 2048 flattened values -> 64 combinations.
        # linear2 is the output layer: 64 combinations -> 1 score.
        self.linear1 = nnx.Linear(8 * 8 * 32, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))  # conv1, then batchnorm, then ReLU
        x = nnx.relu(self.bn2(self.conv2(x)))
        x = nnx.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.shape[0], -1)  # flatten: (batch, 8, 8, 32) -> (batch, 2048)
        x = nnx.relu(self.linear1(x))  # (batch, 64): ReLU switches each combination on/off
        x = self.linear2(x)            # (batch, 1): the score
        x = jnp.tanh(x)               # squish the score into -1 (black winning) .. +1 (white winning)
        return x[:, 0]                 # (batch,): one score per board


if __name__ == "__main__":
    print("JAX", jax.__version__, "devices:", jax.devices())
