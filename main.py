import jax
import jax.numpy as jnp
import chess
from flax import nnx
import optax

# TODO: Encode the board.

class ChessCNN(nnx.Module):
    """Convolutional value network: (batch, 8, 8, 12) board planes -> (batch,) scores in [-1, 1]."""

    def __init__(self, channels: int = 64, *, rngs: nnx.Rngs):
        # Convolutional layers: 3x3 kernels, stride 1, SAME padding keeps the 8x8 grid.
        # Each layer's input channels match the previous layer's output channels.
        self.conv1 = nnx.Conv(12, channels, kernel_size=(3, 3), strides=1, padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), strides=1, padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(channels, channels, kernel_size=(3, 3), strides=1, padding="SAME", rngs=rngs)

        # Batch normalization after each conv layer, one per output channel.
        # Use model.train() during training and model.eval() for inference.
        self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
        self.bn2 = nnx.BatchNorm(channels, rngs=rngs)
        self.bn3 = nnx.BatchNorm(channels, rngs=rngs)

        # Fully connected head: flattened feature maps -> hidden layer -> single score.
        self.linear1 = nnx.Linear(8 * 8 * channels, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x):
        # TODO.

if __name__ == "__main__":
    print("JAX", jax.__version__, "devices:", jax.devices())
