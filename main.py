import numpy as np
import jax
import jax.numpy as jnp
import chess
from flax import nnx
import optax


def encode_board(board: chess.Board) -> jnp.ndarray:
    """Turn a board into 12 planes of 8x8: shape (8, 8, 12)."""
    # Start with all zeros: 8 rows (ranks), 8 columns (files), 12 piece planes.
    planes = np.zeros((8, 8, 12), dtype=np.float32)

    # piece_map() gives {square: piece} for every piece on the board.
    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)  # 0 = rank 1, 7 = rank 8
        col = chess.square_file(square)  # 0 = a-file, 7 = h-file

        # Planes 0-5 are white pawn, knight, bishop, rook, queen, king.
        # piece_type is 1-6, so subtract 1 to get 0-5.
        plane = piece.piece_type - 1

        # Planes 6-11 are the same pieces for black.
        if piece.color == chess.BLACK:
            plane += 6

        # Mark "this piece is on this square".
        planes[row, col, plane] = 1.0

    # Convert to a JAX array so it can go on the GPU.
    return jnp.array(planes)

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

        # Head: one linear layer, a weight matrix applied to all 8 x 8 x 32 = 2048
        # flattened output feature map values at once.
        self.linear = nnx.Linear(8 * 8 * 32, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))  # conv1, then batchnorm, then ReLU
        x = nnx.relu(self.bn2(self.conv2(x)))
        x = nnx.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.shape[0], -1)  # flatten: (batch, 8, 8, 32) -> (batch, 2048)
        x = self.linear(x)             # (batch, 1): the score
        x = jnp.tanh(x)                # squish the score into -1 (black winning) .. +1 (white winning)
        return x[:, 0]                 # (batch,): one score per board


if __name__ == "__main__":
    print("JAX", jax.__version__, "devices:", jax.devices())