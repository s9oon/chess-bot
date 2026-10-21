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

        # Value head: flattened feature maps -> hidden layer -> single score.
        self.linear1 = nnx.Linear(8 * 8 * channels, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x):
        # TODO.
        pass

if __name__ == "__main__":
    print("JAX", jax.__version__, "devices:", jax.devices())
    model = ChessCNN(channels=64, rngs=nnx.Rngs(0))
