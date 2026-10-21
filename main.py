import numpy as np
import jax
import jax.numpy as jnp
import chess
from flax import nnx
import optax


PIECE_NAMES = ["white pawn", "white knight", "white bishop", "white rook", "white queen", "white king",
               "black pawn", "black knight", "black bishop", "black rook", "black queen", "black king"]


def encode_board(board: chess.Board, debug: bool = False) -> jnp.ndarray:
    """Turn a board into 12 planes of 8x8: shape (8, 8, 12).

    debug=True prints what was built. Leave it off when encoding thousands of boards.
    """
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

    if debug:
        np.set_printoptions(precision=2, suppress=True, linewidth=120)
        print("\nencoded shape:", planes.shape)  # (8, 8, 12)

        for i in [0, 6]:  # just white pawns and black pawns
            # planes[:, :, i] = sheet i as an 8x8 grid; [::-1] puts rank 8 at the top like the board
            print(f"\nsheet {i}: {PIECE_NAMES[i]}")
            print(planes[:, :, i][::-1])

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

    board = chess.Board()
    x = encode_board(board, debug=True)  # prints the sheets

    # --- The layers: what each one holds (its learnable weights) ---
    print("\nlayer weights:")
    total = 0
    for name in ["conv1", "conv2", "conv3", "bn1", "bn2", "bn3", "linear1", "linear2"]:
        layer = getattr(model, name)  # same as model.conv1, model.conv2, ...
        params = nnx.state(layer, nnx.Param)  # just the learnable numbers
        for param_name, value in params.items():
            print(f"  {name}.{param_name}: {value[...].shape}")
            total += value[...].size
    print("total learnable numbers:", total)
