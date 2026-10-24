import numpy as np
import jax.numpy as jnp
import chess


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


def encode_fen(fen: str) -> jnp.ndarray:
    """Turn a FEN string (a position as text, e.g. from Lichess) into (8, 8, 12) planes."""
    # python-chess reads the FEN into a board, then it's encoded like any other board.
    return encode_board(chess.Board(fen))
