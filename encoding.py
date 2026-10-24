import numpy as np
import jax.numpy as jnp
import chess

# 0-11 pieces, 12 whose move, 13-16 castling rights, 17 en passant square.
NUM_PLANES = 18


def encode_board(board: chess.Board) -> np.ndarray:
    """Turn a board into 18 planes of 8x8: shape (8, 8, 18)."""
    planes = np.zeros((8, 8, NUM_PLANES), dtype=np.float32)

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

    # Plane 12: whose move. All 1s if white to move, all 0s if black.
    if board.turn == chess.WHITE:
        planes[:, :, 12] = 1.0

    # Planes 13-16: castling rights. All 1s if that side can still castle that way.
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[:, :, 13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[:, :, 14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[:, :, 15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[:, :, 16] = 1.0

    # Plane 17: en passant. A single 1 on the square a pawn could capture onto, if any.
    if board.ep_square is not None:
        planes[chess.square_rank(board.ep_square), chess.square_file(board.ep_square), 17] = 1.0

    # Stays on the CPU as a NumPy array: copying boards to the GPU one at a time is slow.
    return planes


def encode_fen(fen: str) -> np.ndarray:
    """Turn a FEN string (a position as text, e.g. from Lichess) into (8, 8, 18) planes."""
    # python-chess reads the FEN into a board, then it's encoded like any other board.
    return encode_board(chess.Board(fen))


def encode_batch(boards: list[chess.Board]) -> jnp.ndarray:
    """Encode many boards on the CPU, then upload them to the GPU in one go: (batch, 8, 8, 18)."""
    return jnp.asarray(np.stack([encode_board(b) for b in boards]))
