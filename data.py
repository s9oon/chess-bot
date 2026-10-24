import json
from pathlib import Path

import numpy as np

from encoding import NUM_PLANES, encode_fen

DATA_DIR = Path(__file__).parent / "data"
EVALS_FILE = DATA_DIR / "evals_10M.jsonl"  # Lichess positions with Stockfish evaluations


def best_score(evals: list[dict]) -> float:
    """Turn a position's Stockfish evaluations into one target score in -1 .. +1 (white's point of view)."""
    deepest = max(evals, key=lambda e: e["depth"])  # the evaluation Stockfish searched furthest
    best_line = deepest["pvs"][0]                   # its first line = Stockfish's best move

    if "mate" in best_line:
        # Mate in N has no centipawn value: treat it as completely winning or losing.
        return 1.0 if best_line["mate"] > 0 else -1.0

    # Centipawns (100 = one pawn up) squashed into -1 .. +1, the same range as the network's tanh.
    return float(np.tanh(best_line["cp"] / 400))


BOARD_SHAPE = (8, 8, NUM_PLANES)
PACKED_SIZE = 8 * 8 * NUM_PLANES // 8  # 1152 values of 0/1 packed 8 per byte = 144 bytes per board


def unpack_boards(packed: np.ndarray) -> np.ndarray:
    """Unpack (batch, 144) packed boards back into (batch, 8, 8, 18) float32 feature maps."""
    return np.unpackbits(packed, axis=1).reshape(-1, *BOARD_SHAPE).astype(np.float32)


def load_positions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the first n positions as (boards, scores), both on the CPU.

    boards: (n, 144) packed feature maps: every 0/1 stored as a single bit, 144 bytes per board
            instead of 1152, so millions of positions fit in RAM. Unpack batches with unpack_boards.
    scores: (n,) target scores in -1 .. +1.

    Encoding is slow, so the result is cached in data/ and loaded straight from there next time.
    The plane count is in the cache name, so changing the encoding builds a fresh cache.
    """
    cache = DATA_DIR / f"encoded_{n}_{NUM_PLANES}planes.npz"
    if cache.exists():
        saved = np.load(cache)
        return saved["boards"], saved["scores"]

    boards = np.zeros((n, PACKED_SIZE), dtype=np.uint8)
    scores = np.zeros(n, dtype=np.float32)

    # Read one line (one position) at a time, so the whole file never sits in memory.
    with open(EVALS_FILE) as f:
        for i, line in enumerate(f):
            if i == n:
                break
            position = json.loads(line)
            boards[i] = np.packbits(encode_fen(position["fen"]).astype(np.uint8))
            scores[i] = best_score(position["evals"])
            if (i + 1) % 50_000 == 0:
                print(f"  encoded {i + 1:,} / {n:,}")

    if i < n:
        # The file ran out early: keep only the positions actually read.
        boards, scores = boards[:i + 1], scores[:i + 1]

    np.savez(cache, boards=boards, scores=scores)
    return boards, scores


if __name__ == "__main__":
    boards, scores = load_positions(200_000)
    print("boards:", boards.shape, boards.dtype, f"{boards.nbytes / 1e6:.0f} MB (packed)")
    print("unpacked batch of 4:", unpack_boards(boards[:4]).shape)
    print("scores:", scores.shape, scores.dtype, f"min {scores.min():.2f}  max {scores.max():.2f}  mean {scores.mean():.2f}")
