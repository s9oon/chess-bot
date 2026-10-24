import json
from pathlib import Path

import numpy as np

from encoding import encode_fen

DATA_DIR = Path(__file__).parent / "data"
EVALS_FILE = DATA_DIR / "evals_1M.jsonl"  # Lichess positions with Stockfish evaluations


def best_score(evals: list[dict]) -> float:
    """Turn a position's Stockfish evaluations into one target score in -1 .. +1 (white's point of view)."""
    deepest = max(evals, key=lambda e: e["depth"])  # the evaluation Stockfish searched furthest
    best_line = deepest["pvs"][0]                   # its first line = Stockfish's best move

    if "mate" in best_line:
        # Mate in N has no centipawn value: treat it as completely winning or losing.
        return 1.0 if best_line["mate"] > 0 else -1.0

    # Centipawns (100 = one pawn up) squashed into -1 .. +1, the same range as the network's tanh.
    return float(np.tanh(best_line["cp"] / 400))


def load_positions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the first n positions as (boards, scores), both on the CPU.

    boards: (n, 8, 8, 12) feature maps, stored as uint8 (0/1) to save memory.
    scores: (n,) target scores in -1 .. +1.

    Encoding is slow, so the result is cached in data/ and loaded straight from there next time.
    Delete the cache if the encoding changes.
    """
    cache = DATA_DIR / f"encoded_{n}.npz"
    if cache.exists():
        saved = np.load(cache)
        return saved["boards"], saved["scores"]

    boards = np.zeros((n, 8, 8, 12), dtype=np.uint8)
    scores = np.zeros(n, dtype=np.float32)

    # Read one line (one position) at a time, so the whole file never sits in memory.
    with open(EVALS_FILE) as f:
        for i, line in enumerate(f):
            if i == n:
                break
            position = json.loads(line)
            boards[i] = encode_fen(position["fen"])
            scores[i] = best_score(position["evals"])
            if (i + 1) % 50_000 == 0:
                print(f"  encoded {i + 1:,} / {n:,}")

    np.savez(cache, boards=boards, scores=scores)
    return boards, scores


if __name__ == "__main__":
    boards, scores = load_positions(200_000)
    print("boards:", boards.shape, boards.dtype, f"{boards.nbytes / 1e6:.0f} MB")
    print("scores:", scores.shape, scores.dtype, f"min {scores.min():.2f}  max {scores.max():.2f}  mean {scores.mean():.2f}")
