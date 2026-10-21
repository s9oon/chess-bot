# Playground for looking at the arrays inside the network, step by step.
# Run it (from chess-bot/):  python dev_logging/explore.py
# Each "# %%" line starts a cell you can run on its own in VS Code (needs ipykernel).

import numpy as np
import jax.numpy as jnp
import chess
from flax import nnx

# Print 2 decimal places, no scientific notation (applies to JAX arrays too).
np.set_printoptions(precision=2, suppress=True, linewidth=120)

PIECES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def show(name, arr):
    print(f"\n=== {name} ===  shape={arr.shape}")


# %% 1. Board -> planes  (8, 8, 12)
board = chess.Board()
board.push_san("e4")  # play 1. e4 so the board isn't perfectly symmetric

planes = np.zeros((8, 8, 12), dtype=np.float32)
for square, piece in board.piece_map().items():
    row = chess.square_rank(square)
    col = chess.square_file(square)
    plane = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
    planes[row, col, plane] = 1.0
planes = jnp.array(planes)

print(board)
show("planes", planes)
# planes[:, :, 0] = every row, every column, plane 0 (white pawns) -> an 8x8 grid.
# [::-1] flips it so rank 8 is at the top, like the board above.
print("white pawns (plane 0):\n", planes[:, :, 0][::-1])
print("black knights (plane 7):\n", planes[:, :, 7][::-1])
print("the 12 numbers for square e4 (one per plane):", planes[3, 4])

# %% 2. Add a batch dimension  (1, 8, 8, 12)
x = planes[None]
show("batched input", x)

# %% 3. Conv layer  (1, 8, 8, 12) -> (1, 8, 8, 64)
rngs = nnx.Rngs(0)
conv1 = nnx.Conv(12, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
show("conv1 kernel (the weights)", conv1.kernel[...])  # (3, 3, 12, 64): a 3x3 window, 12 planes in, 64 features out
h = nnx.relu(conv1(x))
show("after conv1 + relu", h)
print("first 8 of the 64 features for square e4:", h[0, 3, 4, :8])
print("(random for now; after training these mean things like 'attacked', 'defended', ...)")

# %% 4. Flatten  (1, 8, 8, 64) -> (1, 4096)
flat = h.reshape(h.shape[0], -1)
show("flattened", flat)
print("same numbers, just one long row: 8 * 8 * 64 =", 8 * 8 * 64)

# %% 5. Linear = weighted sum. The chess material count IS a linear layer:
counts = jnp.array([8.0, 2, 2, 2, 1])   # pawns, knights, bishops, rooks, queens
weights = jnp.array([1.0, 3, 3, 5, 9])  # how much each one is worth
print("\n=== material count as a linear layer ===")
print("counts * weights =", counts * weights)
print("sum               =", jnp.sum(counts * weights))
print("counts @ weights  =", counts @ weights, " (@ does multiply-then-sum in one go)")

# %% 6. A real Linear layer  (1, 4096) -> (1, 64)
linear1 = nnx.Linear(8 * 8 * 64, 64, rngs=rngs)
show("linear1 W", linear1.kernel[...])  # (4096, 64): 64 different 'scorecards', each with 4096 weights
show("linear1 b", linear1.bias[...])    # (64,): one bias per scorecard
by_hand = flat @ linear1.kernel[...] + linear1.bias[...]
by_layer = linear1(flat)
show("linear1 output", by_layer)
print("by hand  :", by_hand[0, :6])
print("by layer :", by_layer[0, :6], " <- identical: nnx.Linear is just x @ W + b")

# %% 7. Final score  (1, 64) -> (1, 1) -> tanh
linear2 = nnx.Linear(64, 1, rngs=rngs)
raw = linear2(nnx.relu(by_layer))
score = jnp.tanh(raw)
show("score", score)
print("raw:", raw, " after tanh:", score, " (untrained, so meaningless for now)")
print("tanh squashes any number into -1..1:", jnp.tanh(jnp.array([-100.0, -1, 0, 1, 100])))
