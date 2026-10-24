"""Every setting you might want to tweak, in one place."""
from pathlib import Path

ROOT = Path(__file__).parent

# --- Data ---
DATA_DIR = ROOT / "data"
EVALS_FILE = DATA_DIR / "evals_10M.jsonl"  # Lichess positions with Stockfish evaluations
NUM_POSITIONS = 3_000_000                  # how many positions to load and train on
CP_SCALE = 400                             # target = tanh(centipawns / CP_SCALE)
VAL_FRACTION = 0.05                        # share of positions held back for validation

# --- Model ---
CHANNELS = 64  # kernels per conv layer = output feature maps per layer
HIDDEN = 64    # numbers in the head's hidden linear layer

# --- Training ---
LEARNING_RATE = 1e-3     # how big each weight nudge is
BATCH_SIZE = 256         # boards per training step
MAX_EPOCHS = 1000        # passes over the training set (the time limit usually stops it first)
TRAIN_MINUTES = 20       # stop after this long
SAVE_EVERY_MINUTES = 5   # how often to save the latest weights

# --- Checkpoints ---
CHECKPOINT_DIR = ROOT / "checkpoints"
LATEST = CHECKPOINT_DIR / "latest.npz"  # saved every few minutes and at the end
BEST = CHECKPOINT_DIR / "best.npz"      # saved whenever validation loss improves
