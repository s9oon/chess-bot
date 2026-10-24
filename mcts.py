"""Monte Carlo Tree Search, guided by the trained network.

The network only outputs a score (no policy head), so when a position is expanded,
every legal move is scored in one batched GPU call. Those scores become:
  - the move priors: which moves to explore first
  - each move's starting value, before it has been visited

All values inside the tree are from the point of view of the side to move, -1 .. +1.
"""
import math

import numpy as np
import chess
from flax import nnx

from config import C_PUCT, MCTS_SIMULATIONS, PRIOR_SHARPNESS
from encoding import NUM_PLANES, encode_board
from model import ChessCNN

EVAL_BATCH = 256  # fixed batch size so jit compiles once (no position has more than 218 legal moves)


@nnx.jit
def _forward(model: ChessCNN, boards):
    return model(boards)


def evaluate(model: ChessCNN, boards: list[chess.Board]) -> np.ndarray:
    """Network scores for up to 256 boards, from white's point of view."""
    x = np.zeros((EVAL_BATCH, 8, 8, NUM_PLANES), dtype=np.float32)
    for i, b in enumerate(boards):
        x[i] = encode_board(b)
    return np.asarray(_forward(model, x))[:len(boards)]


def side_value(white_score: float, turn: bool) -> float:
    """Turn a white's-point-of-view score into the side to move's point of view."""
    return white_score if turn == chess.WHITE else -white_score


class Node:
    """One position in the search tree."""

    def __init__(self, prior: float, value: float, terminal: bool = False):
        self.prior = prior          # how promising the move into this node looked before searching
        self.value = value          # network (or exact) value for the side to move here
        self.terminal = terminal    # game over here: value is exact
        self.visits = 0
        self.value_sum = 0.0        # summed values for the player who moved INTO this node
        self.children: dict[chess.Move, Node] = {}

    def q(self) -> float:
        """Average value for the player who moved into this node."""
        if self.visits == 0:
            return -self.value      # not searched yet: fall back on the network's opinion
        return self.value_sum / self.visits


def expand(node: Node, board: chess.Board, model: ChessCNN) -> None:
    """Create a child for every legal move, scoring all of them in one batched network call."""
    moves = list(board.legal_moves)
    children_boards, terminal = [], {}
    for i, move in enumerate(moves):
        board.push(move)
        if board.is_checkmate():
            terminal[i] = -1.0      # side to move is mated
        elif board.is_game_over():
            terminal[i] = 0.0       # stalemate / draw
        children_boards.append(board.copy(stack=False))
        board.pop()

    scores = evaluate(model, children_boards)
    child_turn = not board.turn
    values = np.array([terminal.get(i, side_value(s, child_turn)) for i, s in enumerate(scores)])

    # Priors: a softmax over how good each move looks for the player making it (-child value).
    logits = -values * PRIOR_SHARPNESS
    priors = np.exp(logits - logits.max())
    priors /= priors.sum()

    for i, move in enumerate(moves):
        # float(): plain Python numbers, not NumPy ones (keeps maths simple and JSON-friendly).
        node.children[move] = Node(float(priors[i]), float(values[i]), terminal=i in terminal)


def select(node: Node) -> tuple[chess.Move, Node]:
    """PUCT: pick the child with the best mix of 'looks good' (q) and 'barely explored' (prior / visits)."""
    sqrt_visits = math.sqrt(node.visits)

    def score(child: Node) -> float:
        return child.q() + C_PUCT * child.prior * sqrt_visits / (1 + child.visits)

    return max(node.children.items(), key=lambda item: score(item[1]))


def search(board: chess.Board, model: ChessCNN, simulations: int = MCTS_SIMULATIONS) -> Node:
    """Run MCTS from this position and return the root, whose children hold the visit counts."""
    root = Node(1.0, side_value(float(evaluate(model, [board])[0]), board.turn))
    if board.is_game_over():
        root.terminal = True
        return root
    expand(root, board, model)

    for _ in range(simulations):
        node, b, path = root, board.copy(), [root]

        # 1. Selection: walk down the most promising line until reaching an unexpanded position.
        while node.children and not node.terminal:
            move, node = select(node)
            b.push(move)
            path.append(node)

        # 2. Expansion: give the new position its children (and their network scores).
        if not node.terminal:
            expand(node, b, model)

        # 3. Backup: the position's value flows back up, flipping sides at every level.
        value = node.value
        for n in reversed(path):
            n.visits += 1
            n.value_sum += -value   # stored for the player who moved into n
            value = -value

    return root


def is_mate(child: Node) -> bool:
    """This move checkmates: the opponent is left with value -1 and no moves."""
    return child.terminal and child.value == -1.0


def best_move(root: Node) -> chess.Move:
    """Play a checkmate if there is one, otherwise the most visited move."""
    return max(root.children.items(), key=lambda item: (is_mate(item[1]), item[1].visits))[0]


def ranked_moves(root: Node, board: chess.Board) -> list[dict]:
    """Root moves, checkmates first, then most visited, with each move's value from white's point of view."""
    total = max(sum(c.visits for c in root.children.values()), 1)
    sign = 1 if board.turn == chess.WHITE else -1
    moves = sorted(root.children.items(), key=lambda item: (is_mate(item[1]), item[1].visits), reverse=True)
    return [{"uci": m.uci(), "san": board.san(m), "share": c.visits / total, "eval": sign * c.q()}
            for m, c in moves]


def position_eval(root: Node, board: chess.Board) -> float:
    """The search's overall verdict on this position, from white's point of view."""
    if root.terminal or not root.children:
        return side_value(root.value, board.turn)
    visited = [c for c in root.children.values() if c.visits]
    if not visited:
        return side_value(root.value, board.turn)
    value = sum(c.visits * c.q() for c in visited) / sum(c.visits for c in visited)
    return side_value(value, board.turn)
