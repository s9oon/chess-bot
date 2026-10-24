"""Local server to play against the bot. Run: python server.py, then open http://localhost:8000"""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess

from config import BEST, CP_SCALE, NUM_SUGGESTIONS, PORT, ROOT
from mcts import best_move, position_eval, ranked_moves, search
from model import load_model

model = load_model(BEST)
model.eval()

lock = threading.Lock()  # requests run on separate threads; only one may touch the game at once
game = {"board": chess.Board(), "player": chess.WHITE, "last_move": None, "eval": 0.0, "suggestions": []}


def to_pawns(score: float) -> float:
    """Undo the training squash (tanh(cp / CP_SCALE)) to show the score in pawns."""
    score = max(min(score, 0.999), -0.999)
    return CP_SCALE * math.atanh(score) / 100


def bot_move() -> None:
    board = game["board"]
    move = best_move(search(board, model))
    board.push(move)
    game["last_move"] = move.uci()


def analyse_for_player() -> None:
    """Search the player's position: gives the eval bar and the suggested moves."""
    board = game["board"]
    root = search(board, model)
    game["eval"] = position_eval(root, board)
    game["suggestions"] = ranked_moves(root, board)[:NUM_SUGGESTIONS] if not board.is_game_over() else []


def state() -> dict:
    board = game["board"]
    return {
        "fen": board.fen(),
        "player": "white" if game["player"] == chess.WHITE else "black",
        "legal": [m.uci() for m in board.legal_moves] if board.turn == game["player"] else [],
        "last_move": game["last_move"],
        "eval": game["eval"],                     # -1 .. +1, white's point of view
        "eval_pawns": to_pawns(game["eval"]),
        "suggestions": [{**s, "eval_pawns": to_pawns(s["eval"])} for s in game["suggestions"]],
        "result": board.result() if board.is_game_over() else None,
    }


def new_game(player: bool) -> None:
    game.update(board=chess.Board(), player=player, last_move=None, eval=0.0, suggestions=[])
    if player == chess.BLACK:
        bot_move()  # the bot plays white, so it moves first
    analyse_for_player()


def play(uci: str) -> bool:
    board = game["board"]
    move = chess.Move.from_uci(uci)
    if board.turn != game["player"] or move not in board.legal_moves:
        return False
    board.push(move)
    game["last_move"] = uci
    if not board.is_game_over():
        bot_move()
    analyse_for_player()
    return True


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (ROOT / "web" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            with lock:
                self.send_json(state())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        with lock:
            if self.path == "/new":
                new_game(chess.BLACK if data.get("color") == "black" else chess.WHITE)
                self.send_json(state())
            elif self.path == "/move":
                if play(data.get("uci", "")):
                    self.send_json(state())
                else:
                    self.send_json({"error": "illegal move"}, 400)
            else:
                self.send_error(404)

    def log_message(self, *args):
        pass  # keep the terminal quiet


if __name__ == "__main__":
    with lock:
        analyse_for_player()  # warm up the network and give the first suggestions
    print(f"Play at http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
