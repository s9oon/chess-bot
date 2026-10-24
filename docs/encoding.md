# Board encoding

A network only understands numbers, so `encode_board()` in
[`encoding.py`](../encoding.py) turns a chess position into an array:

```python
x = encode_board(chess.Board())   # shape (8, 8, 18), on the CPU
x = encode_fen(fen)               # same, from a FEN string (e.g. Lichess data)
x = encode_batch(boards)          # many boards: (batch, 8, 8, 18), uploaded to the GPU
```

## 18 input feature maps

The board is split into **18 grids of 8x8**: 12 for the pieces and 6 for the
rest of the game state.

| Maps | What |
|---|---|
| 0-11 | pieces |
| 12 | whose move |
| 13-16 | castling rights |
| 17 | en passant |

In the convolution paper's terms ([Dumoulin & Visin, 2016][paper]) these are
the network's **input feature maps**, also called **input channels**. The
paper's figures usually draw a single feature map; we have 18 stacked together.

### Piece maps (0-11)

One per piece type and colour. Each grid has a `1` where that piece is and `0`
everywhere else.

![Each map marks where one piece type is; the maps are stacked into one array](images/encoding.png)

Left to right: the printed maps for white pawns (map 0) and black pawns
(map 6), what those `1`s mean on the board, and all the maps stacked into one
array. The network gets the whole stack at once.

| Map | Piece | Map | Piece |
|---|---|---|---|
| 0 | white pawn | 6 | black pawn |
| 1 | white knight | 7 | black knight |
| 2 | white bishop | 8 | black bishop |
| 3 | white rook | 9 | black rook |
| 4 | white queen | 10 | black queen |
| 5 | white king | 11 | black king |

### Game state maps (12-17)

The pieces alone don't tell the whole story: the same position can be winning
or losing depending on these.

| Map | Meaning |
|---|---|
| 12 | whose move: **all 1s** if white to move, **all 0s** if black |
| 13 | white can castle kingside: all 1s if yes |
| 14 | white can castle queenside: all 1s if yes |
| 15 | black can castle kingside: all 1s if yes |
| 16 | black can castle queenside: all 1s if yes |
| 17 | en passant: a single `1` on the square a pawn could capture onto, if any |

Maps 12-16 are a yes/no for the whole board, so the whole map is filled with the
same value. That way every kernel sees it, wherever it sits on the board.

## The shape: `(8, 8, 18)`

```
planes[row, col, map]
       │    │    └─ which map (0-17)
       │    └────── file: 0 = a ... 7 = h
       └─────────── rank: 0 = rank 1 ... 7 = rank 8
```

For example, `planes[1, 4, 0]` is the white-pawn map on e2: `1.0` in the
starting position.

The network takes boards in batches, which adds a dimension at the front:
`(batch, 8, 8, 18)`. For one board: `encode_board(board)[None]`.

## Why one map per piece?

A single grid with pawn = 1, knight = 2 ... would make the network treat a
knight as "two pawns". Separate maps keep each piece type independent. This is
called **one-hot encoding**: each square has at most one `1` across the 12
piece maps, and an empty square is all `0`s.

## How the code works

1. Start with an `(8, 8, 18)` array of zeros.
2. For every piece on the board, work out its rank (row), file (column) and
   map (`piece_type - 1`, plus 6 for black), and put a `1` there.
3. Fill map 12 if it's white to move, and maps 13-16 for each castling right
   that's still available.
4. Put a `1` on map 17 at the en passant square, if there is one.
5. Return it as a NumPy array, still on the CPU.

The array is built with NumPy because NumPy arrays can be changed in place;
JAX arrays can't.

## Batching before the GPU

Copying data to the GPU has a fixed cost each time, so copying boards one at a
time is slow. `encode_batch` encodes every board on the CPU first, stacks them,
then uploads the whole batch to the GPU in **one** copy:

```
1000 boards, one-by-one upload:  ~420 ms
1000 boards, encode_batch:        ~25 ms
```

[paper]: https://arxiv.org/abs/1603.07285
