# Board encoding

A neural network only understands numbers, so before a position can go into
`ChessCNN` it has to be turned into an array. That's the job of
`encode_board()` in [`main.py`](../main.py).

```python
x = encode_board(board)   # board: chess.Board  ->  x: shape (8, 8, 12)
```

## The idea: 12 sheets

The board is split into **12 sheets**, one for each piece type and colour.
Each sheet is an 8x8 grid that answers one yes/no question for every square,
e.g. *"is there a white pawn here?"*: `1` = yes, `0` = no.

![Each sheet marks where one piece type is; the 12 sheets are stacked into one array](images/encoding.png)

From left to right:

1. **The printed sheets.** Sheet 0 (white pawns) has a row of `1`s on rank 2;
   sheet 6 (black pawns) has a row of `1`s on rank 7.
2. **What they mean on the board.** Each `1` is a square with that piece on it.
3. **The stack.** All 12 sheets are stacked on top of each other into **one**
   array. The network gets the whole stack at once, never a sheet on its own.

| Sheet | Piece | Sheet | Piece |
|---|---|---|---|
| 0 | white pawn | 6 | black pawn |
| 1 | white knight | 7 | black knight |
| 2 | white bishop | 8 | black bishop |
| 3 | white rook | 9 | black rook |
| 4 | white queen | 10 | black queen |
| 5 | white king | 11 | black king |

## The shape: `(8, 8, 12)`

The result is a **3D array**: you need three numbers to point at one value.

```
planes[row, col, sheet]
       │    │    └─ which piece type (0-11, table above)
       │    └────── file: 0 = a ... 7 = h
       └─────────── rank: 0 = rank 1 ... 7 = rank 8
```

For example, `planes[1, 4, 0]` = rank 2, e-file, white-pawn sheet = `1.0` in
the starting position (there's a pawn on e2).

When the network is trained, boards go in as a **batch**, which adds one more
dimension at the front: `(batch, 8, 8, 12)`. For a single board:

```python
x = encode_board(board)[None]   # (8, 8, 12) -> (1, 8, 8, 12)
```

## Why sheets instead of one grid of numbers?

The obvious alternative is one 8x8 grid with pawn = 1, knight = 2, and so on.
The problem is that the network would treat those as amounts: a knight would
look like "two pawns" and a king like "six pawns", which means nothing.

Separate sheets keep every piece type independent, so the network learns what
each one is worth on its own. This is called **one-hot encoding**: on each
square, at most one of the 12 sheets is `1` (only one piece fits on a square),
and an empty square is all `0`s.

Colours get separate sheets because a white rook and a black rook matter in
opposite directions for the score.

## How the code works

```python
planes = np.zeros((8, 8, 12), dtype=np.float32)        # 1. start with all zeros

for square, piece in board.piece_map().items():        # 2. every piece on the board
    row = chess.square_rank(square)                    #    which rank
    col = chess.square_file(square)                    #    which file
    plane = piece.piece_type - 1                       #    pawn..king -> 0..5
    if piece.color == chess.BLACK:
        plane += 6                                     #    black pieces -> 6..11
    planes[row, col, plane] = 1.0                      # 3. mark "this piece is here"

return jnp.array(planes)                               # 4. JAX array, ready for the GPU
```

- The array is built with **NumPy** because NumPy arrays can be changed in
  place. JAX arrays can't, so it's converted to JAX only at the end.
- `encode_board` is **not** `jit`-compiled. It works on a python-chess
  `Board`, which JAX can't compile, and it's cheap anyway (at most 32 pieces).
  The expensive part, the network, is what gets `jit`.

## Seeing it for yourself

Pass `debug=True` to print the shape and the pawn sheets:

```python
x = encode_board(chess.Board(), debug=True)
```

```
encoded shape: (8, 8, 12)

sheet 0: white pawn
[[0. 0. 0. 0. 0. 0. 0. 0.]
 ...
 [1. 1. 1. 1. 1. 1. 1. 1.]
 [0. 0. 0. 0. 0. 0. 0. 0.]]
```

The sheets are printed with `[::-1]` so rank 8 is at the top, like a normal
board diagram. Without it they'd be upside down, because row 0 is rank 1.

Leave `debug` off (the default) when encoding lots of positions, or you'll get
thousands of printouts.
