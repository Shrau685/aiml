# tic_tac_toe.py (no lru_cache)
import math

LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

def show(board):
    print("\n".join(" | ".join(board[i*3+j] or str(i*3+j) for j in range(3)) for i in range(3)))

def winner(board):
    for a,b,c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return "Draw" if all(board) else None

def minimax(board, player):
    w = winner(board)
    if w == "X":   return 1, None
    if w == "O":   return -1, None
    if w == "Draw":return 0, None

    best_val = -math.inf if player == "X" else math.inf
    best_move = None
    next_player = "O" if player == "X" else "X"

    for i, cell in enumerate(board):
        if cell is None:
            nb = list(board)
            nb[i] = player
            val, _ = minimax(tuple(nb), next_player)
            if player == "X" and val > best_val:
                best_val, best_move = val, i
            elif player == "O" and val < best_val:
                best_val, best_move = val, i
    return best_val, best_move

if __name__ == "__main__":
    board = (None,) * 9
    player = "X"  # AI = X, Human = O

    while True:
        show(board)
        w = winner(board)
        if w:
            print(f"Result: {w}")
            break

        if player == "X":
            _, move = minimax(board, "X")
            print(f"AI plays: {move}")
        else:
            try:
                move = int(input("Your move (0–8): "))
            except ValueError:
                print("Enter a number between 0–8.")
                continue

        if 0 <= move < 9 and board[move] is None:
            board = tuple(board[i] if i != move else player for i in range(9))
            player = "O" if player == "X" else "X"
        else:
            print("Invalid move, try again.")
