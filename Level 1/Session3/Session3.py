import random

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)


def check_winner(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True

    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True

    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True

    return False


def get_empty_cells(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == " "]


def is_board_full(board):
    return all(board[i][j] != " " for i in range(3) for j in range(3))


def get_ai_move(board):
    empty_cells = get_empty_cells(board)
    return random.choice(empty_cells)


def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    players = ["X", "O"]
    random.shuffle(players)
    current_player = players[0]


    print("Welcome to Tic-Tac-Toe!")
    print_board(board)


    while True:
        if current_player == "X":
            row, col = map(int, input("Enter your move (row col): ").split())
            if board[row][col] != " ":
                print("Invalid move. Try again.")
                continue
            board[row][col] = "X"
        else:
            print("AI is making a move...")
            row, col = get_ai_move(board)
            board[row][col] = "O"
            print(f"AI placed O at ({row}, {col})")


        print_board(board)


        if check_winner(board, current_player):
            print(f"{current_player} wins!")
            break


        if is_board_full(board):
            print("It's a draw!")
            break


        current_player = "X" if current_player == "O" else "O"


play_game()




