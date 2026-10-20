"""Checkers in the terminal: you against the computer."""

from display import print_board, print_history
from move_input import ask_move
from saving import save_game, start_game


def human_turn(game, computer):
    """The player's move or command. False when they quit."""
    answer = ask_move(game)
    if answer == "quit":
        return False
    if answer == "undo":
        game.undo()
        if game.turn == computer.color:   # take back the computer's reply too
            game.undo()
    elif answer == "history":
        print_history(game)
    elif answer == "save":
        save_game(game, computer)
        print("Game saved.")
    else:
        game.play(answer)
    return True


def main():
    game, computer = start_game()
    while not game.is_over():
        print_board(game.board)
        if game.turn == computer.color:
            move = computer.choose_move(game.board)
            print(f"The computer plays {move.to_text()}.")
            game.play(move)
        elif not human_turn(game, computer):
            return
    print_board(game.board)
    winner = game.winner()
    print(f"{winner} wins!" if winner else "A draw.")


if __name__ == "__main__":
    main()
