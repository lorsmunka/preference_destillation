"""A text adventure in the terminal: beat the monsters, find the keys and climb the tower."""

from game import Game


def main():
    game = Game()
    print("You wake up in a cold cellar. Type help for the commands.")
    print(game.current_room().describe())
    while not game.over:
        player = game.player
        print(game.handle(input(f"\n[HP {player.hp}, {player.score.points} points] > ")))
    score = game.player.score
    print(f"Final score: {score.points}. Rank: {score.rank()}.")


if __name__ == "__main__":
    main()
