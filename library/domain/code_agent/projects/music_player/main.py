"""A music player in the terminal: scan the music folder, play, rate, make playlists."""

from config import MUSIC_FOLDER
from now_playing import print_now_playing
from player import Player
from playlists import Playlist
from scanner import scan_folder
from search import songs_for
from storage import load_library, save_library

HELP_TEXT = "scan, play <what>, skip, back, shuffle, repeat, rate <1-5>, add <playlist>, quit"


def run_command(command, argument, library, player):
    """Do one command. False when the user quits."""
    song = player.playing
    actions = {"skip": player.skip, "back": player.back,
               "shuffle": player.queue.toggle_shuffle, "repeat": player.queue.cycle_repeat}
    if command == "quit":
        save_library(library)
        return False
    if command == "scan":
        added = library.add_scanned(scan_folder(MUSIC_FOLDER))
        print(f"{added} new songs, {library.remove_missing()} missing ones removed.")
    elif command == "play":
        player.play_songs(songs_for(library, argument))
    elif command in actions:
        actions[command]()
    elif command == "rate" and song and argument.isdigit():
        print("Rated." if song.rate(int(argument)) else "Give 1 to 5 stars.")
    elif command == "add" and song and argument:
        playlist = library.playlists.setdefault(argument, Playlist(argument))
        print("Added." if playlist.add_song(song) else "It is in that playlist already.")
    else:
        print(HELP_TEXT)
    print_now_playing(player)
    return True


def main():
    library, player = load_library(), Player()
    print(f"{len(library.songs)} songs in the library. Type help for the commands.")
    while True:
        command, _, argument = input("> ").strip().partition(" ")
        if not run_command(command.lower(), argument.strip(), library, player):
            break


if __name__ == "__main__":
    main()
