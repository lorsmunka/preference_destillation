"""The library on disk: the songs and the playlists in one JSON file."""

import json
import os

from config import LIBRARY_FILE
from music_library import MusicLibrary
from playlists import Playlist
from songs import Song


def save_library(library, path=LIBRARY_FILE):
    playlists = {name: playlist.paths for name, playlist in library.playlists.items()}
    data = {"songs": [song.to_dict() for song in library.all_songs()], "playlists": playlists}
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def load_library(path=LIBRARY_FILE):
    """The saved MusicLibrary, or an empty one the first time."""
    if not os.path.exists(path):
        return MusicLibrary()
    with open(path) as file:
        data = json.load(file)
    songs = {item["path"]: Song(**item) for item in data["songs"]}
    playlists = {name: Playlist(name, paths) for name, paths in data["playlists"].items()}
    return MusicLibrary(songs, playlists)
