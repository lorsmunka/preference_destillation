"""The MusicLibrary: every known song by its file path, and the playlists."""

import os


class MusicLibrary:
    def __init__(self, songs=None, playlists=None):
        self.songs = songs or {}           # file path -> Song
        self.playlists = playlists or {}   # name -> Playlist

    def all_songs(self):
        return list(self.songs.values())

    def add_scanned(self, scanned):
        """Add the new songs of a scan; known songs keep their plays and
        ratings. Returns how many songs are new."""
        new_songs = [song for song in scanned if song.path not in self.songs]
        for song in new_songs:
            self.songs[song.path] = song
        return len(new_songs)

    def remove_missing(self):
        """Forget the songs whose files are gone. Returns how many."""
        missing = [path for path in self.songs if not os.path.exists(path)]
        for path in missing:
            del self.songs[path]
        return len(missing)
