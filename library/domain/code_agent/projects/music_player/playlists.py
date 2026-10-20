"""A Playlist: a name and songs in the listener's order (kept as file paths)."""


class Playlist:
    def __init__(self, name, paths=None):
        self.name = name
        self.paths = paths or []

    def add_song(self, song):
        """Add a song at the end. False when it is in the playlist already."""
        if song.path in self.paths:
            return False
        self.paths.append(song.path)
        return True

    def remove_song(self, position):
        """Take out the song at a position (1 is the first)."""
        if 1 <= position <= len(self.paths):
            del self.paths[position - 1]

    def move_song(self, old_position, new_position):
        """Move a song up or down the list, like dragging it."""
        path = self.paths.pop(old_position - 1)
        self.paths.insert(new_position - 1, path)

    def song_list(self, library):
        """The playlist's songs; songs gone from the library are left out."""
        return [library.songs[path] for path in self.paths if path in library.songs]
