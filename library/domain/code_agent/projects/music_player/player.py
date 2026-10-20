"""The Player: the song playing now and the queue it comes from."""

from play_queue import PlayQueue


class Player:
    def __init__(self):
        self.queue = PlayQueue([])
        self.playing = None   # the Song playing, None when stopped

    def play_songs(self, songs, shuffled=False):
        """Start a new queue: a playlist, search results, recommendations."""
        self.queue = PlayQueue(songs)
        if shuffled:
            self.queue.toggle_shuffle()
        return self.start(self.queue.current())

    def start(self, song):
        """Play a song, which counts as a play; None stops the player."""
        self.playing = song
        if song is not None:
            song.count_play()
        return song

    def skip(self):
        return self.start(self.queue.next_song())

    def back(self):
        return self.start(self.queue.previous_song())
