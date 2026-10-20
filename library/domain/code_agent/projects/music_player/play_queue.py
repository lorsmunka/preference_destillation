"""The play queue: the songs to play in order, with shuffle and repeat."""

import random

REPEAT_MODES = ("off", "all", "one")


class PlayQueue:
    def __init__(self, songs):
        self.queued = list(songs)   # the order the songs were queued in
        self.order = list(songs)    # the order they play in
        self.position = 0
        self.shuffled = False
        self.repeat = "off"

    def current(self):
        return self.order[self.position] if self.order else None

    def next_song(self):
        """Go to the next song; None after the last one, unless repeating."""
        if not self.order or self.repeat == "one":
            return self.current()
        if self.position + 1 == len(self.order) and self.repeat == "off":
            return None
        self.position = (self.position + 1) % len(self.order)
        return self.current()

    def previous_song(self):
        self.position = max(0, self.position - 1)
        return self.current()

    def toggle_shuffle(self):
        """Shuffle the songs (the current one stays first), or unshuffle them."""
        current = self.current()
        self.shuffled = not self.shuffled
        rest = [song for song in self.queued if song is not current]
        if self.shuffled:
            random.shuffle(rest)
            self.order = ([current] if current else []) + rest
            self.position = 0
        else:
            self.order = list(self.queued)
            self.position = self.order.index(current) if current else 0

    def cycle_repeat(self):
        """off -> all -> one -> off."""
        self.repeat = REPEAT_MODES[(REPEAT_MODES.index(self.repeat) + 1) % len(REPEAT_MODES)]

    def upcoming(self, count=3):
        return self.order[self.position + 1:self.position + 1 + count]
