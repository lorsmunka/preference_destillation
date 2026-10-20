"""A Song: one music file with its title, artist, genre, plays and rating."""

from dataclasses import asdict, dataclass
from datetime import date

from config import MAX_STARS


@dataclass
class Song:
    path: str
    title: str
    artist: str
    genre: str
    added: str = ""        # the day it was found, like "2026-10-21"
    plays: int = 0
    rating: int = 0        # 0 = not rated, else 1 to 5 stars
    last_played: str = ""

    @property
    def label(self):
        return f"{self.artist} - {self.title}"

    def count_play(self):
        self.plays += 1
        self.last_played = date.today().isoformat()

    def rate(self, stars):
        """Give the song 1 to MAX_STARS stars; False for any other number."""
        if not 1 <= stars <= MAX_STARS:
            return False
        self.rating = stars
        return True

    def to_dict(self):
        return asdict(self)
