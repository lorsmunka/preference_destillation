"""Recommendations: songs of the listener's favourite genres, rarely played."""

from collections import Counter
from datetime import date

from config import GOOD_RATING, RECOMMEND_COUNT

FAVOURITE_GENRES = 3


def genre_scores(library):
    """How much each genre is liked: its plays, counted twice on well rated songs."""
    scores = Counter()
    for song in library.all_songs():
        scores[song.genre] += song.plays * (2 if song.rating >= GOOD_RATING else 1)
    return scores


def favourite_genres(library):
    top = genre_scores(library).most_common(FAVOURITE_GENRES)
    return [genre for genre, score in top if score > 0]


def recommend(library, count=RECOMMEND_COUNT):
    """Rarely played songs of the favourite genres, not played today and not
    rated badly: the favourite genre first, then the best rated."""
    genres = favourite_genres(library)
    today = date.today().isoformat()
    picks = [song for song in library.all_songs() if song.genre in genres and song.plays < 3
             and song.last_played != today and song.rating not in (1, 2)]
    return sorted(picks, key=lambda song: (genres.index(song.genre), -song.rating))[:count]
