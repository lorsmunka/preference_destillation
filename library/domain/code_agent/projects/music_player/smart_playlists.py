"""Smart playlists: made fresh from the plays, ratings and dates every time."""

from datetime import date, timedelta

from config import GOOD_RATING, RECENT_DAYS, SMART_SIZE


def most_played(library):
    played = [song for song in library.all_songs() if song.plays > 0]
    return sorted(played, key=lambda song: song.plays, reverse=True)[:SMART_SIZE]


def recently_added(library):
    """Songs found in the last RECENT_DAYS days, the newest first."""
    since = (date.today() - timedelta(days=RECENT_DAYS)).isoformat()
    recent = [song for song in library.all_songs() if song.added >= since]
    return sorted(recent, key=lambda song: song.added, reverse=True)[:SMART_SIZE]


def top_rated(library):
    """Well rated songs: the best rated first, then the most played."""
    rated = [song for song in library.all_songs() if song.rating >= GOOD_RATING]
    return sorted(rated, key=lambda song: (song.rating, song.plays), reverse=True)[:SMART_SIZE]


def never_played(library):
    return [song for song in library.all_songs() if song.plays == 0][:SMART_SIZE]


def smart_playlist(library, name):
    """The songs of the smart playlist with this name (None for another name)."""
    makers = {"most played": most_played, "recently added": recently_added,
              "top rated": top_rated, "never played": never_played}
    return makers[name](library) if name in makers else None
