"""Searching the library: every word typed must be in the title, artist or genre."""

from recommendations import recommend
from smart_playlists import smart_playlist


def matches(song, words):
    text = f"{song.title} {song.artist} {song.genre}".lower()
    return all(word in text for word in words)


def relevance(song, words):
    """Title hits count most, then artist hits; more plays win a tie."""
    title, artist = song.title.lower(), song.artist.lower()
    return sum(word in title for word in words), sum(word in artist for word in words), song.plays


def search_songs(library, query):
    words = query.lower().split()
    found = [song for song in library.all_songs() if words and matches(song, words)]
    return sorted(found, key=lambda song: relevance(song, words), reverse=True)


def songs_for(library, name):
    """What "play <name>" plays: a playlist, a smart playlist, the
    recommendations, or else the search results for the words."""
    if name in library.playlists:
        return library.playlists[name].song_list(library)
    if name == "recommended":
        return recommend(library)
    smart = smart_playlist(library, name)
    return smart if smart is not None else search_songs(library, name)
