"""The "now playing" display: the song, its stars, the modes, what comes next."""

from config import MAX_STARS


def star_text(rating):
    """3 -> "***--", 0 -> "not rated"."""
    return "*" * rating + "-" * (MAX_STARS - rating) if rating else "not rated"


def now_playing_lines(player):
    song, queue = player.playing, player.queue
    if song is None:
        return ["Nothing is playing."]
    shuffle_text = "on" if queue.shuffled else "off"
    lines = [f"Now playing: {song.label} [{song.genre}]",
             f"  {star_text(song.rating)}, played {song.plays} times",
             f"  song {queue.position + 1} of {len(queue.order)}, "
             f"shuffle {shuffle_text}, repeat {queue.repeat}"]
    coming = queue.upcoming()
    if coming:
        lines.append("  next: " + ", ".join(other.label for other in coming))
    return lines


def print_now_playing(player):
    print("\n".join(now_playing_lines(player)))
