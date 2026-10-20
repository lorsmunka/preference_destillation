"""Finding songs in the music folder: files named "Artist - Title.mp3",
a folder per genre, like music/rock/Queen - Bohemian Rhapsody.mp3."""

import os
import re
from datetime import date

from config import AUDIO_EXTENSIONS, UNKNOWN
from songs import Song

TRACK_NUMBER = re.compile(r"^\d+(\s*[.-]\s*|\s+)")   # "01 - " or "07 " at the start


def split_file_name(file_name):
    """"03 - Queen - Bohemian_Rhapsody.mp3" -> ("Queen", "Bohemian Rhapsody")."""
    name = os.path.splitext(file_name)[0].replace("_", " ")
    artist, separator, title = TRACK_NUMBER.sub("", name).partition(" - ")
    if not separator:
        return UNKNOWN, artist.strip()
    return artist.strip(), title.strip()


def genre_of(folder, directory):
    """The first folder below the music folder is the genre."""
    relative = os.path.relpath(directory, folder)
    return UNKNOWN if relative == "." else relative.split(os.sep)[0].lower()


def scan_folder(folder):
    """A Song for every audio file under the folder."""
    today = date.today().isoformat()
    found = []
    for directory, _, file_names in os.walk(folder):
        for file_name in sorted(file_names):
            if file_name.lower().endswith(AUDIO_EXTENSIONS):
                artist, title = split_file_name(file_name)
                path = os.path.join(directory, file_name)
                found.append(Song(path, title, artist, genre_of(folder, directory), today))
    return found
