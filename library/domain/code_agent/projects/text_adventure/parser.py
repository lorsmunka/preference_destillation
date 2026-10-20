"""Turning what the player types into a verb and a noun: "take the brass key" -> ("take", "brass key")."""

from config import DIRECTIONS

SYNONYMS = {"get": "take", "grab": "take", "walk": "go", "l": "look", "i": "inventory",
            "hit": "attack", "fight": "attack", "eat": "use", "unlock": "use", "?": "help"}
FILLER_WORDS = {"the", "a", "an", "to", "at", "with", "door"}
HELP_TEXT = "go <way> (or n, s, e, w, up, down), take, drop, use, attack, look, inventory, save, load"


def parse_command(text):
    """(verb, noun): synonyms become the main verb, filler words go and "n" alone means go north."""
    words = [word for word in text.lower().split() if word not in FILLER_WORDS]
    if not words:
        return "", ""
    verb = SYNONYMS.get(words[0], words[0])
    if verb in DIRECTIONS or verb in DIRECTIONS.values():
        return "go", direction_name(verb)
    noun = " ".join(words[1:])
    if verb == "go":
        noun = direction_name(noun)
    return verb, noun


def direction_name(word):
    """"n" -> "north"; a full direction name stays as it is."""
    return DIRECTIONS.get(word, word)


def show_help(game, noun):
    return HELP_TEXT
