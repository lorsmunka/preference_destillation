"""How people say things, for v2's commands (2026-11-09, PLAN.md "v2: the user's notes"): colour and shape words,
the check that a written colour or shape matches them, the way a block is called in an edit ("this knight", "it"),
casual verbs and typos.

A colour word is a spec, not one hex: "light red" accepts every colour whose hue is near red and whose lightness is
light enough, so Gemma's own choice of shade is never marked wrong.
"""

import colorsys
import random
import re
from typing import Dict, List, Optional, Tuple

# name: (hue in degrees or None for a grey, lightness range, saturation range)
COLOURS = {
    "red": (0, (0.25, 0.8), (0.45, 1)), "scarlet": (5, (0.3, 0.7), (0.6, 1)), "crimson": (350, (0.2, 0.6), (0.5, 1)),
    "coral": (12, (0.5, 0.85), (0.5, 1)), "salmon": (8, (0.55, 0.9), (0.4, 1)), "orange": (28, (0.3, 0.8), (0.55, 1)),
    "peach": (25, (0.65, 0.95), (0.4, 1)), "amber": (40, (0.3, 0.75), (0.6, 1)), "gold": (46, (0.3, 0.75), (0.55, 1)),
    "yellow": (54, (0.35, 0.9), (0.55, 1)), "lime": (100, (0.3, 0.85), (0.5, 1)), "olive": (65, (0.15, 0.45), (0.25, 1)),
    "green": (128, (0.2, 0.8), (0.3, 1)), "mint": (138, (0.55, 0.92), (0.3, 1)), "emerald": (150, (0.25, 0.65), (0.4, 1)),
    "teal": (175, (0.2, 0.7), (0.35, 1)), "turquoise": (176, (0.35, 0.8), (0.4, 1)), "cyan": (188, (0.35, 0.85), (0.5, 1)),
    "sky blue": (200, (0.5, 0.9), (0.45, 1)), "blue": (220, (0.2, 0.8), (0.4, 1)), "navy": (225, (0.08, 0.35), (0.3, 1)),
    "indigo": (257, (0.2, 0.7), (0.35, 1)), "violet": (279, (0.3, 0.85), (0.35, 1)),
    "lavender": (262, (0.65, 0.93), (0.25, 1)), "purple": (275, (0.2, 0.8), (0.3, 1)),
    "magenta": (300, (0.3, 0.75), (0.5, 1)), "pink": (330, (0.45, 0.93), (0.35, 1)), "rose": (345, (0.4, 0.85), (0.45, 1)),
    "maroon": (350, (0.1, 0.35), (0.35, 1)), "brown": (28, (0.12, 0.45), (0.25, 1)), "beige": (40, (0.7, 0.92), (0.15, 0.7)),
    "white": (None, (0.9, 1), (0, 1)), "black": (None, (0, 0.14), (0, 1)), "grey": (None, (0.2, 0.85), (0, 0.22)),
    "silver": (None, (0.6, 0.92), (0, 0.22)),
}
GREY_SPELLING = {"gray": "grey"}
LIGHT = ("light", "pale", "pastel", "soft", "baby")
DARK = ("dark", "deep")
BRIGHT = ("bright", "neon", "vivid", "hot")
HUE_TOLERANCE = 28
# the common colours, for the generators (the rarer names above are for reading what a user may type)
COMMON = ("red", "orange", "yellow", "green", "teal", "cyan", "blue", "purple", "pink", "white", "black", "grey",
          "lime", "brown", "gold", "navy", "magenta", "coral", "violet", "mint", "maroon", "sky blue", "turquoise",
          "lavender", "crimson", "indigo")


def hex_rgb(value: str) -> Optional[Tuple[float, float, float]]:
    """"RGB", "RRGGBB" or "AARRGGBB" (with or without #) -> (r, g, b) in 0..1; None for anything else."""
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    elif len(value) == 4:
        value = "".join(char * 2 for char in value[1:])
    elif len(value) == 8:
        value = value[2:]
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        return None
    return tuple(int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))


def hls(value: str) -> Optional[Tuple[float, float, float]]:
    rgb = hex_rgb(value)
    if rgb is None:
        return None
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    return hue * 360, lightness, saturation


def colour_spec(words: str) -> Optional[Dict]:
    """"pinkish, light red" -> {"words", "hue", "tolerance", "light", "sat"}; None when no colour name is in it."""
    text = words.lower().replace(",", " ")
    for wrong, right in GREY_SPELLING.items():
        text = re.sub(rf"\b{wrong}\b", right, text)
    found = [name for name in sorted(COLOURS, key=len, reverse=True) if re.search(rf"\b{name}\b", text)]
    found = [name for name in found if not any(name != other and name in other for other in found)]
    if not found:
        return None
    # the last colour word is the colour; an "-ish" word before it pulls the hue towards itself
    positions = {name: text.rfind(name) for name in found}
    main = max(found, key=lambda name: positions[name])
    hue, light, saturation = COLOURS[main]
    tolerance = HUE_TOLERANCE
    leaning = re.search(r"\b([a-z]+)ish\b", text)
    lean_name = leaning.group(1) if leaning else None
    if lean_name and lean_name + "ish" != main:
        lean_name = {"redd": "red", "blu": "blue", "purpl": "purple", "orang": "orange",
                     "gray": "grey"}.get(lean_name, lean_name)
        if lean_name in COLOURS and COLOURS[lean_name][0] is not None and hue is not None:
            other = COLOURS[lean_name][0]
            difference = ((other - hue + 180) % 360) - 180
            hue = (hue + difference / 2) % 360
            tolerance += 8
    if re.search(rf"\b({'|'.join(LIGHT)})(er)?\b|\blighter\b", text):
        light = (max(light[0], 0.6), 0.97)
    if re.search(rf"\b({'|'.join(DARK)})(er)?\b|\bdarker\b", text):
        light = (0.05, min(light[1], 0.45))
    if re.search(rf"\b({'|'.join(BRIGHT)})\b", text):
        saturation = (max(saturation[0], 0.6), 1)
    if re.search(r"\b\w+ish\b", text) and not lean_name:
        tolerance += 10
    return {"words": words, "hue": hue, "tolerance": tolerance, "light": light, "sat": saturation}


def colour_matches(value: str, spec: Dict) -> bool:
    found = hls(value)
    if found is None:
        return False
    hue, lightness, saturation = found
    if not spec["light"][0] - 0.03 <= lightness <= spec["light"][1] + 0.03:
        return False
    if spec["hue"] is None:
        return saturation <= spec["sat"][1] + 0.05 or lightness > 0.9 or lightness < 0.14
    if not spec["sat"][0] - 0.05 <= saturation:
        return False
    return abs(((hue - spec["hue"] + 180) % 360) - 180) <= spec["tolerance"]


def colour_name(value: str) -> Optional[str]:
    """What a person would call the colour: "light blue", "dark red", "grey" (for "from red to blue")."""
    found = hls(value)
    if found is None:
        return None
    hue, lightness, saturation = found
    if lightness > 0.93:
        return "white"
    if lightness < 0.1:
        return "black"
    if saturation < 0.25:
        return "light grey" if lightness > 0.7 else "dark grey" if lightness < 0.3 else "grey"
    names = [(0, "red"), (25, "orange"), (52, "yellow"), (90, "lime"), (130, "green"), (172, "teal"), (190, "cyan"),
             (220, "blue"), (265, "purple"), (300, "magenta"), (330, "pink"), (360, "red")]
    name = min(names, key=lambda pair: abs(pair[0] - hue))[1]
    if name == "orange" and lightness < 0.35:
        return "brown"
    if lightness > 0.72:
        return "light " + name
    if lightness < 0.3:
        return "dark " + name
    return name


def random_colour_words(random_generator: random.Random) -> str:
    """A colour as a person says it: "blue", "light red", "pinkish light red", "dark teal", "bright green"."""
    name = random_generator.choice(COMMON)
    roll = random_generator.random()
    chromatic = COLOURS[name][0] is not None and name not in ("navy", "maroon", "brown", "mint", "lavender")
    if chromatic and roll < 0.2:
        name = random_generator.choice(LIGHT[:3] + ("light", "light")) + " " + name
    elif chromatic and roll < 0.35:
        name = random_generator.choice(DARK + ("dark",)) + " " + name
    elif chromatic and roll < 0.42:
        name = random_generator.choice(BRIGHT) + " " + name
    if chromatic and random_generator.random() < 0.15:
        lean = random_generator.choice(("pinkish", "reddish", "bluish", "greenish", "purplish", "yellowish"))
        lean_hue = COLOURS[{"pinkish": "pink", "reddish": "red", "bluish": "blue", "greenish": "green",
                            "purplish": "purple", "yellowish": "yellow"}[lean]][0]
        main_hue = COLOURS[name.split()[-1] if name.split()[-1] in COLOURS else name.split(" ", 1)[-1]][0]
        near = main_hue is not None and abs(((lean_hue - main_hue + 180) % 360) - 180) <= 75
        if lean[:-3] not in name and near:
            name = lean + random_generator.choice((" ", ", ")) + name
    if name.endswith("grey") and random_generator.random() < 0.4:
        name = name[:-4] + "gray"
    return name


# ---------- shapes ----------

SHAPES = {   # word: (keyword, how to check it)
    "square": "rectangle", "rectangle": "rectangle", "box": "rectangle", "block": "rectangle", "platform": "rectangle",
    "brick": "rectangle", "circle": "circle", "ball": "circle", "dot": "circle", "orb": "circle", "bubble": "circle",
    "triangle": "triangle", "star": "star", "diamond": "diamond", "gem": "diamond", "ring": "ring", "donut": "ring",
    "hexagon": "polygon", "pentagon": "polygon", "octagon": "polygon", "cross": "cross", "plus sign": "cross",
}
POLYGON_SIDES = {"hexagon": 6, "pentagon": 5, "octagon": 8}
SHAPE_NAMES = ("square", "rectangle", "circle", "triangle", "star", "diamond", "ring", "hexagon")


def shape_of(entity_node) -> Optional[object]:
    type_block = entity_node.child("type") if entity_node is not None else None
    return type_block.children[0] if type_block is not None and type_block.children else None


def numbers(node) -> List[float]:
    return [value for value in node.args if isinstance(value, float)]


def shape_word(shape) -> Optional[str]:
    """What a person calls a written shape: "square", "rectangle", "circle", "hexagon", ..."""
    if shape is None:
        return None
    values = numbers(shape)
    if shape.key == "rectangle" and len(values) >= 2:
        return "square" if values[0] == values[1] else "rectangle"
    if shape.key == "polygon" and values:
        return {6: "hexagon", 5: "pentagon", 8: "octagon"}.get(int(values[0]), "polygon")
    if shape.key in ("circle", "triangle", "star", "diamond", "ring", "cross"):
        return shape.key
    return None


def shape_matches(shape, word: str) -> bool:
    if shape is None:
        return False
    keyword = SHAPES.get(word, word)
    if shape.key != keyword:
        return False
    values = numbers(shape)
    if word == "square":
        return len(values) >= 2 and abs(values[0] - values[1]) <= 0.15 * max(values[0], values[1], 1)
    if word in POLYGON_SIDES:
        return bool(values) and int(values[0]) == POLYGON_SIDES[word]
    return True


def shape_size(shape) -> Optional[float]:
    """The largest side of a shape's box in px."""
    if shape is None:
        return None
    values = numbers(shape)
    if not values:
        return None
    if shape.key in ("circle", "ring"):
        return values[0] * 2
    if shape.key in ("star", "polygon"):
        return values[1] * 2 if len(values) > 1 else None
    if shape.key in ("rectangle", "diamond", "cross"):
        return max(values[:2])
    if shape.key == "text":
        return values[0]
    return values[0]


# ---------- how a block is called ----------

def noun_of(entity_id: Optional[str]) -> Optional[str]:
    """A word for an entity from its id ("darkKnight" -> "dark knight", "moth" -> "moth"); None when the id is no
    word ("c00", "p1")."""
    if not entity_id:
        return None
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", entity_id).replace("_", " ").lower().split()
    words = [word for word in words if word.isalpha()]
    if not words or len(words[-1]) < 3 or not re.search(r"[aeiouy]", words[-1]):
        return None
    return " ".join(words)


def pointer(random_generator: random.Random, noun: Optional[str], shape: Optional[str] = None,
            entity_id: Optional[str] = None) -> str:
    """How the user points at the block under the cursor: "this", "it", "this knight", "the knight", "#knight",
    "this circle"."""
    options = ["this", "this", "it", "this one"]
    if noun:
        options += [f"this {noun}", f"this {noun}", f"the {noun}"]
    if entity_id:
        options.append(f"#{entity_id}")
    if shape:
        options += [f"this {shape}"]
    return random_generator.choice(options)


# ---------- casual wording and typos ----------

NEIGHBOURS = {
    "a": "qsz", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh", "h": "gj", "i": "uo", "j": "hk",
    "k": "jl", "l": "k", "m": "n", "n": "bm", "o": "ip", "p": "o", "r": "et", "s": "ad", "t": "ry", "u": "yi",
    "v": "cb", "w": "qe", "y": "tu", "z": "x",
}
MISSPELLINGS = {
    "change": ["chnage", "chage", "cahnge"], "colour": ["color", "colur"], "color": ["colour", "colr"],
    "circle": ["cirlce", "circel"], "triangle": ["triange", "traingle"], "square": ["sqaure", "squre"],
    "enemy": ["enemey", "enmy"], "enemies": ["enemys", "enemeis"], "victory": ["vectory", "victroy"],
    "the": ["teh", "th"], "with": ["wiht", "wth"], "when": ["wehn", "whne"], "player": ["palyer", "plaer"],
    "faster": ["fater", "fastr"], "bigger": ["biger", "bgger"], "sound": ["sond", "suond"], "create": ["creat", "craete"],
    "level": ["levle", "lvl"], "make": ["mak", "amke"], "purple": ["purpel", "puprle"], "yellow": ["yelow", "yellw"],
    "please": ["pls", "plz", "pleas"], "something": ["somthing", "smth"], "because": ["bc", "cuz"],
    "orange": ["orang", "ornage"], "diamond": ["daimond", "diamon"], "background": ["backround", "bg"],
    "text": ["txt", "tetx"], "really": ["realy"], "move": ["moev"], "right": ["rihgt"], "left": ["lfet"],
}
PROTECTED = re.compile(r'"[^"]*"|#\w+')


def typo(text: str, random_generator: random.Random) -> str:
    """One or two slips a fast typist makes, never inside a quoted text or a #name."""
    protected = [match.span() for match in PROTECTED.finditer(text)]
    words = [(match.start(), match.end()) for match in re.finditer(r"[A-Za-z]{3,}", text)
             if not any(start <= match.start() < end for start, end in protected)]
    if not words:
        return text
    for _ in range(1 if random_generator.random() < 0.75 else 2):
        start, end = random_generator.choice(words)
        word = text[start:end]
        lower = word.lower()
        if lower in MISSPELLINGS and random_generator.random() < 0.6:
            changed = random_generator.choice(MISSPELLINGS[lower])
        else:
            position = random_generator.randrange(1, len(word))
            kind = random_generator.choice(("swap", "drop", "double", "neighbour"))
            if kind == "swap" and position < len(word):
                changed = word[:position - 1] + word[position] + word[position - 1] + word[position + 1:]
            elif kind == "drop":
                changed = word[:position] + word[position + 1:]
            elif kind == "double":
                changed = word[:position] + word[position] + word[position:]
            else:
                letter = word[position].lower()
                changed = word[:position] + random_generator.choice(NEIGHBOURS.get(letter, letter)) + word[position + 1:]
        text = text[:start] + changed + text[end:]
        shift = len(changed) - len(word)
        words = [(word_start + (shift if word_start > start else 0), word_end + (shift if word_start > start else 0))
                 for word_start, word_end in words if (word_start, word_end) != (start, end)]
        if not words:
            break
    return text


ASKING_VERBS = ("add", "make", "create", "put", "change", "turn", "give", "spawn", "insert", "play", "draw", "place",
                "end", "shake", "flash", "set", "send", "write", "remove", "delete", "double", "half", "paint")


def casual(text: str, random_generator: random.Random, typo_share: float = 0.25) -> str:
    """The same ask with the small things people do: a please, a question, no capital, no full stop, a typo. "can
    you" and "i want to" only go in front of an ask that starts with a verb."""
    starts_with_verb = text.split(" ", 1)[0] in ASKING_VERBS
    roll = random_generator.random() if starts_with_verb else 0.06 + random_generator.random() * 0.94
    if roll < 0.06:
        text = "can you " + text + random_generator.choice(("?", "", " pls"))
    elif roll < 0.1:
        text = text + random_generator.choice((" pls", " please", " plz"))
    elif roll < 0.13:
        text = random_generator.choice(("i want to ", "i'd like to ", "lets ", "now ", "ok ") if starts_with_verb
                                       else ("now ", "ok ", "hmm ")) + text
    if random_generator.random() < 0.12:
        text = text[0].upper() + text[1:]
    if random_generator.random() < 0.06:
        text += random_generator.choice((".", "!", "!!"))
    if random_generator.random() < typo_share:
        text = typo(text, random_generator)
    return re.sub(r"\s+", " ", text).strip()


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"
