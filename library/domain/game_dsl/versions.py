"""Seeded versions of a game: every name (entity and template ids, bars, signals, per-entity vars) renamed
consistently and every colour replaced consistently, so the student cannot learn a game's names and colours by
heart. Numbers, texts, sounds and the order of things stay as written (changing them could break the play). A
version is kept only when it still compiles and passes every check, like the game itself."""

import random
from typing import Dict, List, Optional, Set

from .language import COLOR, Node, parse_program, walk
from .manual import sound_and_music_names

# names the compiler or the engine give a meaning to: never renamed
FIXED_NAMES = {"lives", "padUp", "padDown", "padLeft", "padRight", "floor", "wallL", "wallR", "wallT", "wallB",
               "stick", "__waves__", ""}
# argument positions that hold a text shown to the player, never a name
DISPLAY_TEXT = {"text": {0}, "settext": {1}, "gameover": {0}, "lives": {1}, "hostile": {1}, "goal": {5},
                "hostilepatrol": {2}}
WORDS = ("ace acorn amber anchor ant apex arch arrow ash aster atom axe badge bait bale balm bark basil bat beacon bead "
         "beam bean bear bee beetle bell berry birch bison blade blaze bloom boar bolt bone boot bough boulder bow box "
         "brick bridge brook bud bug bulb bunny burr cab cactus cake camel candle cannon canoe cap car card cargo carp "
         "cart cask cat cedar cell chalk charm chick chip cider cinder clam claw clay cliff clock cloud clove coal "
         "cobra cocoa coin colt comet cone coral cork crab crane crate crest crow crown cube cub cup curl dart dash "
         "dawn deer den dew dial dice dime disc dock dot dove drake drift drill drop drum duck dune dusk eagle echo "
         "eel egg elk elm ember fang fawn feather fern ferry fig fin finch fire flag flake flame flint float flute "
         "foam fog fork fox frog frost fuse gale gate gecko gem ghost gill glade glow gnat goat gong goose gourd "
         "grain grape grub gull gust hail hare harp hatch hawk haze hedge helm hen heron hill hive hog hook horn "
         "husk ice imp inch ink iris isle ivy jade jar jay jelly jet jewel kale kelp kettle key kite kiwi knot lake "
         "lamb lamp lance lark latch lava leaf ledge lemon lens lid lily lime lint lion lizard llama lock log loom "
         "lynx mace mango maple marble mask mast meadow melon mesa mill mint mist mole moon moose moss moth mouse "
         "mule nectar nest net newt nib node nut oak oar oat olive onion opal orb orca otter owl ox paddle pail palm "
         "pan panda pear pearl pebble pecan peg pelican pepper perch pike pine pip pit plum pod pollen pond pony "
         "poppy pot puck pug pump quail quartz quill rabbit raft rail rain ram raven ray reed reef ridge ring river "
         "robin rock rook root rope rose rover ruby rune rush sage sail salt sand sap seal seed shard shell ship "
         "shrub silk slate sled slug snail spark spike spoon spore spray sprout squid star stem stone stork straw "
         "stump swan tack tadpole tar thorn tide tiger toad torch tower trout tulip tusk twig urchin vale vane vine "
         "violet viper wagon wasp wave wax web whale wheat whisk wick willow wing wisp wolf wren yak yarn yew zebra").split()
PALETTE = ("ef4444 f97316 f59e0b eab308 84cc16 22c55e 10b981 14b8a6 06b6d4 0ea5e9 3b82f6 6366f1 8b5cf6 a855f7 "
           "d946ef ec4899 f43f5e fca5a5 fdba74 fde68a bef264 86efac 5eead4 67e8f9 93c5fd a5b4fc c4b5fd f0abfc "
           "f9a8d4 991b1b 9a3412 92400e 3f6212 166534 115e59 155e75 1e40af 3730a3 5b21b6 86198f 9d174d 0f172a "
           "1e293b 334155 475569 64748b 94a3b8 cbd5e1 e2e8f0 f8fafc 111827 1f2937 fafaf9 78716c 44403c 292524 "
           "fbbf24 4ade80 38bdf8 c084fc fb7185 a3e635 2dd4bf facc15").split()


def game_names(root: Node) -> Set[str]:
    """Every name the game defines: entity and template ids, bars, signals, per-entity vars."""
    names = set()
    for node, _ in walk(root):
        first = node.first_string()
        if node.key in ("id", "paddle", "goal", "hostilepatrol", "bar", "var", "broadcast", "onsignal", "lock") \
                and first:
            names.add(first)
        if node.key in ("clearfullrows", "clearfullcolumns", "clearfullboxes", "slide", "slideall") and node.args \
                and isinstance(node.args[-1], str):
            names.add(node.args[-1])
    keeps_hero = any(node.key == "goal" or (node.key == "key" and len(node.args) < 2) for node, _ in walk(root))
    if keeps_hero:
        names.discard("hero")
    # a name that is also a sound or a music track stays (sound("pop") must keep working)
    return names - FIXED_NAMES - set(sound_and_music_names())


def new_name(old: str, used: Set[str], random_generator: random.Random) -> str:
    for _ in range(100):
        word = random_generator.choice(WORDS)
        style = random_generator.random()
        if style < 0.45:
            name = word
        elif style < 0.7:
            name = word + str(random_generator.randint(1, 9))
        elif style < 0.85:
            name = word + random_generator.choice("ABCDEFGH")
        else:
            name = word + random_generator.choice(WORDS).capitalize()
        if any(character.isdigit() for character in old[-1:]) and not name[-1].isdigit():
            name += old[-1]   # keep a numbered family numbered (guard1, guard2 -> owl1, owl2)
        if name not in used:
            return name
    raise RuntimeError("ran out of names")


def make_version(source: str, seed: int) -> str:
    """One seeded version: names and colours replaced, everything else as written."""
    random_generator = random.Random(seed)
    root = parse_program(source)
    names = sorted(game_names(root))
    used = set(names) | FIXED_NAMES | {"hero"}
    renames: Dict[str, str] = {}
    families: Dict[str, str] = {}
    for name in names:
        stem = name.rstrip("0123456789")
        follower = families[stem] + name[len(stem):] if stem != name and stem in families else None
        if follower is not None and follower not in used:     # guard2 follows guard1's new stem
            renames[name] = follower
        else:
            renames[name] = new_name(name, used, random_generator)
            families.setdefault(stem, renames[name].rstrip("0123456789"))
        used.add(renames[name])
    colours: Dict[str, str] = {}
    edits = []
    for node, _ in walk(root):
        if node.key in ("sound", "music"):
            continue
        shown = DISPLAY_TEXT.get(node.key, set())
        for position, (value, (start, end)) in enumerate(zip(node.args, node.arg_spans)):
            if not isinstance(value, str) or position in shown:
                continue
            if value in renames:
                edits.append((start, end, f'"{renames[value]}"'))
            elif COLOR.match(value) and value.lstrip("#").lower() not in ("00000000", "0000"):
                digits = value.lstrip("#")
                if digits.lower() not in colours:
                    colours[digits.lower()] = random_generator.choice(PALETTE)
                alpha = digits[:2] if len(digits) == 8 else ""
                edits.append((start, end, f'"{alpha}{colours[digits.lower()]}"'))
    for start, end, text in sorted(edits, reverse=True):
        source = source[:start] + text + source[end:]
    return source


def make_versions(source: str, count: int, seed: int, check) -> List[str]:
    """Up to `count` versions that pass `check(source) -> list of problems`; failed ones are dropped."""
    candidates = [make_version(source, seed * 1000 + index) for index in range(count)]
    results = check({str(index): candidate for index, candidate in enumerate(candidates)})
    return [candidates[int(name)] for name, found in results.items() if not found]
