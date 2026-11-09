"""v2's problems (2026-11-09, PLAN.md "v2: the user's notes"): asks from the user's point of view. A whole game
version (nothing is cut out), the cursor at a place where a user could type, and an ask; Gemma writes the block
(there is no real one). The input is built exactly as the app builds it (live.insert_input / live.edit_input).

Each ask made by rule carries what the block must do ("expect"), checked at home next to the compiler, so the
blocks that do not do what was asked can be dropped too (v1 only checked that they compiled; its student made a
"twice as big" thing smaller).

Units (the PLAN.md v2 item each one answers):
  colour_shape  /insert a thing of a colour, shape, size: "a pinkish, light red circle character"        (4)
  character     the same with a behaviour: "... that chases the player", "... that moves left and right"  (3)
  naming        /insert "create a hero": id hero, or hero_2 when hero is taken                              (5)
  construct     /insert a construct by its keyword or in words: "add an OnValue", "add a timer"            (6)
  sound         /insert or /edit a sound: "play a coin sound", "add a sound when this happens"             (3)
  edit          /edit said the way people point: "change this from a square to a circle", "make it blue",
                "change the text to "winners"", "twice as big", "faster", "remove the sound"                (8)
  text          /insert a text: "add text that reflects victory", "text here for victory"                  (8)
  level         /insert a whole Level: "create a level like level 2 but harder"                            (7)
  open          asks without a rule check (written by a model or by hand): the user's own ideas            (2)
Every ask goes through wording.casual (please, a question, no capital, typos)                              (1)

    python -m library.domain.game_dsl.asks test [per unit] [folder]   -> runs/game_v2/<folder>/asks.jsonl, prompts
    (on the card: python experiments/gemma_vllm_run.py <prompts file>, writer_prompts.jsonl first)
    python -m library.domain.game_dsl.asks writer_asks [folder]       -> Gemma's own asks + their prompts (round 2)
    python -m library.domain.game_dsl.asks check [folder path]        -> per unit: complete, kept, follows

First test (2026-11-09, A40 + vLLM, runs/game_v2/test_b = the "card2" prompt with all its conventions), the share
of blocks that compile, pass the checks AND do what was asked: colour_shape 55%, character 52%, naming 78%,
construct 65%, sound 78%, edit 72%, text 82%, level 35%, open (hand-written) 78%, open written by Gemma 55%.
"""

import json
import random
import re
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

from . import GameDslDomain
from .gemma_writes import HOLLOW, block_of
from .language import Node, parse_fragment, parse_program, walk
from .live import edit_input, insert_input
from .problems import OUTPUT as PROBLEMS, label
from .wording import (COLOURS, SHAPE_NAMES, article, casual, colour_matches, colour_name, colour_spec, noun_of,
                      pointer, random_colour_words, shape_matches, shape_of, shape_size, shape_word)

REPO = Path(__file__).resolve().parents[3]
OUTPUT = REPO / "runs" / "game_v2"
CARD_SOUNDS = ("beep", "bell", "blip", "bubble", "bump", "buzz", "clang", "click", "coin", "death", "glass", "hurt",
               "jump", "laser", "lose", "pickup", "pop", "sparkle", "thud", "win", "win2")
MUSIC = ("bg_1", "bg_2", "bg_3", "bg_4", "bg_5", "bg_6", "bg_ambient", "bg_nature")
FRESH_NOUNS = ("hero", "enemy", "coin", "ghost", "boss", "wall", "platform", "gem", "key", "door", "rock", "bullet",
               "zombie", "slime", "bat", "spider", "robot", "alien", "dragon", "knight", "wizard", "cat", "dog", "fish",
               "bird", "tree", "cloud", "heart", "bomb", "rocket", "ninja", "pirate", "pumpkin", "apple", "mushroom",
               "crystal", "shield", "sword", "flag", "portal", "spike", "ladder", "balloon", "button", "monster",
               "player", "target", "blob", "snake", "frog", "car", "ship", "asteroid", "lamp", "trap", "chest")
NAMES_GIVEN = ("knight", "bob", "max", "luna", "rex", "zippy", "sparky", "hero", "tank", "pixel", "blinky", "rocky",
               "nova", "bolt", "mochi", "ziggy", "buddy", "shadow", "pebble", "comet", "player2", "boss", "speedy")
THING_NOUNS = ("character", "enemy", "thing", "monster", "guy", "coin", "pickup", "obstacle", "target", "blob",
               "friend", "bot", "player", "block", "creature")
WORDS_TO_SAY = ("winners", "GAME OVER", "you win!", "nice!", "try again", "level up", "ouch", "boom", "ready?", "go!",
                "victory", "well done", "oops", "paused", "almost!", "too slow", "perfect", "you lost", "new record",
                "watch out!", "hurry up", "SAFE", "danger", "round 2", "bonus", "combo!", "keep going", "the end")
THEMES = ("victory", "winning", "losing", "a warning", "getting ready", "a welcome", "how to play", "the next level",
          "running out of time", "a high score", "being hit", "a bonus", "the boss")
VERBS = ("add ", "add ", "create ", "make ", "put ", "insert ", "new ", "", "", "place ", "give me ", "i need ",
         "spawn ", "draw ")
SIZE_WORDS = {"big": "big", "large": "big", "huge": "big", "giant": "big", "small": "small", "tiny": "small",
              "little": "small", "mini": "small"}


# ---------- the game and its places ----------

@dataclass
class Spot:
    kind: str                  # insert: top, level, events, actions, bar; edit: entity, event
    offset: int
    indent: str = ""
    node: Optional[Node] = None   # edit: the block under the cursor; insert: the block the cursor is in
    owner: Optional[Node] = None  # the Entity the place belongs to (events, actions, edit of an event)


class Game:
    def __init__(self, version_id: str, source: str, split: str):
        self.version_id, self.source, self.split = version_id, source, split
        self.root = parse_program(source)
        self.entities = [node for node, _ in walk(self.root) if node.key == "entity" and node.child("id")]
        self.ids = {node.child("id").first_string() for node in self.entities} - {None}
        self.names = set(self.ids)
        for node, _ in walk(self.root):
            if node.key in ("bar", "broadcast", "onsignal", "paddle", "goal", "hostilepatrol", "var") \
                    and node.first_string():
                self.names.add(node.first_string())
        self.bars = [node.first_string() for node, _ in walk(self.root) if node.key == "bar" and node.first_string()]
        self.signals = sorted({node.first_string() for node, _ in walk(self.root)
                               if node.key in ("broadcast", "onsignal") and node.first_string()})
        self.templates = [node.child("id").first_string() for node in self.entities if node.child("template")]
        self.levels = [child for child in self.root.children if child.key == "level"]
        self.keys = Counter(node.key for node, _ in walk(self.root))
        self.player = self.find_player()
        sizes = [shape_size(shape_of(node)) for node in self.entities if shape_of(node) is not None
                 and shape_of(node).key != "text"]
        sizes = [size for size in sizes if size and size >= 8]
        self.typical_size = median(sizes) if sizes else 40.0
        self.text_entities = [node.child("id").first_string() for node in self.entities
                              if shape_of(node) is not None and shape_of(node).key == "text"]

    def find_player(self) -> Optional[str]:
        for node, _ in walk(self.root):
            if node.key in ("joystick", "dpad", "camerafollow", "paddle") and node.first_string() in self.ids:
                return node.first_string()
        for node in self.entities:
            movement = node.child("movement")
            if movement is not None and (movement.child("control") or movement.child("gridmove")):
                return node.child("id").first_string()
        return None

    def free_name(self, word: str) -> str:
        """The naming rule (PLAN.md v2 item 5): the word itself, else word_2, word_3, ..."""
        if word not in self.names:
            return word
        number = 2
        while f"{word}_{number}" in self.names:
            number += 1
        return f"{word}_{number}"

    def line_start(self, position: int) -> int:
        return self.source.rfind("\n", 0, position) + 1

    def own_line(self, node: Node) -> bool:
        return self.source[self.line_start(node.start):node.start].strip() == ""

    def indent_at(self, position: int) -> str:
        line = self.source[self.line_start(position):].split("\n", 1)[0]
        return line[:len(line) - len(line.lstrip(" \t"))]

    def insert_spots(self) -> List[Spot]:
        """Every line start inside a Window, Level, Events, Actions or Bar where a new block can go: before a child
        that starts its own line, or before a closing brace on its own line."""
        spots = []
        kinds = {"window": "top", "level": "level", "events": "events", "actions": "actions", "bar": "bar"}

        def visit(node: Node, owner: Optional[Node]):
            if node.key == "entity":
                owner = node
            if node.key in kinds and not (node.key == "window" and node is not self.root):
                lined = [child for child in node.children if self.own_line(child)]
                for child in lined:
                    spots.append(Spot(kinds[node.key], self.line_start(child.start), self.indent_at(child.start),
                                      node, owner))
                closing = node.end - 1
                if self.source[closing:closing + 1] == "}" and \
                        self.source[self.line_start(closing):closing].strip() == "" and lined:
                    spots.append(Spot(kinds[node.key], self.line_start(closing), self.indent_at(lined[-1].start),
                                      node, owner))
            for child in node.children:
                visit(child, owner)
        visit(self.root, None)
        return spots

    def edit_spots(self) -> List[Spot]:
        spots = []

        def visit(node: Node, owner: Optional[Node]):
            if node.key == "entity" and node.child("id"):
                spots.append(Spot("entity", node.start, node=node, owner=node))
                owner = node
            elif node.key in ("oncollision", "ontapped", "onelapsedtime", "onsignal", "ongridstep", "ongridblocked",
                              "onswipe", "sequence"):
                spots.append(Spot("event", node.start, node=node, owner=owner))
            for child in node.children:
                visit(child, owner)
        visit(self.root, None)
        return spots

    def entity_spots(self, spots: List[Spot]) -> List[Spot]:
        """Where a new Entity may go: inside a Level when the game has Levels (the Window's own would be ignored)."""
        return [spot for spot in spots if spot.kind == ("level" if self.levels else "top")]


def the_player(game: Game, random_generator: random.Random) -> str:
    noun = noun_of(game.player)
    options = ["the player", "the player", "me", f"#{game.player}"] + ([f"the {noun}"] if noun else [])
    return random_generator.choice(options)


def entity_called(game: Game, entity_id: str, random_generator: random.Random) -> str:
    noun = noun_of(entity_id)
    return random_generator.choice([f"#{entity_id}", f"the {noun}"] if noun else [f"#{entity_id}"])


# ---------- the units: each returns (task, said, expect) or None ----------

def thing_words(random_generator: random.Random, game: Game) -> Tuple[str, Dict]:
    """A colour + shape (+ size, + a noun) as a person says it, and what the Entity must then be."""
    colour = random_colour_words(random_generator)
    shape = random_generator.choice(SHAPE_NAMES + ("ball", "box", "circle", "square"))
    size = random_generator.choice(list(SIZE_WORDS)) if random_generator.random() < 0.35 else ""
    noun = random_generator.choice(THING_NOUNS) if random_generator.random() < 0.55 else ""
    forms = [f"{size} {colour} {shape} {noun}", f"{noun or 'thing'} that is a {size} {colour} {shape}",
             f"{size} {shape} in {colour}", f"{colour} {size} {shape}", f"{size} {colour} {shape}",
             f"{colour}, {size} {shape} {noun}"]
    words = re.sub(r"\s+", " ", random_generator.choice(forms)).strip().strip(",").strip()
    name = noun if noun and re.search(rf"\b{noun}\b", words) else "thing" if words.startswith("thing") else shape
    if random_generator.random() < 0.2:          # named by the user: "a red circle named knight" (2026-11-09)
        name = random_generator.choice(NAMES_GIVEN)
        words += random_generator.choice([f" named {name}", f" called {name}", f", name it {name}",
                                          f" with the name {name}", f" ({name})"])
    expect = {"top": "entity", "shape": shape, "colour": colour_spec(colour), "new_id": True,
              "id": game.free_name(name)}
    if size:
        expect["size"] = SIZE_WORDS[size]
    return words, expect


def with_verb(words: str, random_generator: random.Random) -> str:
    verb = random_generator.choice(VERBS)
    if verb in ("", "new ") or words.startswith(("a ", "an ")):
        return verb + words
    return f"{verb}{article(words)} {words}"


def unit_colour_shape(game: Game, spot: Spot, random_generator: random.Random):
    words, expect = thing_words(random_generator, game)
    tail = random_generator.choice(["", "", "", " here", " please", " at the top", " on the left"])
    if tail == " at the top":
        expect["where"] = "top"
    elif tail == " on the left":
        expect["where"] = "left"
    return "insert", with_verb(words, random_generator) + tail, expect


def unit_character(game: Game, spot: Spot, random_generator: random.Random):
    words, expect = thing_words(random_generator, game)
    behaviours = [("that moves left and right", {"has_any": ["automove", "path", "setvelocity"]}),
                  ("that falls down", {"has_any": ["automove", "gravity", "setvelocity", "applyimpulse"]}),
                  ("that bounces around", {"has_any": ["automove", "setvelocity"]}),
                  ("that disappears when tapped", {"has": ["ontapped"], "has_any": ["deleteentity", "setvisible"]}),
                  ("that just sits there", {"lacks": ["movement"]}),
                  ("that i can control", {"has_any": ["control", "gridmove"]}),
                  ("that blinks every second", {"has": ["onelapsedtime"]}),
                  ("controlled by dragging", {"has": ["control"]}),
                  (", controlled by dragging it left and right", {"arg": ["movementcontrol", 0, 2]}),
                  ("that i can drag up and down", {"arg": ["movementcontrol", 0, 1]})]
    if game.player:
        player = the_player(game, random_generator)
        behaviours += [(f"that chases {player}", {"chase": game.player}),
                       (f"that hurts {player} when they touch", {"collision": game.player}),
                       (f"that {player} can collect", {"collision": game.player})]
        behaviours = [behaviour for behaviour in behaviours if behaviour[0] != "that i can control"]
    if game.bars:
        bar = random_generator.choice(game.bars)
        behaviours.append((f"that adds a point to {random_generator.choice(['#' + bar, 'the ' + bar])} when tapped",
                           {"has": ["ontapped", "updatebar"]}))
    clause, extra = random_generator.choice(behaviours)
    expect.update(extra)
    return "insert", (with_verb(words, random_generator) + " " + clause).replace(" , ", ", "), expect


def unit_naming(game: Game, spot: Spot, random_generator: random.Random):
    taken = sorted(entity_id for entity_id in game.ids if noun_of(entity_id) == entity_id)
    if taken and random_generator.random() < 0.4:
        noun = random_generator.choice(taken)
    else:
        noun = random_generator.choice([noun for noun in FRESH_NOUNS if noun not in game.names] or FRESH_NOUNS)
    adjective = random_generator.choice(["", "", "", "big ", "small ", "red ", "fast ", "scary ", "little ", "blue ",
                                         "angry ", "golden "])
    forms = [f"create {article(adjective + noun)} {adjective}{noun}", f"add {article(adjective + noun)} {adjective}{noun}",
             f"{adjective}{noun}", f"new {adjective}{noun}", f"make {article(adjective + noun)} {adjective}{noun} here",
             f"i want {article(adjective + noun)} {adjective}{noun}", f"put {article(adjective + noun)} {adjective}{noun} here",
             f"another {adjective}{noun}" if noun in game.ids else f"{article(noun)} {noun} pls"]
    expect = {"top": "entity", "id": game.free_name(noun), "new_id": True}
    return "insert", random_generator.choice(forms), expect


def unit_construct(game: Game, spot: Spot, random_generator: random.Random):
    options = []      # (said forms, expect)
    seconds = random_generator.choice([1, 2, 3, 5, 0.5])
    say_seconds = {1: "1 second", 0.5: "half a second"}.get(seconds, f"{seconds:g} seconds")
    others = sorted(game.ids - ({spot.owner.child("id").first_string()} if spot.owner is not None else set()))
    if spot.kind == "events":
        if others:
            other = random_generator.choice(others)
            called = entity_called(game, other, random_generator)
            options.append(([f"add an OnCollision with #{other}", f"oncollision {other}", f"when it hits {called}",
                              f"add a collision with {called}", f"on collision with {called}",
                              f"what happens when it touches {called}"], {"top": "oncollision",
                                                                          "arg": ["oncollision", 0, other]}))
        options.append(([f"add a timer", f"add an OnElapsedTime", f"every {say_seconds}",
                         f"add a timer that runs every {say_seconds}", f"OnElapsedTime {seconds:g}",
                         f"do something every {say_seconds}"], {"top": "onelapsedtime"}))
        options.append((["add an OnTapped", "when tapped", "add a tap event", "when i tap it", "ontapped",
                         "on tap"], {"top": "ontapped"}))
        signal = random_generator.choice(game.signals) if game.signals and random_generator.random() < 0.7 else \
            random_generator.choice(["boom", "reset", "powerup", "alarm", "start"])
        options.append(([f'add an OnSignal for "{signal}"', f"listen for the {signal} signal",
                         f'onsignal "{signal}"', f"when the {signal} signal comes"],
                        {"top": "onsignal", "arg": ["onsignal", 0, signal]}))
        options.append((["add a Sequence", "add a sequence with two phases", "a sequence of 3 steps",
                         "add a Sequence: first flash, then shake"], {"top": "sequence"}))
        direction = random_generator.choice(["left", "right", "up", "down"])
        options.append(([f"add an OnSwipe {direction}", f"when i swipe {direction}", f"on swipe {direction}"],
                        {"top": "onswipe", "arg": ["onswipe", 0, direction]}))
    elif spot.kind == "actions":
        options.append((["add a randomAction", "pick a random action", "do one of two things at random",
                         "random action here", "randomly either flash or shake"], {"top": "randomaction"}))
        options.append(([f"add a delay of {say_seconds}", f"wait {say_seconds} then flash", f"delay {seconds:g}",
                         f"after {say_seconds} play a pop sound"], {"top": "delay"}))
        options.append((["shake the screen", "add a screen shake", "add a shake", "shake", "small shake"],
                        {"top": "shake"}))
        colour = random_colour_words(random_generator)
        options.append(([f"add a burst", f"particles here", f"burst of {colour} particles",
                         f"add a {colour} burst"], {"top": "burst"}))
        options.append(([f"flash {colour}", "add a flash", f"make it flash {colour}"], {"top": "flash"}))
        word = random_generator.choice(WORDS_TO_SAY)
        options.append(([f'end the game with "{word}"', "game over here", f'add a gameOver that says "{word}"',
                         "the player loses here"], {"top": "gameover"}))
        options.append((["add a loopCount of 3", "repeat the next action 3 times", "do it three times"],
                        {"top": "loopcount"}))
        if game.bars:
            bar = random_generator.choice(game.bars)
            amount = random_generator.choice([1, 2, 5, 10, -1])
            options.append(([f"add {amount} to the {bar}" if amount > 0 else f"take 1 from the {bar}",
                             f"add an updateBar for {bar}", f"increase #{bar} by {abs(amount)}" if amount > 0 else
                             f"decrease #{bar} by 1", f"{bar} +{amount}" if amount > 0 else f"{bar} -1"],
                            {"top": "updatebar", "arg": ["updatebar", 0, bar]}))
            options.append(([f"add a whenBar for {bar}", f"if the {bar} is above 5 then flash",
                             f"only when #{bar} is 0"], {"top": "whenbar", "arg": ["whenbar", 0, bar]}))
        if game.templates:
            template = random_generator.choice(game.templates)
            options.append(([f"spawn a #{template}", f"add a spawn of {template}", f"spawn {template} here",
                             f"make a new {template} appear"], {"top_any": ["spawn", "spawnhere", "spawnatempty"],
                                                                "arg_any": [template]}))
        signal = random_generator.choice(game.signals) if game.signals else "boom"
        options.append(([f'broadcast "{signal}"', f"send the {signal} signal", f"tell everyone {signal}"],
                        {"top": "broadcast", "arg": ["broadcast", 0, signal]}))
        if game.text_entities:
            text_id = random_generator.choice(game.text_entities)
            options.append(([f'set the {text_id} text to "{word}"', f'make #{text_id} say "{word}"',
                             f'change the text of {text_id} to "{word}"'], {"top": "settext", "literal": word}))
        if game.levels:
            options.append((["go to the next level", "next level", "add a nextLevel"], {"top": "nextlevel"}))
    elif spot.kind == "bar":
        value = random_generator.choice([0, 3, 5, 10, 20, 50])
        options.append(([f"add an OnValue", f"add an OnValue at {value}", f"when it reaches {value}",
                         f"onvalue {value}", f"when this hits {value} end the game", f"at {value} play a win sound"],
                        {"top": "onvalue"}))
    elif spot.kind in ("top", "level"):
        name = random_generator.choice(["coins", "time", "health", "energy", "ammo", "gems", "points", "fuel"])
        if spot.kind == "top":
            options.append(([f"add a bar called {name}", f"add a {name} counter", "add a Bar", f"track {name}",
                             f"new bar for {name}"], {"top": "bar"}))
            if game.bars and not game.keys["winat"]:
                bar = random_generator.choice(game.bars)
                options.append(([f"win when the {bar} reaches 20", "add a winAt", f"winning at 20 {bar}"],
                                {"top": "winat"}))
            if not game.keys["lives"]:
                options.append((["give the player 3 lives", "add lives", "3 lives"], {"top": "lives"}))
        if spot.kind == "level" or not game.levels:
            options.append((["add a Wave", "add a wave of enemies every 2 seconds", "add a wave",
                             "a wave that spawns things every second"], {"top": "wave"}))
            options.append((["add walls around the screen", "add a wallBox", "wall box around everything"],
                            {"top": "wallbox"}))
            if game.player:
                options.append((["add a HostilePatrol", "add a guard that patrols", "a patrolling guard",
                                 "add an enemy that walks between two points and catches the player"],
                                {"top_any": ["hostilepatrol", "entity"]}))
    if not options:
        return None
    forms, expect = random_generator.choice(options)
    return "insert", random_generator.choice(forms), expect


def unit_sound(game: Game, spot: Spot, random_generator: random.Random):
    name = random_generator.choice(CARD_SOUNDS)
    if spot.kind == "actions":
        named = random_generator.random() < 0.65
        forms = [f"play a {name} sound", f"{name} sound", f"add the {name} sound here", f"play {name}",
                 f"make a {name} noise"] if named else \
            ["sound effect here", "make a noise", "add a sound", "play a sound", "some sound here pls"]
        expect = {"top": "sound", **({"arg": ["sound", 0, name]} if named else {})}
        return "insert", random_generator.choice(forms), expect
    if spot.kind == "top":
        if game.keys["music"]:
            return None
        track = random_generator.choice(MUSIC)
        return "insert", random_generator.choice(["add background music", "some music pls", f"play {track}",
                                                   "add music"]), {"top": "music"}
    if spot.kind == "event":
        sounds = [node for node, _ in walk(spot.node) if node.key == "sound" and node.first_string()]
        if sounds and random_generator.random() < 0.6:
            old = sounds[0].first_string()
            name = random_generator.choice([sound for sound in CARD_SOUNDS if sound != old])
            forms = [f"change the sound to {name}", f"use {name} instead", f"{name} instead of {old}",
                     f"make it go {name}", f"swap the {old} sound for {name}"]
            return "edit", random_generator.choice(forms), {"arg_any": [name], "keep_top": True}
        forms = [f"add a sound to this", f"play a {name} sound when this happens", f"make it go {name}",
                 f"add a {name} sound", f"{name} sound here too", "make some noise when this happens"]
        said = random_generator.choice(forms)
        expect = {"more": "sound", "keep_top": True}
        if name in said:
            expect["arg_any"] = [name]
        return "edit", said, expect
    if spot.kind == "entity":
        return "edit", random_generator.choice(["add sounds", "give this some sounds", "add sound effects to this",
                                                 f"add sounds to {pointer(random_generator, noun_of(entity_id(spot.node)), None, entity_id(spot.node))}",
                                                 "this needs sounds"]), {"more": "sound", "keep_id": True}
    return None


def entity_id(node: Optional[Node]) -> Optional[str]:
    return node.child("id").first_string() if node is not None and node.key == "entity" and node.child("id") else None


def unit_edit(game: Game, spot: Spot, random_generator: random.Random):
    node = spot.node
    if spot.kind == "entity":
        noun = noun_of(entity_id(node))
        shape = shape_of(node)
        this = pointer(random_generator, noun, shape_word(shape), entity_id(node))
        its = random_generator.choice(["its", f"the {noun}'s" if noun else "its", "its"])
        options = []
        colour_node = node.child("color")
        if colour_node is not None and colour_node.first_string() and colour_name(colour_node.first_string()):
            old = colour_name(colour_node.first_string())
            new = random_colour_words(random_generator)
            while new.split()[-1] in old:
                new = random_colour_words(random_generator)
            forms = [f"make {this} {new}", f"{this} should be {new}", f"change {this} from {old} to {new}",
                     f"edit {this} from {old} to {new}", f"change the colour to {new}", f"colour it {new}",
                     f"{new} instead of {old}", f"make it {new} instead", f"turn {this} {new}", f"paint {this} {new}",
                     f"change {its} color to {new}", f"{new}", f"i want {this} to be {new}"]
            options.append((forms, {"colour": colour_spec(new), "keep_id": True}))
        old_shape = shape_word(shape)
        if old_shape in SHAPE_NAMES + ("pentagon", "cross"):
            new_shape = random_generator.choice([word for word in SHAPE_NAMES if word != old_shape])
            forms = [f"change {this} from {article(old_shape)} {old_shape} to {article(new_shape)} {new_shape}",
                     f"make it {article(new_shape)} {new_shape}", f"turn {this} into {article(new_shape)} {new_shape}",
                     f"{new_shape} instead", f"change the shape to {article(new_shape)} {new_shape}",
                     f"can it be {article(new_shape)} {new_shape}?", f"i want {article(new_shape)} {new_shape} not "
                     f"{article(old_shape)} {old_shape}", f"{new_shape} shape pls"]
            options.append((forms, {"shape": new_shape, "keep_id": True}))
        if shape is not None and shape.key != "text" and shape_size(shape):
            change = random_generator.choice(["grow", "shrink", "double", "half"])
            forms = {"grow": [f"make {this} bigger", "bigger", f"{this} is too small", "make it larger",
                              "a bit bigger pls", "increase the size"],
                     "shrink": [f"make {this} smaller", "smaller", f"{this} is too big", "shrink it", "make it tiny",
                                "a bit smaller"],
                     "double": ["make it twice as big", f"double {its} size", f"{this} twice as big", "2x bigger"],
                     "half": ["make it half the size", f"half {its} size", f"{this} half as big", "halve the size"]}
            options.append((forms[change], {"size_change": change, "keep_id": True}))
        movement = node.child("movement")
        speed = movement.child("speed") if movement is not None else None
        if speed is not None and speed.args and isinstance(speed.args[0], float) and speed.args[0] > 0:
            change = random_generator.choice(["faster", "slower", "double", "half"])
            forms = {"faster": [f"make {this} faster", "faster", "speed it up", f"{this} is too slow", "faster pls"],
                     "slower": [f"make {this} slower", "slower", "slow it down", f"{this} is too fast",
                                "a bit slower"],
                     "double": ["twice as fast", f"double {its} speed", "2x speed"],
                     "half": ["half speed", f"half {its} speed", "half as fast"]}
            options.append((forms[change], {"speed_change": change, "keep_id": True}))
            options.append(([f"make {this} stop moving", "stop it from moving", "it shouldn't move",
                             f"{this} should stay still"], {"still": True, "keep_id": True}))
        if game.player and entity_id(node) != game.player:
            player = the_player(game, random_generator)
            options.append(([f"make {this} chase {player}", f"it should follow {player}",
                             f"make it go after {player}"], {"chase": game.player, "keep_id": True}))
        if shape is not None and shape.key == "text":
            word = random_generator.choice(WORDS_TO_SAY)
            theme = random_generator.choice(THEMES)
            options.append(([f'change the text to "{word}"', f'change to "{word}"', f'make it say "{word}"',
                             f'the text should say "{word}"', f'text: "{word}"', f'write "{word}" instead'],
                            {"literal": word, "keep_id": True}))
            options.append(([f"change the text to something about {theme}", f"make the text about {theme}",
                             f"different text, about {theme}", f"text for {theme}"],
                            {"literal_changed": True, "keep_id": True}))
        if not options:
            return None
        forms, expect = random_generator.choice(options)
        return "edit", random_generator.choice(forms), expect
    if spot.kind == "event":
        options = []
        strings = [(inner, inner.args[1] if inner.key == "settext" else inner.first_string())
                   for inner, _ in walk(node) if inner.key in ("gameover", "settext")
                   and (inner.key == "gameover" or len(inner.args) > 1)]
        if strings:
            word = random_generator.choice(WORDS_TO_SAY)
            theme = random_generator.choice(THEMES)
            options.append(([f'change the message to "{word}"', f'make it say "{word}"', f'change to "{word}"',
                             f'the text should be "{word}"'], {"literal": word, "keep_top": True}))
            options.append(([f"change the text to something about {theme}", f"make the message about {theme}",
                             f"a different message, about {theme}"], {"literal_changed": True, "keep_top": True}))
        colour = random_colour_words(random_generator)
        options.append((["also shake the screen", "add a shake here", "shake too", "and shake the screen"],
                        {"more": "shake", "keep_top": True}))
        options.append(([f"add a {colour} burst here", f"also a burst of {colour} particles", "add particles too"],
                        {"more": "burst", "keep_top": True}))
        if any(inner.key == "sound" for inner, _ in walk(node)):
            options.append((["remove the sound", "no sound here", "make it silent", "delete the sound"],
                            {"lacks": ["sound"], "keep_top": True}))
        if not any(inner.key == "gameover" for inner, _ in walk(node)):
            options.append((["make it end the game", "this should be game over", "game over when this happens"],
                            {"has": ["gameover"], "keep_top": True}))
        if game.bars:
            bar = random_generator.choice(game.bars)
            options.append(([f"also add 1 to {bar}", f"and give a point to #{bar}", f"increase {bar} here"],
                            {"more": "updatebar", "keep_top": True}))
        forms, expect = random_generator.choice(options)
        return "edit", random_generator.choice(forms), expect
    return None


def unit_text(game: Game, spot: Spot, random_generator: random.Random):
    word = random_generator.choice(WORDS_TO_SAY)
    theme = random_generator.choice(THEMES)
    options = [([f"add text that reflects {theme}", f"text here for {theme}", f"some text about {theme}",
                 f"put a message about {theme} here", f"a title for {theme}"], {}),
               ([f'put a big "{word}" text here', f'add a title that says "{word}"', f'add a label: "{word}"',
                 f'write "{word}" here', f'text "{word}"', f'"{word}" text at the top'], {"literal": word})]
    forms, extra = random_generator.choice(options)
    return "insert", random_generator.choice(forms), {"top": "entity", "shape_key": "text", "new_id": True, **extra}


def unit_level(game: Game, spot: Spot, random_generator: random.Random):
    if spot.kind != "top" or not game.levels:
        return None
    number = random_generator.randint(1, len(game.levels))
    forms = ["add a new level", f"create a level like level {number} but harder", "another level with more enemies",
             f"level {len(game.levels) + 1}", "make a harder level", f"a level like level {number} but faster",
             "add an easy first level", "new level pls", "add a bonus level with lots of coins",
             "create a level with moving platforms", "create a level", "create a simple level",
             "a simple level with a few platforms", f"copy level {number} but make it harder",
             f"level {len(game.levels) + 1} with lava", "a short bonus level", "make a new level with 3 enemies",
             "create an easy level for beginners", "add a level where everything is faster",
             "new level, more walls and one exit", "a level with a maze"]
    return "insert", random_generator.choice(forms), {"top": "level", "min_entities": 2}


# ---------- the user's asks of 2026-11-09 evening: control, buttons, rules ("up the score when touched") ----------

def unit_control(game: Game, spot: Spot, random_generator: random.Random):
    """/edit an Entity into something the player drags (Control), or /insert a joystick or dpad for one."""
    if spot.kind == "entity":
        movement = spot.node.child("movement")
        if movement is not None and movement.child("control"):
            return None
        this = pointer(random_generator, noun_of(entity_id(spot.node)), shape_word(shape_of(spot.node)),
                       entity_id(spot.node))
        options = [([f"make {this} controllable", f"make it controllable by dragging", f"let me drag {this}",
                     f"i want to control {this}", f"make {this} the player", "make it draggable",
                     f"control {this} with my finger"], {"has": ["control"], "keep_id": True}),
                   ([f"make {this} controllable left and right only", "drag it sideways only",
                     f"let me move {this} left and right"], {"arg": ["movementcontrol", 0, 2], "keep_id": True}),
                   ([f"make {this} controllable up and down", "drag it up and down only"],
                    {"arg": ["movementcontrol", 0, 1], "keep_id": True})]
        forms, expect = random_generator.choice(options)
        return "edit", random_generator.choice(forms), expect
    if spot.kind in ("top", "level") and not game.keys["joystick"] and not game.keys["dpad"]:
        movers = sorted(game.ids - set(game.templates))
        target = game.player if game.player and random_generator.random() < 0.7 else \
            (random_generator.choice(movers) if movers else None)
        if target is None:
            return None
        called = entity_called(game, target, random_generator)
        options = [([f"add a joystick for {called}", f"control {called} with a joystick", "joystick pls",
                     f"make {called} controllable with a joystick", f"add a joystick so i can move {called}"],
                    {"top": "joystick", "arg": ["joystick", 0, target]}),
                   ([f"add a dpad for {called}", f"arrow buttons to move {called}", f"control {called} with a dpad",
                     f"add arrow pads for {called}"], {"top": "dpad", "arg": ["dpad", 0, target]})]
        forms, expect = random_generator.choice(options)
        return "insert", random_generator.choice(forms), expect
    return None


def unit_button(game: Game, spot: Spot, random_generator: random.Random):
    word = random_generator.choice(("jump", "start", "go", "fire", "reset", "shop", "next", "play", "boost", "help"))
    options = [(["create a button", "add a button", "button here", "a button pls", "make a button"], {}),
               (["add a button that restarts the game", "a restart button", "restart button in the corner"],
                {"has": ["restart"]}),
               ([f'make a button that says "{word}"', f'a "{word}" button', f'add a button with the text "{word}"'],
                {"literal": word}),
               ([f"a big {colour} button in the corner" for colour in ("green", "red", "blue", "yellow")],
                {"has_any": ["color"]})]
    if game.bars:
        bar = random_generator.choice(game.bars)
        options.append(([f"a button that adds a point to {bar}", f"add a button, each tap +1 {bar}",
                         f"button that increases #{bar}"], {"has": ["updatebar"], "arg_any": [bar]}))
    if game.player:
        called = the_player(game, random_generator)
        options.append(([f"add a jump button for {called}", f"a button that makes {called} jump",
                         f"button to push {called} up"], {"has_any": ["applyimpulse", "setvelocity"]}))
    if game.templates:
        template = random_generator.choice(game.templates)
        options.append(([f"a button that spawns a {template}", f"add a button, tap it to spawn #{template}"],
                        {"has_any": ["spawn", "spawnhere", "spawnatempty"], "arg_any": [template]}))
    forms, extra = random_generator.choice(options)
    expect = {"top": "entity", "has": ["ontapped"] + extra.pop("has", []), "new_id": True, **extra}
    if random_generator.random() < 0.15:
        name = random_generator.choice(("start", "go", "reset", "jumpButton", "fire", "menu"))
        return "insert", random_generator.choice(forms) + f" named {name}", {**expect, "id": game.free_name(name)}
    return "insert", random_generator.choice(forms), expect


def unit_rules(game: Game, spot: Spot, random_generator: random.Random):
    """Game rules said in passing: "up the score when touched", "lose a life when it touches me"."""
    owner = spot.owner if spot.kind == "events" else spot.node
    if owner is None or owner.key != "entity" or entity_id(owner) == game.player:
        return None
    bars = [bar for bar in game.bars if bar != "lives"]
    options = []
    called = the_player(game, random_generator) if game.player else "the player"
    if bars:
        options.append((["up the score when touched", "touching this gives a point", "+1 score when touched",
                         f"add a point when {called} touches this", "give points for touching it",
                         "score goes up when you touch it"], {"has": ["updatebar"]}))
    if game.keys["lives"]:
        options.append(([f"lose a life when {called} touches it", "touching this costs a life",
                         "take a life away when touched"], {"has": ["updatebar"], "arg_any": ["lives"]}))
    options.append((["end the game when touched", f"game over when {called} touches this",
                     "touching this should kill you"], {"has": ["gameover"]}))
    options.append((["it should disappear when touched", "remove it when touched", "make it vanish on touch"],
                    {"has_any": ["deleteentity", "setvisible", "setenable"]}))
    forms, specific = random_generator.choice(options)
    if spot.kind == "events":
        said = random_generator.choice(forms).replace(" this", " it")
        return "insert", said, {"top_any": ["oncollision", "ontapped"], **specific}
    return "edit", random_generator.choice(forms), {**specific, "touch_event": True, "keep_id": True}


# the user's own ideas, said the way a person making the game would (hand-written 2026-11-09 for the first test; the
# full set is to come from a model that reads each game): no rule check, only the compiler and the checks
OPEN_INSERTS = (
    "add a power-up that makes {player} faster for 5 seconds", "add a boss that shows up after 30 seconds",
    "make the game get harder over time", "add a coin that appears somewhere random every 3 seconds",
    "add a timer that counts down from 60 and ends the game", "add a big red flash when {player} gets hit",
    "add an enemy that shoots at {player}", "add a heart pickup that gives an extra life",
    "add a wall in the middle", "add some platforms to jump on", "add a moving obstacle",
    "spawn a few rocks that fall from the top", "add a shield that protects {player} once",
    "add a button that restarts the game when tapped", "add a score text in the corner",
    "add stars in the background that twinkle", "make a portal that teleports {player} to the other side",
    "add a trap that slows {player} down", "add a ghost that goes through walls and chases {player}",
    "add a bonus that doubles the points for 10 seconds", "add a finish line at the top",
    "add 5 coins in a row", "add a bouncy pad that throws {player} up", "add an exploding barrel",
    "add a snake enemy that moves in a zigzag", "add a key and a door it opens",
    "add a lava floor that ends the game", "add a slow cloud that drifts across the sky",
    "make an enemy that splits into two when hit", "add a countdown 3 2 1 go at the start",
    "add a pause text that shows when tapped", "add a magnet pickup", "add a little dog that follows {player}",
    "add a cannon that fires balls every 2 seconds", "add a falling star that gives 5 points when caught",
    "add a spinning blade obstacle", "add a sign that says how to play", "add a treasure chest worth 10 points",
    "add confetti when you win", "add a moving platform that goes up and down",
)
OPEN_EDITS = (
    "make this harder to hit", "this should only appear after 10 seconds", "make it explode when it dies",
    "give it 3 hit points", "make it move faster over time", "make this wobble a bit",
    "it should bounce off the walls", "make it change colour when touched", "make it blink",
    "it should respawn after 3 seconds", "make this worth more points", "make it zigzag",
    "make it follow {player} but slower", "it should play a sound and disappear when tapped",
    "make it a bit more fun", "make it go back and forth", "make it glow when {player} is near",
    "make this flash when something hits it", "make it shrink each time it is hit", "make it random each time",
)


def unit_open(game: Game, spot: Spot, random_generator: random.Random):
    pool = OPEN_EDITS if spot.kind in ("entity", "event") else OPEN_INSERTS
    if spot.kind == "event":
        return None
    said = random_generator.choice([ask for ask in pool if game.player or "{player}" not in ask])
    if game.player:
        said = said.replace("{player}", the_player(game, random_generator))
    return ("edit" if spot.kind == "entity" else "insert"), said, {}


UNITS = {"colour_shape": unit_colour_shape, "character": unit_character, "naming": unit_naming,
         "construct": unit_construct, "sound": unit_sound, "edit": unit_edit, "text": unit_text, "level": unit_level,
         "open": unit_open, "control": unit_control, "button": unit_button, "rules": unit_rules}


def unit_spots(unit: str, game: Game, inserts: List[Spot], edits: List[Spot]) -> List[Spot]:
    if unit in ("colour_shape", "character", "naming", "text", "button"):
        return game.entity_spots(inserts)
    if unit == "control":
        return [spot for spot in edits if spot.kind == "entity"] + game.entity_spots(inserts)
    if unit == "rules":
        return [spot for spot in edits if spot.kind == "entity"] + [spot for spot in inserts if spot.kind == "events"]
    if unit == "construct":
        return [spot for spot in inserts if spot.kind in ("events", "actions", "bar", "top", "level")]
    if unit == "sound":
        return [spot for spot in inserts if spot.kind in ("actions", "top")] + edits
    if unit == "edit":
        return edits
    if unit == "open":
        return game.entity_spots(inserts) + [spot for spot in edits if spot.kind == "entity"]
    if unit == "level":
        return [spot for spot in inserts if spot.kind == "top" and game.levels and spot.node is game.root and
                any(child.key == "level" and child.start >= spot.offset for child in game.root.children)
                or (spot.kind == "top" and game.levels and spot.offset >= game.levels[-1].end)]
    return []


# ---------- the student's (and Gemma's) input, the block put back, the checks ----------

def command_line(task: str, said: str) -> str:
    return f"/{task} {said}"


def student_text(game: Game, ask: Dict) -> Tuple[str, int, int, str]:
    """(input, start, end, indent): the app replaces source[start:end] with the block at that indentation."""
    command = command_line(ask["task"], ask["said"])
    if ask["task"] == "edit":
        text, start, end = edit_input(game.source, ask["offset"], command)
        return text, start, end, game.indent_at(start if game.source[start:start + 1] != "\n" else start + 1)
    text, start = insert_input(game.source, ask["offset"], command, ask["indent"])
    return text, start, start, ask["indent"]


def indented(block: str, indent: str) -> str:
    import textwrap
    return "\n".join(indent + line if line.strip() else line for line in textwrap.dedent(block).splitlines())


def spliced(game_source: str, ask: Dict, block: str) -> str:
    start, end = ask["start"], ask["end"]
    return game_source[:start] + indented(block, ask["indent"]) + "\n" + game_source[end:]


def strings_in(nodes: List[Node]) -> List[str]:
    return [value for top in nodes for node, _ in walk(top) if node.key != "id"
            for value in node.args if isinstance(value, str)]


def first_entity(nodes: List[Node]) -> Optional[Node]:
    return next((node for top in nodes for node, _ in walk(top) if node.key == "entity"), None)


def speed_of(entity: Optional[Node]) -> Optional[float]:
    movement = entity.child("movement") if entity is not None else None
    speed = movement.child("speed") if movement is not None else None
    return speed.args[0] if speed is not None and speed.args and isinstance(speed.args[0], float) else None


def ratio_ok(change: str, ratio: Optional[float]) -> bool:
    if ratio is None:
        return False
    return {"grow": ratio >= 1.15, "faster": ratio >= 1.1, "shrink": ratio <= 0.87, "slower": ratio <= 0.91,
            "double": 1.6 <= ratio <= 2.6, "half": 0.35 <= ratio <= 0.65}[change]


def failed_checks(ask: Dict, block: str, before: Optional[str], game: Game) -> List[str]:
    """The expectations the block does not meet (empty = it does what was asked)."""
    expect = ask.get("expect") or {}
    try:
        nodes = parse_fragment(block)
    except ValueError:
        return ["parse"]
    keys = Counter(node.key for top in nodes for node, _ in walk(top))
    tops = [node.key for node in nodes]
    entity = first_entity(nodes) if expect.get("top", "entity") == "entity" or ask["task"] == "edit" else None
    old = parse_fragment(before) if before else []
    old_keys = Counter(node.key for top in old for node, _ in walk(top))
    failed = []
    if "top" in expect and expect["top"] not in tops:
        failed.append("top")
    if "top_any" in expect and not set(expect["top_any"]) & set(tops):
        failed.append("top")
    if expect.get("keep_top") and old and (not nodes or nodes[0].key != old[0].key):
        failed.append("keep_top")
    if expect.get("keep_id") and entity_id(first_entity(nodes)) != entity_id(first_entity(old)):
        failed.append("keep_id")
    if "id" in expect and entity_id(entity) != expect["id"]:
        failed.append("id")
    if expect.get("new_id") and (entity_id(entity) is None or entity_id(entity) in game.names):
        failed.append("new_id")
    if "shape" in expect and not shape_matches(shape_of(entity), expect["shape"]):
        failed.append("shape")
    if "shape_key" in expect and (shape_of(entity) is None or shape_of(entity).key != expect["shape_key"]):
        failed.append("shape")
    if expect.get("colour"):
        colour = entity.child("color").first_string() if entity is not None and entity.child("color") else None
        if colour is None or not colour_matches(colour, colour_spec(expect["colour"]["words"])):
            failed.append("colour")
    if "size" in expect:
        size = shape_size(shape_of(entity))
        big = size is not None and size >= 1.3 * game.typical_size
        small = size is not None and size <= 0.8 * game.typical_size
        if not (big if expect["size"] == "big" else small):
            failed.append("size")
    if "size_change" in expect:
        new_size, old_size = shape_size(shape_of(entity)), shape_size(shape_of(first_entity(old)))
        if not ratio_ok(expect["size_change"], new_size / old_size if new_size and old_size else None):
            failed.append("size_change")
    if "speed_change" in expect:
        new_speed, old_speed = speed_of(entity), speed_of(first_entity(old))
        if not ratio_ok(expect["speed_change"], new_speed / old_speed if new_speed and old_speed else None):
            failed.append("speed_change")
    if expect.get("still") and (speed_of(entity) or 0) > 0 and not (keys["automove"] == 0 and keys["path"] == 0
                                                                   and keys["chase"] == 0):
        failed.append("still")
    if "where" in expect and entity is not None and entity.child("startposition"):
        values = [value for value in entity.child("startposition").args if isinstance(value, float)]
        width, height = window_size(game)
        if len(values) >= 2 and not ((values[1] >= 0.6 * height) if expect["where"] == "top"
                                     else (values[0] <= 0.4 * width)):
            failed.append("where")
    for key in expect.get("has", []):
        if not keys[key]:
            failed.append("has:" + key)
    if "has_any" in expect and not any(keys[key] for key in expect["has_any"]):
        failed.append("has_any")
    for key in expect.get("lacks", []):
        if keys[key]:
            failed.append("lacks:" + key)
    if "more" in expect and keys[expect["more"]] <= old_keys[expect["more"]]:
        failed.append("more:" + expect["more"])
    if "arg" in expect:
        key, index, value = expect["arg"]
        found = [node for top in nodes for node, _ in walk(top) if node.key == key and len(node.args) > index]
        if not any(str(node.args[index]).lower() == str(value).lower() or
                   (isinstance(node.args[index], float) and str(value).replace(".", "", 1).isdigit()
                    and node.args[index] == float(value)) for node in found):
            failed.append("arg")
    if "arg_any" in expect and not set(expect["arg_any"]) & set(strings_in(nodes)):
        failed.append("arg")
    if "chase" in expect and not any(node.key == "chase" and node.first_string() == expect["chase"]
                                     for top in nodes for node, _ in walk(top)):
        failed.append("chase")
    if "collision" in expect and not any(node.key == "oncollision" and node.first_string() == expect["collision"]
                                         for top in nodes for node, _ in walk(top)):
        failed.append("collision")
    if "literal" in expect and expect["literal"].lower().strip() not in \
            [value.lower().strip() for value in strings_in(nodes)]:
        failed.append("literal")
    if expect.get("literal_changed") and set(strings_in(nodes)) == set(strings_in(old)):
        failed.append("literal_changed")
    # "when touched": by the player (a collision) or by a finger (a tap), either is right; it must be a new one
    if expect.get("touch_event") and keys["oncollision"] + keys["ontapped"] <= old_keys["oncollision"] \
            + old_keys["ontapped"]:
        failed.append("touch_event")
    if "min_entities" in expect and keys["entity"] + keys["hostilepatrol"] + keys["wave"] < expect["min_entities"]:
        failed.append("min_entities")
    return failed


def window_size(game: Game) -> Tuple[float, float]:
    for node in game.root.children:
        if node.key == "size" and len(node.args) >= 2:
            return node.args[0] * 100, node.args[1] * 100
    return 800.0, 1000.0


# ---------- making the asks ----------

GAMES = Path(__file__).resolve().parent / "games"
VERSIONS_ROUND2 = PROBLEMS / "versions_round2.jsonl"   # the round-2 games' versions (v1's versions.jsonl untouched)
ROUND2_VERSIONS_PER_GAME = 20
ROUND2_TEST_SHARE = 15     # percent of the round-2 (and round-3) games held out, by a stable hash of their key


def versions_path(round_number: int) -> Path:
    return PROBLEMS / ("versions.jsonl" if round_number == 1 else f"versions_round{round_number}.jsonl")


def load_games(split: Optional[str] = None, originals_only: bool = False, ids: Optional[set] = None,
               rounds: Tuple[int, ...] = (1, 2, 3)) -> List[Game]:
    games = []
    paths = [versions_path(round_number) for round_number in rounds if versions_path(round_number).exists()]
    for path in paths:
        with open(path, encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                if split and record["split"] != split:
                    continue
                if originals_only and not record["id"].endswith("|v0"):
                    continue
                if ids is not None and record["id"] not in ids:
                    continue
                games.append(Game(record["id"], record["source"], record["split"]))
    return games


def round_versions(round_number: int = 2):
    """Seeded versions of the round-2 / round-3 games (games/round<n>/<family>/, written by Claude agents
    2026-11-09) -> problems/versions_round<n>.jsonl; only games that pass every check, as in round 1."""
    from .check import check_sources
    from .versions import make_versions
    output = versions_path(round_number)
    if output.exists():
        print(f"{output} exists: asks are keyed to its versions")
        return
    found = []
    for index_path in sorted((GAMES / f"round{round_number}").glob("*/index.jsonl")):
        family = index_path.parent.name
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                path = index_path.parent / f"{item['slug']}.dsl"
                if path.exists():
                    found.append((family, item["slug"], path.read_text(encoding="utf-8")))
    clean = check_sources({f"{family}/{slug}": source for family, slug, source in found})
    rows, counts = [], Counter()
    for game_index, (family, slug, source) in enumerate(found):
        key = f"round{round_number}/{family}/{slug}"
        if clean[f"{family}/{slug}"]:
            counts["not clean"] += 1
            continue
        split = "test" if zlib.crc32(key.encode()) % 100 < ROUND2_TEST_SHARE else "train"
        count = ROUND2_VERSIONS_PER_GAME if split == "train" else 5
        versions = [source] + make_versions(source, count - 1, 1000 + 2000 * round_number + game_index,
                                            check_sources)
        rows += [{"id": f"{key}|v{index}", "game": key, "family": family, "split": split, "source": version}
                 for index, version in enumerate(versions)]
        counts[split] += 1
        print(f"{key}: {len(versions)} versions ({split})", flush=True)
    append_jsonl(output, rows)
    print(f"{dict(counts)} games, {len(rows):,} versions -> {output}")


def make_ask(game: Game, unit: str, random_generator: random.Random, typo_share: float = 0.25) -> Optional[Dict]:
    inserts, edits = game.insert_spots(), game.edit_spots()
    spots = unit_spots(unit, game, inserts, edits)
    random_generator.shuffle(spots)
    for spot in spots[:6]:
        made = UNITS[unit](game, spot, random_generator)
        if made is None:
            continue
        task, said, expect = made
        if (task == "edit") != (spot.kind in ("entity", "event")):
            continue
        plain = said
        said = casual(said, random_generator, typo_share)
        given = (expect.get("id") or "").rsplit("_", 1)[0]
        if given and given in plain and given not in said:      # a typo must not change a name the user gave
            said = casual(plain, random_generator, 0)
        ask = {"version": game.version_id, "split": game.split, "unit": unit, "task": task, "said": said,
               "offset": spot.offset, "indent": spot.indent, "spot": spot.kind, "expect": expect,
               "cursor_label": label(spot.node, game.source) if spot.node is not None else ""}
        text, start, end, indent = student_text(game, ask)
        ask.update({"text": text, "start": start, "end": end, "indent": indent,
                    "before": game.source[start:end] if task == "edit" else None})
        ask["id"] = f"{game.version_id}|{unit}|{zlib.crc32((said + str(spot.offset)).encode()) % 10**8}"
        return ask
    return None


def budget(ask: Dict) -> int:
    """Gemma's token budget: an edit rewrites its block (~2.8 characters per token), a Level is long."""
    if ask["task"] == "edit":
        return min(2000, int(len(ask["before"]) / 2.8 * 1.4) + 200)
    return {"level": 2000, "construct": 500, "sound": 300, "open": 1200, "gemma_open": 1200}.get(ask["unit"], 700)


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def append_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def test(per_unit: int = 50, folder: Path = OUTPUT / "test"):
    """Asks for Gemma's first v2 test, from the original (v0) train games, and the prompts for the card. Refused in a
    folder that already has Gemma's replies (they belong to its asks)."""
    if (folder / "vllm_written.jsonl").exists():
        print(f"{folder} already has Gemma's replies to its asks: use another folder")
        return
    random_generator = random.Random(2026_11_09)
    games = load_games("train", originals_only=True)
    domain = GameDslDomain("card2")
    asks = []
    for unit in UNITS:
        made, tries = 0, 0
        while made < (per_unit if unit != "edit" else per_unit * 2) and tries < per_unit * 20:
            tries += 1
            ask = make_ask(random_generator.choice(games), unit, random_generator)
            if ask is not None:
                asks.append(ask)
                made += 1
        print(f"{unit}: {made} asks")
    for path in (folder / "asks.jsonl", folder / "prompts.jsonl"):
        path.unlink(missing_ok=True)       # derived: rebuilt each time
    append_jsonl(folder / "asks.jsonl", asks)
    append_jsonl(folder / "prompts.jsonl", [{"id": ask["id"], "command": command_line(ask["task"], ask["said"]),
                                             "prompt": domain.teacher_prompt(ask["text"]), "budget": budget(ask)}
                                            for ask in asks])
    print(f"{len(asks)} asks -> {folder}")


# ---------- the full v2 corpus (the user chose ~50k asks, 2026-11-09) ----------

FULL_FOLDER = OUTPUT / "full"
# rule asks per unit on the train games (+ ~2.5k open asks from the subagents, open_collect); the test games get
# TEST_PER_UNIT of each
FULL_COUNTS = {"colour_shape": 6000, "character": 6000, "naming": 4000, "construct": 9000, "sound": 5000,
               "edit": 10000, "text": 3500, "level": 2000, "open": 1000}
TEST_PER_UNIT = 150


def build_rule_asks(folder: Path = FULL_FOLDER):
    """The rule asks of the full corpus -> rule_asks.jsonl (refused when it exists: Gemma's replies are keyed to
    its ids)."""
    path = folder / "rule_asks.jsonl"
    if path.exists():
        print(f"{path} exists: Gemma's replies are keyed to its asks; use another folder")
        return
    random_generator = random.Random(2026_11_12)
    asks, seen = [], set()
    for split, counts in (("train", FULL_COUNTS), ("test", {unit: TEST_PER_UNIT for unit in FULL_COUNTS})):
        games = load_games(split)
        for unit, count in counts.items():
            made, tries = 0, 0
            while made < count and tries < count * 20:
                tries += 1
                ask = make_ask(random_generator.choice(games), unit, random_generator)
                if ask is None or ask["id"] in seen:
                    continue
                seen.add(ask["id"])
                asks.append(ask)
                made += 1
            print(f"{split} {unit}: {made:,}", flush=True)
    append_jsonl(path, asks)
    print(f"{len(asks):,} rule asks -> {path}")


def build_prompts(folder: Path = FULL_FOLDER):
    """Every ask (the rule asks + the subagents' open asks) -> asks.jsonl and prompts.jsonl (Gemma's card2 prompt;
    rebuilt each time, the ids stay)."""
    asks = read_jsonl(folder / "rule_asks.jsonl") + open_collect()
    domain = GameDslDomain("card2")
    for name in ("asks.jsonl", "prompts.jsonl"):
        (folder / name).unlink(missing_ok=True)      # derived from rule_asks.jsonl and the answers
    append_jsonl(folder / "asks.jsonl", asks)
    append_jsonl(folder / "prompts.jsonl", [{"id": ask["id"], "command": command_line(ask["task"], ask["said"]),
                                             "prompt": domain.teacher_prompt(ask["text"]), "budget": budget(ask)}
                                            for ask in asks])
    counts = Counter((ask["split"], ask["unit"]) for ask in asks)
    print(f"{len(asks):,} asks -> {folder}: {dict(sorted(counts.items()))}")


# ---------- round 2 (the user, 2026-11-09 evening: more games, more kinds of asks, up to ~$30 in all) ----------

FULL2_FOLDER = OUTPUT / "full2"
OPEN2_FOLDER = OUTPUT / "open2"
# (split, game rounds, asks per unit): every unit on the round-2 games; the three new units also on round 1's
ROUND2_PLAN = (
    ("train", (2,), {"colour_shape": 1500, "character": 2000, "naming": 1000, "construct": 3500, "sound": 2000,
                     "edit": 4500, "text": 1000, "level": 800, "open": 500, "control": 1500, "button": 1500,
                     "rules": 2000}),
    ("train", (1,), {"control": 2500, "button": 2500, "rules": 3000}),
    ("test", (2,), {unit: 100 for unit in ("colour_shape", "character", "naming", "construct", "sound", "edit",
                                           "text", "level", "open", "control", "button", "rules")}),
    ("test", (1,), {"control": 100, "button": 100, "rules": 100}),
)


def build_rule_asks_round2(folder: Path = FULL2_FOLDER):
    path = folder / "rule_asks.jsonl"
    if path.exists():
        print(f"{path} exists: Gemma's replies are keyed to its asks; use another folder")
        return
    random_generator = random.Random(2026_11_13)
    asks, seen = [], set()
    for split, rounds, counts in ROUND2_PLAN:
        games = load_games(split, rounds=rounds)
        for unit, count in counts.items():
            made, tries = 0, 0
            while made < count and tries < count * 20:
                tries += 1
                ask = make_ask(random_generator.choice(games), unit, random_generator)
                if ask is None or ask["id"] in seen:
                    continue
                seen.add(ask["id"])
                asks.append(ask)
                made += 1
            print(f"{split} round {rounds} {unit}: {made:,}", flush=True)
    append_jsonl(path, asks)
    print(f"{len(asks):,} rule asks -> {path}")


def build_prompts_round2(folder: Path = FULL2_FOLDER):
    asks = read_jsonl(folder / "rule_asks.jsonl") + open_collect(OPEN2_FOLDER)
    domain = GameDslDomain("card2")
    for name in ("asks.jsonl", "prompts.jsonl"):
        (folder / name).unlink(missing_ok=True)      # derived from rule_asks.jsonl and the answers
    append_jsonl(folder / "asks.jsonl", asks)
    append_jsonl(folder / "prompts.jsonl", [{"id": ask["id"], "command": command_line(ask["task"], ask["said"]),
                                             "prompt": domain.teacher_prompt(ask["text"]), "budget": budget(ask)}
                                            for ask in asks])
    counts = Counter((ask["split"], ask["unit"]) for ask in asks)
    print(f"{len(asks):,} asks -> {folder}: {dict(sorted(counts.items()))}")


# ---------- round 3 (the user, 2026-11-09 night: usage cap 50%): ~58 more games, rewordings, round-1 open asks ----------

FULL3_FOLDER = OUTPUT / "full3"
ROUND3_UNIT_COUNTS = {"colour_shape": 600, "character": 800, "naming": 500, "construct": 1200, "sound": 700,
                      "edit": 1500, "text": 500, "level": 300, "open": 200, "control": 600, "button": 600, "rules": 800}
ROUND3_PLAN = (("train", (3,), ROUND3_UNIT_COUNTS),
               ("test", (3,), {unit: 60 for unit in ROUND3_UNIT_COUNTS}))


def build_rule_asks_round3(folder: Path = FULL3_FOLDER):
    path = folder / "rule_asks.jsonl"
    if path.exists():
        print(f"{path} exists: Gemma's replies are keyed to its asks; use another folder")
        return
    random_generator = random.Random(2026_11_17)
    asks, seen = [], set()
    for split, rounds, counts in ROUND3_PLAN:
        games = load_games(split, rounds=rounds)
        for unit, count in counts.items():
            made, tries = 0, 0
            while made < count and tries < count * 20:
                tries += 1
                ask = make_ask(random_generator.choice(games), unit, random_generator)
                if ask is None or ask["id"] in seen:
                    continue
                seen.add(ask["id"])
                asks.append(ask)
                made += 1
            print(f"{split} round {rounds} {unit}: {made:,}", flush=True)
    append_jsonl(path, asks)
    print(f"{len(asks):,} rule asks -> {path}")


def build_prompts_round3(folder: Path = FULL3_FOLDER):
    asks = read_jsonl(folder / "rule_asks.jsonl") + reword_collect() + open_collect(OUTPUT / "open3")
    domain = GameDslDomain("card2")
    for name in ("asks.jsonl", "prompts.jsonl"):
        (folder / name).unlink(missing_ok=True)      # derived from rule_asks.jsonl and the answers
    append_jsonl(folder / "asks.jsonl", asks)
    append_jsonl(folder / "prompts.jsonl", [{"id": ask["id"], "command": command_line(ask["task"], ask["said"]),
                                             "prompt": domain.teacher_prompt(ask["text"]), "budget": budget(ask)}
                                            for ask in asks])
    counts = Counter((ask["split"], ask["unit"], ask.get("writer", "rule")) for ask in asks)
    print(f"{len(asks):,} asks -> {folder}: {dict(sorted(counts.items()))}")


# ---------- rule asks reworded by Claude subagents (round 3, 2026-11-09: the templates' wording is narrow) ----------

REWORD_FOLDER = OUTPUT / "reword"
REWORDS_PER_FILE = 600
REWORD_UNITS = ("colour_shape", "character", "naming", "construct", "sound", "edit", "text", "level", "control",
                "button", "rules")
KEYWORD_WORDS = re.compile(r"\b(On[A-Z]\w*|when[A-Z]\w*|random[A-Z]\w*|Wave|Sequence|Phase|Bar|wallBox|HostilePatrol|"
                           r"loopCount|nextLevel|gameOver|updateBar|winAt)\b")

REWORD_INSTRUCTIONS = """You are writing training data for a code assistant inside the editor of QR Game Studio, an app \
with a small game language. A user types an ask at their cursor (/insert: new code at the cursor; /edit: change the \
block under the cursor). Below are asks a script made from templates; their wording is too uniform.

Rewrite each ask the way a different real person would type the same request: change the sentence structure, the \
verb (or drop it), the word order, the tone (terse, chatty, a question, a little command), sometimes add a short \
reason or filler ("for the boss fight", "pls", "ok so"), sometimes a typo. Keep the meaning exactly, and keep \
every word listed after "keep:" exactly as written (names, colours, shapes, sizes, numbers, quoted texts, #names, \
construct names). Do not add new requirements. No code, no hex colours. One line per ask.

Answers: append to {answers}, one line per ask: <number><TAB><your rewrite> (no quotes, no /insert or /edit). \
Append after each part you read, all its asks in one go (Bash: cat >> file <<'EOF' ... EOF; never `cat >>` \
without the heredoc); never rewrite the file.
"""


def keep_words(ask: Dict) -> List[str]:
    """What a rewrite must keep so the ask's rule check still holds."""
    said, expect = ask["said"], ask.get("expect") or {}
    keep = re.findall(r'"[^"]*"', said) + re.findall(r"#\w+", said) + re.findall(r"\b\d+(?:\.\d+)?\b", said)
    keep += KEYWORD_WORDS.findall(said)
    words = said.lower()
    given = (expect.get("id") or "").rsplit("_", 1)[0]
    if given and (re.search(rf"\b{re.escape(given)}\b", words) or ask["unit"] == "naming"):
        keep.append(given)
    values = [expect["arg"][2]] if "arg" in expect else []
    values += list(expect.get("arg_any", [])) + ([expect["literal"]] if "literal" in expect else [])
    keep += [str(value) for value in values if isinstance(value, str) and str(value).lower() in words]
    if expect.get("colour"):
        keep += [word for word in re.split(r"[\s,]+", expect["colour"]["words"]) if word]
    if expect.get("shape") and expect["shape"] in words:
        keep.append(expect["shape"])
    for size in SIZE_WORDS:
        if re.search(rf"\b{size}\b", words):
            keep.append(size)
    return list(dict.fromkeys(keep))


def reword_tasks(files: int = 12, folder: Path = REWORD_FOLDER):
    if (folder / "sources.jsonl").exists():
        print(f"{folder / 'sources.jsonl'} exists: its answers may be on the way")
        return
    random_generator = random.Random(2026_11_15)
    pool = [ask for path in (FULL_FOLDER / "rule_asks.jsonl", FULL2_FOLDER / "rule_asks.jsonl")
            for ask in read_jsonl(path) if ask["split"] == "train" and ask["unit"] in REWORD_UNITS]
    by_unit = {unit: [ask for ask in pool if ask["unit"] == unit] for unit in REWORD_UNITS}
    per_unit = files * REWORDS_PER_FILE // len(REWORD_UNITS) + 1
    chosen = [ask for unit in REWORD_UNITS for ask in random_generator.sample(by_unit[unit],
                                                                             min(per_unit, len(by_unit[unit])))]
    random_generator.shuffle(chosen)
    chosen = chosen[:files * REWORDS_PER_FILE]
    sources = []
    for file_index in range(files):
        answers = (folder / f"tasks_{file_index:02d}_answers.tsv").resolve().as_posix()
        lines = [REWORD_INSTRUCTIONS.format(answers=answers)]
        for ask in chosen[file_index * REWORDS_PER_FILE:(file_index + 1) * REWORDS_PER_FILE]:
            number = len(sources) + 1
            keep = keep_words(ask)
            lines.append(f"{number}\t/{ask['task']}\t{ask['said']}" + (f"\tkeep: {' | '.join(keep)}" if keep else ""))
            sources.append({"number": number, "id": ask["id"], "keep": keep})
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"tasks_{file_index:02d}.txt", "x", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")
    append_jsonl(folder / "sources.jsonl", sources)
    print(f"{len(sources):,} asks to reword in {files} files -> {folder}")


def reword_collect(folder: Path = REWORD_FOLDER) -> List[Dict]:
    """The rewrites -> new asks (unit unchanged, the original's expect kept), only when every keep word is there."""
    sources = {source["number"]: source for source in read_jsonl(folder / "sources.jsonl")}
    originals = {ask["id"]: ask for path in (FULL_FOLDER / "rule_asks.jsonl", FULL2_FOLDER / "rule_asks.jsonl")
                 for ask in read_jsonl(path)}
    games = {game.version_id: game for game in load_games(ids={originals[source["id"]]["version"]
                                                               for source in sources.values()})}
    asks, dropped, seen = [], Counter(), set()
    for path in sorted(folder.glob("tasks_*_answers.tsv")):
        for line in open(path, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].strip().isdigit() or int(parts[0]) not in sources:
                dropped["format"] += 1
                continue
            source = sources[int(parts[0])]
            said = re.sub(r"^/(insert|edit)\s*", "", " ".join(parts[1:]).strip().strip('"')).strip()
            if not said or len(said) > 300 or re.search(r"[{}]|\w\(|\b[0-9a-fA-F]{6}\b", said):
                dropped["code or empty"] += 1
                continue
            if any(word.lower() not in said.lower() for word in source["keep"]):
                dropped["lost a keep word"] += 1
                continue
            original = originals[source["id"]]
            ask = {**original, "said": said, "writer": "claude-reword", "reworded_from": original["id"]}
            text, start, end, indent = student_text(games[original["version"]], ask)
            ask.update({"text": text, "start": start, "end": end, "indent": indent,
                        "id": f"{original['version']}|{original['unit']}|r"
                              f"{zlib.crc32((said + str(original['offset'])).encode()) % 10**8}"})
            if ask["id"] in seen:
                continue
            seen.add(ask["id"])
            asks.append(ask)
    print(f"{len(asks):,} reworded asks; dropped {dict(dropped)}")
    return asks


# ---------- Gemma as the writer of the open asks (tried in the first test: it is already on the card) ----------

WRITER_PROMPT = """{card}

---

You are making a small game in QR Game Studio's editor; its code is below. The editor has a code assistant: you put \
your cursor somewhere, type a one-line command, and it writes the code.

{where}

Write {count} different commands you might type there: things this game does not have yet that would make it more \
fun, or changes you would want. Mix how you say them, the way real people type: some short and casual, some with \
details (how many, how fast, what colour, what happens), one with a typo. Never write code, keyword names or hex \
colours. One command per line, nothing else.

```
{game}
```"""
WRITER_COUNT = 4
WRITER_MAX_CHARS = 12000


def writer_prompts(folder: Path, spots_count: int = 24):
    """Spots for Gemma to write open asks for: half /insert (where a new Entity may go), half /edit (an Entity)."""
    random_generator = random.Random(2026_11_10)
    games = [game for game in load_games("train", originals_only=True) if len(game.source) <= WRITER_MAX_CHARS]
    card = (Path(__file__).resolve().parent / "reference_card.md").read_text(encoding="utf-8")
    rows = []
    for index in range(spots_count):
        game = random_generator.choice(games)
        if index % 2 == 0:
            spot = random_generator.choice(game.entity_spots(game.insert_spots()))
            marked = game.source[:spot.offset] + spot.indent + "<cursor>\n" + game.source[spot.offset:]
            where = ("Your cursor is at <cursor> in the code, on an empty line: the assistant writes new code there "
                     "(an /insert).")
            task = "insert"
        else:
            spot = random_generator.choice([spot for spot in game.edit_spots() if spot.kind == "entity"])
            marked = game.source[:spot.offset] + "<cursor>" + game.source[spot.offset:]
            where = (f"Your cursor is at <cursor>, on the Entity {label(spot.node, game.source)}: the assistant "
                     f"rewrites that Entity as you ask (an /edit; you may call it \"this\" or \"it\").")
            task = "edit"
        rows.append({"id": f"{game.version_id}|writer|{task}|{spot.offset}|{spot.indent}", "command": "",
                     "prompt": WRITER_PROMPT.format(card=card, where=where, count=WRITER_COUNT, game=marked),
                     "budget": 400})
    path = folder / "writer_prompts.jsonl"
    path.unlink(missing_ok=True)       # derived: rebuilt each time
    append_jsonl(path, rows)
    print(f"{len(rows)} writer prompts -> {path}")


def writer_asks(folder: Path, written_name: str = "vllm_written.jsonl"):
    """Gemma's open asks (the writer rows of the replies file) -> asks of unit "gemma_open" + the prompts for Gemma
    to answer them (prompts_round2.jsonl)."""
    domain = GameDslDomain("card2")
    games = {game.version_id: game for game in load_games("train", originals_only=True)}
    asks = []
    for row in read_jsonl(folder / written_name):
        if "|writer|" not in row["id"]:
            continue
        version, place = row["id"].rsplit("|writer|", 1)
        task, offset, indent = place.split("|")
        game = games[version]
        for line in (row.get("reply") or "").split("\n"):
            said = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip('"').strip()
            said = re.sub(r"^/(insert|edit)\s*", "", said)
            if not said or len(said) > 300 or re.search(r"[{}]|\w\(", said):
                continue
            ask = {"version": version, "split": game.split, "unit": "gemma_open", "task": task, "said": said,
                   "offset": int(offset), "indent": indent, "spot": "writer", "expect": {}, "cursor_label": ""}
            text, start, end, ask_indent = student_text(game, ask)
            ask.update({"text": text, "start": start, "end": end, "indent": ask_indent,
                        "before": game.source[start:end] if task == "edit" else None,
                        "id": f"{version}|gemma_open|{zlib.crc32((said + offset).encode()) % 10**8}"})
            asks.append(ask)
    append_jsonl(folder / "asks.jsonl", asks)
    path = folder / "prompts_round2.jsonl"
    path.unlink(missing_ok=True)       # derived: rebuilt each time
    append_jsonl(path, [{"id": ask["id"], "command": command_line(ask["task"], ask["said"]),
                         "prompt": domain.teacher_prompt(ask["text"]), "budget": budget(ask)} for ask in asks])
    print(f"{len(asks)} asks written by Gemma -> {path}")


# ---------- open asks written by Claude subagents (the user's choice, 2026-11-09) ----------

OPEN_FOLDER = OUTPUT / "open"
PLACES_PER_FILE = 40
ASKS_PER_PLACE = 4
# where the open asks go: new things where an Entity may go, new events and actions, changes to entities and events
PLACE_KINDS = {"entity_place": 0.35, "events": 0.15, "actions": 0.15, "entity": 0.25, "event": 0.10}

OPEN_INSTRUCTIONS = """You are writing training data for a code assistant inside the editor of QR Game Studio, an app \
with a small game language. A user puts the cursor somewhere in their game, types an ask, and the assistant writes \
the code: /insert writes new code at the cursor, /edit rewrites the block under the cursor.

Each place below shows what the assistant sees: WINDOW (window size in units of 100 px, background, gravity, ...), \
NAMES (one line per entity / group / bar / signal: its kind, shape, movement and events), CURSOR (where the cursor \
is), BLOCK (the block under or around the cursor; <cursor> marks the spot). The game itself is not shown.

For each place, write {asks} different asks a person making this game might type there, from their point of view: \
"here is my game; what would I want to add or change right here?" New features, tweaks, characters, enemies, \
pickups, sounds, effects, texts, rules, difficulty: things the game does not have yet, not the blocks it already \
has.
- /insert places: the ask must be about new code AT the cursor. CURSOR says where: at the top level or inside a \
Level, a new thing (an entity, a group, a bar, a wave, a whole new level when the game has Levels); inside an \
Events block, a new event of the entity in BLOCK ("when it hits the wall ..."); inside an Actions block, more \
actions for that moment ("also shake the screen"). Never a change to something elsewhere in the game.
- /edit places: change the block under the cursor. Point at it the way people do: "this", "it", "this knight", \
"the ball", sometimes #id.
- Vary the wording a lot across the asks: with or without a verb ("add", "create", "make", "put", none at all), \
short and casual ("boss here pls"), detailed (how many, how fast, what colour, what happens), a question ("can you \
make ..."), mostly lower case, now and then a typo. About one ask in five names a construct of the language \
("add an OnValue at 10", "add a timer", "add a Wave", "add an OnCollision with #wall").
- Ask only for what this language can do: shapes (rectangle, circle, triangle, star, diamond, ring, polygon, cross, \
text, emoji text), colours, sizes, positions, movement (automatic, player controlled, paths, chasing, grid steps), \
collisions, taps, swipes, timers, signals, bars and counters (score, lives, time), win and lose, sounds and music, \
particles, shake, flash, spawning copies, levels. No images, no online play, no menus or settings screens.
- Never write code, braces, brackets after a word, or hex colours; colour names are fine.
- A #name only for a name in NAMES.

Answers: append to {answers}, one line per ask: <place number><TAB><ask> (no quotes around it, no verb like /insert \
in front). Append after each part you read, all its places in one go (Bash: cat >> file <<'EOF' ... EOF; never \
`cat >>` without the heredoc); never rewrite the file.
"""

OPEN_AGENT_PROMPT = """Read the instructions at the top of {tasks} and do every place in it ({places} places, \
{asks} asks each). The file is {size} KB: read it in a few big parts (Read with offset/limit, ~600 lines at a time), \
and after each part append the answers for all its places in one go (one heredoc). Do not run any code, touch any \
file other than the answers file the instructions name, or use git. When done, reply with how many asks you wrote \
and any place whose instructions were impossible to follow (one line each)."""


def place_spots(game: Game, kind: str) -> List[Spot]:
    if kind == "entity_place":
        return game.entity_spots(game.insert_spots())
    if kind in ("events", "actions"):
        return [spot for spot in game.insert_spots() if spot.kind == kind]
    return [spot for spot in game.edit_spots() if spot.kind == kind]


def open_tasks(files: int = 14, test_files: int = 2, folder: Path = OPEN_FOLDER, rounds: Tuple[int, ...] = (1,),
               seed: int = 2026_11_11):
    """Task files for the subagents: PLACES_PER_FILE places each, from random versions (train files from train
    games, the last test_files from the held-out games) -> tasks_NN.txt + places.jsonl."""
    if (folder / "places.jsonl").exists():
        print(f"{folder / 'places.jsonl'} exists: its answers may be on the way; use another folder")
        return
    random_generator = random.Random(seed)
    by_split = {"train": load_games("train", rounds=rounds), "test": load_games("test", rounds=rounds)}
    places = []
    for file_index in range(files + test_files):
        split = "train" if file_index < files else "test"
        answers = (folder / f"tasks_{file_index:02d}_answers.tsv").resolve().as_posix()
        lines = [OPEN_INSTRUCTIONS.format(asks=ASKS_PER_PLACE, answers=answers)]
        made = 0
        while made < PLACES_PER_FILE:
            game = random_generator.choice(by_split[split])
            kind = random_generator.choices(list(PLACE_KINDS), list(PLACE_KINDS.values()))[0]
            spots = place_spots(game, kind)
            if not spots:
                continue
            spot = random_generator.choice(spots)
            task = "edit" if kind in ("entity", "event") else "insert"
            ask = {"task": task, "said": "", "offset": spot.offset, "indent": spot.indent}
            try:
                text, *_ = student_text(game, ask)
            except (ValueError, TypeError):
                continue
            number = len(places) + 1
            seen = text.rsplit("\nCOMMAND", 1)[0]
            lines.append(f"===== PLACE {number} | /{task}\n{seen}\n")
            places.append({"number": number, "file": file_index, "version": game.version_id, "split": split,
                           "task": task, "offset": spot.offset, "indent": spot.indent, "kind": kind})
            made += 1
        (folder / f"tasks_{file_index:02d}.txt").parent.mkdir(parents=True, exist_ok=True)
        with open(folder / f"tasks_{file_index:02d}.txt", "x", encoding="utf-8") as file:
            file.write("\n".join(lines))
    append_jsonl(folder / "places.jsonl", places)
    print(f"{len(places)} places in {files + test_files} files -> {folder}")


def open_collect(folder: Path = OPEN_FOLDER) -> List[Dict]:
    """The subagents' answers -> asks of unit "open" (no rule check), with their inputs."""
    places = {place["number"]: place for place in read_jsonl(folder / "places.jsonl")}
    games = {game.version_id: game for game in load_games(ids={place["version"] for place in places.values()})}
    asks, dropped = [], Counter()
    for path in sorted(folder.glob("tasks_*_answers.tsv")):
        for line in open(path, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].strip().isdigit() or int(parts[0]) not in places:
                dropped["format"] += 1
                continue
            place = places[int(parts[0])]
            said = re.sub(r"^/(insert|edit)\s*", "", " ".join(parts[1:]).strip().strip('"')).strip()
            if not said or len(said) > 300 or re.search(r"[{}]|\w\(|\b[0-9a-fA-F]{6}\b", said):
                dropped["code or empty"] += 1
                continue
            if re.search(r"\.\.\.\s*no\b|\bi mean\b|\bno wait\b|\?\s*no\b|,\s*no,", said):   # an agent's self-correction
                dropped["self-correction"] += 1
                continue
            game = games[place["version"]]
            if any(name not in game.names | {"lives", "score"} for name in re.findall(r"#(\w+)", said)):
                dropped["unknown #name"] += 1
                continue
            ask = {"version": place["version"], "split": place["split"], "unit": "open", "task": place["task"],
                   "said": said, "offset": place["offset"], "indent": place["indent"], "spot": place["kind"],
                   "expect": {}, "cursor_label": "", "writer": "claude-subagent"}
            text, start, end, indent = student_text(game, ask)
            ask.update({"text": text, "start": start, "end": end, "indent": indent,
                        "before": game.source[start:end] if place["task"] == "edit" else None,
                        "id": f"{place['version']}|open|{zlib.crc32((said + str(place['offset'])).encode()) % 10**8}"})
            asks.append(ask)
    print(f"{len(asks):,} open asks from the subagents; dropped {dict(dropped)}")
    return asks


def without_semicolons(block: str) -> str:
    """Gemma's most common slip: actions separated by ";" (the language has none); taken out, never inside a text."""
    parts = re.split(r'("[^"\n]*")', block)
    return "".join(part if index % 2 else part.replace(";", " ") for index, part in enumerate(parts))


def repaired(ask: Dict, block: str, game: Game) -> Tuple[str, List[str]]:
    """Two slips mended by rule instead of dropping the block: ";" between actions, and a new Entity that does not
    follow the naming rule (PLAN.md v2 item 5: "triangle_1" when "triangle" is free; renamed to the rule's id)."""
    repairs = []
    if ";" in re.sub(r'"[^"\n]*"', "", block):
        block = without_semicolons(block)
        repairs.append("semicolons")
    expected = (ask.get("expect") or {}).get("id")
    if expected:
        try:
            written = entity_id(first_entity(parse_fragment(block)))
        except ValueError:
            written = None
        if written and written != expected and written not in game.names - game.ids | {expected} \
                and (written not in game.ids or written.startswith(expected.split("_")[0])):
            block = re.sub(r'"' + re.escape(written) + r'"', f'"{expected}"', block)
            repairs.append("id")
    return block, repairs


def check(folder: Path = OUTPUT / "test", written_name: str = "vllm_written.jsonl"):
    """Gemma's replies: the block into its game (where the app would put it), the app's compiler and checks (a
    block is kept when it adds no problem the game did not have), then the rule checks of the ask."""
    from .check import check_sources
    asks = {ask["id"]: ask for ask in read_jsonl(folder / "asks.jsonl")}
    replies = {row["id"]: row for row in read_jsonl(folder / written_name) if row.get("reply")}
    games = {game.version_id: game for game in load_games() if game.version_id in {ask["version"] for ask in
                                                                                   asks.values()}}
    baseline = {version: Counter(found) for version, found in
                check_sources({version: games[version].source for version in games}).items()}
    rows, sources = [], {}
    for ask_id, ask in asks.items():
        reply = replies.get(ask_id)
        if reply is None:
            continue
        block = block_of(reply["reply"])
        repairs = []
        if block is not None:
            block, repairs = repaired(ask, block, games[ask["version"]])
        rows.append({"id": ask_id, "unit": ask["unit"], "task": ask["task"], "said": ask["said"], "block": block,
                     "repairs": repairs, "finish": reply.get("finish")})
        if block is not None:
            sources[ask_id] = spliced(games[ask["version"]].source, ask, block)
    found = check_sources(sources) if sources else {}
    for row in rows:
        ask = asks[row["id"]]
        if row["block"] is None:
            row.update({"kept": False, "new_problems": ["no complete block"], "failed": ["no block"]})
            continue
        added = Counter(found[row["id"]]) - baseline[ask["version"]]
        row["new_problems"] = sorted(item for item in added.elements() if not item.startswith(HOLLOW))
        row["kept"] = not row["new_problems"]
        row["failed"] = failed_checks(ask, row["block"], ask["before"], games[ask["version"]])
        row["follows"] = not row["failed"]
    path = folder / "checked.jsonl"
    path.unlink(missing_ok=True)        # derived from the replies
    append_jsonl(path, rows)
    print(f"{'unit':<14}{'asks':>6}{'complete':>10}{'kept':>8}{'follows':>9}{'both':>7}  most failed checks")
    for unit in list(UNITS) + ["gemma_open"]:
        unit_rows = [row for row in rows if row["unit"] == unit]
        if not unit_rows:
            continue
        count = len(unit_rows)
        share = lambda test: f"{sum(map(test, unit_rows)) / count:.0%}"
        failed = Counter(check for row in unit_rows for check in row.get("failed", []))
        print(f"{unit:<14}{count:>6}{share(lambda row: row['block'] is not None):>10}{share(lambda row: row['kept']):>8}"
              f"{share(lambda row: row.get('follows', False)):>9}"
              f"{share(lambda row: row['kept'] and row.get('follows', False)):>7}  {failed.most_common(4)}")
    print(f"-> {path}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "test"
    folder_name = sys.argv[3] if len(sys.argv) > 3 else "test"
    if action == "test" and not (OUTPUT / folder_name / "vllm_written.jsonl").exists():
        test(int(sys.argv[2]) if len(sys.argv) > 2 else 50, OUTPUT / folder_name)
        writer_prompts(OUTPUT / folder_name)
    elif action == "test":
        print(f"{OUTPUT / folder_name} already has Gemma's replies to its asks: name another folder")
    elif action == "writer_asks":
        writer_asks(OUTPUT / (sys.argv[2] if len(sys.argv) > 2 else "test"))
    elif action == "build":
        build_rule_asks()
    elif action == "prompts":
        build_prompts()
    elif action == "round2_versions":
        round_versions(2)
    elif action == "round3_versions":
        round_versions(3)
    elif action == "build2":
        build_rule_asks_round2()
    elif action == "open_tasks2":
        open_tasks(12, 2, OPEN2_FOLDER, rounds=(2,), seed=2026_11_14)
    elif action == "prompts2":
        build_prompts_round2()
    elif action == "build3":
        build_rule_asks_round3()
    elif action == "prompts3":
        build_prompts_round3()
    elif action == "reword_tasks":
        reword_tasks()
    elif action == "open_tasks3":
        open_tasks(8, 0, OUTPUT / "open3", rounds=(1,), seed=2026_11_16)
    elif action == "open_tasks":
        open_tasks()
    elif action == "open_collect":
        open_collect()
    elif action == "check":
        check(Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT / "test")
