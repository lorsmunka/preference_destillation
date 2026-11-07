"""The games the data is made from.

  games/human/   the app's 26 built-in games (written by the user), exported with `python -m
                 library.domain.game_dsl.games`; index.json gives each one's name, description and split
  games/new/<family>/<slug>.dsl   new games written for the data (by Claude agents, 2026-11-07), one folder per
                 family of genres, each with index.jsonl ({"slug", "title", "brief"} per game)

Test = the 8 held-out human games (HELD_OUT): never shown as examples when new games are written, never
recombined; they measure how the student does on games written by a person it never saw.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .compiler import dump_built_in_games

GAMES = Path(__file__).resolve().parent / "games"
HUMAN = GAMES / "human"
NEW = GAMES / "new"
HELD_OUT = ["Snake", "Sokoban", "Pinball-lite", "Brick Breaker", "Tower Defense", "Reaction Test", "Open World",
            "Sundown Herd"]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def export_human_games() -> None:
    HUMAN.mkdir(parents=True, exist_ok=True)
    dump_path = HUMAN / "built_in.json"
    dump_built_in_games(dump_path)
    games = json.loads(dump_path.read_text(encoding="utf-8"))
    dump_path.unlink()
    index = []
    for game in games:
        name = slug(game["name"])
        (HUMAN / f"{name}.dsl").write_text(game["source"].strip() + "\n", encoding="utf-8")
        index.append({"slug": name, "name": game["name"], "description": game["description"],
                      "split": "test" if game["name"] in HELD_OUT else "train"})
    (HUMAN / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(index)} games -> {HUMAN} ({sum(item['split'] == 'test' for item in index)} held out)")


def human_games(split: Optional[str] = None) -> List[Dict]:
    index = json.loads((HUMAN / "index.json").read_text(encoding="utf-8"))
    return [{**item, "origin": "human", "family": "human",
             "source": (HUMAN / f"{item['slug']}.dsl").read_text(encoding="utf-8")}
            for item in index if split is None or item["split"] == split]


def new_games() -> List[Dict]:
    games = []
    for index_path in sorted(NEW.glob("*/index.jsonl")):
        family = index_path.parent.name
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            path = index_path.parent / f"{item['slug']}.dsl"
            if path.exists():
                games.append({**item, "origin": "agent", "family": family,
                              "source": path.read_text(encoding="utf-8")})
    return games


if __name__ == "__main__":
    export_human_games()
