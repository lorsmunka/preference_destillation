"""The game-language agent: parser, the checks the app's compiler does not make, versions and problems. CPU only;
the tests that run the app's compiler are skipped when its standalone build (compiler.CHECKER) is missing.

    python -m unittest discover -s tests -p "test_game_dsl.py"
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.domain.game_dsl.compiler import CHECKER, compile_programs
from library.domain.game_dsl.games import HUMAN, human_games
from library.domain.game_dsl.language import defined_ids, hollow, parse_fragment, parse_program, problems, walk
from library.domain.game_dsl.manual import load_reference
from library.domain.game_dsl.problems import insert_candidates, make_edit, make_insert, student_input
from library.domain.game_dsl.versions import game_names, make_version

BALL = ('Window {\n size(8, 8)\n Entity { id("ball") startPosition(400, 400) color("4ade80") Type { circle(20) }\n'
        '  Movement { speed(4) AutoMove { direction(1, 1) } }\n'
        '  Events { OnCollision { Actions { bounceBack() } } } }\n}')
GAME = ('Window {\n    size(8, 12)\n    lives(3, "OUT")\n'
        '    Entity { id("hero") startPosition(400, 60) color("38bdf8") Type { rectangle(60, 30) }\n'
        '        Movement { speed(6) Control { movementControl(2) } } }\n'
        '    Entity { id("rock") template(true) color("ef4444") Type { circle(20) }\n'
        '        Movement { speed(3) AutoMove { direction(0, -1) } }\n'
        '        Events {\n            OnCollision("hero") { Actions { sound("hurt") updateBar("lives", -1) '
        'deleteEntity() } }\n            OnCollision { Actions { deleteEntity() } }\n        } }\n'
        '    Entity { id("dropper") startPosition(400, 1100) color("00000000") Type { circle(1) }\n'
        '        Events { OnElapsedTime(1) { repeatCount(-1) Actions { spawn("rock", random(40, 760), 1150) } } } }\n'
        '}\n')
HAVE_HUMAN = (HUMAN / "index.json").exists()


class ParserTest(unittest.TestCase):
    def test_spans_cover_the_block(self):
        root = parse_program(BALL)
        entity = root.children[1]
        self.assertTrue(BALL[entity.start:entity.end].startswith("Entity {"))
        self.assertTrue(BALL[entity.start:entity.end].endswith("} }"))
        speed = next(node for node, _ in walk(root) if node.key == "speed")
        self.assertEqual(BALL[speed.arg_spans[0][0]:speed.arg_spans[0][1]], "4")

    def test_random_and_comments(self):
        nodes = parse_fragment('spawn("a", random(1, 5), 3)  # a comment with { braces\n')
        self.assertEqual(nodes[0].args[1], ("random", ["1", "5"]))
        start, end = nodes[0].arg_spans[1]
        self.assertEqual('spawn("a", random(1, 5), 3)'[start:end], "random(1, 5)")

    def test_node_in_parentheses_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_program('Window { Entity { Type(circle(20)) } }')


@unittest.skipUnless(HAVE_HUMAN, "games/human not exported")
class ChecksTest(unittest.TestCase):
    def setUp(self):
        self.reference = load_reference()

    def found(self, source):
        root = parse_program(source)
        return problems([root], None, self.reference, defined_ids(root))

    def test_what_the_compiler_lets_through(self):
        self.assertEqual(self.found(BALL), {})
        self.assertIn("placement", self.found(BALL.replace("Actions { bounceBack() }", "bounceBack()")))
        self.assertIn("arity", self.found(BALL.replace("circle(20)", "circle()")))
        self.assertIn("color", self.found(BALL.replace('"4ade80"', '"notacolor"')))
        self.assertIn("reference", self.found(BALL.replace("AutoMove { direction(1, 1) }", 'chase("nobody")')))

    def test_hollow(self):
        self.assertEqual(hollow(parse_program(GAME)), [])
        self.assertTrue(any("never spawned" in item for item in
                            hollow(parse_program(GAME.replace('spawn("rock"', 'spawn("other"')))))
        self.assertTrue(any("without an id" in item for item in hollow(parse_program(
            GAME.replace('lives(3, "OUT")', 'Bar("b") { OnValue(2) { Actions { burst(5, "fff") } } }\n'
                                            '    winAt("b", 9)\n    lives(3, "OUT")')))))
        randomly_nothing = GAME.replace('sound("hurt")', 'randomAction { Actions { } Actions { sound("hurt") } }')
        self.assertEqual(hollow(parse_program(randomly_nothing)), [])


@unittest.skipUnless(HAVE_HUMAN, "games/human not exported")
class VersionsAndProblemsTest(unittest.TestCase):
    def test_names_and_colours_replaced_consistently(self):
        version = make_version(GAME, 3)
        root = parse_program(version)
        self.assertEqual(game_names(root) & {"rock", "dropper"}, set())
        self.assertIn('sound("hurt")', version)                 # sounds are never renamed
        self.assertIn('lives(3, "OUT")', version)               # texts and fixed names stay
        rock = sorted(game_names(parse_program(GAME)))
        self.assertEqual(len(game_names(root)), len(rock))

    def test_insert_and_edit_problems(self):
        root = parse_program(GAME)
        random_generator = random.Random(0)
        kinds = {candidate[0] for candidate in insert_candidates(root)}
        self.assertTrue({"entity", "event", "actions", "window"} <= kinds)
        entity_cut = next(candidate for candidate in insert_candidates(root) if candidate[0] == "entity"
                          and candidate[1][-1].children[candidate[2]].child("id").first_string() == "rock")
        problem = make_insert(GAME, root, entity_cut, "t|i0")
        self.assertTrue(problem["target"].startswith('Entity { id("rock")'))
        self.assertEqual(problem["new_names"], ["rock"])
        self.assertIn("hero", problem["named_blocks"])
        self.assertIn("COMMAND /insert", student_input(problem, "/insert falling #rock"))
        node = parse_program(GAME).children[2]
        edit = make_edit(GAME, root, node, [root], "t|e0", random_generator, ["coin", "hurt"])
        self.assertIsNotNone(edit)
        self.assertNotEqual(edit["before"], edit["target"])


@unittest.skipUnless(CHECKER.exists() and HAVE_HUMAN, "the app's compiler is not built")
class CompilerTest(unittest.TestCase):
    def test_every_human_game_and_version_compiles(self):
        sources = {}
        for game in human_games():
            sources[game["slug"]] = game["source"]
            sources[game["slug"] + "|v"] = make_version(game["source"], 1)
        results = compile_programs(sources)
        self.assertEqual([name for name, (ok, _) in results.items() if not ok], [])


if __name__ == "__main__":
    unittest.main()
