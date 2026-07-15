import re
from typing import Optional

from library.shared.metrics.task import MathMetric
from ..base import Domain


# Two worked examples shown to the teacher before the real problem, so it fills in the
# scaffold (A=, B=, ...) immediately instead of writing prose.
_MASTER_PROMPT = """Follow this example exactly:

Problem: "Lisa has 3 bags with 5 apples each. She eats 2 apples. How many are left?"
A=?
B=?
C=?
D=A*B=?
E=D-C=?
Solution: ?;

Calculations:
A=3
B=5
C=2
D=A*B=15
E=D-C=13
Solution: 13;

Problem: "Tom has 15 stickers. He gives away 7. Sarah has 12 stickers. Does Tom have more stickers than Sarah?"
A=?
B=?
C=?
D=A-B=?
E=D>C=?
Solution: ?;

Calculations:
A=15
B=7
C=12
D=A-B=8
E=D>C=False
Solution: False;

"""


class MathWordProblemDomain(Domain):
    name = "math_word_problem"
    stop_token = ";"
    max_steps = 350
    max_input_tokens = 500
    max_seq_length = 210  # tuned: well below input+steps; problems rarely fill the scaffold

    # Populated from an empirical test (200 examples, 0 failures): 31 unique tokens
    # observed; F, G added for robustness (+2 beyond the max scaffold variable E).
    example_responses = [
        "100\nB=50\nC=A+B=150\nSolution: 150;",
        "87\nB=43\nC=A-B=44\nSolution: 44;",
        "6\nB=9\nC=A*B=54\nSolution: 54;",
        "900\nB=300\nC=A/B=3\nSolution: 3;",
        "4\nB=7\nC=10\nD=A*B=28\nE=D+C=38\nSolution: 38;",
        "3\nB=5\nC=2\nD=A*B=15\nE=D-C=13\nSolution: 13;",
        "93\nB=67\nC=63\nD=A+B=160\nE=D*C=10080\nSolution: 10080;",
        "20\nB=8\nC=5\nD=A-B=12\nE=D*C=60\nSolution: 60;",
        "933\nB=844\nC=208\nD=A+B=1777\nE=D-C=1569\nSolution: 1569;",
        "939\nB=313\nC=110\nD=A/B=3\nE=D+C=113\nSolution: 113;",
        "50\nB=30\nC=A>B=True\nSolution: True;",
        "20\nB=70\nC=A>B=False\nSolution: False;",
        "15\nB=7\nC=12\nD=A-B=8\nE=D>C=False\nSolution: False;",
        "90\nB=10\nC=30\nD=A-B=80\nE=D>C=True\nSolution: True;",
        "8\nB=4\nC=50\nD=A*B=32\nE=D>C=False\nSolution: False;",
        "7\nB=9\nC=40\nD=A*B=63\nE=D>C=True\nSolution: True;",
        "30\nB=50\nC=A>B=False\nSolution: False;",
        "80\nB=20\nC=A>B=True\nSolution: True;",
    ]

    prompt_tokens = [
        "Follow", "this", "example", "exactly",
        "Problem", "Solution", "Calculations",
        "True", "False", "▁True", "▁False",
    ]

    extra_auxiliary_tokens = [
        "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", ".",
    ]

    def teacher_prompt(self, text: str) -> str:
        # text includes the full scaffold (Problem: "..."\nA=?\n...\nSolution: ?;).
        # Append "Calculations:\nA=" so the teacher starts filling in values immediately.
        return _MASTER_PROMPT + text + "\n\nCalculations:\nA="

    def task_metric(self) -> MathMetric:
        return MathMetric()

    def build_corpus(self, count: Optional[int] = None) -> None:
        from .corpus import build_corpus
        build_corpus(self.corpus_path, count=count)

    def structural_validity(self, output: str) -> bool:
        return bool(re.search(r"Solution:\s*[^;\n]+", output))
