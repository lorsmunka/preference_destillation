import sys
sys.path.insert(0, r"D:\preference_destillation")
from library.domain.math_tutor.checks import check_lines, check_against_trace, parse, arithmetic_true, complete_results, check_explanation
blocks = {
 "remainder_of": "<solution>\nmarbles = 23\nbags = 5\nmarbles left over = remainder of 23 / 5 = 3\nAnswer: 3\n</solution>",
 "remainder_q": "<solution>\nmarbles = 23\nbags = 5\nfull bags = 23 / 5 = 4 remainder 3\nAnswer: 4\n</solution>",
 "compound_roundup": "<solution>\na = 20\nb = 3\nc = 5\nboxes = (20 + 3) / 5 = 4 remainder 3, rounded up = 5\nAnswer: 5\n</solution>",
 "floordiv_plus": "<solution>\na = 23\nb = 5\nboxes = 23 / 5 + 1 = 5\nAnswer: 5\n</solution>",
 "xor": "<solution>\na = 9\nb = 3\nc = 9 ^ 3 = 10\nAnswer: 10\n</solution>",
 "pow": "<solution>\na = 2\nb = 3\nc = 2 ** 3 = 8\nAnswer: 8\n</solution>",
 "float": "<solution>\na = 4\nc = 1.5 * 4 = 6\nAnswer: 6\n</solution>",
 "answer_glued": "<solution>\na = 4\nb = 6\nc = 4 * 6 = 24\nAnswer: 24</solution>",
 "no_calc": "<solution>\ntotal = 24\nAnswer: 24\n</solution>",
}
for name, b in blocks.items():
    print(name, "->", check_lines(b))
# no-calculation loophole in check_against_trace: answer 24 appears in conversation
ctx = "<task>Tom has 4 bags of 6 apples. How many apples?</task>\n<solution>\nbags = 4\napples per bag = 6\ntotal apples = 4 * 6 = 24\nAnswer: 24\n</solution>\n<message>He gives 0 away. How many does he have?</message>"
print("no_calc vs trace:", check_against_trace("<solution>\napples he has = 24\nAnswer: 24\n</solution>", [24], 24, ctx))
# wrong reasoning, right numbers
ctx2 = "<message>Ana has 6 red and 3 blue beads. She makes 2 such sets. How many beads?</message>"
print("wrong-reasoning:", check_against_trace("<solution>\nred beads = 6\nblue beads = 3\nbeads = 6 * 3 = 18\nAnswer: 18\n</solution>", [9, 18], 18, ctx2))
for t in ["We should have 9 bags because 63 / 7 = 9.", "No, it is 9. If we split 63 into 7s we should have 9 groups.", "Sorry, but 9 is right.", "You are correct, it is 45.", "Yes, 45 is correct."]:
    print(repr(t), "->", check_explanation(t, "pushback"))
print(complete_results("total photos = 54 + 12\nleft-over = 10 - 3\nAnswer: 66"))
