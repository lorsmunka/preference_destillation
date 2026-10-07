import sys, pathlib
sys.path.insert(0, r"D:\preference_destillation\experiments")
import _common
scratch = pathlib.Path(r"C:\Users\Ors\AppData\Local\Temp\claude\D--preference-destillation\ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f\scratchpad\review_tooling\results")
_common.RESULTS_DIR = scratch
scratch.mkdir(exist_ok=True)
_common.write_rows("t.csv", ["run", "a", "b"], [{"run": f"r{i}", "a": i, "b": i} for i in range(5)])
try:
    _common.write_rows("t.csv", ["run", "a"], [{"run": "r9", "a": 9}])   # a column was dropped from the script
except Exception as e:
    print("raised:", type(e).__name__, e)
print(open(scratch / "t.csv").read())
