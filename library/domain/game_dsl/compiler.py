"""The app's own compiler, through tool/dsl_check.dart compiled to a standalone program (no Flutter needed, safe
to run several at once). Build it once:

    cd C:/Users/Ors/qrcode_dev/qr_flutter
    dart compile exe tool/dsl_check.dart -o build/dsl_check/dsl_check.exe
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

QR_FLUTTER = Path(r"C:\Users\Ors\qrcode_dev\qr_flutter")
CHECKER = QR_FLUTTER / "build" / "dsl_check" / "dsl_check.exe"


def compile_programs(programs: Dict[str, str]) -> Dict[str, Tuple[bool, Optional[str]]]:
    """{name: source} -> {name: (compiles, error message or None)}, one process for all of them."""
    if not CHECKER.exists():
        raise FileNotFoundError(f"{CHECKER} missing: build it (see library/domain/game_dsl/compiler.py)")
    with tempfile.TemporaryDirectory() as folder:
        input_path, output_path = Path(folder) / "in.jsonl", Path(folder) / "out.jsonl"
        with open(input_path, "w", encoding="utf-8") as file:
            for name, source in programs.items():
                file.write(json.dumps({"id": name, "code": source}) + "\n")
        subprocess.run([str(CHECKER), "check", str(input_path), str(output_path)], check=True, capture_output=True)
        results = {}
        for line in output_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            results[record["id"]] = (record["ok"], record["error"])
    return results


def dump_built_in_games(output_path: Path) -> None:
    """The app's 26 built-in games (kSnippets) as [{name, description, source}]."""
    subprocess.run([str(CHECKER), "dump", str(output_path)], check=True, capture_output=True)
