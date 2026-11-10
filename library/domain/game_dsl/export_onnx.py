"""The game agent for the app (2026-11-08, the user: "ship it as a lazily loaded asset", built on-device by the
qr_flutter session): the trained student as one ONNX graph with a key/value cache, its vocabularies, Gemma 4's
tokenizer, the license, a manifest, and reference cases the app's Dart port is checked against.

  qr_flutter/web/game_agent/manifest.json        {"latest", "versions": {"v1": {"files": [name, size, sha256]}}}
  qr_flutter/web/game_agent/v1/model.onnx        inputs  input_ids int32 [1, n] (compact input ids),
                                                         past_key_values.{i}.key / .value float32 [1, 4, past, 64]
                                                 outputs logits float32 [1, 5704] (the last position only),
                                                         present.{i}.key / .value (past + n positions)
                                                 past = 0 is the prompt's pass; positions (RoPE) come from past
  qr_flutter/web/game_agent/v1/tokenizer.json    Gemma 4's own (the app's Dart BPE matches it exactly)
  qr_flutter/web/game_agent/v1/config.json       prompt delimiter, stop rule, sampling, vocabularies, sizes
  qr_flutter/web/game_agent/v1/LICENSE, NOTICE   Apache 2.0 (Gemma 4's license) and the attribution
  qr_flutter/test/assistant/reference_cases.json the live inputs, ids, greedy outputs from the exported graph

    python -m library.domain.game_dsl.export_onnx [run name]      (default: game-agent-kd-m-s1)

v2 (2026-11-10, the user's QR-Flower models, thesis table 9.9): the same package per model under its own id, with
its display name and parameter count in the manifest, its reference cases in reference_cases_<id>.json; the
manifest's "latest" stays as it was (the app's model picker chooses):

    python -m library.domain.game_dsl.export_onnx qr-flower-mini-1 | qr-flower-1 | qr-flower-larger-1
"""

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import GameDslDomain, GameDslV2Domain

QR_FLUTTER = Path(r"C:\Users\Ors\qrcode_dev\qr_flutter")
PACKAGE_DIR = QR_FLUTTER / "web" / "game_agent"
REFERENCE_CASES = QR_FLUTTER / "test" / "assistant" / "reference_cases.json"
VERSION = "v1"
# the v2 models in the app (the user, 2026-11-10): package id -> (run, display name)
V2_MODELS = {"qr-flower-mini-1": ("game-agent-v2-s5-s1", "QR-Flower-Mini-1"),
             "qr-flower-1": ("game-agent-v2-s15-s1", "QR-Flower-1"),
             "qr-flower-larger-1": ("game-agent-v2-s30-s1", "QR-Flower-Larger-1")}
TOKENIZER = Path(__file__).resolve().parents[3] / "models" / "gemma-4-12B-it" / "tokenizer.json"
OPSET = 17
NOTICE_TEXT = """QR Game Studio game agent ({version})

A small model ({parameters} parameters) distilled from Google's Gemma 4 12B (google/gemma-4-12B-it): it was trained on
Gemma's own answers and probabilities for game-language commands. Gemma 4 is provided under the Apache License 2.0
(https://ai.google.dev/gemma/docs/gemma_4_license); a copy of the license is in LICENSE. tokenizer.json is Gemma 4's
tokenizer, unchanged.
"""


class StepModel(nn.Module):
    """One decoding step of the student over n new tokens, given the cached keys / values of the earlier ones."""

    def __init__(self, transformer):
        super().__init__()
        self.model = transformer
        self.layers = len(transformer.transformer_layers)
        attention = transformer.transformer_layers[0].attention
        self.heads, self.head_dim = attention.num_heads, attention.head_dim

    def forward(self, input_ids, *past):
        model = self.model
        n = input_ids.shape[1]
        past_length = past[0].shape[2]
        x = model.input_embedding(input_ids.long())
        positions = torch.arange(n, dtype=torch.float32) + past_length
        frequencies = positions[:, None] * model.rotary_embedding.inv_freq[None, :]
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cos, sin = embedding.cos()[None, None], embedding.sin()[None, None]
        query_positions = torch.arange(n) + past_length
        key_positions = torch.arange(past_length + n)
        mask = (key_positions[None, :] > query_positions[:, None]).float() * -1e9
        presents = []
        for index, layer in enumerate(model.transformer_layers):
            attention = layer.attention
            normed = layer.norm1(x)
            shape = (1, n, self.heads, self.head_dim)
            query = attention.query(normed).view(shape).transpose(1, 2)
            key = attention.key(normed).view(shape).transpose(1, 2)
            value = attention.value(normed).view(shape).transpose(1, 2)
            query = query * cos + rotate_half(query) * sin
            key = key * cos + rotate_half(key) * sin
            key = torch.cat((past[2 * index], key), dim=2)
            value = torch.cat((past[2 * index + 1], value), dim=2)
            presents += [key, value]
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim) + mask
            attended = torch.matmul(torch.softmax(scores, dim=-1), value)
            x = x + attention.output(attended.transpose(1, 2).reshape(1, n, self.heads * self.head_dim))
            x = x + layer.feed_forward(layer.norm2(x))
        logits = model.output_projection(model.layer_norm(x))[:, -1, :]
        return (logits, *presents)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def names(layers):
    past = [f"past_key_values.{i}.{part}" for i in range(layers) for part in ("key", "value")]
    present = [f"present.{i}.{part}" for i in range(layers) for part in ("key", "value")]
    return past, present


def export(student, path):
    step = StepModel(student.model).eval()
    past_names, present_names = names(step.layers)
    dummy_ids = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    dummy_past = [torch.zeros(1, step.heads, 2, step.head_dim) for _ in past_names]
    dynamic = {"input_ids": {1: "n"}}
    dynamic.update({name: {2: "past"} for name in past_names})
    dynamic.update({name: {2: "total"} for name in present_names})
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(step, (dummy_ids, *dummy_past), str(path), input_names=["input_ids", *past_names],
                          output_names=["logits", *present_names], dynamic_axes=dynamic, opset_version=OPSET,
                          do_constant_folding=True, dynamo=False)
    return step


class OnnxGenerator:
    """Greedy decoding with the exported graph (the app's loop, in Python), for the reference cases."""

    def __init__(self, path, student, domain):
        import onnxruntime
        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.student, self.domain = student, domain
        transformer = student.model
        self.layers = len(transformer.transformer_layers)
        attention = transformer.transformer_layers[0].attention
        self.heads, self.head_dim = attention.num_heads, attention.head_dim
        self.past_names, self.present_names = names(self.layers)

    def run(self, ids, past):
        feed = {"input_ids": np.array([ids], dtype=np.int32)}
        feed.update(dict(zip(self.past_names, past)))
        outputs = self.session.run(None, feed)
        return outputs[0][0], outputs[1:]

    def generate(self, text, max_new_tokens=None):
        student = self.student
        prompt_ids = student.tokenizer.encode(self.domain.student_prompt(text), add_special_tokens=False)
        input_ids = student.model.remap_input_tokens(prompt_ids)
        past = [np.zeros((1, self.heads, 0, self.head_dim), dtype=np.float32) for _ in self.past_names]
        logits, past = self.run(input_ids, past)
        indices, token_ids, first_logits, text_so_far = [], [], [], ""
        for step in range(max_new_tokens or self.domain.max_steps):
            if step < 3:
                top = np.argsort(-logits)[:5]
                first_logits.append([[int(index), float(logits[index])] for index in top])
            index = int(np.argmax(logits))
            token_id = student.output_token_ids[index]
            indices.append(index)
            token_ids.append(token_id)
            text_so_far += student.tokenizer.decode([token_id])
            if self.domain.is_stop("", text_so_far):
                break
            logits, past = self.run(student.model.remap_input_tokens([token_id]), past)
        return {"prompt_ids": prompt_ids, "input_ids": input_ids, "greedy_output_indices": indices,
                "greedy_text": student.tokenizer.decode(token_ids, skip_special_tokens=True),
                "first_logits_top5": first_logits}


def reference_problems(count=44):
    """Test problems spread over the kinds, as (source, offset, command, indent, kind) the way the app sends them."""
    from .gemma_writes import load_versions, version_of
    from .problems import OUTPUT
    rows = [json.loads(line) for line in open(GameDslDomain("card").corpus_path, encoding="utf-8")]
    tests = [row for row in rows if row["split"] == "test"]
    by_kind = {}
    for row in tests:
        by_kind.setdefault(row["turn_type"], []).append(row)
    picked = []
    while len(picked) < count and any(by_kind.values()):
        for kind in sorted(by_kind):
            if by_kind[kind] and len(picked) < count:
                picked.append(by_kind[kind].pop(len(by_kind[kind]) // 2))
    wanted = {row["item_id"] for row in picked}
    problems = {}
    for line in open(OUTPUT / "problems.jsonl", encoding="utf-8"):
        problem = json.loads(line)
        if problem["id"] in wanted:
            problems[problem["id"]] = problem
    sources = load_versions({version_of(item) for item in wanted})
    cases = []
    for row in picked:
        problem = problems[row["item_id"]]
        source = sources[version_of(problem["id"])]
        command = row["text"].splitlines()[-1][len("COMMAND "):]
        start, end = problem["cut"]
        if problem["task"] == "insert":
            rest = source[start:]
            cases.append({"source": source[:start] + source[end:], "offset": start, "command": command,
                          "indent": rest[:len(rest) - len(rest.lstrip(" \t"))], "kind": row["turn_type"]})
        else:
            game = source[:start] + problem["before"] + source[end:]
            cases.append({"source": game, "offset": start + 1, "command": command, "indent": "",
                          "kind": row["turn_type"]})
    return cases


EDGE_CASES = [   # digits, runs of spaces, tabs, newlines, accents, emoji, an unseen word
    ("Window {\n    size(8, 10)\n    background(\"0f172a\")\n\tEntity { id(\"player\")  startPosition(350,   100) "
     "color(\"22c55e\") Type { text(\"🐸 béka\", 40) } Control { tapMove() } }\n\n\n}\n",
     "/insert a xylophone-shaped bar #score2 that starts at 0 and goes up to 12345"),
    ("Window {\n    size(8, 10)\n    Entity { id(\"érme\") startPosition(200, 600) color(\"facc15\") Type { circle(20) } }\n}\n",
     "/insert egy piros pénz ami 1 pontot ad 💰 when the player touches it"),
    ("Window {\n  size(7, 10)\n  Entity { id(\"rock\") startPosition(10, 900) color(\"78716c\") Type { circle(30) } }\n}\n",
     "/edit make #rock fall at speed 2.5"),
]


def main(argument="game-agent-kd-m-s1"):
    """A v2 package id (V2_MODELS) or a v1 run name."""
    if argument in V2_MODELS:
        run_name, display_name = V2_MODELS[argument]
        package(run_name, GameDslV2Domain(), argument, display_name,
                QR_FLUTTER / "test" / "assistant" / f"reference_cases_{argument}.json", set_latest=False)
    else:
        package(argument, GameDslDomain("card"), VERSION, None, REFERENCE_CASES, set_latest=True)


def package(run_name, domain, version, display_name, reference_cases_path, set_latest):
    from library import RunStore
    from library.model.student import StudentModel
    from .live import edit_input, insert_input
    store = RunStore()
    _, checkpoint_path = store.resolve_checkpoint(run_name, "latest")
    student = StudentModel.from_run(store.get(run_name), checkpoint_path, domain=domain, device="cpu")
    parameter_count = sum(parameter.numel() for parameter in student.model.parameters())
    version_dir = PACKAGE_DIR / version
    model_path = version_dir / "model.onnx"
    export(student, model_path)
    generator = OnnxGenerator(model_path, student, domain)

    cases = reference_problems()
    for source, command in EDGE_CASES:
        offset = source.rindex("\n}") + 1 if command.startswith("/insert") else source.index("id(") + 1
        cases.append({"source": source, "offset": offset, "command": command,
                      "indent": "    " if command.startswith("/insert") else "", "kind": "edge"})
    same = 0
    for case in cases:
        if case["command"].startswith("/edit"):
            case["input_text"] = edit_input(case["source"], case["offset"], case["command"])[0]
        else:
            case["input_text"] = insert_input(case["source"], case["offset"], case["command"], case["indent"])[0]
        case.update(generator.generate(case["input_text"]))
        torch_text = student.generate(case["input_text"], max_new_tokens=domain.max_steps).text
        case["torch_matches"] = torch_text == case["greedy_text"]
        same += case["torch_matches"]
    print(f"exported {model_path} ({model_path.stat().st_size / 1e6:.1f} MB); ONNX greedy = PyTorch greedy on "
          f"{same}/{len(cases)} reference cases")

    transformer = student.model
    attention = transformer.transformer_layers[0].attention
    config = {"version": version, "run": run_name, "prompt_delimiter": domain.student_prompt(""), "stop": "```",
              "stop_count": 2, "max_new_tokens": domain.max_steps, "attempts": 3, "temperature": 0.7,
              "input_vocabulary": {str(gemma_id): compact for gemma_id, compact in
                                   transformer.input_token_mapping.items()},
              "unknown_input_id": transformer.unknown_input_id,
              "output_vocabulary": list(student.output_token_ids),
              "layers": len(transformer.transformer_layers), "heads": attention.num_heads,
              "head_dim": attention.head_dim}
    (version_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    shutil.copyfile(TOKENIZER, version_dir / "tokenizer.json")
    license_text = Path(__file__).resolve().parents[3] / "LICENSE-APACHE-2.0.txt"
    if license_text.exists():
        shutil.copyfile(license_text, version_dir / "LICENSE")
    (version_dir / "NOTICE").write_text(NOTICE_TEXT.format(version=display_name or version,
                                                           parameters=f"{parameter_count / 1e6:.2f}M"), encoding="utf-8")

    files = []
    for path in sorted(version_dir.iterdir()):
        files.append({"name": path.name, "size": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest_path = PACKAGE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"versions": {}}
    if set_latest:
        manifest["latest"] = version
    entry = {"files": files}
    if display_name:
        entry.update({"name": display_name, "parameters": parameter_count})
    manifest["versions"][version] = entry
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    reference_cases_path.parent.mkdir(parents=True, exist_ok=True)
    reference_cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    print(f"{len(files)} files in {version_dir} ({sum(file['size'] for file in files) / 1e6:.1f} MB); "
          f"{len(cases)} reference cases -> {reference_cases_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "game-agent-kd-m-s1")

