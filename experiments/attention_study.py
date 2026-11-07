"""Where the students look, next to Gemma (2026-11-05, user: "visualize how KL-s, CE-s, scaling changes how
the model attends to different text ... also visualize Gemma, where it attends. Since we don't distil
attention patterns, it'll be interesting to see if they learn the same or different things: different
seeds, sizes, losses"; "as a baseline we should see the difference between same settings, different
seeds ... then see how settings change the pattern"; Gemma in bf16 on a rented card, never the local
4-bit one).

Nothing in training touches attention: the students learn Gemma's next-word lists (KL) or its written
words (CE) only. So every likeness of attention, between seeds, losses, sizes or to Gemma, is found by
the students themselves. Read on fixed test records ("probes", never trained on):
  math   80 replies of the math tutor (32 solutions, 32 explanations, 16 routing tags)
  code   24 scope replies of the coding agent (6 per test project): the whole project, then the lines

Attention is kept only where a model writes the reply (the query rows that predict a reply token), over
the student's own tokens (math) or the project's lines (code). Gemma reads an instruction header and a
chat template the students never see: that weight goes into one "outside the student's view" column,
measured, then left out (each row renormalised) when Gemma is compared with a student.

    python experiments/attention_study.py            # what is missing: probes, students, then the report
    python experiments/attention_study.py probes     # the test records, with Gemma's prompt aligned to the student's tokens
    python experiments/attention_study.py students   # every student's attention on the probes (local GPU, ~min)
    python experiments/attention_study.py teacher    # Gemma 4 12B, bf16, on the probes: a rented card (>= 40 GB)
    python experiments/attention_study.py report     # metrics, comparisons, the page

Files: runs/attention_study/ (probes.pt, students/<key>.pt, teacher.pt: gitignored, large), the page and
its tables in experiments/results/attention/.
"""

import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from _common import REPO_ROOT, RESULTS_DIR

import torch

TEACHER = "google/gemma-4-12B-it"
TEACHERS = {"math": TEACHER, "code": TEACHER, "sentiment": "google/gemma-3-4b-it"}   # the model that wrote each kind's data
STUDY_DIR = REPO_ROOT / "runs" / "attention_study"
PROBES_PATH = STUDY_DIR / "probes.pt"
STUDENTS_DIR = STUDY_DIR / "students"
TEACHER_PATH = STUDY_DIR / "teacher.pt"
TEACHER_PATHS = {TEACHER: TEACHER_PATH, "google/gemma-3-4b-it": STUDY_DIR / "teacher_gemma3.pt"}
TEACHER_MEMORY_GB = {TEACHER: 40, "google/gemma-3-4b-it": 16}   # bf16 weights plus the attention maps
PAGE_DIR = RESULTS_DIR / "attention"
PROBE_QUOTAS = {"math": {"solve": 32, "explain": 32, "route": 16},
                "code": {"shop": 6, "todo": 6, "spreadsheet": 6, "travel_site": 6},   # code: scope replies per test project
                "sentiment": {"comment": 48}}
PROBE_SEED = 0
DOMAIN_BATCHES = {"math": "math_tutor_full_v1", "code": "code_agent", "sentiment": "reddit_comment_sentiment"}
SENTIMENT_ARMS = {"pure-kl": "pure KL", "pure-ce": "pure CE", "kl-to-ce": "KL to CE (0.9 to 0.1)", "ce-to-kl": "CE to KL (0.1 to 0.9)",
                  "flat-50": "flat 50/50"}
SENTIMENT_DATA = [256, 512, 1024, 2048]


@dataclass
class Student:
    key: str          # file name and label key
    run: Optional[str]  # None = untrained (random weights, built like `like`)
    domain: str       # "math" or "code"
    label: str
    epoch: Optional[int] = None   # checkpoint_epoch_<epoch>.pt; None = the last one saved
    like: Optional[str] = None    # untrained: the run whose settings and vocabularies it copies
    seed: int = 1


def math_students() -> List[Student]:
    students = []
    for arm, label in (("kl", "KL"), ("ce", "CE"), ("kl-to-ce", "KL to CE")):
        for seed in ((1, 2, 3) if arm != "kl-to-ce" else (1, 2)):
            students.append(Student(f"math-{arm}-m-s{seed}", f"math-agent-{arm}-m-s{seed}", "math", f"{label} m s{seed}"))
    for size in ("xs", "s", "m", "l"):
        students.append(Student(f"math-size-{size}", f"math-full-kd-{'' if size == 's' else size + '-'}s1", "math", f"KD 0.9 to 0.1, {size} (10k problems)"))
    students.append(Student("math-size-s-ce", "math-full-ce-s1", "math", "CE, s (10k problems)"))
    students.append(Student("math-size-xl-pilot", "math-full-pilot_nf4-kd-xl-s1", "math", "KD 0.9 to 0.1, xl (pilot data)"))
    for arm in ("kl", "ce"):
        for seed in (1, 2, 3):   # seeds 2-3 added 2026-11-07 (chapter 9 plan #6: n=3 per pass)
            for epoch in range(9):   # pass 10 is the final student above
                students.append(Student(f"math-{arm}-m-s{seed}-pass{epoch + 1}", f"math-agent-{arm}-m-s{seed}", "math",
                                        f"{arm.upper()} m s{seed} after pass {epoch + 1}", epoch=epoch))
    students.append(Student("math-untrained-m", None, "math", "untrained m (random weights)", like="math-agent-kl-m-s1"))
    return students


def code_students() -> List[Student]:
    students = [Student(f"code-size-{size}", f"code-agent-pilot2_nf4-kd-{'' if size == 's' else size + '-'}s1", "code", f"KD 0.9 to 0.1, {size} (pilot data)")
                for size in ("xs", "s", "m", "l", "xl")]
    students.append(Student("code-size-s-ce", "code-agent-pilot2_nf4-ce-s1", "code", "CE, s (pilot data)"))
    students.append(Student("code-kl-m", "code-agent-kl-m-s1", "code", "KL m, full data, after pass 8", epoch=7))
    students.append(Student("code-untrained-m", None, "code", "untrained m (random weights)", like="code-agent-kl-m-s1"))
    return students


def sentiment_students() -> List[Student]:
    """Part 1's sentiment students (Gemma 3 4B teacher, 48 wide, 3 layers, 1 head; they learned only the class
    prior, code review X-C1). E-01: five KL / CE schedules x 10 seeds, 2,048 comments, 3 passes, every seed
    reading the data in the same order (no shuffle), so seeds differ only in the start weights. Idea D: the
    same student on 256 to 2,048 comments, 5 seeds (completed runs only)."""
    from library.training.training_runner import is_completed
    students = [Student(f"sent-{arm}-s{seed}", f"e01-reddit-{arm}-s{seed}", "sentiment", f"{label} s{seed}", seed=seed)
                for arm, label in SENTIMENT_ARMS.items() for seed in range(1, 11)]
    students += [Student(f"sent-data{count}-s{seed}", f"ideaD-data{count:05d}-s{seed}", "sentiment", f"{count} comments s{seed}", seed=seed)
                 for count in SENTIMENT_DATA for seed in range(1, 6) if is_completed(f"ideaD-data{count:05d}-s{seed}")]
    students.append(Student("sent-untrained", None, "sentiment", "untrained (random weights)", like="e01-reddit-pure-kl-s1"))
    return students


def all_students() -> List[Student]:
    return math_students() + code_students() + sentiment_students()


# ── probes ──────────────────────────────────────────────────────────────────────────────────────────

def tokenizer(teacher: str = TEACHER):
    from transformers import AutoTokenizer
    from library.shared.config import teacher_weights_path
    return AutoTokenizer.from_pretrained(teacher_weights_path(teacher))


def sample_test_records(kind: str) -> List[dict]:
    """A seeded sample of test records, PROBE_QUOTAS[kind] per turn type (math) or per test project (code
    scope replies), at most two per batch file so they spread over many conversations."""
    from library.shared.batch_files import batch_paths, read_batch
    from library.shared.config import batches_dir
    quotas = dict(PROBE_QUOTAS[kind])
    paths = batch_paths(batches_dir(DOMAIN_BATCHES[kind], TEACHERS[kind]))
    # the math test records sit in the tail; sentiment has no split labels: its test set is the fixed tail of
    # 1,024 comments (32 files), the default max_test_examples the E-01 and idea D students were tested on
    paths = paths[-700:] if kind == "math" else paths[-32:] if kind == "sentiment" else paths
    random.Random(PROBE_SEED).shuffle(paths)
    chosen = []
    for path in paths:
        taken = 0
        for number, record in enumerate(read_batch(path)):
            if (kind != "sentiment" and record.get("split") != "test") or taken == 2:
                continue
            if kind == "sentiment":
                bucket, item_id, turn_type = "comment", f"{path.stem}#{number}", "classify"
            else:
                bucket = record["turn_type"] if kind == "math" else (record["item_id"].split("~")[0] if record["turn_type"] == "scope" else None)
                item_id, turn_type = record["item_id"], record["turn_type"]
            if quotas.get(bucket, 0) > 0:
                quotas[bucket] -= 1
                taken += 1
                chosen.append({"kind": kind, "bucket": bucket, "item_id": item_id, "turn_type": turn_type,
                               "sentence": record["sentence"], "reply": record["model_response"],
                               "reply_tokens": [step["token"] for step in record["steps"]]})
        if not any(quotas.values()):
            return chosen
    raise SystemExit(f"{kind}: not enough test records for the quotas, still missing {quotas}")


def scope_target_lines(reply: str) -> Dict[str, set]:
    """Gemma's scope reply ("file.py 12-30" per line) -> {file: {line numbers}}."""
    targets: Dict[str, set] = {}
    for line in reply.splitlines():
        match = re.match(r"\s*(\S+\.(?:py|html))\s+(.+)", line)
        if not match:
            continue
        for part in match.group(2).split(","):
            numbers = [int(number) for number in re.findall(r"\d+", part)]
            if numbers:
                targets.setdefault(match.group(1), set()).update(range(numbers[0], numbers[-1] + 1))
    return targets


def describe_lines(sentence: str, targets: Dict[str, set]):
    """Per text line of a code record: its label and region (target line / rest of the target file /
    other files / the message / markup)."""
    labels, regions, current_file = [], [], None
    for text in sentence.split("\n"):
        file_match = re.match(r'<file name="([^"]+)">', text)
        number_match = re.match(r"\s*(\d+):", text)
        if file_match:
            current_file = file_match.group(1)
            labels.append(text), regions.append("markup")
        elif text.startswith("<message>"):
            labels.append(text), regions.append("message")
        elif number_match and current_file is not None:
            number = int(number_match.group(1))
            labels.append(f"{current_file} {number}")
            if number in targets.get(current_file, ()):
                regions.append("target lines")
            else:
                regions.append("target file" if current_file in targets else "other files")
        else:
            labels.append(text[:40]), regions.append("markup")
    return labels, regions


def math_regions(sentence: str) -> List[str]:
    """Per character of a math conversation: which part it belongs to (the pupil's new message, the
    newest task, or the earlier conversation)."""
    regions = ["earlier conversation"] * len(sentence)
    message_start = sentence.rfind("<message>")
    task_start = sentence.rfind("<task>")
    task_end = sentence.find("</task>", task_start) + len("</task>") if task_start >= 0 else -1
    for index in range(max(task_start, 0), task_end if task_start >= 0 else 0):
        regions[index] = "newest task"
    for index in range(max(message_start, 0), len(sentence) if message_start >= 0 else 0):
        regions[index] = "message"
    return regions


def token_category(token: str) -> str:
    text = token.replace("▁", " ").strip()
    if text.isdigit():
        return "number"
    if text and all(character in "<>/" for character in text) or text in ("task", "solution", "question", "explanation", "message", "out_of_scope", "scope"):
        return "markup"
    if not text:
        return "space"
    return "word"


def build_probe(record: dict, gemma_tokenizer) -> dict:
    """The student's and Gemma's token ids of one record, the query rows (the positions that predict a reply
    token) and the column groups both are summed into: math = one group per student token; code = one
    group per text line of the project, then the end of the input, then each reply token. The last
    group is "outside the student's view" (Gemma's instruction header, chat template start, <bos>)."""
    from library.domain import as_domain
    kind, sentence = record["kind"], record["sentence"]
    domain = as_domain(DOMAIN_BATCHES[kind])
    student_text = sentence + "\n\n"
    student_encoding = gemma_tokenizer(student_text, add_special_tokens=False, return_offsets_mapping=True)
    sentence_ids = student_encoding["input_ids"]
    reply_ids_student = [(gemma_tokenizer.encode(token, add_special_tokens=False) or [gemma_tokenizer.unk_token_id])[0]
                         for token in record["reply_tokens"]]
    reply_ids_teacher = gemma_tokenizer.convert_tokens_to_ids(record["reply_tokens"])
    reply_count = len(reply_ids_student)

    if kind == "sentiment":
        # Gemma 3 4B wrote these labels in March 2026 (data_generation/model_handler.py then): the plain prompt,
        # no chat template, the tokenizer's own <bos> in front, bf16
        prompt_text = domain.teacher_prompt(sentence)
        teacher_encoding = gemma_tokenizer(prompt_text, add_special_tokens=True, return_offsets_mapping=True)
    else:
        prompt_text = gemma_tokenizer.apply_chat_template([{"role": "user", "content": domain.teacher_prompt(sentence)}],
                                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
        teacher_encoding = gemma_tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = teacher_encoding["input_ids"]
    if gemma_tokenizer.bos_token_id is not None and prompt_ids[:1] != [gemma_tokenizer.bos_token_id]:
        raise ValueError("the prompt has no <bos>: the offsets below would be off by one")
    conversation_start = prompt_text.find(sentence)
    if conversation_start < 0 or prompt_text.find(sentence, conversation_start + 1) >= 0:
        raise ValueError(f"{record['item_id']}: the conversation is not found exactly once in Gemma's prompt")
    # sentiment: the field list after the comment is instructions the student never sees; only the closing
    # "JSON:" line plays the part of the student's end of input
    end_of_input_start = prompt_text.rfind("JSON:") if kind == "sentiment" else conversation_start + len(sentence)

    # which student token holds each character of the student's text
    token_of_character = [0] * len(student_text)
    for token_index, (start, end) in enumerate(student_encoding["offset_mapping"]):
        for character in range(start, end):
            token_of_character[character] = token_index
    line_of_character = []
    for line_index, line in enumerate(sentence.split("\n")):
        line_of_character.extend([line_index] * (len(line) + 1))
    line_count = len(sentence.split("\n"))

    if kind in ("math", "sentiment"):
        student_positions = len(sentence_ids) + reply_count - 1
        group_count = student_positions + 1
        student_groups = list(range(student_positions))
        def conversation_group(character): return token_of_character[character]
        end_of_input_group = token_of_character[len(sentence)]
        def reply_group(index): return len(sentence_ids) + index
    else:
        group_count = line_count + reply_count + 1
        student_groups = [line_of_character[start] if start < len(sentence) else line_count
                          for start, _ in student_encoding["offset_mapping"]]
        student_groups += [line_count + 1 + index for index in range(reply_count - 1)]
        def conversation_group(character): return line_of_character[character]
        end_of_input_group = line_count
        def reply_group(index): return line_count + 1 + index
    outside_group = group_count - 1

    teacher_groups = []
    conversation_end = conversation_start + len(sentence)
    for start, end in teacher_encoding["offset_mapping"]:
        if start < conversation_end and end > conversation_start and end > start:   # overlaps the student's text
            teacher_groups.append(conversation_group(max(start, conversation_start) - conversation_start))
        elif start >= end_of_input_start and start >= conversation_end:
            teacher_groups.append(end_of_input_group)
        else:
            teacher_groups.append(outside_group)
    teacher_groups += [reply_group(index) for index in range(reply_count - 1)]

    probe = dict(record)
    probe.update({
        "student_ids": sentence_ids + reply_ids_student[:-1],
        "student_queries": [len(sentence_ids) - 1 + index for index in range(reply_count)],
        "student_groups": student_groups,
        "teacher_ids": prompt_ids + reply_ids_teacher[:-1],
        "teacher_queries": [len(prompt_ids) - 1 + index for index in range(reply_count)],
        "teacher_groups": teacher_groups,
        "group_count": group_count,
        "outside_group": outside_group,
        "target_token_ids": reply_ids_student,
    })
    if kind == "math":
        tokens = gemma_tokenizer.convert_ids_to_tokens(probe["student_ids"])
        character_regions = math_regions(sentence)
        probe["group_labels"] = tokens + ["(outside)"]
        probe["group_token_ids"] = probe["student_ids"] + [-1]
        probe["group_regions"] = ([character_regions[start] if start < len(sentence) else "end of input"
                                   for start, _ in student_encoding["offset_mapping"]]
                                  + ["reply so far"] * (reply_count - 1) + ["outside"])
        probe["group_categories"] = [token_category(token) for token in tokens] + ["outside"]
    elif kind == "sentiment":
        tokens = gemma_tokenizer.convert_ids_to_tokens(probe["student_ids"])
        probe["group_labels"] = tokens + ["(outside)"]
        probe["group_token_ids"] = probe["student_ids"] + [-1]
        probe["group_regions"] = (["comment" if start < len(sentence) else "end of input" for start, _ in student_encoding["offset_mapping"]]
                                  + ["reply so far"] * (reply_count - 1) + ["outside"])
        probe["group_categories"] = [token_category(token) for token in tokens] + ["outside"]
    else:
        labels, regions = describe_lines(sentence, scope_target_lines(record["reply"]))
        reply_tokens = record["reply_tokens"][:-1]
        probe["group_labels"] = labels + ["(end of input)"] + reply_tokens + ["(outside)"]
        probe["group_regions"] = regions + ["end of input"] + ["reply so far"] * len(reply_tokens) + ["outside"]
        probe["group_token_ids"] = [-1] * group_count
    check_probe(probe)
    return probe


def check_probe(probe: dict) -> None:
    assert len(probe["student_groups"]) == len(probe["student_ids"]), "a student position without a group"
    assert len(probe["teacher_groups"]) == len(probe["teacher_ids"]), "a teacher position without a group"
    assert max(probe["student_groups"]) < probe["outside_group"] and max(probe["teacher_groups"]) <= probe["outside_group"]
    assert len(probe["group_labels"]) == len(probe["group_regions"]) == probe["group_count"]
    assert probe["student_queries"][-1] == len(probe["student_ids"]) - 1
    assert probe["teacher_queries"][-1] == len(probe["teacher_ids"]) - 1
    # the conversation's tokens should mostly coincide one to one (same tokenizer, same text)
    if probe["kind"] in ("math", "sentiment"):
        conversation = [group for group in probe["teacher_groups"] if group != probe["outside_group"]]
        assert len(set(conversation)) >= 0.95 * len(probe["student_ids"]), "Gemma's tokens cover too few student tokens"


def make_probes() -> List[dict]:
    """The probes, built once; a kind added later is appended, so earlier probes keep their index (the
    students' and Gemma's saved rows are keyed by it)."""
    probes = torch.load(PROBES_PATH, weights_only=False) if PROBES_PATH.exists() else []
    missing = [kind for kind in PROBE_QUOTAS if not any(probe["kind"] == kind for probe in probes)]
    if not missing:
        return probes
    for kind in missing:
        kind_tokenizer = tokenizer(TEACHERS[kind])
        probes += [build_probe(record, kind_tokenizer) for record in sample_test_records(kind)]
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(probes, PROBES_PATH)
    print(f"{len(probes)} probes ({', '.join(missing)} added) -> {PROBES_PATH}")
    return probes


# ── students ────────────────────────────────────────────────────────────────────────────────────────

def load_student(student: Student, device: str):
    from library import RunStore
    from library.model.custom_student import build_student
    from library.model.student import StudentModel
    from library.domain import as_domain
    store = RunStore()
    if student.run is None:
        run = store.get(student.like)
        torch.manual_seed(student.seed)
        model = build_student(run.info, as_domain(run.info["domain"]), StudentModel._input_vocabulary(run)).to(device)
        return model.eval()
    run = store.get(student.run)
    checkpoints = sorted((Path(run.run_dir) / "checkpoints").glob("checkpoint_epoch_*.pt"),
                         key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    path = checkpoints[-1] if student.epoch is None else Path(run.run_dir) / "checkpoints" / f"checkpoint_epoch_{student.epoch}.pt"
    return StudentModel.from_run(run, path, device=device).model


def student_rows(student: Student, probes: List[dict], device: str) -> Dict[int, torch.Tensor]:
    from library.tooling.attention import student_attention
    model = load_student(student, device)
    rows = {}
    for index, probe in enumerate(probes):
        if probe["kind"] != student.domain:
            continue
        input_ids = torch.tensor([model.remap_input_tokens(probe["student_ids"])], device=device)
        _, grouped = student_attention(model, input_ids, probe["student_queries"], probe["student_groups"], probe["group_count"])
        rows[index] = grouped.half().cpu()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def run_students(probes: List[dict]) -> None:
    STUDENTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for student in all_students():
        path = STUDENTS_DIR / f"{student.key}.pt"
        if path.exists():
            continue
        started = time.time()
        rows = student_rows(student, probes, device)
        torch.save(rows, path)
        print(f"{student.key}: {len(rows)} probes, {time.time() - started:.1f} s")


# ── teacher (a rented card: Gemma 4 12B in bf16 needs ~24 GB plus the attention maps) ─────────────────

def teacher_rows(teacher: str) -> Dict[int, torch.Tensor]:
    path = TEACHER_PATHS[teacher]
    return torch.load(path, mmap=True, weights_only=False) if path.exists() else {}


def run_teacher(probes: List[dict], teacher: str) -> None:
    """One teacher's attention on the probes of the kinds it wrote, in bf16 (never a 4-bit copy), on a card
    big enough for it (TEACHER_MEMORY_GB); resumes where its file stops."""
    from transformers import AutoModelForCausalLM
    from library.shared.config import teacher_weights_path
    from library.tooling.attention import TeacherAttention
    needed = TEACHER_MEMORY_GB[teacher]
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < needed * 2**30:
        raise SystemExit(f"{teacher} needs bf16 on a card with >= {needed} GB (never a 4-bit copy)")
    path = TEACHER_PATHS[teacher]
    rows = dict(torch.load(path, weights_only=False)) if path.exists() else {}
    todo = [index for index, probe in enumerate(probes) if TEACHERS[probe["kind"]] == teacher and index not in rows]
    if not todo:
        print(f"{teacher}: every probe done")
        return
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(teacher_weights_path(teacher), dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager").eval()
    print(f"{teacher} loaded in {time.time() - started:.0f} s on {torch.cuda.get_device_name(0)}")
    recorder = TeacherAttention(model)
    print(f"{len(recorder.layers)} attention layers")
    for count, index in enumerate(todo, start=1):
        probe = probes[index]
        probe_started = time.time()
        grouped = recorder.rows(probe["teacher_ids"], probe["teacher_queries"], probe["teacher_groups"], probe["group_count"])
        total = grouped.sum(-1)
        if not torch.allclose(total, torch.ones_like(total), atol=2e-2):
            raise SystemExit(f"probe {index}: the rows add up to {float(total.min()):.3f}-{float(total.max()):.3f}, not 1")
        rows[index] = grouped.half()
        print(f"probe {index} ({probe['kind']}, {len(probe['teacher_ids'])} tokens): {time.time() - probe_started:.1f} s")
        if count % 20 == 0:
            torch.save(rows, path)
    torch.save(rows, path)
    print(f"{teacher}: rows for {len(rows)} probes -> {path} ({time.time() - started:.0f} s)")


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else None
    probes = make_probes()
    if step in (None, "students"):
        run_students(probes)
    if step == "teacher":   # optional second argument: one teacher, e.g. google/gemma-3-4b-it (default: each in turn)
        for teacher in ([sys.argv[2]] if len(sys.argv) > 2 else sorted(set(TEACHERS.values()))):
            run_teacher(probes, teacher)
    if step in (None, "report"):
        from attention_report import report
        report(probes)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
