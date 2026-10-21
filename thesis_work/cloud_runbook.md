# Cloud runbook v2: the two agents' data runs on rented GPU time (2026-10-20)

Direction (user, 2026-10-20): both agents work end to end locally first, then the large runs on
the cloud. What runs there, in order:

1. **Coding agent pass**: Gemma does both steps (scope, then explanation / found) of all 12,960
   requests (20 base projects, 16 of them Python and 4 plain HTML sites: 16 train x 190 versions +
   4 test x 50, 4 requests each):
   `library/domain/code_agent/conversations/v1.jsonl`.
2. **Math tutor full pass**: Gemma tutors every train and test problem (29,747 + 1,250, new
   problems only, no copies): `library/domain/math_tutor/items/conversations/full_v1.jsonl`.
3. The two output vocabularies (from those replies), then the teacher data (Gemma reads its own
   replies back) and the students: `experiments/code_agent_trial.py`, `experiments/math_full.py`.
4. The size sweep: the kd student at four more sizes, ~1M to ~100M parameters (step 6b).

Only while you are present; stop with Ctrl+C or a `STOP` file in the working folder. Every pass
resumes where it stopped (items already in the file are skipped; a chunk is saved every 16 / 32).

## 0. Rent
- 1x RTX 5090 (32 GB) or bigger (A100 / H100 40-80 GB). Full precision (bf16) needs ~26 GB: a pass
  picks bf16 by itself on a card with 30 GB or more, and NF4 below (`teacher_setup`).
- CUDA 12.8+ image, PyTorch >= 2.7 (Blackwell needs cu128), **Python >= 3.12** (the coding agent's
  project versions rename names inside f-strings; 3.12 tokenizes them).
- Disk >= 400 GB: weights 24 GB; teacher data ~270 GB for the math tutor (1.4 MB per reply,
  ~195,000 replies, measured on the slice) and ~35 GB for the coding agent.
- Give Claude SSH access (host, port, user, key) if Claude should drive it; otherwise run the
  commands below yourself.

## 1. Code
Clone (after `git push` from home; say the word and Claude pushes) or copy a tarball:
```
git clone https://github.com/lorsmunka/preference_destillation.git && cd preference_destillation
# or: at home  git archive HEAD -o pd.tar.gz;  on the cloud  mkdir preference_destillation && tar xzf pd.tar.gz -C preference_destillation
```

## 2. Environment
```
pip install -r requirements.txt
huggingface-cli login            # your token; the Gemma 4 licence must be accepted on HF
python -m unittest discover -s tests     # 150 tests, CPU, ~25 s
```

## 3. Derived files (gitignored, rebuilt the same every time, ~10 min)
```
python -m library.domain.math_tutor.prune         # -> items.jsonl (34,002 items)
python -m library.domain.code_agent.requests      # -> items.jsonl of the coding agent (12,960 requests)
```

## 4. Check before the long runs (~15 min)
```
python -m library.domain.code_agent.harness pilot_cloud 16 8   # 16 train + 8 test requests, 8 per model call
```
Look at the printout after the chunk (scopes, places hit, decision right) and at `nvidia-smi`
(raise the batch while memory stays below ~28 GB). Note the seconds per round: that sets the
time of step 5. Then stop it (Ctrl+C); the pilot_cloud file is a trial and is never used again.

## 5. The passes (tmux / screen, so a dropped SSH does not kill them)
```
python -m library.domain.code_agent.harness v1 - 8        # coding agent, 8 requests per model call
python -m library.domain.math_tutor.full conversations 16 # math tutor, 16 replies per model call
```
Measured locally (NF4 on the 12 GB 4070, for scale only): coding agent ~45 s per whole-project
scope and ~20 s per answer one at a time; math tutor ~14 min per 32 conversations at 4 per call.

## 6. Vocabularies, teacher data, students
```
python -m library.domain.code_agent.vocabulary
python -m library.domain.math_tutor.agentic
python experiments/code_agent_trial.py v1     # corpus, teacher data, kd + ce students, report
python experiments/math_full.py               # the same for the math tutor
```
Training the math students locally would take ~16 h each (the slice's 15,301 replies took ~75 min
for 10 passes), so train them on the cloud too.

## 6b. The size sweep (the same two scripts, after kd and ce)
For the question "how does the student improve with its size?" (added 2026-10-20), both scripts then
train the kd student at four more sizes (`experiments/_common.py`, `STUDENT_SIZES`), smallest first,
one seed each. Run names: `math-full-kd-<size>-s1`, `code-agent-v1-kd-<size>-s1`.

| size | layers x width x heads | math tutor | coding agent | learning rate |
|---|---|---|---|---|
| xs | 2 x 64 x 2 | 2.6M | 1.2M | 1.5e-3 |
| s (kd and ce) | 4 x 128 x 4 | 6.0M | 3.1M | 1.5e-3 |
| m | 6 x 256 x 4 | 16M | 10M | 1e-3 |
| l | 8 x 512 x 8 | 53M | 42M | 7e-4 |
| xl | 12 x 640 x 10 | 103M | 89M | 5e-4 |

The coding agent's numbers are from the local trial of 2026-10-21 (20 projects, ~7,900-token vocabularies);
v1's vocabulary is rebuilt from the cloud replies, so they shift a little.

Memory is no limit: PyTorch's fused attention (2026-10-20) trains a 6,000-token coding-agent input
in 0.6 GB (s) to 5.3 GB (xl); math tutor inputs need at most 1.1 GB. Time is the cost. Rough hours
for 10 passes on a 4070-class card (time per example measured on the 4070, plus the slice's
per-example overhead; a 5090 / H100 should be faster on the large sizes):

| | xs | s | m | l | xl | the sweep (xs, m, l, xl) |
|---|---|---|---|---|---|---|
| math tutor (~187,000 train replies) | 10 | 15 | 20 | 24 | 33 | ~90 h |
| coding agent (24,320 train records) | 1.5 | 3.4 | 6 | 17 | 32 | ~55 h |

Every run resumes after a stop. To end the sweep early, delete the remaining sizes from
`STUDENT_SIZES` and rerun: the report then covers the runs that are left.

## 7. Bring home, then stop paying
```
# from home: the conversations (small, also committed), the runs (checkpoints ~70 MB each), the results
scp <user>@<host>:preference_destillation/library/domain/code_agent/conversations/v1.* library/domain/code_agent/conversations/
scp <user>@<host>:preference_destillation/library/domain/math_tutor/items/conversations/full_v1.* library/domain/math_tutor/items/conversations/
scp -r <user>@<host>:preference_destillation/runs/code-agent-v1-* <user>@<host>:preference_destillation/runs/math-full-* runs/
scp <user>@<host>:preference_destillation/experiments/results/*.csv experiments/results/
```
The teacher data (~300 GB) only if more students are to be trained at home. Check file counts and
`info.json` on both sides, then destroy the instance.

## Notes
- One teacher setup per data file: the setup is written next to each conversations file
  (`<name>.setup.json`) and a run on another setup refuses. The teacher data is read back with the
  same setup. The slice's local data (NF4) and the cloud data (bf16) stay apart: state it in the thesis.
- Earlier plan (v1 of this runbook, before 2026-10-13): `generate_data.py` with gold targets; replaced
  by the passes (training only on Gemma's own replies, decided 2026-10-13).
