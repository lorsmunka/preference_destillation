# Cloud runbook — teacher data on a rented RTX 5090 (plan step 7)

What runs: `python generate_data.py` → code_recall (10k) then math_tutor (all turns), Gemma 4
12B in bf16, 16-32 inputs per generate call. Output: `batches/<domain>/google_gemma-4-12B-it/`.
Only while you are present; stop with Ctrl+C or a `STOP` file (resumes where it stopped).

## 0. Rent
- 1× RTX 5090 (32 GB), CUDA 12.8+ image with PyTorch ≥ 2.7 (Blackwell needs cu128).
- Disk ≥ 80 GB (weights 24 GB + data ~5 GB + env).
- Give Claude SSH access (host, port, user, key) if Claude should drive it; otherwise run the
  commands below yourself.

## 1. Code
Either clone (after `git push` from home — say the word and Claude pushes):
```
git clone https://github.com/lorsmunka/preference_destillation.git && cd preference_destillation
```
or copy a tarball of the committed code (no weights, no batches):
```
# at home:   git archive HEAD -o pd.tar.gz
# on cloud:  mkdir preference_destillation && tar xzf pd.tar.gz -C preference_destillation
```

## 2. Environment
```
pip install -r requirements.txt
huggingface-cli login            # your token; the Gemma 4 licence must be accepted on HF
```
No `models/` folder on the cloud → weights load from the HF id `google/gemma-4-12B-it`.

## 3. Corpora (gitignored, rebuilt deterministically)
```
python -c "from library.domain import MATH_TUTOR, CODE_RECALL; MATH_TUTOR.build_corpus(); CODE_RECALL.build_corpus()"
```

## 4. Check before the long run (~5 min)
- 64 turns of each domain through the real generator path; look at kept / answer_correct
  (scratchpad smoke script, adapted: `generator_smoke.py`).
- Watch `nvidia-smi`: if memory stays under ~26 GB at batch 16, set
  `generation_batch_size=32` in `generate_data.py`.

## 5. Run
```
python generate_data.py          # tmux / screen so a dropped SSH does not kill it
```
code_recall first (small), then math_tutor. Completed jobs are skipped on a re-run.

## 6. Bring the data home, then stop paying
```
# from home:
scp -r <user>@<host>:preference_destillation/batches/code_recall  batches/
scp -r <user>@<host>:preference_destillation/batches/math_tutor   batches/
```
Check file counts and `info.json` on both sides, then destroy the instance.

## Notes
- Teacher precision differs from the earlier Gemma 4 data (Virág: NF4 on the 4070); within
  each domain all data comes from one setup. State it in the thesis.
- Every math_tutor / code_recall example carries `ground_truth` and `answer_correct`.
- The train/test cut for training is set from the batches afterwards (first batch holding
  a test turn), so no test turn lands in a training batch.
