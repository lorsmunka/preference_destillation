# Top-k Accuracy Analysis

Interactive analysis tool for comparing student checkpoints against the teacher
logits stored in each run's designated corpus.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe top-k-accruacy-analsy\main.py
```

The script writes and reuses `config.json`. You can edit that file directly, or
let the interactive prompts overwrite it. To run exactly what is already in the
config without prompts:

```powershell
.\.venv\Scripts\python.exe top-k-accruacy-analsy\main.py --from-config
```

Metrics:

- `top_k_accuracy`: how often the actual teacher-generated target token appears
  in the student's top-k predictions.
- `teacher_student_top_k_overlap`: average fraction of the teacher top-k token
  set that also appears in the student top-k token set.
- `mean_target_rank`: average 1-based rank of the actual target token under the
  student's full output distribution. Lower is better.

Outputs are overwritten on each run in `outputs/`:

- `top_k_results.json`
- `top_k_results.csv`
- `top_k_accuracy.png`
- `teacher_student_top_k_overlap.png`
- `mean_target_rank.png`
