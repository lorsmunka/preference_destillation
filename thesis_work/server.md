# Server — access & management

> Skeleton — fill in the specifics. Keep only what's needed to start/resume a run and find its
> artifacts. (Placeholders marked `<...>`.)

## Access
- Connect: `<ssh user@host>`
- Auth / VPN: `<key path, jump host, VPN, ...>`
- Repo location on server: `<path>`

## Environment
- GPU(s): `<model, VRAM, count>`
- Python env: `.venv` — `source .venv/bin/activate`
- Install package (editable): `pip install -e .`
- Teacher model access: Gemma is gated — `<HF token / cache location>`

## Running (Python entry scripts — no queue JSON)
- Generate distillation data: `python generate_data.py`   (edit the `JOBS` list)
- Train: `python train.py`   (edit the `RUNS` list)
- Analyze / evaluate: `python analyze.py`   (or `from library import Analysis, RunStore`)
- Graceful stop: press `%` (or Ctrl+C in headless mode) — saves a temp checkpoint; rerunning resumes.

## Artifacts & housekeeping
- Runs: `runs/<run_name>/` → `info.json`, `logs/` (JSONL + progress png), `checkpoints/`
- Distillation data: `batches/<domain>/<teacher>/`
- Disk / cleanup policy: `<where big batches+checkpoints live, what to prune, backups>`
- Long-run monitoring: `<tmux/screen session, nohup, log tail command>`
