"""Live training dashboard — one local browser tab, near-zero hardware cost.

    python progress.py            # http://localhost:8765  (opens the tab)
    python progress.py 9000       # another port

Reads only what training already writes: the newest in-progress run's info.json, its
logs/state.json and the tail of logs/training.jsonl, the other runs of the same experiment
(cached, re-read only when a run finishes), the tail of the newest console log, and one
nvidia-smi query every 5 s. No torch, no matplotlib, no GPU. Local only.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
RUNS_DIR = Path("runs")
LOG_DIRS = [Path("experiments"), Path(".")]
CHART_POINTS = 400          # train_batch records kept for the chart
CONSOLE_LINES = 40
TAIL_BYTES = 96 * 1024


# ── file readers (tails only) ─────────────────────────────────────────

def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return default


def tail_lines(path, max_bytes=TAIL_BYTES):
    try:
        with open(path, "rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - max_bytes))
            data = file.read()
        lines = data.decode("utf-8", errors="replace").splitlines()
        return lines[1:] if size > max_bytes else lines   # drop a possibly cut first line
    except OSError:
        return []


def tail_records(path, record_type=None, limit=CHART_POINTS):
    records = []
    for line in tail_lines(path):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record_type is None or record.get("type") == record_type:
            records.append(record)
    return records[-limit:]


def all_records(path, record_type):
    records = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") == record_type:
                    records.append(record)
    except OSError:
        pass
    return records


def minutes_between(start, end):
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60
    except (TypeError, ValueError):
        return None


# ── run discovery ─────────────────────────────────────────────────────

def run_infos():
    """{run_name: (info dict, info.json mtime)} for every run directory."""
    result = {}
    if not RUNS_DIR.is_dir():
        return result
    for run_dir in RUNS_DIR.iterdir():
        info_path = run_dir / "info.json"
        try:
            mtime = info_path.stat().st_mtime
        except OSError:
            continue
        info = read_json(info_path)
        if info:
            result[run_dir.name] = (info, mtime)
    return result


def current_run(infos):
    """The most recently started run that is still in progress, else the newest run."""
    in_progress = [(info.get("started_at") or "", name) for name, (info, _) in infos.items() if info.get("status") == "in_progress"]
    if in_progress:
        return max(in_progress)[1]
    if infos:
        return max((mtime, name) for name, (_, mtime) in infos.items())[1]
    return None


class ExperimentCache:
    """Per-run summaries for the table; a run is re-read only when its info.json changes."""

    def __init__(self):
        self.summaries = {}   # run name -> (mtime, summary)

    def summary(self, name, info, mtime):
        cached = self.summaries.get(name)
        if cached and cached[0] == mtime:
            return cached[1]
        evals = all_records(RUNS_DIR / name / "logs" / "training.jsonl", "eval_epoch")
        best = max((e.get("task_accuracy", 0.0) for e in evals), default=None)
        final = evals[-1] if evals else {}
        summary = {
            "run": name,
            "status": info.get("status"),
            "domain": info.get("domain"),
            "kl": f"{info.get('kl_ratio_start')}->{info.get('kl_ratio_end')}",
            "minutes": minutes_between(info.get("started_at"), info.get("completed_at")),
            "task_best": best,
            "task_final": final.get("task_accuracy"),
            "overlap_final": final.get("teacher_student_overlap"),
            "student_entropy_final": final.get("student_entropy"),
            "epochs_done": len(evals),
        }
        self.summaries[name] = (mtime, summary)
        return summary


EXPERIMENT_CACHE = ExperimentCache()

EXPERIMENTS_DIR = Path("experiments")
LEDGER_PATH = Path("thesis_work") / "experiment_ledger.txt"
DESCRIPTION_CACHE = {}   # tag -> {"script", "docstring", "ledger"}
RUN_BOOKKEEPING_KEYS = {"run_name", "description", "status", "started_at", "completed_at", "model_info", "experiment"}


def experiment_description(tag):
    """The script that declares this tag (its path + docstring) and the ledger block for it."""
    if tag in DESCRIPTION_CACHE:
        return DESCRIPTION_CACHE[tag]
    import ast
    script, docstring = None, None
    if EXPERIMENTS_DIR.is_dir():
        for path in sorted(EXPERIMENTS_DIR.glob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if re.search(r"^EXPERIMENT\s*=\s*[\"']" + re.escape(tag) + r"[\"']", source, re.MULTILINE):
                script = str(path)
                try:
                    docstring = ast.get_docstring(ast.parse(source))
                except SyntaxError:
                    docstring = None
                break
    ledger = None
    number = re.match(r"e(\d+)-", tag or "")
    if number and LEDGER_PATH.exists():
        key = f"E-{int(number.group(1)):02d}"
        block, inside = [], False
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            if re.match(r"^E-\d+", line):
                if inside:
                    break
                inside = line.startswith(key + " ")
            if inside:
                block.append(line)
        ledger = "\n".join(block).strip() or None
    DESCRIPTION_CACHE[tag] = {"script": script, "docstring": docstring, "ledger": ledger}
    return DESCRIPTION_CACHE[tag]


def config_split(run_infos_for_tag):
    """Which config fields are shared by every run of the experiment, and which vary (the arms)."""
    values = {}
    for info in run_infos_for_tag:
        for key, value in info.items():
            if key in RUN_BOOKKEEPING_KEYS:
                continue
            values.setdefault(key, set()).add(json.dumps(value, sort_keys=True))
    shared = {k: json.loads(next(iter(v))) for k, v in values.items() if len(v) == 1}
    varying = {k: sorted(json.loads(x) for x in v) for k, v in values.items() if len(v) > 1}
    return shared, varying


# ── gpu sampler (one process spawn per 5 s, in its own thread) ────────

GPU = {"available": False}


def sample_gpu_forever():
    while True:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4).stdout.strip()
            util, used, total, temp = [float(x) for x in out.split(",")]
            GPU.update(available=True, utilization=util, memory_used=used, memory_total=total, temperature=temp)
        except Exception:
            GPU.update(available=False)
        time.sleep(5)


# ── state assembly ────────────────────────────────────────────────────

def planned_run_count(log_path):
    """Experiment scripts announce '<tag>: N runs ...' on their first line."""
    if not log_path:
        return None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as file:
            head = file.read(400)
    except OSError:
        return None
    match = re.search(r"(\d+) runs", head)
    return int(match.group(1)) if match else None


def newest_console_log():
    candidates = []
    for directory in LOG_DIRS:
        if directory.is_dir():
            candidates += [p for p in directory.glob("*.log") if not p.name.startswith("progress")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def batches_per_epoch(info, batch_records):
    max_examples = info.get("max_training_examples")
    batch_size = info.get("batch_size", 32)
    ratio = info.get("training_test_ratio", 0.8)
    if max_examples:
        return int((max_examples // batch_size) * ratio)
    return max((r.get("batch", 0) for r in batch_records), default=0)


def build_state():
    infos = run_infos()
    name = current_run(infos)
    state = {"now": datetime.now().isoformat(timespec="seconds"), "gpu": GPU, "run": None}

    log_path = newest_console_log()
    state["console"] = {"file": str(log_path) if log_path else None,
                        "lines": tail_lines(log_path)[-CONSOLE_LINES:] if log_path else []}

    if name is None:
        return state

    info, _ = infos[name]
    logs = RUNS_DIR / name / "logs"
    batch_records = tail_records(logs / "training.jsonl", "train_batch")
    eval_records = tail_records(logs / "training.jsonl", "eval_epoch", limit=50)
    run_state = read_json(logs / "state.json", {}) or {}
    last = batch_records[-1] if batch_records else {}
    per_epoch = batches_per_epoch(info, batch_records)
    epoch = run_state.get("current_epoch", last.get("epoch", 0))
    batch = run_state.get("current_batch", last.get("batch", 0))
    total_epochs = info.get("epoch_count", 0)
    progress = ((epoch * per_epoch) + batch) / (total_epochs * per_epoch) if total_epochs and per_epoch else 0.0

    state["run"] = {
        "name": name,
        "status": info.get("status"),
        "experiment": info.get("experiment"),
        "domain": info.get("domain"),
        "description": info.get("description"),
        "started_at": info.get("started_at"),
        "completed_at": info.get("completed_at"),
        "epoch": epoch, "epochs": total_epochs, "batch": batch, "batches_per_epoch": per_epoch,
        "progress": min(1.0, progress),
        "last": {k: last.get(k) for k in ("loss", "kl_loss", "ce_loss", "accuracy", "learning_rate",
                                           "kl_ratio", "temperature", "teacher_temperature", "attention_noise",
                                           "time_seconds", "steps")},
        "chart": [{"x": r.get("epoch", 0) * per_epoch + r.get("batch", 0), "loss": r.get("loss"),
                   "accuracy": r.get("accuracy"), "kl_ratio": r.get("kl_ratio")} for r in batch_records],
        "evals": [{"epoch": e.get("epoch"), "task_accuracy": e.get("task_accuracy"),
                   "student_accuracy": e.get("student_accuracy"), "teacher_forced_accuracy": e.get("teacher_forced_accuracy"),
                   "overlap": e.get("teacher_student_overlap"), "student_entropy": e.get("student_entropy"),
                   "teacher_entropy": e.get("teacher_entropy")} for e in eval_records],
    }

    # The experiment this run belongs to: every sibling run, cached summaries.
    tag = info.get("experiment")
    siblings = [(n, i, m) for n, (i, m) in infos.items() if i.get("experiment") == tag]
    rows = sorted((EXPERIMENT_CACHE.summary(n, i, m) for n, i, m in siblings), key=lambda r: r["run"])
    completed = [r for r in rows if r["status"] == "completed" and r["minutes"]]
    mean_minutes = sum(r["minutes"] for r in completed) / len(completed) if completed else None
    shared, varying = config_split([i for _, i, _ in siblings])
    per_group = {}
    for row in rows:
        group = f"{row['domain']} · kl {row['kl']}"
        bucket = per_group.setdefault(group, {"completed": 0, "seen": 0})
        bucket["seen"] += 1
        bucket["completed"] += row["status"] == "completed"
    state["experiment"] = {
        "tag": tag,
        "rows": rows,
        "completed": len(completed),
        "total": planned_run_count(log_path) or len(rows),
        "mean_minutes": mean_minutes,
        "shared_config": shared,
        "varying_config": varying,
        "groups": per_group,
        **experiment_description(tag),
    }
    return state


# ── http ──────────────────────────────────────────────────────────────

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Training progress</title>
<style>
:root{--bg:#0f141a;--panel:#161d26;--ink:#dfe6ee;--dim:#8a99a8;--rule:#26313d;--acc:#5aa9ff;--ok:#7fd48a;--warn:#f0b35c;--mono:"IBM Plex Mono",Consolas,monospace}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:18px 22px 40px}
h1{font-size:16px;margin:0 0 4px;font-weight:600}.sub{color:var(--dim);font-size:12.5px;margin:0 0 14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.wide{grid-column:1/-1}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:12px 14px}
.label{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);margin:0 0 6px}
.big{font-family:var(--mono);font-size:20px}.mono{font-family:var(--mono)}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:8px 14px}.metrics div span{display:block;color:var(--dim);font-size:11px}
.bar{height:8px;background:#0b0f14;border-radius:4px;overflow:hidden;margin:6px 0}.bar i{display:block;height:100%;background:var(--acc)}
.gpu i{background:var(--ok)}
canvas{width:100%;height:220px;display:block}
table{width:100%;border-collapse:collapse;font-size:12.5px}th{color:var(--dim);font-weight:500;text-align:left;padding:4px 6px;border-bottom:1px solid var(--rule);font-size:11px;letter-spacing:.05em;text-transform:uppercase}
td{padding:3px 6px;border-bottom:1px solid #1d2630;font-family:var(--mono);font-size:12px}td.r{text-align:right}tr.now td{color:var(--acc)}
pre{margin:0;font:11.5px/1.45 var(--mono);color:#b9c6d2;white-space:pre-wrap;word-break:break-all;max-height:320px;overflow:auto}
.tick{color:var(--dim);font-size:12px;font-family:var(--mono)}
.about{display:grid;grid-template-columns:1fr 1fr;gap:12px 18px;margin-top:10px}.about pre{background:#0b0f14;padding:8px 10px;border-radius:4px}
details summary{font-size:13px}
</style></head><body><div class="wrap">
<h1 id="title">Training progress</h1><p class="sub" id="subtitle">waiting for data…</p>
<div class="grid">
 <div class="panel wide"><p class="label">Now</p><div class="big mono" id="runname">—</div>
  <div class="bar"><i id="bar" style="width:0%"></i></div>
  <div class="tick" id="progress">—</div>
  <div class="metrics" id="metrics" style="margin-top:10px"></div></div>
 <div class="panel wide"><p class="label">Current run — loss (blue) and token accuracy (green) per batch · KL ratio (amber, dashed)</p><canvas id="chart" width="1120" height="220"></canvas></div>
 <div class="panel"><p class="label">Epoch evaluations</p><table id="evals"></table></div>
 <div class="panel"><p class="label">GPU</p><div id="gpu">—</div></div>
 <div class="panel wide"><p class="label" id="explabel">Experiment</p><div class="tick" id="eta"></div>
  <details id="about" style="margin-top:8px"><summary style="cursor:pointer;color:var(--acc)">About this experiment — what is being tested, the arms, the shared config</summary>
   <div class="about"><div><p class="label">Script</p><div class="tick" id="script"></div><pre id="docstring" style="max-height:260px"></pre></div>
   <div><p class="label">Varies across runs (the arms)</p><table id="varying"></table><p class="label" style="margin-top:10px">Progress per group</p><table id="groups"></table></div>
   <div><p class="label">Shared by every run</p><table id="shared"></table></div>
   <div><p class="label">Experiment ledger entry</p><pre id="ledger" style="max-height:260px"></pre></div></div></details>
  <div style="max-height:300px;overflow:auto;margin-top:10px"><table id="exp"></table></div></div>
 <div class="panel wide"><p class="label" id="conlabel">Console</p><pre id="console"></pre></div>
</div></div>
<script>
const $=id=>document.getElementById(id);let lastState=null,lastFetch=0;
const f=(v,d=4)=>v==null?'—':(typeof v==='number'?v.toFixed(d):v);
const pct=v=>v==null?'—':(v*100).toFixed(1)+'%';
function draw(chart){const c=$('chart'),ctx=c.getContext('2d');const W=c.width,H=c.height;ctx.clearRect(0,0,W,H);if(!chart||chart.length<2)return;
 const L=46,R=46,T=10,B=22,w=W-L-R,h=H-T-B;const xs=chart.map(p=>p.x);const x0=Math.min(...xs),x1=Math.max(...xs);
 const losses=chart.map(p=>p.loss).filter(v=>v!=null);const lmax=Math.max(...losses),lmin=Math.min(...losses);
 const X=x=>L+(x-x0)/Math.max(1,x1-x0)*w;const Y=(v,lo,hi)=>T+h-(v-lo)/Math.max(1e-9,hi-lo)*h;
 ctx.strokeStyle='#26313d';ctx.lineWidth=1;for(let i=0;i<=4;i++){const y=T+h*i/4;ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(L+w,y);ctx.stroke();}
 ctx.fillStyle='#8a99a8';ctx.font='11px monospace';ctx.textAlign='right';for(let i=0;i<=4;i++){ctx.fillText((lmax-(lmax-lmin)*i/4).toFixed(3),L-4,T+h*i/4+4);}
 ctx.textAlign='left';for(let i=0;i<=4;i++){ctx.fillText((100-25*i)+'%',L+w+4,T+h*i/4+4);}
 ctx.textAlign='center';ctx.fillText('batch '+x0,L,H-6);ctx.fillText('batch '+x1,L+w,H-6);
 const line=(key,color,lo,hi,dash)=>{ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.setLineDash(dash||[]);ctx.beginPath();let s=false;
  for(const p of chart){if(p[key]==null)continue;const x=X(p.x),y=Y(p[key],lo,hi);s?ctx.lineTo(x,y):ctx.moveTo(x,y);s=true;}ctx.stroke();ctx.setLineDash([]);};
 line('loss','#5aa9ff',lmin,lmax);line('accuracy','#7fd48a',0,1);line('kl_ratio','#f0b35c',0,1,[4,4]);}
function render(s){lastState=s;const r=s.run;
 $('subtitle').textContent=(s.console.file?('console: '+s.console.file+' · '):'')+'updated '+s.now.replace('T',' ');
 if(!r){$('runname').textContent='no runs found';return;}
 $('runname').textContent=r.name+'   ['+r.status+']';$('bar').style.width=(r.progress*100).toFixed(1)+'%';
 const el=r.started_at?((Date.now()-Date.parse(r.started_at))/60000):null;
 $('progress').textContent=`epoch ${r.epoch+1}/${r.epochs} · batch ${r.batch}/${r.batches_per_epoch} · ${(r.progress*100).toFixed(1)}% · elapsed ${el==null?'—':el.toFixed(1)+' min'} · ${r.domain} · ${r.description||''}`;
 const m=r.last;$('metrics').innerHTML=[['loss',f(m.loss)],['KL',f(m.kl_loss)],['CE',f(m.ce_loss)],['token acc',pct(m.accuracy)],['LR',f(m.learning_rate,6)],['KL ratio',f(m.kl_ratio,3)],['temp / teacher',f(m.temperature,2)+' / '+f(m.teacher_temperature,2)],['attn noise',f(m.attention_noise,3)],['steps in batch',m.steps??'—'],['batch time',f(m.time_seconds,2)+' s']].map(([k,v])=>`<div><span>${k}</span><b class="mono">${v}</b></div>`).join('');
 draw(r.chart);
 $('evals').innerHTML='<tr><th>epoch</th><th>task</th><th>student</th><th>TF</th><th>overlap</th><th>H stu</th><th>H tea</th></tr>'+r.evals.map(e=>`<tr><td>${e.epoch+1}</td><td>${pct(e.task_accuracy)}</td><td>${pct(e.student_accuracy)}</td><td>${pct(e.teacher_forced_accuracy)}</td><td>${pct(e.overlap)}</td><td>${f(e.student_entropy,3)}</td><td>${f(e.teacher_entropy,3)}</td></tr>`).join('');
 const g=s.gpu;$('gpu').innerHTML=g.available?`<div class="tick">${g.utilization.toFixed(0)}% util · ${(g.memory_used/1024).toFixed(1)} / ${(g.memory_total/1024).toFixed(1)} GB · ${g.temperature.toFixed(0)}°C</div><div class="bar gpu"><i style="width:${g.utilization}%"></i></div><div class="bar"><i style="width:${g.memory_used/g.memory_total*100}%"></i></div>`:'nvidia-smi unavailable';
 const e=s.experiment;if(e){$('explabel').textContent=`Experiment ${e.tag} — ${e.completed}/${e.total} runs completed`;
  const rem=e.total-e.completed;$('eta').textContent=e.mean_minutes?`mean ${e.mean_minutes.toFixed(1)} min/run · ${rem} left ≈ ${(rem*e.mean_minutes/60).toFixed(1)} h · ETA ${new Date(Date.now()+rem*e.mean_minutes*60000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}`:'';
  $('script').textContent=e.script||'(no experiments/*.py declares this tag)';$('docstring').textContent=e.docstring||'';$('ledger').textContent=e.ledger||'(no ledger entry)';
  const kv=obj=>Object.entries(obj||{}).map(([k,v])=>`<tr><td>${k}</td><td>${Array.isArray(v)?v.map(x=>JSON.stringify(x)).join(' · '):JSON.stringify(v)}</td></tr>`).join('');
  $('varying').innerHTML=kv(e.varying_config)||'<tr><td>—</td></tr>';$('shared').innerHTML=kv(e.shared_config);
  $('groups').innerHTML=Object.entries(e.groups||{}).map(([g,c])=>`<tr><td>${g}</td><td class="r">${c.completed}/${c.seen} done</td></tr>`).join('');
  $('exp').innerHTML='<tr><th>run</th><th>kl</th><th>status</th><th class="r">min</th><th class="r">task best</th><th class="r">task final</th><th class="r">overlap</th><th class="r">H stu</th></tr>'+e.rows.map(x=>`<tr class="${x.run===r.name?'now':''}"><td>${x.run}</td><td>${x.kl}</td><td>${x.status}</td><td class="r">${f(x.minutes,1)}</td><td class="r">${pct(x.task_best)}</td><td class="r">${pct(x.task_final)}</td><td class="r">${pct(x.overlap_final)}</td><td class="r">${f(x.student_entropy_final,2)}</td></tr>`).join('');}
 $('conlabel').textContent='Console — '+(s.console.file||'');$('console').textContent=s.console.lines.join('\\n');$('console').scrollTop=$('console').scrollHeight;}
async function poll(){try{const s=await(await fetch('/api/state')).json();lastFetch=Date.now();render(s);}catch(err){$('subtitle').textContent='server unreachable — '+err;}}
setInterval(poll,3000);poll();
setInterval(()=>{if(lastState&&lastState.run&&lastState.run.started_at){const el=(Date.now()-Date.parse(lastState.run.started_at))/60000;const t=$('progress').textContent.replace(/elapsed [^·]+/,'elapsed '+el.toFixed(1)+' min');$('progress').textContent=t;}},1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/state"):
            body = json.dumps(build_state()).encode("utf-8")
            self._send(200, "application/json", body)
        elif self.path == "/" or self.path.startswith("/?"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):   # keep the terminal quiet
        pass


def main():
    threading.Thread(target=sample_gpu_forever, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Training dashboard at {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
