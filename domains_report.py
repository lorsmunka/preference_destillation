"""Domain documentation page — one local HTML file describing every registered domain with
its real data: the task, the teacher prompt, the vocabulary, the datasets on disk, the
measured teacher entropy, and a selectable set of real examples (input -> teacher response).

    python domains_report.py        # writes thesis_work/domains.html, open it in a browser

Local file only. Reads batches/ (first batch file per dataset), the completed runs' eval
logs, experiments/results/domain_entropy.csv, and the domain classes. Nothing is uploaded.

Teacher entropy has one source in every report (code review X-M8): the test-split eval
logs of the completed runs (library.tooling.run.test_split_teacher_entropy). Only a dataset
no run has been trained on yet falls back to domain_entropy.csv (measure_entropy.py, the
first 256 examples); the page says which sample each number is.
"""

import csv
import html
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer

from library import RunStore
from library.domain import get_domain, registered_domains
from library.shared.batch_files import batch_paths, read_batch, read_batch_texts
from library.shared.config import BATCHES_DIR, MODEL_NAME, PROJECT_ROOT
from library.shared.metrics.task import extract_answer
from library.shared.vocabulary import build_vocabulary
from library.tooling.run import test_split_teacher_entropy

OUTPUT = PROJECT_ROOT / "thesis_work" / "domains.html"
ENTROPY_CSV = PROJECT_ROOT / "experiments" / "results" / "domain_entropy.csv"
EXAMPLES_PER_DATASET = 12
PLACEHOLDER = "<input text>"


def entropy_rows():
    rows = {}
    if ENTROPY_CSV.exists():
        with open(ENTROPY_CSV, "r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                rows[row["key"]] = row
    return rows


def dataset_size(directory):
    files = batch_paths(directory)
    if not files:
        return 0, 0
    last = len(read_batch_texts(files[-1]))
    first = len(read_batch_texts(files[0]))
    return len(files), first * (len(files) - 1) + last


def ground_truth(domain):
    """{text: (answer, attribute)} for domains whose corpus stores the truth (Virág)."""
    truth = {}
    corpus = Path(domain.corpus_path)
    if not corpus.exists():
        return truth
    with open(corpus, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if "answer" in record:
                truth[record["text"]] = (record["answer"], record.get("attribute", ""))
    return truth


def examples_for(domain, directory):
    files = batch_paths(directory)
    if not files:
        return []
    truth = ground_truth(domain)
    examples = []
    for record in read_batch(files[0])[:EXAMPLES_PER_DATASET]:
        item = {"input": record["sentence"], "response": record["model_response"], "steps": len(record["steps"])}
        if record.get("thinking"):
            item["thinking"] = record["thinking"]
        if truth and record["sentence"] in truth:
            answer, attribute = truth[record["sentence"]]
            item["truth"] = answer
            item["attribute"] = attribute
            item["correct"] = extract_answer(record["model_response"]) == answer.lower()
        examples.append(item)
    return examples


def teacher_entropy(domain_name, teacher, csv_row, completed_runs):
    """(nats, exp(H), which sample) for one dataset: the test-split eval logs of the
    completed runs on it, else the measure_entropy.py row."""
    runs = [run for run in completed_runs if run.domain == domain_name and run.teacher_model == teacher]
    measured = test_split_teacher_entropy(runs)
    if measured:
        return measured["nats"], measured["exp_nats"], measured["sample"]
    if csv_row:
        return (float(csv_row["mean_nats"]), float(csv_row["exp_H"]),
                f"no trained run yet: first {csv_row['examples']} examples of the batches (measure_entropy.py)")
    return None, None, "not measured"


def describe(domain_name, tokenizers, entropies, completed_runs):
    domain = get_domain(domain_name)
    doc = inspect.getdoc(sys.modules[type(domain).__module__]) or ""
    datasets = []
    domain_dir = BATCHES_DIR / domain.name
    if domain_dir.is_dir():
        for teacher_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            teacher = teacher_dir.name.replace("_", "/", 1)
            batches, examples = dataset_size(teacher_dir)
            if not batches:
                continue
            entropy = entropies.get(domain.name, {})
            entropy_nats, exp_entropy, entropy_source = teacher_entropy(domain.name, teacher, entropy, completed_runs)
            tokenizer = tokenizers.setdefault(teacher, AutoTokenizer.from_pretrained(teacher))
            vocabulary = build_vocabulary(tokenizer, domain)
            datasets.append({
                "teacher": teacher, "batches": batches, "examples": examples,
                "vocab_size": vocabulary["vocab_size"],
                "entropy_nats": entropy_nats,
                "exp_H": exp_entropy,
                "entropy_source": entropy_source,
                "one_hot": float(entropy["one_hot_share"]) if entropy else None,
                "examples_list": examples_for(domain, teacher_dir),
            })
    if not datasets:   # no data yet: still show the vocabulary the default teacher would build
        tokenizer = tokenizers.setdefault(MODEL_NAME, AutoTokenizer.from_pretrained(MODEL_NAME))
        vocabulary = build_vocabulary(tokenizer, domain)
        datasets.append({"teacher": None, "batches": 0, "examples": 0, "vocab_size": vocabulary["vocab_size"],
                         "entropy_nats": None, "exp_H": None, "entropy_source": "no data", "one_hot": None,
                         "examples_list": []})
    fields = {}
    for key, row in entropies.items():
        if key.startswith(domain.name + "/"):
            fields[key.split("/", 1)[1]] = float(row["exp_H"])
    return {
        "name": domain.name,
        "class": type(domain).__name__,
        "doc": doc,
        "stop_token": domain.stop_token,
        "max_steps": domain.max_steps,
        "metric": type(domain.task_metric()).__name__,
        "structural_validity": domain.structural_validity("") is not None,
        "prompt": domain.teacher_prompt(PLACEHOLDER),
        "datasets": datasets,
        "fields_exp_H": fields,
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Domains</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#f5f5f5;--surface:#fff;--ink:rgba(0,0,0,.87);--dim:rgba(0,0,0,.6);--faint:rgba(0,0,0,.38);--rule:rgba(0,0,0,.12);--primary:#1565c0;--primary-soft:#e3f2fd;--code:#263238;--codeink:#eceff1;--ok:#2e7d32;--bad:#c62828;--mono:"Roboto Mono",Consolas,monospace;--shadow:0 1px 3px rgba(0,0,0,.12),0 1px 2px rgba(0,0,0,.24)}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 Roboto,"Segoe UI",sans-serif}
.appbar{background:var(--primary);color:#fff;box-shadow:0 2px 4px rgba(0,0,0,.2)}.appbar .in{max-width:1040px;margin:0 auto;padding:18px 24px 0}
.appbar h1{font-size:20px;font-weight:500;margin:0;letter-spacing:.01em}.appbar .sub{color:rgba(255,255,255,.75);font-size:12.5px;margin:2px 0 12px}
.tabs{display:flex;flex-wrap:wrap;gap:0}.tab{background:none;border:none;border-bottom:2px solid transparent;padding:12px 18px;font:500 13px Roboto,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.7);cursor:pointer;transition:background .15s}
.tab:hover{background:rgba(255,255,255,.08)}.tab[aria-selected=true]{color:#fff;border-bottom-color:#fff}
.wrap{max-width:1040px;margin:0 auto;padding:22px 24px 60px}
.card{background:var(--surface);border-radius:8px;box-shadow:var(--shadow);padding:18px 22px;margin:0 0 16px}
h2{font-size:20px;font-weight:500;margin:0 0 6px}h3{font-size:12px;font-weight:500;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin:0 0 10px}
p{max-width:72ch;color:var(--dim);margin:0 0 10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th{text-align:left;font-size:12px;font-weight:500;color:var(--dim);padding:8px 12px;border-bottom:1px solid var(--rule)}td{padding:9px 12px;border-bottom:1px solid var(--rule);vertical-align:top}tr:last-child td{border-bottom:none}td.m{font-family:var(--mono);font-size:12.5px}td.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
pre{background:var(--code);color:var(--codeink);padding:12px 14px;border-radius:6px;font:12.5px/1.55 var(--mono);white-space:pre-wrap;word-break:break-word;margin:0 0 10px;max-height:340px;overflow:auto}
.ex{display:grid;grid-template-columns:1fr 1fr;gap:14px}.ex pre{max-height:380px}
select{font:14px Roboto,sans-serif;padding:9px 12px;border:1px solid var(--rule);border-radius:4px;background:var(--surface);color:var(--ink);max-width:100%;outline:none}select:focus{border-color:var(--primary);box-shadow:0 0 0 2px var(--primary-soft)}
.ok{color:var(--ok);font-weight:500}.bad{color:var(--bad);font-weight:500}
.badge{display:inline-block;font:500 11px Roboto,sans-serif;letter-spacing:.04em;padding:3px 10px;border-radius:12px;background:var(--primary-soft);color:var(--primary);margin-right:6px}
@media(max-width:760px){.ex{grid-template-columns:1fr}}
</style></head><body>
<div class="appbar"><div class="in"><h1>Domains</h1><p class="sub">Every registered task, its teacher prompt, vocabulary, datasets and real examples · generated __DATE__ by domains_report.py · local file</p><div class="tabs" id="tabs"></div></div></div>
<div class="wrap"><div id="panel"></div>
<script>
const DOMAINS=__DATA__;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const f=(v,d=3)=>v==null?'—':Number(v).toFixed(d);
function render(i){const d=DOMAINS[i];document.querySelectorAll('.tab').forEach((t,j)=>t.setAttribute('aria-selected',j===i));
 let h=`<div class="card"><h2>${esc(d.name)} <span class="badge">${esc(d.class)}</span></h2><p>${esc(d.doc)}</p></div>`;
 h+=`<div class="card"><h3>Spec</h3><table><tr><th>stop token</th><th>max steps</th><th>task metric</th><th>structural check</th></tr><tr><td class="m">${esc(JSON.stringify(d.stop_token))}</td><td class="n">${d.max_steps}</td><td class="m">${esc(d.metric)}</td><td>${d.structural_validity?'yes':'no (free-form)'}</td></tr></table></div>`;
 h+=`<div class="card"><h3>Teacher prompt</h3><pre>${esc(d.prompt)}</pre></div>`;
 h+=`<div class="card"><h3>Datasets on disk</h3><table><tr><th>teacher</th><th class="n">batches</th><th class="n">examples</th><th class="n">output vocab</th><th class="n">teacher H (nats)</th><th class="n">exp(H)</th><th class="n">one-hot steps</th></tr>`+d.datasets.map(s=>`<tr><td class="m">${esc(s.teacher||'(none yet — vocab shown for the default teacher)')}</td><td class="n">${s.batches}</td><td class="n">${s.examples.toLocaleString()}</td><td class="n">${s.vocab_size}</td><td class="n">${f(s.entropy_nats,4)}</td><td class="n">${f(s.exp_H,3)}</td><td class="n">${s.one_hot==null?'—':(s.one_hot*100).toFixed(1)+'%'}</td></tr>`).join('')+`</table>`
 +d.datasets.filter(s=>s.teacher).map(s=>`<p style="margin-top:8px">Teacher H (${esc(s.teacher)}): ${esc(s.entropy_source)}. One-hot steps: first 256 examples (measure_entropy.py).</p>`).join('');
 const fk=Object.keys(d.fields_exp_H||{});if(fk.length){h+=`<h3 style="margin-top:16px">exp(H) per output field (first 256 examples, measure_entropy.py)</h3><table><tr>${fk.map(k=>`<th>${esc(k)}</th>`).join('')}</tr><tr>${fk.map(k=>`<td class="n">${f(d.fields_exp_H[k],3)}</td>`).join('')}</tr></table>`;}
 h+=`</div>`;
 d.datasets.forEach((s,si)=>{if(!s.examples_list.length)return;h+=`<div class="card"><h3>Examples — ${esc(s.teacher)}</h3><p>Pick one of the first ${s.examples_list.length} stored examples. Input on the left, the teacher's response on the right${s.examples_list[0].truth!==undefined?', with the generator\\'s ground truth':''}.</p><select id="sel-${i}-${si}" onchange="showEx(${i},${si},this.value)">${s.examples_list.map((e,k)=>`<option value="${k}">${k+1}. ${esc(e.input.replace(/\\s+/g,' ').slice(0,90))}${e.correct===false?'  ✗':''}</option>`).join('')}</select><div id="ex-${i}-${si}" style="margin-top:12px"></div></div>`;});
 document.getElementById('panel').innerHTML=h;d.datasets.forEach((s,si)=>{if(s.examples_list.length)showEx(i,si,0);});}
function showEx(i,si,k){const e=DOMAINS[i].datasets[si].examples_list[k];let extra='';
 if(e.truth!==undefined)extra+=`<div style="margin:6px 0">ground truth: <b class="m">${esc(e.truth)}</b> <span class="badge">${esc(e.attribute)}</span> ${e.correct?'<span class="ok">teacher correct</span>':'<span class="bad">teacher wrong</span>'}</div>`;
 if(e.thinking)extra+=`<h3>Thinking trace</h3><pre>${esc(e.thinking)}</pre>`;
 document.getElementById(`ex-${i}-${si}`).innerHTML=`<div class="ex"><div><div class="badge">input</div><pre>${esc(e.input)}</pre></div><div><div class="badge">teacher response · ${e.steps} steps</div><pre>${esc(e.response)}</pre>${extra}</div></div>`;}
const tabs=document.getElementById('tabs');DOMAINS.forEach((d,i)=>{const b=document.createElement('button');b.className='tab';b.textContent=d.name;b.onclick=()=>render(i);tabs.appendChild(b);});render(0);
</script></div></body></html>"""


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    entropies = entropy_rows()
    tokenizers = {}
    completed_runs = RunStore().completed()
    domains = [describe(name, tokenizers, entropies, completed_runs) for name in registered_domains()]
    page = PAGE.replace("__DATA__", json.dumps(domains, ensure_ascii=False).replace("</", "<\\/")) \
               .replace("__DATE__", datetime.now().strftime("%Y-%m-%d %H:%M"))
    OUTPUT.write_text(page, encoding="utf-8")
    for domain in domains:
        sets = ", ".join(f"{s['teacher']}: {s['examples']:,} ex / V={s['vocab_size']}" for s in domain["datasets"] if s["teacher"])
        print(f"{domain['name']:26} {sets or '(no data yet)'}")
    print(f"\n-> {OUTPUT}")


if __name__ == "__main__":
    main()
