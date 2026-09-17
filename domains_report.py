"""Domain documentation page — one local HTML file describing every registered domain with
its real data: the task, the teacher prompt, the vocabulary, the datasets on disk, the
measured teacher entropy, and a selectable set of real examples (input -> teacher response).

    python domains_report.py        # writes thesis_work/domains.html, open it in a browser

Local file only. Reads batches/ (first batch file per dataset), experiments/results/
domain_entropy.csv, and the domain classes. Nothing is uploaded anywhere.
"""

import csv
import html
import inspect
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer

from library.domain import get_domain, registered_domains
from library.shared.batch_files import batch_paths, read_batch, read_batch_texts
from library.shared.config import BATCHES_DIR, MODEL_NAME
from library.shared.vocabulary import build_vocabulary

OUTPUT = Path("thesis_work") / "domains.html"
ENTROPY_CSV = Path("experiments") / "results" / "domain_entropy.csv"
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
            match = re.search(r"(?:Answer:)?\s*([^;\n]+);", record["model_response"])
            item["truth"] = answer
            item["attribute"] = attribute
            item["correct"] = bool(match) and match.group(1).strip().lower() == answer.lower()
        examples.append(item)
    return examples


def describe(domain_name, tokenizers, entropies):
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
            tokenizer = tokenizers.setdefault(teacher, AutoTokenizer.from_pretrained(teacher))
            vocabulary = build_vocabulary(tokenizer, domain)
            datasets.append({
                "teacher": teacher, "batches": batches, "examples": examples,
                "vocab_size": vocabulary["vocab_size"],
                "entropy_nats": float(entropy["mean_nats"]) if entropy else None,
                "exp_H": float(entropy["exp_H"]) if entropy else None,
                "one_hot": float(entropy["one_hot_share"]) if entropy else None,
                "examples_list": examples_for(domain, teacher_dir),
            })
    if not datasets:   # no data yet: still show the vocabulary the default teacher would build
        tokenizer = tokenizers.setdefault(MODEL_NAME, AutoTokenizer.from_pretrained(MODEL_NAME))
        vocabulary = build_vocabulary(tokenizer, domain)
        datasets.append({"teacher": None, "batches": 0, "examples": 0, "vocab_size": vocabulary["vocab_size"],
                         "entropy_nats": None, "exp_H": None, "one_hot": None, "examples_list": []})
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
        "max_input_tokens": domain.max_input_tokens,
        "metric": type(domain.task_metric()).__name__,
        "structural_validity": domain.structural_validity("") is not None,
        "prompt": domain.teacher_prompt(PLACEHOLDER),
        "datasets": datasets,
        "fields_exp_H": fields,
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Domains</title>
<style>
:root{--bg:#eceff2;--panel:#fff;--ink:#16212c;--dim:#5b6b7a;--faint:#8797a5;--rule:#d3dbe2;--acc:#1c5896;--code:#151d25;--codeink:#d3dce4;--ok:#2f7a3d;--bad:#a83a3a;--mono:"IBM Plex Mono",Consolas,monospace}
@media(prefers-color-scheme:dark){:root{--bg:#101720;--panel:#18212b;--ink:#e1e8ee;--dim:#a1b1bf;--faint:#74869a;--rule:#2b3743;--acc:#6ca9e8;--code:#0b1117;--ok:#7fd48a;--bad:#f2a8a8}}
body{margin:0;background:var(--bg);color:var(--ink);font:14.5px/1.55 -apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:26px 24px 60px}h1{font-size:21px;margin:0 0 4px}.sub{color:var(--faint);font-size:12.5px;margin:0 0 18px}
.tabs{display:flex;flex-wrap:wrap;gap:2px;border-bottom:1px solid var(--rule);margin-bottom:22px}.tab{background:none;border:none;border-bottom:2px solid transparent;margin-bottom:-1px;padding:8px 13px;font:inherit;font-size:13.5px;color:var(--dim);cursor:pointer}
.tab[aria-selected=true]{color:var(--acc);border-bottom-color:var(--acc);font-weight:600}
h2{font-size:17px;margin:0 0 4px}h3{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);margin:22px 0 8px}
p{max-width:70ch;color:var(--dim);margin:0 0 10px}
table{border-collapse:collapse;width:100%;background:var(--panel);font-size:13px;border:1px solid var(--rule);border-radius:4px}th{text-align:left;font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--faint);padding:7px 11px;border-bottom:1px solid var(--rule)}td{padding:6px 11px;border-bottom:1px solid var(--rule);vertical-align:top}tr:last-child td{border-bottom:none}td.m{font-family:var(--mono);font-size:12px}td.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
pre{background:var(--code);color:var(--codeink);padding:12px 14px;border-radius:4px;font:12.5px/1.55 var(--mono);white-space:pre-wrap;word-break:break-word;margin:0 0 10px;max-height:340px;overflow:auto}
.ex{display:grid;grid-template-columns:1fr 1fr;gap:12px}.ex pre{max-height:380px}
select{font:inherit;font-size:13px;padding:5px 8px;border:1px solid var(--rule);border-radius:4px;background:var(--panel);color:var(--ink);max-width:100%}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.badge{display:inline-block;font:11px var(--mono);padding:2px 7px;border:1px solid var(--rule);border-radius:3px;color:var(--dim);margin-right:6px}
@media(max-width:760px){.ex{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>Domains</h1><p class="sub">Every registered task, its teacher prompt, vocabulary, datasets and real examples · generated __DATE__ by domains_report.py · local file</p>
<div class="tabs" id="tabs"></div><div id="panel"></div>
<script>
const DOMAINS=__DATA__;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const f=(v,d=3)=>v==null?'—':Number(v).toFixed(d);
function render(i){const d=DOMAINS[i];document.querySelectorAll('.tab').forEach((t,j)=>t.setAttribute('aria-selected',j===i));
 let h=`<h2>${esc(d.name)} <span class="badge">${esc(d.class)}</span></h2><p>${esc(d.doc)}</p>`;
 h+=`<h3>Spec</h3><table><tr><th>stop token</th><th>max steps</th><th>max input tokens</th><th>task metric</th><th>structural check</th></tr><tr><td class="m">${esc(JSON.stringify(d.stop_token))}</td><td class="n">${d.max_steps}</td><td class="n">${d.max_input_tokens}</td><td class="m">${esc(d.metric)}</td><td>${d.structural_validity?'yes':'no (free-form)'}</td></tr></table>`;
 h+=`<h3>Teacher prompt</h3><pre>${esc(d.prompt)}</pre>`;
 h+=`<h3>Datasets on disk</h3><table><tr><th>teacher</th><th class="n">batches</th><th class="n">examples</th><th class="n">output vocab</th><th class="n">teacher H (nats)</th><th class="n">exp(H)</th><th class="n">one-hot steps</th></tr>`+d.datasets.map(s=>`<tr><td class="m">${esc(s.teacher||'(none yet — vocab shown for the default teacher)')}</td><td class="n">${s.batches}</td><td class="n">${s.examples.toLocaleString()}</td><td class="n">${s.vocab_size}</td><td class="n">${f(s.entropy_nats,4)}</td><td class="n">${f(s.exp_H,3)}</td><td class="n">${s.one_hot==null?'—':(s.one_hot*100).toFixed(1)+'%'}</td></tr>`).join('')+`</table>`;
 const fk=Object.keys(d.fields_exp_H||{});if(fk.length){h+=`<h3>exp(H) per output field</h3><table><tr>${fk.map(k=>`<th>${esc(k)}</th>`).join('')}</tr><tr>${fk.map(k=>`<td class="n">${f(d.fields_exp_H[k],3)}</td>`).join('')}</tr></table>`;}
 d.datasets.forEach((s,si)=>{if(!s.examples_list.length)return;h+=`<h3>Examples — ${esc(s.teacher)}</h3><p>Pick one of the first ${s.examples_list.length} stored examples. Input on the left, the teacher's response on the right${s.examples_list[0].truth!==undefined?', with the generator\\'s ground truth':''}.</p><select id="sel-${i}-${si}" onchange="showEx(${i},${si},this.value)">${s.examples_list.map((e,k)=>`<option value="${k}">${k+1}. ${esc(e.input.replace(/\\s+/g,' ').slice(0,90))}${e.correct===false?'  ✗':''}</option>`).join('')}</select><div id="ex-${i}-${si}" style="margin-top:8px"></div>`;});
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
    domains = [describe(name, tokenizers, entropies) for name in registered_domains()]
    page = PAGE.replace("__DATA__", json.dumps(domains, ensure_ascii=False).replace("</", "<\\/")) \
               .replace("__DATE__", datetime.now().strftime("%Y-%m-%d %H:%M"))
    OUTPUT.write_text(page, encoding="utf-8")
    for domain in domains:
        sets = ", ".join(f"{s['teacher']}: {s['examples']:,} ex / V={s['vocab_size']}" for s in domain["datasets"] if s["teacher"])
        print(f"{domain['name']:26} {sets or '(no data yet)'}")
    print(f"\n-> {OUTPUT}")


if __name__ == "__main__":
    main()
