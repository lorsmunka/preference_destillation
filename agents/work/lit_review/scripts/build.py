import sys, json, re, html, os
sys.path.insert(0, 'scripts')
sys.stdout.reconfigure(encoding='utf-8')
from entries import ENTRIES, THEMES
from sections import TLDR, CLAIMS, COLUMNS, POSITIONING, NOT_DONE, OUTLINE, RISKS, UNVERIFIED

html_out, bib_out = sys.argv[1], sys.argv[2]
for path in (html_out, bib_out):
    if os.path.exists(path) and '--allow-overwrite' not in sys.argv:
        raise SystemExit(f'{path} exists; refusing to overwrite')

meta = {}
for name in ['data/meta1.json', 'data/meta2.json', 'data/meta3.json']:
    meta.update(json.load(open(name, encoding='utf-8')))

by_key = {}
for entry in ENTRIES:
    if entry['key'] in by_key:
        raise SystemExit('duplicate key ' + entry['key'])
    by_key[entry['key']] = entry
    if 'arxiv' in entry:
        record = meta[entry['arxiv']]
        entry.setdefault('title', record['title'])
        entry.setdefault('authors', record['authors'])
        entry.setdefault('links', [("arXiv", f"https://arxiv.org/abs/{entry['arxiv']}")])
        entry.setdefault('verified', 'arXiv API record' + (' + venue cross-check' if entry['bib']['type'] != 'misc' else ''))


def surname(name):
    name = name.strip()
    if name.startswith('{'):
        return name.strip('{}')
    parts = name.split()
    if len(parts) >= 3 and parts[-2][0].islower():
        return ' '.join(parts[-2:])
    return parts[-1]


def short_label(entry):
    names = entry['authors']
    others = entry['bib'].get('author_others')
    if entry['key'].startswith('gemma'):
        return f"Gemma Team {entry['year']}"
    if entry['key'] == 'deepseek2025r1':
        return 'DeepSeek-AI 2025'
    if len(names) == 1 and not others:
        return f"{surname(names[0])} {entry['year']}"
    if len(names) == 2 and not others:
        return f"{surname(names[0])} & {surname(names[1])} {entry['year']}"
    return f"{surname(names[0])} et al. {entry['year']}"


def display_authors(entry):
    if entry.get('display_authors'):
        return entry['display_authors']
    names = [n.strip('{}') for n in entry['authors']]
    if len(names) <= 3:
        return ', '.join(names)
    return ', '.join(names[:3]) + ' et al.'


def link_refs(text):
    def replace(match):
        key = match.group(1)
        if key not in by_key:
            raise SystemExit('unknown ref ' + key)
        return f'<a class="cite" href="#{key}">{html.escape(short_label(by_key[key]))}</a>'
    return re.sub(r'\[\[([a-z0-9]+)\]\]', replace, text)


def ref_list(keys):
    return ', '.join(f'<a class="cite" href="#{key}">{html.escape(short_label(by_key[key]))}</a>' for key in keys)


# ---------------- BibTeX ----------------
def bib_escape(text):
    text = text.replace('\\', '\\textbackslash{}') if '\\url' not in text else text
    for char in ['&', '%', '#', '_']:
        text = text.replace(char, '\\' + char)
    text = text.replace('>', '$>$')
    return text


def bib_entry(entry):
    bib = dict(entry['bib'])
    entry_type = bib.pop('type')
    others = bib.pop('author_others', False)
    fields = []
    authors = ' and '.join(entry['authors']) + (' and others' if others else '')
    fields.append(('author', authors))
    fields.append(('title', '{' + bib_escape(entry['title']) + '}'))
    for name in ['booktitle', 'journal', 'series', 'volume', 'pages', 'publisher']:
        if name in bib:
            fields.append((name, bib_escape(bib[name])))
    fields.append(('year', str(entry['year'])))
    if 'doi' in bib:
        fields.append(('doi', bib['doi']))
    if 'arxiv' in entry:
        fields.append(('eprint', entry['arxiv']))
        fields.append(('archivePrefix', 'arXiv'))
        if entry_type == 'misc' and 'howpublished' not in bib:
            fields.append(('howpublished', f"arXiv preprint arXiv:{entry['arxiv']}"))
    if 'howpublished' in bib:
        fields.append(('howpublished', bib['howpublished']))
    fields.append(('url', bib.get('url') or entry['links'][0][1]))
    if 'note' in bib:
        fields.append(('note', bib_escape(bib['note']) if '\\url' not in bib['note'] else bib['note']))
    body = ',\n'.join(f'  {name} = {{{value}}}' for name, value in fields)
    return f'@{entry_type}{{{entry["key"]},\n{body}\n}}\n'


bib_lines = ['% literature.bib -- references for the literature review (thesis_work/literature_review.html).',
             '% Every entry was opened and checked on 2026-10-07/08: arXiv records via the arXiv API;',
             '% venues via arXiv comments, Semantic Scholar, ACL Anthology, PMLR, icml.cc; software via GitHub/docs.',
             '% UTF-8 encoded (use biber/biblatex or inputenc utf8). Grouped by theme as in the HTML page.', '']
theme_names = {code: name for code, name, _ in THEMES}
for code, name, _ in THEMES:
    bib_lines.append(f'% ---------------- {code}. {name} ----------------')
    for entry in [e for e in ENTRIES if e['theme'] == code]:
        bib_lines.append(bib_entry(entry))
with open(bib_out, 'x' if '--allow-overwrite' not in sys.argv else 'w', encoding='utf-8') as handle:
    handle.write('\n'.join(bib_lines))

# ---------------- HTML ----------------
STATUS_LABEL = {'yes': 'yes', 'partial': 'partly', 'no': 'no', 'unknown': 'not stated', 'na': 'n/a'}
VERDICT_LABEL = {'defend': 'Defensible', 'reword': 'Reword', 'avoid': 'Do not claim as new'}

css = """
:root { --bg:#f7f7f4; --card:#ffffff; --fg:#1c1c1e; --muted:#5f5f66; --border:#e1e1dc; --code:#f0f0ec;
  --accent:#2a5bb0; --accent-soft:#e6eefb; --yes-bg:#e1f3e7; --yes-fg:#1d6a3a; --part-bg:#fdf0d2; --part-fg:#7a5300;
  --no-bg:#f1f1ee; --no-fg:#6a6a70; --unk-bg:#ece7f8; --unk-fg:#5b3fa0; --avoid-bg:#fbe3e0; --avoid-fg:#9b2c22;
  --this-row:#eef4ff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141416; --card:#1c1c20; --fg:#ececef;
  --muted:#a0a0a8; --border:#2f2f35; --code:#26262b; --accent:#8db4ff; --accent-soft:#1d2a44; --yes-bg:#173322;
  --yes-fg:#86dba6; --part-bg:#3a2d0f; --part-fg:#f0c868; --no-bg:#25252a; --no-fg:#a3a3aa; --unk-bg:#2a2340;
  --unk-fg:#c3b0ff; --avoid-bg:#3d1e1b; --avoid-fg:#ff9d91; --this-row:#1b2538; } }
:root[data-theme="dark"] { --bg:#141416; --card:#1c1c20; --fg:#ececef; --muted:#a0a0a8; --border:#2f2f35; --code:#26262b;
  --accent:#8db4ff; --accent-soft:#1d2a44; --yes-bg:#173322; --yes-fg:#86dba6; --part-bg:#3a2d0f; --part-fg:#f0c868;
  --no-bg:#25252a; --no-fg:#a3a3aa; --unk-bg:#2a2340; --unk-fg:#c3b0ff; --avoid-bg:#3d1e1b; --avoid-fg:#ff9d91; --this-row:#1b2538; }
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body { margin:0; background:var(--bg); color:var(--fg); font:15.5px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:1040px; margin:0 auto; padding:28px 16px 80px; }
h1 { font-size:28px; line-height:1.25; margin:0 0 6px; }
h2 { font-size:21px; margin:44px 0 10px; padding-top:6px; border-top:1px solid var(--border); }
h3 { font-size:17px; margin:30px 0 6px; }
p { margin:8px 0; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
.lede { color:var(--muted); font-size:16.5px; margin:0 0 14px; }
.meta { color:var(--muted); font-size:13.5px; }
nav.toc { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:10px 16px; margin:16px 0; }
nav.toc ol { margin:4px 0; padding-left:22px; columns:2; column-gap:28px; }
nav.toc li { margin:2px 0; break-inside:avoid; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px 18px; }
.tldr li { margin:7px 0; }
.cite { font-size:0.93em; white-space:nowrap; }
code { font:13px ui-monospace,Consolas,monospace; background:var(--code); padding:1px 5px; border-radius:5px; overflow-wrap:anywhere; }
.theme-intro { color:var(--muted); margin:4px 0 12px; }
.refs { display:grid; gap:10px; }
article.ref { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:11px 15px; scroll-margin-top:12px; }
article.ref:target { outline:2px solid var(--accent); }
article.ref h4 { margin:0 0 2px; font-size:15.5px; line-height:1.4; }
article.ref .who { font-size:13.5px; color:var(--muted); }
article.ref .sum { margin:6px 0 4px; }
article.ref .rel { margin:4px 0; padding:6px 10px; background:var(--accent-soft); border-radius:7px; font-size:14.5px; }
article.ref .foot { font-size:12.5px; color:var(--muted); margin-top:4px; }
.table-wrap { overflow-x:auto; border:1px solid var(--border); border-radius:10px; background:var(--card); -webkit-overflow-scrolling:touch; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th, td { text-align:left; vertical-align:top; padding:7px 9px; border-bottom:1px solid var(--border); }
th { font-size:11.5px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); background:var(--card); }
table.pos { min-width:980px; }
table.pos td:first-child { font-weight:600; min-width:150px; }
table.pos td .note { display:block; color:var(--muted); font-size:12px; line-height:1.35; margin-top:2px; }
tr.this td { background:var(--this-row); }
.pill { display:inline-block; font-size:11.5px; font-weight:600; padding:0 7px; border-radius:20px; line-height:1.7; }
.pill.yes { background:var(--yes-bg); color:var(--yes-fg); } .pill.partial { background:var(--part-bg); color:var(--part-fg); }
.pill.no { background:var(--no-bg); color:var(--no-fg); } .pill.unknown, .pill.na { background:var(--unk-bg); color:var(--unk-fg); }
.pill.defend { background:var(--yes-bg); color:var(--yes-fg); } .pill.reword { background:var(--part-bg); color:var(--part-fg); }
.pill.avoid { background:var(--avoid-bg); color:var(--avoid-fg); }
table.claims { min-width:720px; } table.claims td:first-child { width:36%; }
.legend { font-size:13px; color:var(--muted); margin:8px 0; }
.outline li { margin:10px 0; } .outline .hu { color:var(--muted); font-style:italic; }
.outline .refs-line { font-size:13.5px; }
.risk { margin:10px 0; } .risk h4 { margin:0 0 4px; font-size:15.5px; }
.risk p { margin:4px 0; font-size:14.5px; }
ul.tight li { margin:5px 0; }
@media (max-width: 640px) { body { font-size:15px; } h1 { font-size:23px; } nav.toc ol { columns:1; }
  article.ref { padding:10px 12px; } }
"""

parts = []
add = parts.append
add('<!doctype html><html lang="en"><head><meta charset="utf-8">')
add('<meta name="viewport" content="width=device-width, initial-scale=1">')
add('<title>Literature Review Groundwork</title>')
add(f'<style>{css}</style></head><body><main>')
add('<h1>Literature review groundwork</h1>')
add('<p class="lede">Distilling a large language model into small from-scratch students and a maths-tutor agent: '
    'the state of the field, where this thesis sits, an annotated bibliography and a proposed chapter outline.</p>')
add(f'<p class="meta">{len(ENTRIES)} references, each opened and checked on 7-8 October 2026 (arXiv records via the arXiv API; '
    'venues via arXiv comments, Semantic Scholar, ACL Anthology, PMLR and icml.cc; software via GitHub and its docs). '
    'BibTeX: <code>thesis_work/literature.bib</code>. Builds on <code>thesis_work/novelty_check.md</code> and does not repeat it.</p>')
add('<nav class="toc"><b>Contents</b><ol>'
    '<li><a href="#tldr">TL;DR</a></li><li><a href="#claims">Claims check</a></li>'
    '<li><a href="#positioning">Positioning table</a></li><li><a href="#risks">Risks to the claims</a></li>'
    '<li><a href="#outline">Proposed chapter outline</a></li><li><a href="#bibliography">Annotated bibliography</a></li>'
    '<li><a href="#unverified">Unverified and left out</a></li></ol></nav>')

add('<h2 id="tldr">TL;DR</h2><div class="card tldr"><ul>')
for item in TLDR:
    add(f'<li>{link_refs(item)}</li>')
add('</ul></div>')

add('<h2 id="claims">Claims check</h2>')
add('<p>What the thesis can say, what it should reword, and what it must not present as new. References are the ones '
    'a reviewer would cite back.</p>')
add('<div class="table-wrap"><table class="claims"><thead><tr><th>Claim</th><th>Verdict</th><th>Why, and what to cite</th></tr></thead><tbody>')
for claim, verdict, why, keys in CLAIMS:
    add(f'<tr><td>{html.escape(claim)}</td><td><span class="pill {verdict}">{VERDICT_LABEL[verdict]}</span></td>'
        f'<td>{html.escape(why)}<br><span class="meta">{ref_list(keys)}</span></td></tr>')
add('</tbody></table></div>')

add('<h2 id="positioning">Positioning table</h2>')
add('<p>The closest frameworks and methods against the features this thesis combines. "Partly" and "not stated" are '
    'deliberate: they mark where a paper or tool does something nearby, or where the source does not say. The last row '
    'is this thesis, followed by what it does not do.</p>')
add('<p class="legend"><span class="pill yes">yes</span> <span class="pill partial">partly</span> '
    '<span class="pill no">no</span> <span class="pill unknown">not stated</span> <span class="pill na">n/a</span> '
    '&nbsp;Scroll sideways on a phone.</p>')
add('<div class="table-wrap"><table class="pos"><thead><tr><th>Work</th>' + ''.join(f'<th>{html.escape(c)}</th>' for c in COLUMNS) + '</tr></thead><tbody>')
for label, keys, cells in POSITIONING:
    row_class = ' class="this"' if not keys else ''
    label_html = html.escape(label) + (f'<span class="note">{ref_list(keys)}</span>' if keys else '')
    add(f'<tr{row_class}><td>{label_html}</td>')
    for status, note in cells:
        note_html = f'<span class="note">{html.escape(note)}</span>' if note else ''
        add(f'<td><span class="pill {status}">{STATUS_LABEL[status]}</span>{note_html}</td>')
    add('</tr>')
add('</tbody></table></div>')
add('<h3>What this thesis does not do</h3><ul class="tight">')
for item in NOT_DONE:
    add(f'<li>{link_refs(item)}</li>')
add('</ul>')

add('<h2 id="risks">Risks to the claims</h2><p>The sharpest objections the literature supports, each with a mitigation.</p>')
for number, (title, problem, fix) in enumerate(RISKS, 1):
    add(f'<div class="card risk"><h4>{number}. {html.escape(title)}</h4><p>{link_refs(html.escape(problem, quote=False))}</p>'
        f'<p><b>Mitigation:</b> {link_refs(html.escape(fix, quote=False))}</p></div>')

add('<h2 id="outline">Proposed outline of the literature review chapter</h2>')
add('<p>Section numbers assume the review is chapter 2. Hungarian working titles are given for the szakdolgozat. Each '
    'section lists the references it would use; the order follows the topic description: distillation and small-model '
    'training, then agents, then the available tools and the tool choice.</p><ol class="outline">')
for number, title, hungarian, scope, keys in OUTLINE:
    add(f'<li value="{number.split(".")[1]}"><b>{number} {html.escape(title)}</b> <span class="hu">({html.escape(hungarian)})</span><br>'
        f'{html.escape(scope)}<br><span class="refs-line">{ref_list(keys)}</span></li>')
add('</ol>')

add('<h2 id="bibliography">Annotated bibliography</h2>')
for code, name, intro in THEMES:
    theme_entries = [e for e in ENTRIES if e['theme'] == code]
    add(f'<h3 id="theme-{code}">{code}. {html.escape(name)} <span class="meta">({len(theme_entries)})</span></h3>')
    add(f'<p class="theme-intro">{html.escape(intro)}</p><div class="refs">')
    for entry in theme_entries:
        links = ' &middot; '.join(f'<a href="{html.escape(url)}">{html.escape(label)}</a>' for label, url in entry['links'])
        add(f'<article class="ref" id="{entry["key"]}"><h4><a href="{html.escape(entry["links"][0][1])}">{html.escape(entry["title"])}</a></h4>'
            f'<div class="who">{html.escape(display_authors(entry))} &middot; {entry["year"]} &middot; {html.escape(entry["venue"])}</div>'
            f'<p class="sum">{html.escape(entry["summary"])}</p>'
            f'<p class="rel"><b>Relevance:</b> {html.escape(entry["relevance"])}</p>'
            f'<div class="foot">BibTeX <code>{entry["key"]}</code> &middot; {links} &middot; checked via {html.escape(entry["verified"])}</div></article>')
    add('</div>')

add('<h2 id="unverified">Unverified and left out</h2><p>Nothing below is cited as fact on this page without the caveat '
    'shown. The last item lists papers that were opened and checked but left out to keep the list focused.</p><ul class="tight">')
for item in UNVERIFIED[:-1]:
    add(f'<li>{html.escape(item)}</li>')
add('</ul><h3>Checked, not included</h3>')
add(f'<p class="meta">{html.escape(UNVERIFIED[-1])}</p>')
add('<p class="meta" style="margin-top:30px">Generated locally from one source file for both this page and the BibTeX; '
    'no external scripts or fonts.</p>')
add('</main></body></html>')

with open(html_out, 'x' if '--allow-overwrite' not in sys.argv else 'w', encoding='utf-8') as handle:
    handle.write('\n'.join(parts))
print('entries', len(ENTRIES), 'themes', len(THEMES))
