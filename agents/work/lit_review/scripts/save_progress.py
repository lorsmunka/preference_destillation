import sys, json, datetime
sys.path.insert(0, 'scripts')
from entries import ENTRIES, THEMES
status_text = open(sys.argv[1], encoding='utf-8').read()
target = r'D:\preference_destillation\thesis_work\agent_saves\literature_review.md'
meta = {}
for name in ['data/meta1.json', 'data/meta2.json', 'data/meta3.json']:
    meta.update(json.load(open(name, encoding='utf-8')))
def short_authors(entry):
    names = entry.get('authors') or meta[entry['arxiv']]['authors']
    names = [n.strip('{}') for n in names]
    return ', '.join(names[:3]) + (' et al.' if len(names) > 3 or entry.get('bib', {}).get('author_others') else '')
lines = ['', f"## Save {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", '', status_text.strip(), '']
if len(sys.argv) > 2 and sys.argv[2] == 'with-refs':
    lines.append(f'### Verified references ({len(ENTRIES)})')
    theme_names = dict((code, name) for code, name, _ in THEMES)
    for code, name, _ in THEMES:
        lines.append(f'#### {code}. {name}')
        for entry in [e for e in ENTRIES if e['theme'] == code]:
            title = entry.get('title') or meta[entry['arxiv']]['title']
            link = entry['links'][0][1] if entry.get('links') else f"https://arxiv.org/abs/{entry['arxiv']}"
            lines.append(f"- [{entry['key']}] {short_authors(entry)} ({entry['year']}). {title}. {entry['venue']}. {link} -- {entry['relevance']}")
        lines.append('')
with open(target, 'a', encoding='utf-8') as handle:
    handle.write('\n'.join(lines) + '\n')
print('appended', len(lines), 'lines')
