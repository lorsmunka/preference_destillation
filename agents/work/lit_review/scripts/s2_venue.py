import sys, json, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
ids = []
for name in sys.argv[1:-1]:
    ids += [line.split('#')[0].strip() for line in open(name) if line.split('#')[0].strip()]
body = json.dumps({'ids': ['arXiv:' + i for i in ids]}).encode()
request = urllib.request.Request('https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,venue,year,publicationVenue,externalIds,publicationTypes,journal', data=body, headers={'Content-Type': 'application/json'})
import time
for attempt in range(8):
    try:
        data = json.loads(urllib.request.urlopen(request, timeout=120).read()); break
    except Exception as error:
        print('retry', attempt, error); time.sleep(10 + 10 * attempt)
out = {}
for arxiv_id, paper in zip(ids, data):
    out[arxiv_id] = paper
json.dump(out, open(sys.argv[-1], 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
for arxiv_id, paper in out.items():
    if not paper: print(arxiv_id, 'NONE'); continue
    pv = (paper.get('publicationVenue') or {}).get('name', '')
    j = paper.get('journal') or {}
    print(arxiv_id, '|', paper.get('venue'), '|', pv, '|', j.get('name',''), j.get('volume',''), j.get('pages',''), '|', paper.get('externalIds',{}).get('DOI',''), '|', paper.get('externalIds',{}).get('ACL',''))
