import sys, json, time, urllib.request, xml.etree.ElementTree as ET
ids_file, out_file = sys.argv[1], sys.argv[2]
ids = [line.split('#')[0].strip() for line in open(ids_file) if line.split('#')[0].strip()]
ns = {'a': 'http://www.w3.org/2005/Atom', 'x': 'http://arxiv.org/schemas/atom'}
results = {}
for start in range(0, len(ids), 20):
    chunk = ids[start:start+20]
    url = 'https://export.arxiv.org/api/query?max_results=40&id_list=' + ','.join(chunk)
    data = urllib.request.urlopen(url, timeout=90).read()
    root = ET.fromstring(data)
    for entry in root.findall('a:entry', ns):
        arxiv_id = entry.find('a:id', ns).text.rsplit('/abs/', 1)[-1]
        base = arxiv_id.rsplit('v', 1)[0]
        def text(path):
            node = entry.find(path, ns)
            return ' '.join(node.text.split()) if node is not None and node.text else ''
        results[base] = {
            'id': arxiv_id,
            'title': text('a:title'),
            'authors': [ ' '.join(a.find('a:name', ns).text.split()) for a in entry.findall('a:author', ns)],
            'published': text('a:published')[:10],
            'updated': text('a:updated')[:10],
            'journal_ref': text('x:journal_ref'),
            'comment': text('x:comment'),
            'doi': text('x:doi'),
            'summary': text('a:summary'),
        }
    time.sleep(3)
json.dump(results, open(out_file, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
missing = [i for i in ids if i not in results]
print('fetched', len(results), 'missing', missing)
