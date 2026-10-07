import sys, json, time, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding='utf-8')
titles = [line.strip() for line in open(sys.argv[1], encoding='utf-8') if line.strip()]
for title in titles:
    url = 'https://dblp.org/search/publ/api?format=json&h=12&q=' + urllib.parse.quote(title)
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=60).read())
    except Exception as error:
        print('ERR', title, error); time.sleep(3); continue
    hits = data.get('result', {}).get('hits', {}).get('hit', [])
    rows = []
    for hit in hits:
        info = hit['info']
        venue = info.get('venue', '')
        if isinstance(venue, list): venue = '/'.join(venue)
        rows.append(f"{info.get('year')} {venue} [{info.get('type','')}] {info.get('title','')[:70]} {info.get('ee','') if isinstance(info.get('ee',''), str) else info.get('ee')[0]}")
    print('##', title)
    for row in rows[:6]: print('   ', row)
    time.sleep(1.5)
