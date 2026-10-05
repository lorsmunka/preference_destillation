export const meta = {
  name: 'math-tutor-write-c',
  description: 'Math tutor writing run C: the 72 writer and harvester tasks that runs A and B lost to the usage limit, in waves of 10',
  phases: [
    { title: 'Waves', detail: '53 memory batches, 17 seeded batches, 2 harvesters; 10 at a time' },
  ],
}

const REPO = 'D:/preference_destillation'
const BRIEF = `${REPO}/library/domain/math_tutor/writer_brief.md`
const RAW = `${REPO}/library/domain/math_tutor/items/raw`
const HARVESTED = `${REPO}/library/domain/math_tutor/benchmarks/harvested`
const SEEDS = `${REPO}/library/domain/math_tutor/benchmarks/seeds.jsonl`
const SEED_COUNT = 1310
const PER_WRITER = 100

const RESULT = {
  type: 'object',
  properties: { file: { type: 'string' }, written: { type: 'integer' }, valid: { type: 'integer' }, notes: { type: 'string' } },
  required: ['file', 'written', 'valid', 'notes'],
}

const COMMON_RULES = `Rules: work only in ${REPO} (Windows; the Bash tool is Git Bash, use forward slashes). Create only the one file you are given (check first that it does not exist; if it does, stop and report). Create it with the Write tool (a Bash heredoc of this size fails). Do not modify, delete or read any other data file; do not run git; do not upload or post anything anywhere (searching and reading web pages is fine). Return the file path, how many records you wrote, how many pass the check, and short notes on anything hard or unclear.`

const STYLES = [
  'short and direct: mostly 1-2 sentences',
  'small stories: 3-4 sentences with a setting and a reason for what happens',
  'question first or in the middle, facts after it',
  'measurement: lengths, weights, volumes, minutes, pages, steps, laps (no unit conversions)',
  'money: prices, savings, change, tickets, fundraising (whole amounts only)',
  'time and plans: days, mornings, schedules, before and after',
  'everyday talk: what someone says, asks, plans or notices',
  'school textbook register: "A class collected...", "Find the total...", "Work out..."',
]
const CONTEXTS = ['bakery', 'farm', 'school library', 'soccer', 'basketball', 'swimming pool', 'zoo', 'aquarium', 'museum', 'bus', 'train station', 'airport', 'birthday party', 'garden', 'orchard', 'camping', 'hiking trail', 'beach', 'river', 'fishing trip', 'music class', 'art class', 'science fair', 'pet shop', 'dog park', 'toy store', 'bookstore', 'grocery store', 'farmers market', 'kitchen', 'restaurant', 'cafe', 'pizza shop', 'ice cream stand', 'lemonade stand', 'car wash fundraiser', 'school play', 'concert', 'cinema', 'playground', 'skate park', 'snow day', 'rainy day', 'space club', 'robotics club', 'chess club', 'scouts', 'recycling drive', 'food bank', 'post office', 'clinic', 'fire station', 'building site', 'bike shop', 'flower shop', 'pumpkin patch', 'county fair', 'sports day', 'sewing club', 'building blocks', 'video games', 'board games', 'puzzles', 'carnival', 'parade', 'festival', 'harbor', 'lighthouse', 'mountain cabin', 'jungle trip', 'desert trip', 'bee farm', 'chicken coop', 'horse stable', 'vegetable patch', 'greenhouse', 'sticker collection', 'stamp collection', 'kite day', 'theater']

// ---- harvesters (same prompts as run A) ----
const HARVESTERS = [
  { name: 'uk-naep', sources: 'UK Standards and Testing Agency key stage 1 and key stage 2 mathematics past and sample papers on gov.uk (Open Government Licence v3.0; the reasoning papers have word problems), and NAEP grade 4 mathematics released questions (nces.ed.gov NAEP Questions Tool; US federal government work, public domain).' },
  { name: 'other-oer', sources: 'Other openly licensed sources: nzmaths.co.nz (check its licence), Open Up Resources K-5 (CC BY 4.0), Achieve the Core tasks (check licence), Wikibooks / Wikiversity primary maths (CC BY-SA), OER Commons resources with an explicit CC licence, and released state test items that are explicitly public domain or openly licensed (verify on the page).' },
]
function harvestPrompt(h) {
  return `You are harvesting real maths word problems for a university thesis project.

Goal: collect real grade 2-4 arithmetic word problems, copied word for word, from openly licensed or public-domain sources, so they can be used as human-written seed problems. Each must come with a Python solve() function that computes its answer.

Your sources: ${h.sources}

Licences: use a source only if the page or site states an open licence (CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA, Open Government Licence) or it is a US federal government work (public domain). Record the licence string as the site states it. If the licence is unclear, skip that source. NEVER use: k5learning.com, commoncoresheets.com, mathplayground.com, ck12.org, ixl.com, mathocean.com, ask-math.com, dadsworksheets.com, khanacademy.org, math-aids.com, mathworksheets4kids.com, or any copy of the SVAMP, ASDiv, MAWPS or GSM8K datasets.

Keep only problems that fit: a word problem (a short story with a question, not a bare sum), solvable with 1 to 3 operations of + - * and division, whole numbers only, every number in the text at most 99, answer at most 999, no fractions, decimals, measurement conversions, pictures, tables or multiple-choice options needed to solve it. If a problem has several parts, take a part that stands alone and copy the text it needs.

One JSON object per line:
{"id": "hv-${h.name}-0001", "problem": "<the text exactly as on the page, whitespace tidied>", "solve": "def solve():\\n    apples = 7\\n    eaten = 3\\n    return {'answer': apples - eaten}", "stated_answer": 4, "url": "<page url>", "site": "<domain>", "licence": "<licence>", "grade": 3}
- stated_answer: the answer the source gives (answer key), else null. grade: from the source, else null.
- solve() rules (checked by machine): only assignments, + - * // %, whole-number literals and one final return of a dict with 'answer'; no calls, loops, if, imports, floats; every value 0..999; every number in the code must appear in the text (digits or words such as "seven", "a dozen", "twice", "half"; 0 and 1 always allowed).

Aim for 150 to 400 problems. Write them to the NEW file ${HARVESTED}/${h.name}.jsonl. Large files: write the first part with the Write tool, then append further parts with the Edit tool. When done, run from ${REPO}: python -m library.domain.math_tutor.harvest library/domain/math_tutor/benchmarks/harvested/${h.name}.jsonl and fix or remove every failing record.

${COMMON_RULES}`
}

// ---- memory writers (same prompts as run A) ----
const SLICES = {
  S04: { what: 'compare, additive: more / fewer than', unknowns: 'difference 34, larger 33, smaller 33', grades: 'grade 2 for 50, grade 3 for 50', keyword: true, index: 3 },
  S05: { what: 'equal groups: n groups of k', unknowns: 'product 100 (groups, rows, packs, arrays, repeated addition)', grades: 'grade 3 for 70, grade 4 for 30', keyword: false, index: 4 },
  S06: { what: 'partitive division: share equally', unknowns: 'group size 100', grades: 'grade 3 for 70, grade 4 for 30', keyword: false, index: 5 },
  S07: { what: 'quotative division: how many groups, leftovers, round up', unknowns: 'group count 34, remainder 33, round up 33', grades: 'grade 3 for 50, grade 4 for 50', keyword: false, index: 6 },
  S08: { what: 'multiplicative compare: 3 times as many, a third as many', unknowns: 'larger 34, smaller 33, times 33', grades: 'grade 3 for 50, grade 4 for 50', keyword: true, index: 7 },
  S09: { what: 'rates and money: price x quantity, change from a note, per hour', unknowns: 'total 34, change 33, rate 33', grades: 'grade 3 for 50, grade 4 for 50', keyword: false, index: 8 },
  S10: { what: 'multi-step: 2-3 operations mixing the above', unknowns: 'result 100', grades: 'grade 4 for 70, grade 3 for 30', keyword: false, index: 9 },
}
function memoryContexts(sliceIndex, batch) {
  const picked = []
  for (let i = 0; i < 12; i++) picked.push(CONTEXTS[(sliceIndex * 29 + batch * 12 + i) % CONTEXTS.length])
  return picked.join(', ')
}
function writerPrompt(id, batch) {
  const slice = SLICES[id]
  const tag = String(batch + 1).padStart(2, '0')
  const file = `${RAW}/${id}_memory_b${tag}.jsonl`
  const distractors = slice.keyword ? 'none 50, number 20, sentence 15, keyword 15' : 'none 65, number 20, sentence 15 (keyword does not fit this slice)'
  return `You are writing training data for a university thesis project. First read the brief and follow it exactly: ${BRIEF}

Your assignment:
- Source: memory. Put "source": "memory" on every item.
- Slice: ${id} (${slice.what}). Every item is slice ${id}.
- Write 100 items to the NEW file ${file}. ids: ${id}-mem-b${tag}-0001 ... ${id}-mem-b${tag}-0100.
- Mix over the 100 items: unknown: ${slice.unknowns}. ${slice.grades}. distractor: ${distractors}. follow-ups: 0 on 20 items, 1 on 35, 2 on 30, 3 on 15.
- Contexts for this batch (use all of them, spread evenly; you may add a few more): ${memoryContexts(slice.index, batch)}.
- Style emphasis for this batch (most items, not all): ${STYLES[batch % STYLES.length]}.
- Other writers cover the same slice with other contexts and styles; make your 100 items different from each other: no two may share a sentence frame with only nouns or numbers swapped.

When done, run from ${REPO}: python -m library.domain.math_tutor.items library/domain/math_tutor/items/raw/${id}_memory_b${tag}.jsonl and fix every failing item in your file until all pass (or explain any you could not fix).

${COMMON_RULES}`
}

// ---- seeded writers (same prompts as run B) ----
function seededContexts(pass, chunk) {
  const picked = []
  for (let i = 0; i < 15; i++) picked.push(CONTEXTS[(pass * 41 + chunk * 15 + i) % CONTEXTS.length])
  return picked.join(', ')
}
function seededPrompt(pass, chunk) {
  const first = chunk * PER_WRITER + 1
  const last = Math.min(SEED_COUNT, first + PER_WRITER - 1)
  const tag = `p${pass}-b${String(chunk + 1).padStart(2, '0')}`
  const fileName = `seeded_${tag.replace('-', '_')}.jsonl`
  const count = last - first + 1
  const passNote = pass === 1 ? '' : '\n- This is the second pass over these seeds: another writer already reworked each of them once. Avoid the most obvious rewrite; pick a less expected setting and sentence structure.'
  return `You are writing training data for a university thesis project. First read the brief and follow it exactly: ${BRIEF}

Your assignment:
- Source: seeded. Your seeds are lines ${first} to ${last} of ${SEEDS} (one JSON object per line: id, problem, solve, answer, shape). Read exactly those lines (e.g. the Read tool with offset ${first} and limit ${count}; check the first and last seed you got are lines ${first} and ${last}).
- For EACH seed write exactly one new item with the same mathematical structure (the same operations in the same order, as in its "shape"), but a different story, setting, names and wording, and new numbers. Choose the slice (S01-S10), unknown, grade (2-4), distractor and context that fit. Put "source": "seeded:<seed id>".
- Write the ${count} items, in seed order, to the NEW file ${RAW}/${fileName}. ids: sd-${tag}-0001 ... sd-${tag}-${String(count).padStart(4, '0')}.
- follow-ups: about 20% with 0, 35% with 1, 30% with 2, 15% with 3, spread over the slices.
- distractor: if the seed has a number its solution does not use, keep one irrelevant number (in digits) and label it "number"; otherwise mostly "none", with about 1 in 10 "sentence".
- Contexts to draw from (use many of them): ${seededContexts(pass, chunk)}.
- Style emphasis (most items, not all): ${STYLES[(chunk + pass * 3) % STYLES.length]}.
- It must not read like the seed with words swapped: new setting, new sentence structure.${passNote}

When done, run from ${REPO}: python -m library.domain.math_tutor.items library/domain/math_tutor/items/raw/${fileName} and fix every failing item in your file until all pass (or explain any you could not fix).

Rules: work only in ${REPO} (Windows; the Bash tool is Git Bash, use forward slashes). Create only the one file you are given (check first that it does not exist; if it does, stop and report). Create it with the Write tool (a Bash heredoc of this size fails). Do not modify, delete or read any other data file besides your seed lines; do not run git; do not upload or post anything anywhere. Return the file path, how many items you wrote, how many pass the check, and short notes on anything hard or unclear.`
}

// ---- the missing work ----
const tasks = []
for (const h of HARVESTERS) tasks.push({ label: `harvest:${h.name}`, prompt: harvestPrompt(h) })
for (const batch of [2, 4, 5, 6, 7]) tasks.push({ label: `write:S04-b${batch + 1}`, prompt: writerPrompt('S04', batch) })
for (const id of ['S05', 'S06', 'S07', 'S08', 'S09', 'S10']) {
  for (let batch = 0; batch < 8; batch++) tasks.push({ label: `write:${id}-b${batch + 1}`, prompt: writerPrompt(id, batch) })
}
for (const chunk of [5, 7, 8, 9, 12]) tasks.push({ label: `seeded:p1-b${chunk + 1}`, prompt: seededPrompt(1, chunk) })
for (const chunk of [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) tasks.push({ label: `seeded:p2-b${chunk + 1}`, prompt: seededPrompt(2, chunk) })

log(`${tasks.length} tasks in waves of 10`)
const results = []
for (let start = 0; start < tasks.length; start += 10) {
  const wave = tasks.slice(start, start + 10)
  const done = await parallel(wave.map(task => () => agent(task.prompt, { label: task.label, phase: 'Waves', schema: RESULT, agentType: 'general-purpose' })))
  done.forEach((result, index) => results.push({ label: wave[index].label, result }))
  const ok = done.filter(Boolean).length
  log(`wave ${start / 10 + 1}: ${ok}/${wave.length} returned`)
}
const returned = results.filter(r => r.result)
return {
  returned: returned.length,
  failed: results.filter(r => !r.result).map(r => r.label),
  written: returned.reduce((s, r) => s + r.result.written, 0),
  valid: returned.reduce((s, r) => s + r.result.valid, 0),
  results: returned.map(r => ({ label: r.label, ...r.result })),
}