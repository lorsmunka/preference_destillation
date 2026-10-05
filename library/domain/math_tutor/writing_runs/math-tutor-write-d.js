export const meta = {
  name: 'math-tutor-write-d',
  description: 'Math tutor writing run D: the last 32 writer tasks (S09, S10, 17 seeded batches) plus seeded writers on the 843 harvested seeds (2 passes), waves of 10',
  phases: [
    { title: 'Waves', detail: '15 memory batches, 17 seeded batches, 18 harvest-seeded batches; 10 at a time' },
  ],
}

const REPO = 'D:/preference_destillation'
const BRIEF = `${REPO}/library/domain/math_tutor/writer_brief.md`
const RAW = `${REPO}/library/domain/math_tutor/items/raw`
const SEEDS = `${REPO}/library/domain/math_tutor/benchmarks/seeds.jsonl`

const RESULT = {
  type: 'object',
  properties: { file: { type: 'string' }, written: { type: 'integer' }, valid: { type: 'integer' }, notes: { type: 'string' } },
  required: ['file', 'written', 'valid', 'notes'],
}
const COMMON_RULES = `Rules: work only in ${REPO} (Windows; the Bash tool is Git Bash, use forward slashes). Create only the one file you are given (check first that it does not exist; if it does, stop and report). Create it with the Write tool (a Bash heredoc of this size fails). Do not modify, delete or read any other data file; do not run git; do not upload or post anything anywhere. Return the file path, how many records you wrote, how many pass the check, and short notes on anything hard or unclear.`

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

const SLICES = {
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

function seededContexts(pass, key) {
  const picked = []
  for (let i = 0; i < 15; i++) picked.push(CONTEXTS[(pass * 41 + key * 15 + i) % CONTEXTS.length])
  return picked.join(', ')
}
// first/last are 1-based seed line numbers; tag names the file and the ids
function seededPrompt(pass, first, last, tag, styleKey, harvested) {
  const fileName = `seeded_${tag.replace(/-/g, '_')}.jsonl`
  const count = last - first + 1
  const passNote = pass === 1 ? '' : '\n- This is the second pass over these seeds: another writer already reworked each of them once. Avoid the most obvious rewrite; pick a less expected setting and sentence structure.'
  const harvestNote = harvested ? '\n- These seeds were copied from real school materials (US, UK); some use pounds or pence or UK wording. Your items use plain whole numbers and any everyday setting.' : ''
  return `You are writing training data for a university thesis project. First read the brief and follow it exactly: ${BRIEF}

Your assignment:
- Source: seeded. Your seeds are lines ${first} to ${last} of ${SEEDS} (one JSON object per line: id, problem, solve, answer, shape). Read exactly those lines (the Read tool with offset ${first} and limit ${count}; check the first and last seed you got are lines ${first} and ${last}).
- For EACH seed write exactly one new item with the same mathematical structure (the same operations in the same order, as in its "shape"), but a different story, setting, names and wording, and new numbers. Choose the slice (S01-S10), unknown, grade (2-4), distractor and context that fit. Put "source": "seeded:<seed id>".
- Write the ${count} items, in seed order, to the NEW file ${RAW}/${fileName}. ids: sd-${tag}-0001 ... sd-${tag}-${String(count).padStart(4, '0')}.
- follow-ups: about 20% with 0, 35% with 1, 30% with 2, 15% with 3, spread over the slices.
- distractor: if the seed has a number its solution does not use, keep one irrelevant number (in digits) and label it "number"; otherwise mostly "none", with about 1 in 10 "sentence".
- Contexts to draw from (use many of them): ${seededContexts(pass, styleKey)}.
- Style emphasis (most items, not all): ${STYLES[(styleKey + pass * 3) % STYLES.length]}.
- It must not read like the seed with words swapped: new setting, new sentence structure.${passNote}${harvestNote}

When done, run from ${REPO}: python -m library.domain.math_tutor.items library/domain/math_tutor/items/raw/${fileName} and fix every failing item in your file until all pass (or explain any you could not fix).

Rules: work only in ${REPO} (Windows; the Bash tool is Git Bash, use forward slashes). Create only the one file you are given (check first that it does not exist; if it does, stop and report). Create it with the Write tool (a Bash heredoc of this size fails). Do not modify, delete or read any other data file besides your seed lines; do not run git; do not upload or post anything anywhere. Return the file path, how many items you wrote, how many pass the check, and short notes on anything hard or unclear.`
}

const tasks = []
for (let batch = 1; batch < 8; batch++) tasks.push({ label: `write:S09-b${batch + 1}`, prompt: writerPrompt('S09', batch) })
for (let batch = 0; batch < 8; batch++) tasks.push({ label: `write:S10-b${batch + 1}`, prompt: writerPrompt('S10', batch) })
// the original 1,310 seeds: chunk k covers lines 100k+1 .. min(1310, 100k+100)
const chunkRange = k => [k * 100 + 1, Math.min(1310, k * 100 + 100)]
for (const k of [5, 7, 8, 9, 12]) { const [f, l] = chunkRange(k); tasks.push({ label: `seeded:p1-b${k + 1}`, prompt: seededPrompt(1, f, l, `p1-b${String(k + 1).padStart(2, '0')}`, k, false) }) }
for (const k of [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) { const [f, l] = chunkRange(k); tasks.push({ label: `seeded:p2-b${k + 1}`, prompt: seededPrompt(2, f, l, `p2-b${String(k + 1).padStart(2, '0')}`, k, false) }) }
// the 843 harvested seeds: lines 1311..2153, 100 per writer, two passes
for (const pass of [1, 2]) {
  for (let h = 0; h < 9; h++) {
    const first = 1311 + h * 100, last = Math.min(2153, first + 99)
    tasks.push({ label: `seeded:p${pass}-h${h + 1}`, prompt: seededPrompt(pass, first, last, `p${pass}-h${String(h + 1).padStart(2, '0')}`, 20 + h, true) })
  }
}

log(`${tasks.length} tasks in waves of 10`)
const results = []
for (let start = 0; start < tasks.length; start += 10) {
  const wave = tasks.slice(start, start + 10)
  const done = await parallel(wave.map(task => () => agent(task.prompt, { label: task.label, phase: 'Waves', schema: RESULT, agentType: 'general-purpose' })))
  done.forEach((result, index) => results.push({ label: wave[index].label, result }))
  log(`wave ${start / 10 + 1}: ${done.filter(Boolean).length}/${wave.length} returned`)
}
const returned = results.filter(r => r.result)
return {
  returned: returned.length,
  failed: results.filter(r => !r.result).map(r => r.label),
  written: returned.reduce((s, r) => s + r.result.written, 0),
  valid: returned.reduce((s, r) => s + r.result.valid, 0),
  results: returned.map(r => ({ label: r.label, file: r.result.file, written: r.result.written, valid: r.result.valid })),
}