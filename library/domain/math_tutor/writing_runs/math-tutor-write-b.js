export const meta = {
  name: 'math-tutor-write-b',
  description: 'Math tutor writing run B: seeded writers rework each of the 1,310 human seeds twice (two passes, different contexts and styles)',
  phases: [
    { title: 'Seeded pass 1', detail: '14 writers x up to 100 seeds, one new item per seed' },
    { title: 'Seeded pass 2', detail: 'the same seeds again with other contexts and styles' },
  ],
}

const REPO = 'D:/preference_destillation'
const BRIEF = `${REPO}/library/domain/math_tutor/writer_brief.md`
const RAW = `${REPO}/library/domain/math_tutor/items/raw`
const SEEDS = `${REPO}/library/domain/math_tutor/benchmarks/seeds.jsonl`
const SEED_COUNT = 1310
const PER_WRITER = 100

const RESULT = {
  type: 'object',
  properties: { file: { type: 'string' }, written: { type: 'integer' }, valid: { type: 'integer' }, notes: { type: 'string' } },
  required: ['file', 'written', 'valid', 'notes'],
}

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

function contextsFor(pass, chunk) {
  const picked = []
  for (let i = 0; i < 15; i++) picked.push(CONTEXTS[(pass * 41 + chunk * 15 + i) % CONTEXTS.length])
  return picked.join(', ')
}

function seededPrompt(pass, chunk) {
  const first = chunk * PER_WRITER + 1
  const last = Math.min(SEED_COUNT, first + PER_WRITER - 1)
  const tag = `p${pass}-b${String(chunk + 1).padStart(2, '0')}`
  const file = `${RAW}/seeded_${tag.replace('-', '_')}.jsonl`
  const count = last - first + 1
  const passNote = pass === 1 ? '' : '\n- This is the second pass over these seeds: another writer already reworked each of them once. Avoid the most obvious rewrite; pick a less expected setting and sentence structure.'
  return `You are writing training data for a university thesis project. First read the brief and follow it exactly: ${BRIEF}

Your assignment:
- Source: seeded. Your seeds are lines ${first} to ${last} of ${SEEDS} (one JSON object per line: id, problem, solve, answer, shape). Read exactly those lines (e.g. the Read tool with offset ${first - 1} and limit ${count}).
- For EACH seed write exactly one new item with the same mathematical structure (the same operations in the same order, as in its "shape"), but a different story, setting, names and wording, and new numbers. Choose the slice (S01-S10), unknown, grade (2-4), distractor and context that fit. Put "source": "seeded:<seed id>".
- Write the ${count} items, in seed order, to the NEW file ${file}. ids: sd-${tag}-0001 ... sd-${tag}-${String(count).padStart(4, '0')}.
- follow-ups: about 20% with 0, 35% with 1, 30% with 2, 15% with 3, spread over the slices.
- distractor: if the seed has a number its solution does not use, keep one irrelevant number (in digits) and label it "number"; otherwise mostly "none", with about 1 in 10 "sentence".
- Contexts to draw from (use many of them): ${contextsFor(pass, chunk)}.
- Style emphasis (most items, not all): ${STYLES[(chunk + pass * 3) % STYLES.length]}.
- It must not read like the seed with words swapped: new setting, new sentence structure.${passNote}

When done, run from ${REPO}: python -m library.domain.math_tutor.items library/domain/math_tutor/items/raw/${file.split('/').pop()} and fix every failing item in your file until all pass (or explain any you could not fix).

Rules: work only in ${REPO} (Windows; the Bash tool is Git Bash, use forward slashes). Create only the one file you are given (check first that it does not exist; if it does, stop and report). Create it with the Write tool (a Bash heredoc of this size fails). Do not modify, delete or read any other data file besides your seed lines; do not run git; do not upload or post anything anywhere. Return the file path, how many items you wrote, how many pass the check, and short notes on anything hard or unclear.`
}

const chunks = Math.ceil(SEED_COUNT / PER_WRITER)
const tasks = []
for (const pass of [1, 2]) {
  for (let chunk = 0; chunk < chunks; chunk++) {
    tasks.push(() => agent(seededPrompt(pass, chunk), { label: `seeded:p${pass}-b${chunk + 1}`, phase: `Seeded pass ${pass}`, schema: RESULT, agentType: 'general-purpose' }))
  }
}
log(`${tasks.length} seeded writers over ${SEED_COUNT} seeds, two passes`)
const results = (await parallel(tasks)).filter(Boolean)
const written = results.reduce((s, r) => s + r.written, 0)
const valid = results.reduce((s, r) => s + r.valid, 0)
log(`${results.length}/${tasks.length} returned; ${written} written, ${valid} valid`)
return { returned: results.length, failed: tasks.length - results.length, written, valid, results }