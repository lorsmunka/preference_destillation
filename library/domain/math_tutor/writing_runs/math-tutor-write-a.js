export const meta = {
  name: 'math-tutor-write-a',
  description: 'Math tutor writing run A: 4 harvesters collect openly licensed human problems, 80 memory writers write 100 items each across 10 slices',
  phases: [
    { title: 'Harvest', detail: 'copy real grade 2-4 problems word for word from openly licensed sources, with solve()' },
    { title: 'Write', detail: '10 slices x 8 batches x 100 items, from memory, each batch its own contexts and style' },
  ],
}

const REPO = 'D:/preference_destillation'
const BRIEF = `${REPO}/library/domain/math_tutor/writer_brief.md`
const RAW = `${REPO}/library/domain/math_tutor/items/raw`
const HARVESTED = `${REPO}/library/domain/math_tutor/benchmarks/harvested`

const RESULT = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    written: { type: 'integer' },
    valid: { type: 'integer' },
    notes: { type: 'string' },
  },
  required: ['file', 'written', 'valid', 'notes'],
}

const COMMON_RULES = `Rules: work only in ${REPO} (Windows; the Bash tool is Git Bash, use forward slashes). Create only the one file you are given (check first that it does not exist; if it does, stop and report). Create it with the Write tool (a Bash heredoc of this size fails). Do not modify, delete or read any other data file; do not run git; do not upload or post anything anywhere (searching and reading web pages is fine). Return the file path, how many records you wrote, how many pass the check, and short notes on anything hard or unclear.`

const HARVESTERS = [
  { name: 'im', sources: 'Illustrative Mathematics K-5 (curriculum.illustrativemathematics.org/k5, also im.kendallhunt.com/k5): grade 2, 3 and 4 practice problems, cool-downs and activities; and tasks.illustrativemathematics.org grade 2-4 tasks (OA, NBT, MD). Licence: CC BY 4.0.' },
  { name: 'engageny', sources: 'EngageNY / Eureka Math grade 2, 3 and 4 mathematics modules (engageny.org, nysed.gov): Application Problems, Problem Sets, Homework and assessments, often in PDFs. Licence: CC BY-NC-SA 3.0 US.' },
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

Aim for 150 to 400 problems. Write them to the NEW file ${HARVESTED}/${h.name}.jsonl (create the folder if needed). When done, run from ${REPO}: python -m library.domain.math_tutor.harvest library/domain/math_tutor/benchmarks/harvested/${h.name}.jsonl and fix or remove every failing record.

${COMMON_RULES}`
}

const SLICES = [
  { id: 'S01', what: 'change, join: some more arrive', unknowns: 'result 34, change 33, start 33', grades: 'grade 2 for 70, grade 3 for 30', keyword: true },
  { id: 'S02', what: 'change, separate: some leave', unknowns: 'result 34, change 33, start 33', grades: 'grade 2 for 70, grade 3 for 30', keyword: true },
  { id: 'S03', what: 'combine: two parts, one whole', unknowns: 'whole 50, part 50', grades: 'grade 2 for 70, grade 3 for 30', keyword: true },
  { id: 'S04', what: 'compare, additive: more / fewer than', unknowns: 'difference 34, larger 33, smaller 33', grades: 'grade 2 for 50, grade 3 for 50', keyword: true },
  { id: 'S05', what: 'equal groups: n groups of k', unknowns: 'product 100 (groups, rows, packs, arrays, repeated addition)', grades: 'grade 3 for 70, grade 4 for 30', keyword: false },
  { id: 'S06', what: 'partitive division: share equally', unknowns: 'group size 100', grades: 'grade 3 for 70, grade 4 for 30', keyword: false },
  { id: 'S07', what: 'quotative division: how many groups, leftovers, round up', unknowns: 'group count 34, remainder 33, round up 33', grades: 'grade 3 for 50, grade 4 for 50', keyword: false },
  { id: 'S08', what: 'multiplicative compare: 3 times as many, a third as many', unknowns: 'larger 34, smaller 33, times 33', grades: 'grade 3 for 50, grade 4 for 50', keyword: true },
  { id: 'S09', what: 'rates and money: price x quantity, change from a note, per hour', unknowns: 'total 34, change 33, rate 33', grades: 'grade 3 for 50, grade 4 for 50', keyword: false },
  { id: 'S10', what: 'multi-step: 2-3 operations mixing the above', unknowns: 'result 100', grades: 'grade 4 for 70, grade 3 for 30', keyword: false },
]

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

function contextsFor(sliceIndex, batch) {
  const picked = []
  for (let i = 0; i < 12; i++) picked.push(CONTEXTS[(sliceIndex * 29 + batch * 12 + i) % CONTEXTS.length])
  return picked.join(', ')
}

function writerPrompt(slice, sliceIndex, batch) {
  const tag = String(batch + 1).padStart(2, '0')
  const file = `${RAW}/${slice.id}_memory_b${tag}.jsonl`
  const distractors = slice.keyword ? 'none 50, number 20, sentence 15, keyword 15' : 'none 65, number 20, sentence 15 (keyword does not fit this slice)'
  return `You are writing training data for a university thesis project. First read the brief and follow it exactly: ${BRIEF}

Your assignment:
- Source: memory. Put "source": "memory" on every item.
- Slice: ${slice.id} (${slice.what}). Every item is slice ${slice.id}.
- Write 100 items to the NEW file ${file}. ids: ${slice.id}-mem-b${tag}-0001 ... ${slice.id}-mem-b${tag}-0100.
- Mix over the 100 items: unknown: ${slice.unknowns}. ${slice.grades}. distractor: ${distractors}. follow-ups: 0 on 20 items, 1 on 35, 2 on 30, 3 on 15.
- Contexts for this batch (use all of them, spread evenly; you may add a few more): ${contextsFor(sliceIndex, batch)}.
- Style emphasis for this batch (most items, not all): ${STYLES[batch % STYLES.length]}.
- Other writers cover the same slice with other contexts and styles; make your 100 items different from each other: no two may share a sentence frame with only nouns or numbers swapped.

When done, run from ${REPO}: python -m library.domain.math_tutor.items library/domain/math_tutor/items/raw/${slice.id}_memory_b${tag}.jsonl and fix every failing item in your file until all pass (or explain any you could not fix).

${COMMON_RULES}`
}

const tasks = []
for (const h of HARVESTERS) {
  tasks.push(() => agent(harvestPrompt(h), { label: `harvest:${h.name}`, phase: 'Harvest', schema: RESULT, agentType: 'general-purpose' }))
}
SLICES.forEach((slice, sliceIndex) => {
  for (let batch = 0; batch < 8; batch++) {
    tasks.push(() => agent(writerPrompt(slice, sliceIndex, batch), { label: `write:${slice.id}-b${batch + 1}`, phase: 'Write', schema: RESULT, agentType: 'general-purpose' }))
  }
})
log(`${tasks.length} agents: ${HARVESTERS.length} harvesters, ${tasks.length - HARVESTERS.length} memory writers`)
const results = await parallel(tasks)
const done = results.filter(Boolean)
const written = done.reduce((sum, r) => sum + r.written, 0)
const valid = done.reduce((sum, r) => sum + r.valid, 0)
log(`${done.length}/${tasks.length} agents returned; ${written} records written, ${valid} pass their checks`)
return { returned: done.length, failed: tasks.length - done.length, written, valid, results: done }