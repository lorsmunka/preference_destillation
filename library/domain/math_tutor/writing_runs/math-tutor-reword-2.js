export const meta = {
  name: 'math-tutor-reword-2',
  description: 'Math tutor reword pass, continued: the rest of chunks 37 and 40 plus chunks 41-73, two rewordings per train item, waves of 10',
  phases: [
    { title: 'Reword', detail: '35 agents: 2 half chunks + 33 full chunks of 150 frozen train items' },
  ],
}

const REPO = 'D:/preference_destillation'
const BRIEF = `${REPO}/library/domain/math_tutor/writer_brief.md`
const TARGETS = `${REPO}/library/domain/math_tutor/items/reword_targets.jsonl`
const OUT = `${REPO}/library/domain/math_tutor/items/reword`
const TOTAL = 10847

const RESULT = {
  type: 'object',
  properties: { file: { type: 'string' }, written: { type: 'integer' }, valid: { type: 'integer' }, notes: { type: 'string' } },
  required: ['file', 'written', 'valid', 'notes'],
}

function rewordPrompt(first, last, tag) {
  const count = last - first + 1
  const file = `${OUT}/rw_b${tag}.jsonl`
  return `You are rewording maths word problems for a university thesis project. First read the brief: ${BRIEF} — the section "Rewording (a separate job)" is your job; the rest explains the items.

Your items: lines ${first} to ${last} of ${TARGETS} (one JSON object per line: id, slice, distractor, problem, solve, follow_ups). Read exactly those lines (the Read tool with offset ${first} and limit ${count}; check the first and last items are lines ${first} and ${last}).

For EACH item write TWO rewordings, ids "<item id>~w1" and "<item id>~w2", "of": "<item id>":
- Same maths: every number the solve() code uses stays in the text with the same role (digits stay digits; a number word stays that word or becomes the same number in digits), a digit distractor stays one unused digit number, a "sentence" distractor keeps one extra sentence without a number.
- Same follow-ups: one new question per follow-up, in order, asking for the same quantity.
- Really different: ~w1 moves the story to a new setting with new objects and names; ~w2 to yet another setting, with a different sentence structure (for example the question first, or the facts in another order). Each must be clearly different from the original AND from the other rewording; the check below fails anything more than 80% the same.
- Natural, grade-appropriate English; the numbers stay, and the new story must make sense with them.

Write the ${count * 2} compact records (see the brief's format: id, of, problem, follow_up_questions) to the NEW file ${file} (check the file does not exist; if it does, stop and report). Write it in parts: the first 100 records with the Write tool, then append the rest with the Edit tool, 100 at a time.

When done, run from ${REPO}: python -m library.domain.math_tutor.reword library/domain/math_tutor/items/reword/rw_b${tag}.jsonl and fix every failing record in your file until all pass (or explain any you could not fix).

Rules: work only in ${REPO} (Windows; the Bash tool is Git Bash, use forward slashes). Create only that one file. Do not modify, delete or read any other data file besides your target lines; do not run git; do not upload or post anything anywhere. Return the file path, how many records you wrote, how many pass the check, and short notes on anything hard.`
}

const tasks = [
  { label: 'reword:b37b', prompt: rewordPrompt(5451, 5550, '037b') },
  { label: 'reword:b40b', prompt: rewordPrompt(5901, 6000, '040b') },
]
for (let chunk = 40; chunk < 73; chunk++) {
  const first = chunk * 150 + 1
  const last = Math.min(TOTAL, first + 149)
  tasks.push({ label: `reword:b${chunk + 1}`, prompt: rewordPrompt(first, last, String(chunk + 1).padStart(3, '0')) })
}
log(`${tasks.length} rewording agents, waves of 10`)
const results = []
for (let start = 0; start < tasks.length; start += 10) {
  const wave = tasks.slice(start, start + 10)
  const done = await parallel(wave.map(task => () => agent(task.prompt, { label: task.label, phase: 'Reword', schema: RESULT, agentType: 'general-purpose' })))
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