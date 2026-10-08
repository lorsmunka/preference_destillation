# Novelty check: from-scratch, small, domain-only, agentic student (2026-10-07)

Research agent, about 120 tool calls, every reference opened. Absence of evidence, not proof:
industry, unpublished and non-English work is not covered.

## Verdict
**Holds, with caveats, as a combination.** No work found combines all four:
- (a) a randomly initialised student,
- (b) about 1M-100M parameters,
- (c) trained on one narrow domain only,
- (d) a text-generating agent in a harness that picks a response type per message and emits structured output the harness checks,

all with an LLM teacher and token-level distributions. Each part alone has precedent.

## Closest works

| Work | Student | Domain / agentic | Signal | Difference |
|---|---|---|---|---|
| LLM4Teach (IJCAI 2024) https://arxiv.org/abs/2311.13373 | 24K / 10M RL policy | one environment, embodied RL | divergence to LLM's soft action choice, annealed to 0, then RL | **closest from-scratch agentic analogue**; discrete actions, not text |
| Structured Agent Distillation (AAMAS 2026) https://arxiv.org/abs/2505.13820 | GPT-2 120-760M (pretrained, implied) | ALFWorld, WebShop, HotPotQA; ReAct | token KL on cached teacher logits; KL over actions in [ACT] spans | closest on signal and size; pretrained, general |
| STAR https://arxiv.org/abs/2602.03022 | Qwen3 0.6-4B pretrained | function calling, multi-turn | top-k forward KL + tail, then RL | pretrained, larger |
| DeDer (ICML 2024) https://arxiv.org/abs/2412.11499 | t5-small + gpt2, pretrained | ALFRED planning | SFT on rationales | pretrained, no logits |
| TinyAgent https://arxiv.org/abs/2409.00608 | TinyLlama 1.1B pretrained | Mac assistant, single turn | LoRA SFT | pretrained, large |
| Octopus v2 https://arxiv.org/abs/2404.01744 | Gemma-2B pretrained | API calling, single turn | SFT; vocabulary expanded | pretrained |
| Different Teachers... https://arxiv.org/abs/2607.08268 | Qwen3-0.6B pretrained | news -> JSON (sentiment, tone, urgency...) | QLoRA SFT | **almost the part-1 Reddit task: cite** |
| TinyGSM https://arxiv.org/abs/2312.09241 | pretrained + **random-init ablation (App. B)** | grade-school math | SFT (+ verifier) | random init costs about 10 points: 53.1 vs 63.3 (125M), 55.5 vs 65.9 (350M) |
| TinyStories https://arxiv.org/abs/2305.07759 | from scratch, 1M-80M | children's stories | LM loss | not agentic; top-10K token vocabulary |
| TinyDialogues (EMNLP 2024) https://arxiv.org/abs/2408.03617 | from scratch, 124M | child dialogues | LM loss | multi-turn but no agent behaviour |
| Pre-training Distillation https://arxiv.org/abs/2410.16215 | from scratch, 330M-6.8B | general | top-p/top-k logits; scheduled KD share | general domain |
| BabyLM 2025 rules https://arxiv.org/abs/2502.10645 | from scratch, small | general | **bans** using a teacher's distribution | - |

Surveys that support the gap:
- agent-distillation roadmap (arXiv 2609.36630): no from-scratch or sub-1B students;
- on-policy distillation survey (arXiv 2604.00626): smallest students about 124M, all pretrained.

## Suggested novelty sentence
"To our knowledge, no prior work distils a large language model into a randomly initialised student of
tens of millions of parameters, trained only on a single narrow domain, that then operates as a
conversational agent inside a harness, choosing per message among response types and emitting
structured outputs that the harness validates, learning from the teacher's token-level distributions.
Each ingredient has precedent: from-scratch logit distillation for general pretraining (Gemma 2;
Peng et al., 2024), from-scratch small models on LLM-generated narrow-domain text (TinyStories;
TinyGSM's random-initialisation ablation), and agent distillation into pretrained 0.1-7B students
(Kang et al., 2025; Liu et al., 2026; STAR). The closest from-scratch agentic analogue (LLM4Teach)
distils an LLM's action distribution into a reinforcement-learning policy rather than a
text-generating model."

## Do not claim as new
- **Annealed KL/CE mixing:** LLM4Teach anneals its weight; Peng et al. schedule the KD share.
- **Reduced vocabulary in general:**
  - TinyStories keeps the top 10K tokens;
  - cross-tokenizer work (https://arxiv.org/abs/2512.14954) truncates to the first 16k-64k merges;
  - VocabTrim (https://arxiv.org/abs/2506.22694) trims a drafter's output head;
  - Kolesnikova et al. (https://arxiv.org/abs/2205.02340) distil Russian encoders with a reduced vocabulary.
- What was **not** found: a **task-derived output vocabulary** (about 525 tokens for Reddit, about
  18k for the tutor), learned with KL to the teacher's distribution restricted to that subset.

## Define in the thesis
"Agentic" = choosing a response type per message plus structured output that a harness checks and
acts on. A reviewer may otherwise call it "structured intent routing".

## Not verified
- Student initialisation in LLM4Teach and Structured Agent Distillation (not stated).
- phi-1 random init (implied, not quoted).
- Whether TinyGSM App. B numbers include the verifier.
- AlphaGeometry (paywalled).

## Correction (2026-10-08, from the adversarial positioning review)
- "Not found: a task-derived output vocabulary" no longer holds: VocabTailor (arXiv 2508.15229) builds task-derived
  static output heads (training-free). What survives: a from-scratch student whose output head is the task vocabulary,
  trained with KL to the teacher's distribution renormalised over that subset.
- BabyLM 2025's ban on using a teacher's output distribution applies to the Interaction track only (2502.10645).
- Recommended framing and full red-team analysis: thesis_work/thesis_positioning.html.
