# Szakdolgozat

**Title (EN):** Domain-Specific Distillation of Large Language Models into Small From-Scratch Students
**Cím (HU):** Nagy nyelvi modellek doménspecifikus desztillációja alapoktól tanított kis modellekbe

Author: Laczkó Örs · Óbuda University NIK · BSc CS · deadline November 2026 · targets: TDK → OTDK

## One sentence

A research bench for tiny (≈3 M parameter) students trained from scratch to reproduce a
large teacher's output distribution over a reduced, domain-specific vocabulary — and a
systematic study of which small-model training techniques survive at that scale, with the
teacher's entropy as the explanatory variable.

## Chapters (working)

1. Problem — why tiny domain-specific students, why generation rather than classification.
2. Background — knowledge distillation, dark knowledge, reduced vocabularies, entropy.
3. The system — domain abstraction, teacher data with stored logits (bf16), the student,
   the annealable dials, logging/analysis, the dashboard. (The engineering is a contribution.)
4. Domains — reddit JSON, maths, post-generation, judged post-generation, long-context recall;
   each placed on the entropy axis (E-00).
5. Method — seeds, noise floor, what counts as an effect, the battery design.
6. Results — one row per technique: prediction from entropy, then what happened.
7. Discussion — what transferred, what did not, and why; the entropy law's reach and limits.
8. Demo.

## Where things live

- Ideas and their fate: `idea-ledger-new.txt` · durable findings: `findings.txt`
- Experiments: `experiment_ledger.txt` · what is running now: `running.txt`
- Toolkit changes: `features_ledger.txt`
