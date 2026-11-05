# Attention study: novelty and adversarial check (fact sheet, 2026-11-05)

Source of the numbers: `experiments/attention_study.py` + `attention_report.py` (page: `runs/attention_study/attention_study.html`,
tables: `experiments/results/attention_models.csv`, `attention_seed_pairs.csv`). Two adversarial passes: one on the data (new
controls), one on the literature (prior work + a hostile reviewer). Citations were opened by the literature agents (abstract pages);
not re-checked by hand: check each before citing.

## Verdicts

| # | Claim as first stated | Verdict | What survives (narrowest defensible form) |
|---|---|---|---|
| C1 | Logit distillation (KL) does not transfer attention: KL .320 vs CE .313 Gemma likeness | **Narrowed** | Soft labels (KL on Gemma's distribution over the reduced vocabulary) vs hard labels (CE on Gemma's greedy tokens) make no difference to how Gemma-like the students' attention is. Most of the apparent likeness is shared recency: with position removed .182 (KL s1) / .177 (CE s1) vs .089 untrained. Not "logit KD does not transfer attention": CE is also distillation (its targets are Gemma's top word in 99.3% of math positions). |
| C2 | Students resemble Gemma's sliding-window layers (.23), not its full-attention layers (.14) | **Dropped** | It is position: with each head's distance profile removed, .093 vs .070 (KL s1); against Gemma on another record, full .045 = sliding .043. Also an architecture match: the students use full RoPE at base 10k, as Gemma's sliding layers do (the global layers: partial RoPE at 1M, k = v, one KV head). The 1,024-token window never applies on the math records (max 913 tokens). |
| C3 | No student forms retrieval heads; Gemma's heads find the target lines | **Narrowed** | In this small-data regime (16 training projects) no student solved the scope task or formed a target-attending head. Gemma's heads in global layers 30 / 36 find the file first (23/23 records), then the lines (up to 23/23; 67% of the weight). The pilot students learned no attention at all on 6,000-token inputs (equal to the untrained student to 3 decimals); only KL m on the full data learned some (the request 5.4%, its own reply), not the target file (3/23, about chance). Large-scale counterexample: Huber et al. 2026 (1.1B from scratch, ~110B tokens: logit KD does transfer long-context retrieval). The link to 0% accuracy is a coincidence in the data, not a causal test. |
| C4 | The seed decides attention, not the loss (0.60 vs 0.39; 92-100% same heads) | **Holds; not novel as a principle** | With a shared start and data order, changing the loss keeps the same heads in place, and its effect on attention is ~2% of a re-seed (same minus other loss .0036, 95% over probes .0028-.0042; over the 8 students p = .11). Expected from shared-basin results (Neyshabur 2020, Kornblith 2021, Bali 2026). In the math runs "seed" bundles start weights and data order; the sentiment runs (data order fixed, 10 seeds, 5 schedules) separate them: the start weights alone give same seed / other schedule .319 vs same schedule / other seed .141 = other schedule / other seed .141 (schedule effect −.0006, p = .64 over 20,000 relabellings). |
| C5 | KL / CE from one start drift apart, then re-converge (.46 → .56) | **Narrowed** | One pair only. Attention at the start is flat (entropy .999): there is no shared pattern to drift from. What survives: the patterns form in pass 1, then the two converge (.458 → .560, the same on both halves of the probes). |

## Robustness that held (data adversary + bootstrap)

| Check | Result |
|---|---|
| Bootstrap over the 80 math probes (1,000 resamples, matching redone) | same loss / other seed .386 [.380, .394]; other loss / same seed .598 [.592, .604]; other loss / other seed .382 [.376, .391] |
| Per reply type | solve .375 / .582 / .373; explain .428 / .622 / .426; route .391 / .531 / .389 |
| Per layer (matched within layer) | same seed .53-.72 vs other seed .25-.44; same vs other loss within .01 in every layer |
| Gemma's sink heads removed (outside view > .5: 274 of 768) | the conclusions above unchanged |
| Sentiment (Study 7): 50 students, data order fixed; data sizes 256-2,048 | start weights alone double the likeness (.319 vs .141); the same seed on another data size (.345) is more alike than two seeds on one size (.15-.21) |
| KL − CE Gemma likeness, by seed | +.0010, +.0058, +.0142 (bootstrap); with position removed +.0046, +.0033, +.0038; flips for seed 1 on a last-tokens window: read as no difference |

## What a reviewer would ask for (to make any of it more than thesis material)

- 5+ seeds per arm, with start weights and data order varied separately (the sentiment E-01 runs already keep the data order fixed across 10 seeds).
- A no-teacher arm (CE on gold or non-Gemma targets) and an attention-distillation arm as a positive control.
- A null for best-of-768 matching (an unrelated model), position controlled.
- For C3: a causal test (attention loss on the target lines, or transferred retrieval heads: does task accuracy rise?).
- Other teachers and domains.

Assessment (both agents): thesis material; at most a workshop paper, and only C3 with an intervention.

## Sources (opened by the literature agents)

| Use | Reference | In literature.bib |
|---|---|---|
| Head matching method | Li, Yosinski, Clune, Lipson, Hopcroft, "Convergent Learning", ICLR 2016, arXiv 1511.07543 | no |
| Head stability across seeds | Bali, Stanley, Suresh, Bzdok, "Quantifying LLM Attention-Head Stability: Implications for Circuit Universality", arXiv 2602.16740 (2026) | no |
| Shared start stays in one basin | Neyshabur et al., arXiv 2008.11687; Frankle et al., "Linear Mode Connectivity and the Lottery Ticket Hypothesis", ICML 2020, arXiv 1912.05671 (not opened) | no |
| Losses differ only in late layers | Kornblith et al., NeurIPS 2021, arXiv 2010.16402 | no |
| Retrieval heads | Wu et al., "Retrieval Head Mechanistically Explains Long-Context Factuality", arXiv 2404.15574 | no |
| Logit KD transfers retrieval at scale (C3 counterexample) | Huber et al., "Short Data, Long Context: Distilling Positional Knowledge in Transformers", arXiv 2604.06070 (2026) | no |
| Retrieval-aware distillation | Bick, Xing, Gu, "Retrieval-Aware Distillation for Transformer-SSM Hybrids", arXiv 2602.11374 (2026) | no |
| Local vs global layers in hybrids | Yang et al., arXiv 2501.18795 (RoPE layers recency, NoPE layers retrieval) | no |
| Attention sinks | Xiao et al., StreamingLLM, ICLR 2024, arXiv 2309.17453; Gu et al., "When Attention Sink Emerges in Language Models", ICLR 2025, arXiv 2410.10781 | no |
| What distillation passes on | Ojha et al., NeurIPS 2023, arXiv 2205.16004; Abnar, Dehghani, Zuidema, arXiv 2006.00555; Faumi & Sumanathilaka, arXiv 2610.37157 (2026) | no |
| Already cited | Stanton et al. 2021 (`stanton2021does`); Gemma 3 / 4 reports; Kim & Rush 2016 | yes |
