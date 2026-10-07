# slideshow agent save — 2026-10-08 (section 1)

TASK: build thesis_work/how_it_works.html — a local, self-contained, animated step-by-step slideshow explaining the thesis (part 1 + agentic maths tutor + how training will work). Never publish; no git/GPU/data-gen/training.

STATUS
- Draft (all 19 slides written, not yet tested): C:\Users\Ors\AppData\Local\Temp\claude\D--preference-destillation\ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f\scratchpad\slideshow\template.html
  It has a placeholder `/*PERTOKEN_DATA*/null` to be replaced by scratchpad\slideshow\pertoken_js.txt (JSON), then written to thesis_work/how_it_works.html.
- thesis_work/how_it_works.html: NOT written yet (checked absent).
- Next: fill placeholder -> write final file -> Edge headless test (#selftest hash writes report into <pre id=selftest> and data-selftest attr; #N.K opens slide N at step K) -> screenshots -> fixes.
- Pending content update: running.txt changed (STATUS 2026-10-08): thin slice = ~1k problems locally (labels ~3h, threads ~2h, scoring, vocab incl. explanations, one student, eval gold vs own history + SVAMP with unknown tokens); scope frozen (bank 3,418 after 43 exclusions; v2 wish list deferred); open: may cloud runs go unattended; one precision for all final teacher data; user writes ~100 human follow-ups. Slides 18/19 must reflect this.

SLIDES: 1 title · 2 teacher vs student + vocab · 3 KL vs CE (real distribution) · 4 pipeline · 5 part-1 results · 6 agent = model + harness · 7 items/corpus · 8 facts-first + label pass · 9 conversation format (thread drop animation) · 10 one agent turn · 11 template pupil · 12 training records · 13 training interactive (schedules + real per-token loss) · 14 student vs Gemma inputs · 15 eval plan · 16 pilot tiles · 17 cascade (44 vs 2) · 18 planned studies · 19 status/next.

FACTS EXTRACTED (with sources)
- Teacher: part 1 Gemma 3 4B (domain_entropy.csv teacher col); part 2 Gemma 4 12B, 48 layers, hidden 3840, vocab 262,144 (models/gemma-4-12B-it/config.json). Pilots NF4 on 12 GB 4070.
- Student default: hidden 48, 3 layers, 1 head (library/specs.py); 3,416,397 params, 96.0% input embedding (findings F-68a). Defaults: lr 0.0015, warmup 0.1, kl_ratio 0.99->0.5, temp 1->1, batch 32.
- Output vocab: reddit 525 (27 req + 498 aux), math v1 528, post-gen 26,659 (thesis.md); tutor output_tokens.json = 18,357 tokens. Input vocab reddit 68,328; math 1,282 (thesis.md); tutor 11,286; SVAMP 4.1% tokens unseen, 72.5% problems (running.txt step 8).
- Real teacher distribution (post-gen batch_1 record 7 step 65, H=1.60 nats, exp(H)=4.95): "," .4781, "▁about" .1993, "." .1993, "▁<" .0238, "▁and" .0210, ".<" .0112, "▁on" .0112, "▁–" .0099, rest .0461. T=0.5 expH 2.19, T=2 expH 159.4 (computed from stored row; scratch dist.py / temps.py).
- One-hot share (entropy<0.1 nats; measure_entropy.py): reddit 96.4%, math 99.7%, post-gen 20.0%; exp(H) 1.0205 / 1.0028 / 2.734 (domain_entropy.csv).
- E-01 (e01_anneal_direction.csv, recomputed): post-gen overlap pure KL .3052, CE->KL .2984, flat .2913, KL->CE .2839, pure CE .2644; rank 694/743/769/806/1071. Reddit best task acc %: 62.9±0.9, 62.7±1.6, 63.0±0.7, 62.1±0.7, 62.3±2.1 (ce-to-kl, kl-to-ce, flat, pure-ce, pure-kl); math 24.6±1.7, 23.5±1.8, 23.1±1.6, 23.8±1.7, 24.1±1.7. Run length: post-gen 10 ep × 96 batches = 960 steps; reddit/math 3 × 51 = 153.
- Real per-token KL/CE (CPU inference on runs/e01-postgen-pure-kl-s1 epoch0/epoch9 and pure-ce-s1 epoch9 checkpoints, batch_97 example 0, first 14 tokens): scratch pertoken.json / pertoken_js.txt.
- Corpus (domain_specs BUILD RESULT): seeds 2,153; written 12,476; reworded 21,694; pruned 34,170->34,002; train 32,377 / test 1,210 / held-out 415; renumber 96,544 -> 130,546; turns 295,458 + 10,931; held-out turns 3,648; templateness 0.40 vs seeds .54, SVAMP .91, v1 1.00; SVAMP 746 (577 one-step / 169 two-step).
- Label pass sample (commit 7981150): 40 items, 84/91 turns (92%), 37/40 complete, 192 copy turns; sample_labels.jsonl 409.
- Bank (commit ba050d5): 3,461 templates, 1,842 niche, per-kind counts; 43 excluded.
- Pilot summaries: harness_runs/gemma_pilot_{1..5}.json; template_pilot.html (free 175/180 etc.; sweeps v1/v2).
- Cascade: gemma_pilot_5.json, item S08-mem-b05-0022 (48 -> smaller-numbers misroute 2*3=6 -> Sofia 50-6=44, expected 2).
- Token counts (Gemma 4 tokenizer): Ines turn-2 context 101 tokens for the student; teacher prompt with chat template 558.
- Real foci (questions.py on sd-p1-h02-0007): step "divide 63 by 7"; mistake "take 9 away from 54" = 45; what-if 4 -> 11, answer 99.

DESIGN: one HTML file, inline CSS/JS; colour tokens on :root + dark variants (harness.py transcript palette for blocks: task green, question blue, solution purple, explanation orange, harness grey); teacher = #2a78d6, student = #eb6834 (dataviz reference pair). Fragments via data-f, highlights via data-hl, step classes via data-cls="name@from-to"; hooks for slide 3 (bars/toy students/temperature) and slide 13 (schedules + per-token loss). Keys: arrows/space/enter, O overview, T theme; click/swipe; hash #N.K.
