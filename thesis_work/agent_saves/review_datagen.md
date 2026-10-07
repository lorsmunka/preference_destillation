# Data-generation review (library/data_gen, batch_files, vocabulary, build_input_vocabulary, generate_data.py) — FINAL
Saved by the main session from the agent's final report (the agent's own save-file write was blocked).
Nothing critical: teacher-forced scoring, the batched path and the single-step path all align correctly (checked numerically).

## HIGH
H1 VERIFIED | model_handler.py:203-209, :268-272 | Stop strings are rebuilt on every generate() call, and the cache never hits. | transformers 5.17 builds a new StopStringCriteria on every call; its cache key includes the order of tokenizer.get_vocab(), which differs between calls for Gemma 4 (two identical constructions leave 2 cache entries). It costs 0.8 s per call for [";"] and 1.8 s for the agentic tags. MATH_TUTOR is about 19k calls, so roughly 4 h of idle GPU; the harness, thread and label passes pay it too. | Build one StopStringCriteria per handler and pass stopping_criteria=StoppingCriteriaList([...]); for a one-token stop like ";", add its id to eos_token_id.
H2 VERIFIED | generator.py:104-112, 130 | A crash restarts generation from zero. | logger.save() runs only on a graceful stop or at the very end. Simulated crash: batch_1..4.npz on disk but no state.json, so the resume started at 0 and overwrote everything. On the 306k cloud job, a crash at 90% means redoing all of it. | Call logger.save() after each save_batch (batch first, then state); write the npz to a temp file and os.replace it.
H3 VERIFIED, latent | batch_files.py:91-119; example_encoder.py:42 | Batches don't record which reduced vocabulary their logit columns belong to. | Training rebuilds the vocabulary from code and slices with [:vocab_size]. A rebuild of output_tokens.json (frequency-sorted over items.jsonl, still being edited) or any edit to AUXILIARY/WHITESPACE/prompt tokens reorders the columns, so KL targets land on the wrong tokens with no error, while CE (by token string) looks normal. No drift yet in the existing batches. | Store token_list (or its sha1) in the npz meta and info.json; assert equality and width on load.

## MEDIUM
M1 | generate_data.py:53-62 | A graceful Stop / Ctrl+C doesn't stop the job queue: the next job still loads the 12B teacher and runs until its first batch. A second Ctrl+C does nothing. | Check exit_listener at the top of the job loop.
M2 VERIFIED, latent | model_handler.py:116-118, 139/363, 197, 255-260, 292 | BOS handling: Gemma 4 with chat_template=False gets no BOS (and False is the GenerationJob default); Gemma 3 with chat_template=True gets a double BOS. Today's callers are fine. | add_special_tokens=False when the template is applied; otherwise ensure exactly one BOS.
M3 | batch_files.py:103-113 | Per-record metadata (split, item_id, turn_type, teacher mode) is dropped. The math_tutor corpus is 295,458 train + 10,931 test turns, and nothing in the batches marks where test begins; the default 0.8 ratio would move ~16% of train into test. | Carry these fields into the meta and split on them.
M4 | generator.py:121 | Jobs that use the whole corpus never reach "completed" (CODE_RECALL 10k/10k breaks with one skip; MATH_TUTOR max 1,000,000 vs 306k), so every run reloads the 12B teacher for nothing. | Also mark completed when processed >= input_count.

## LOW
- L1 model_handler.py:327-337: validation checks a re-tokenisation of the text, not the stored step tokens; example_encoder.py:43 then quietly swaps an out-of-vocabulary token for the argmax. Validate the stored step tokens.
- L2 generator.py:94-98: answer_correct is always True for forced records (gold vs gold), which inflates teacher accuracy in mixed corpora. Set it to None for forced rows or store the mode.
- L3 saving_handler.py:28-45: info.json lacks provenance (load_in_4bit, chat_template, thinking, batch size, teacher mode), so NF4 and bf16 batches can't be told apart later; started_at and examples_generated reset each session.
- L4 vocabulary.py:213-216: extract_logits_as_vector does one .item() per vocabulary entry (505 or 18,468 GPU syncs per step) in the single-example path. Use logits[ids].float().cpu().tolist().
- L5: the agentic batches are large: 18,468 bf16 columns uncompressed = 33 MB per 32-example batch, about 116 GB for corpus_named.
- L6: np.savez writes are not atomic.
- L7 build_input_vocabulary.py:20: running it as a script fails with ImportError (python -m works); transformer.py:198 names a stale path.
- L8: thinking-mode gaps (unused today): generate_thinking stops only on <channel|>; score_training_examples doesn't reject thinking=True.
- L9: explanation targets are generated in facts-format histories (threads.py:88-94) but scored inside other annotations' histories (conversation.py:137+); "same logits as generation" holds only for facts.

## Checked and fine
- Forced-scoring alignment: a tiny random Gemma4 from the real config, with left padding, explicit positions and logits_to_keep, matched an unpadded reference within 9e-7; a deliberate shift gave 0.3-0.5 difference.
- The batched and single paths match (1.3e-6).
- The prompt/target boundary tokenises identically separately or together (the prompt ends with <channel|>), and generation and scoring use the same build_prompt.
- Reduced vocabularies (505 / 747 / 18,468) have no duplicate or None ids, and every token round-trips.
- bf16 storage is bit-exact; the softcap is applied inside forward; suppress_tokens only touches image/audio tokens.
- The scaffold hold can't loop; batch numbering and graceful resume are correct; the pad fallback to eos is harmless; corpus answers are all ints.

Scratch scripts: scratchpad/review_datagen/ (stopcost*.py H1, resume_test.py H2, drift.py/ties.py H3, bos.py M2, align_*.py alignment, vocab1/roundtrip/perf1/tok1).
