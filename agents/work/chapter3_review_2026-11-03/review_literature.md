# Adversarial review of chapter 3: the literature side

Checked 2026-11-03 against primary pages: arXiv abstracts and HTML, ACL Anthology, PMLR, Hugging Face docs and model cards, GitHub READMEs, release pages and source files. Draft: `chapter3_draft.md`. Reference list: `thesis_draft.md` §13.

Severity tags: WRONG = factually false. MISCITED = bibliographic or attribution error. OVERCLAIM = stronger than the evidence. MISSING = work a KD-literate reviewer would expect. MINOR = precision or wording.

Hungarian fix texts contain no colons and no first person singular. IEEE reference entries are in English and keep the list's existing format, including "Accessed:" and "Available:".

---

## A. Highest priority

### A1. MISSING (high): LLM-era task-specific distillation into small models, including the thesis's own core argument
- **Phrases:** 3.1.1 "Ha viszont lemondunk az általánosságról, és csak egy szűk feladatra (domain-re) szeretnénk fókuszálni, a szükséges tudás egy jóval kisebb modellbe is beférhet." · 3.2 "Lemond az általános képességekről, és cserébe …" · 3.3 "A dolgozat hozzájárulása ezek gyakorlati összekapcsolása …"
- **Problem:** The chapter goes straight from pre-LLM task KD (Kim & Rush, Tang) to general LLM KD (Gemma, R1-Distill). It skips the body of work that is closest to the thesis.
- **Evidence:**
  - **Fu et al., ICML 2023.** PMLR 202:10421–10430, https://proceedings.mlr.press/v202/fu23d.html. The abstract reads "if we specialize their capacity towards a target task, the model can achieve decent performance improvements" and "by intentionally paying the price of decreased generic ability, we can clearly improve across different model scales smaller than 10B". The teacher is GPT-3.5 and the students are T5/FlanT5 of 250M–11B. The HTML (https://arxiv.org/html/2301.12726) says "we use distribution matching as our training objective" over the teacher's top-5 tokens ("the OpenAI API only grants access to the 5 most probable tokens"), with tokenizer alignment "by dynamic programming". The task is math. This is the thesis's central "trade generality for a specialised small model" argument, made by a teacher-distribution method on math.
  - **Hsieh et al., Findings ACL 2023.** pp. 8003–8017, doi:10.18653/v1/2023.findings-acl.507 (https://aclanthology.org/2023.findings-acl.507/). The abstract (arXiv 2305.02301) says the "finetuned 770M T5 model outperforms the few-shot prompted 540B PaLM model" "using only 80% of available data". It also uses SVAMP, the thesis's human test set.
  - **GKD [47] itself.** Already cited, but only as a TRL method. Its students are "T5-small (77M params), T5-base (250M params), and T5-large (800M params)", started from supervised-FT checkpoints, on GSM8K among other tasks (https://arxiv.org/html/2306.13649v3).
- **Fix:** Add two sentences after the Tang sentence in 3.1.1 and two references. Suggested Hungarian:
  > "A nagy nyelvi modellek megjelenése után is folytatódott a feladatspecifikus desztilláció kisebb, de előtanított modellekbe. Hsieh és társai [x] a PaLM 540B indoklásain tanított 770 millió paraméteres T5-modellel egy teszten a few-shot módon promptolt PaLM-ot is megelőzték, Fu és társai [y] pedig a GPT-3.5 öt legvalószínűbb tokenjének eloszlásán tanítottak 250 millió és 11 milliárd paraméter közötti FlanT5-modelleket matematikai feladatokra, kifejezetten az általános képességek rovására. Ezek a tanulómodellek viszont előtanított modellek voltak, és legalább 77 millió paraméteresek (a GKD [47] legkisebb, T5-small tanulója)."

  Then 3.3 can say the specialisation idea is Fu et al.'s and the thesis's step is from-scratch, tiny, task-subset vocabulary.
  - [x] C.-Y. Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y. Fujii, A. Ratner, R. Krishna, C.-Y. Lee, and T. Pfister, "Distilling step-by-step! Outperforming larger language models with less training data and smaller model sizes," in Findings Assoc. Comput. Linguistics: ACL 2023, Toronto, Canada, 2023, pp. 8003–8017, doi: 10.18653/v1/2023.findings-acl.507.
  - [y] Y. Fu, H. Peng, L. Ou, A. Sabharwal, and T. Khot, "Specializing smaller language models towards multi-step reasoning," in Proc. 40th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 202, 2023, pp. 10421–10430. [Online]. Available: https://proceedings.mlr.press/v202/fu23d.html

### A2. WRONG: table 3.1, "Általános LLM-desztilláció" row, column "Kimeneti szótár"
- **Phrase:** "| Általános LLM-desztilláció [29], [30] | … | a tanáré |"
- **Evidence:**
  - DeepSeek-R1-Distill students are "fine-tuned based on open-source models" (Qwen2.5-Math-1.5B/7B, Qwen2.5-14B/32B, Llama-3.1-8B, Llama-3.3-70B-Instruct). The card also says "We slightly change their configs and tokenizers" (https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
  - Config files: DeepSeek-R1 `vocab_size` 129280; R1-Distill-Qwen-1.5B 151936; R1-Distill-Llama-8B 128256. The students keep their own vocabularies, which is possible because they learn from text only.
- **Fix (cell):** "teljes (a Gemma 2-nél a tanáré, a DeepSeek-R1-Distill modelleknél a tanuló saját szótára)"

### A3. WRONG / self-contradiction: 3.2 prose says general distillation uses pretrained students
- **Phrase:** "az általános desztilláció, az ügynökdesztilláció és a desztillációs eszközök pedig előtanított, jellemzően milliárdos nagyságrendű tanulómodellekkel dolgoznak"
- **Evidence:**
  - Gemma 2's 2B and 9B were trained with KD during pre-training, and the ablation is "a 2B model trained over 500B tokens either from scratch or with distillation" (https://arxiv.org/html/2408.00118, Table 6).
  - "All Gemma 3 models are trained with knowledge distillation" (https://arxiv.org/html/2503.19786).
  - The chapter's own table says "alapoktól vagy előtanított modellből".
- **Fix:** "A kvantálás és a finomhangolás nem csökkenti a paraméterek számát, az általános desztilláció milliárdos nagyságrendű, általános célú tanulómodelleket tanít alapoktól vagy előtanított modellből, az ügynökdesztilláció és a desztillációs eszközök pedig előtanított, legalább félmilliárd paraméteres modellekből indulnak."

### A4. WRONG / self-contradiction: 3.3 first sentence
- **Phrase:** "vagy jellemzően előtanított, legalább néhány száz millió paraméteres tanulómodellel és teljes szótárral dolgoznak, mint az általános desztilláció, az ügynökdesztilláció, a kis nyelvi modellek és a desztillációs eszközök, mert céljuk az általános képességek megőrzése."
- **Problems:**
  1. **"a kis nyelvi modellek".** The chapter's own small-LM example, TinyStories [42], uses models trained from scratch "below 10 million total parameters" with "GPT-Neo tokenizer but only keep the top 10K most common tokens" (arXiv 2305.07759 abstract; PDF footnote 2). Its own table row says "1 millió – 1,7 milliárd" and "ritkán csökkentett".
  2. **"mert céljuk az általános képességek megőrzése".** This is false for agent distillation. TinyAgent targets a narrow set of macOS functions, and Kang et al. target specific QA and math tasks. The table itself lists the goal as "ügynöki feladatok". The tools' goal is "tetszőleges".
- **Fix:** "A meglévő megoldások vagy nem csökkentik a modell méretét, mint a kvantálás és a finomhangolás, vagy jellemzően legalább néhány száz millió paraméteres, teljes szótárú tanulómodellel dolgoznak, mint az általános desztilláció, az ügynökdesztilláció és a desztillációs eszközök. Az ennél kisebb, alapoktól tanított nyelvi modellek, például a TinyStories modelljei, csak a tanár szövegéből tanulnak, és nem egy konkrét feladatot sajátítanak el."

### A5. OVERCLAIM: "KL beats CE is the expected direction" (3.3), plus "nem vizsgált méretben"
- **Phrase:** "Ez utóbbi a szakirodalom alapján várható irány, a dolgozat ezt egy a feldolgozott szakirodalomban nem vizsgált méretben és beállításban erősíti meg."
- **Problems:**
  1. **No citation.**
  2. **The size is not new.** Kim & Rush [27] compared word-level KD with the NLL baseline, and Seq-KD with Seq-KD + Word-KD, at 84M, 49M and 8M students trained only on the task. Table 1 (https://ar5iv.labs.arxiv.org/html/1606.07947) gives, for Th→En 2×100 (8M), Baseline 10.6/12.7 and Word-KD 11.8/13.6 BLEU. The 8M student is smaller than the thesis's 16.5M student. Only the setting is new: an LLM teacher, a decoder-only student and a task-subset vocabulary.
  3. **The closest precedent is mixed.** In the thesis the CE target is the teacher's own sampled token. Training data = Gemma's generated replies (see CLAUDE.md and table 3.1, "választott tokenje"). So the CE arm is sequence-level KD, and KL vs CE is "Seq-KD + Word-KD vs Seq-KD". Kim & Rush report that combination for 2×500 as Seq-KD 18.9/19.0 vs Seq-KD + Word-KD 18.7/18.9. Their text says the combination "results in further gains for the 2×300 and 2×100 models (although not for the 2×500 model)".
  4. **Capacity-gap evidence cuts the other way at large teacher/student ratios.**
     - Busbridge [31] (already cited): "improving teacher performance does not always improve student performance, and even reduces student performance eventually" (https://arxiv.org/html/2502.08606v2).
     - Mirzadeh et al., AAAI 2020: "the student network performance degrades when the gap between student and teacher is large" (https://arxiv.org/abs/1902.03393).
     - Cho & Hariharan, ICCV 2019: "more accurate teachers often don't make good teachers" (https://arxiv.org/abs/1910.01348).
     - The thesis's teacher-to-student ratio is about 700x (12B to 16.5M).
  5. Side note for the statistics reviewer: "mind a 30 páros mérésnél" counts 10 evaluation passes × 3 seeds. Measurements within a seed are not independent, so "30 of 30" reads stronger than three independent seed-level comparisons.
- **Fix:** "Ez az irány összhangban van a szakirodalommal [16], [27], [29], [31], ahol a tanár eloszlásának követése jellemzően javít a csak a kiválasztott tokenen történő tanításhoz képest. Kim és Rush [27] eredményeiben viszont a szekvenciaszintű desztilláció mellé adott tokenszintű tag nem minden méretnél hozott további javulást, és nagy méretkülönbségnél a tanár előnye csökkenhet [31]. A dolgozat ezt egy a feldolgozott szakirodalomban nem vizsgált beállításban, nagy nyelvi modell tanárral és alapoktól tanított, szűkített szótárú generatív tanulómodellel mutatja ki."

  Optionally add Cho & Hariharan and Mirzadeh et al. as references (see D6).

### A6. MISCITED: SmolLM2-135M GSM8K 1.4% is not in the paper [41]
- **Phrase:** "a legkisebb, 135 millió paraméteres változat viszont a GSM8K matematikai teszten mindössze 1,4%-os pontosságot ér el" [41]
- **Evidence:**
  - Section 6 of the paper says only "SmolLM2-135M (135M parameters, trained on 2T tokens)" and "More details about SmolLM2-360M and 135M can be found in their respective model cards". It has no 135M GSM8K number (full HTML text searched, https://arxiv.org/html/2502.02737v1).
  - The 1.4 is on the model card, "Base pre-trained model" table, "GSM8K (5-shot) 1.4" (https://huggingface.co/HuggingFaceTB/SmolLM2-135M).
  - The MobileLLM-R1 card reports SmolLM2-135M-base at GSM8K 1.8 (8-shot), so the number depends on the setting.
- **Fix:**
  - Text: "a legkisebb, 135 millió paraméteres változat viszont a modellkártyája szerint a GSM8K matematikai teszten (5 mintapéldával) mindössze 1,4%-os pontosságot ér el [z]".
  - New reference [z]: HuggingFaceTB, "SmolLM2-135M," Hugging Face model card. Accessed: Nov. 3, 2026. [Online]. Available: https://huggingface.co/HuggingFaceTB/SmolLM2-135M

---

## B. Further WRONG / OVERCLAIM items

### B1. WRONG: TinyStories "follows no LLM's generative behaviour" (3.2)
- **Phrase:** "A korai feladatspecifikus desztilláció és a TinyStories apró, előtanítás nélküli modelljei azt mutatják, hogy egy szűk feladathoz kis modell is elég, ezek viszont nem egy nagy nyelvi modell generatív viselkedését követik."
- **Evidence:** TinyStories is "a synthetic dataset of short stories … generated by GPT-3.5 and GPT-4" (arXiv 2305.07759 abstract). Its models imitate an LLM's generations at text level, which is sequence-level distillation in Kim & Rush's sense.
- **Fix:** "… ezek viszont nem egy nagy nyelvi modell eloszlásából tanulnak, a TinyStories modelljei például csak a GPT-3.5 és a GPT-4 által írt szövegből."

### B2. WRONG: TinyStories as "an early example" of vocabulary reduction (3.1.5)
- **Phrase:** "ami a szótárcsökkentés egy korai példája"
- **Evidence:**
  - 2023 is not early. Kim & Rush [27] (2016) already "keep the top 50k most frequent words" with the rest replaced by UNK.
  - Shi & Knight 2017 restrict the NMT target vocabulary at run time, doubling speed with BLEU intact (ACL 2017 short, pp. 574–579, doi:10.18653/v1/P17-2091, https://aclanthology.org/P17-2091/).
  - Ushio et al. 2023 trim a multilingual LM's vocabulary to one language (Findings EMNLP 2023, pp. 14725–14739, doi:10.18653/v1/2023.findings-emnlp.981).
- **Fix:** "ami a szótárcsökkentés egy kis, alapoktól tanított modelleknél alkalmazott példája". Optionally add a clause such as "a gépi fordításban a célszótár szűkítése már korábban is bevett volt [27], [Shi és Knight]".

### B3. OVERCLAIM: from-scratch few-million-parameter students "not documented" in the tools (3.1.6)
- **Phrase:** "Az alapoktól tanított, néhány millió paraméteres tanulómodellek sem dokumentált célpontjai ezeknek az eszközöknek."
- **Evidence:** TextBrewer [53] is cited one paragraph earlier, so it falls under "ezeknek az eszközöknek".
  - The paper states "T3-small and T4-tiny are initialized randomly" (https://ar5iv.labs.arxiv.org/html/2002.12620).
  - The README lists T3-small 17M and T4-Tiny 14M English students, plus 11–14M Chinese ones (https://github.com/airaria/TextBrewer).
  - These sizes fall inside the thesis's own 6–54M range. They are encoder classifiers or MRC models, not generative.
- **Fix:** "A vizsgált mai LLM-desztillációs eszközök dokumentációja alapoktól tanított, néhány millió paraméteres generatív tanulómodellt sem említ. A 2020-as TextBrewer [53] ugyan véletlenszerűen inicializált, 14–17 millió paraméteres tanulómodelleket is bemutat, ezek azonban osztályozó és szövegértő kódoló modellek."

### B4. OVERCLAIM: "current LLM toolkits use fixed weights" (3.1.6)
- **Phrase:** "a vizsgált mai LLM-desztillációs eszközök viszont rögzített súlyokkal és hőmérséklettel dolgoznak"
- **Evidence:**
  - **True for** TRL GKD (`lmbda`, `beta`, `temperature` are scalars), DistillKit (per-loss `weight`, `temperature`), torchtune (`self._kd_ratio = cfg.get("kd_ratio", 0.5)` in `recipes/knowledge_distillation_single_device.py`) and EasyDistill 1.0 (`"kd_ratio": 0.1` in `configs/kd_white_box.json`; `total_loss = (1 - self.kd_ratio) * lm_loss + self.kd_ratio * distil_loss`).
  - **But TRL GOLD has adaptive weights.** It has `xtoken_dynamic_scaling` (default True), "Scale KD by stop_gradient(abs(CE) / abs(KD)) per microbatch to balance loss magnitudes". Also `uld_hybrid_matched_weight`: "If None, uses adaptive weighting based on vocabulary overlap" (https://huggingface.co/docs/trl/main/en/gold_trainer).
  - None of these is a schedule over training time.
- **Fix:** "… a vizsgált mai LLM-desztillációs eszközök dokumentációja viszont nem kínál tanítás közbeni ütemezést a két veszteség arányára és a hőmérsékletre."

### B5. OVERCLAIM: Belcak et al. "most agentic tasks" (3.1.4)
- **Phrase:** "a kis nyelvi modellek (small language model, SLM) a legtöbb ügynöki feladatra elég erősek"
- **Evidence:** The abstract says "sufficiently powerful, inherently more suitable, and necessarily more economical for many invocations in agentic systems". V1 says "principally sufficiently powerful to handle language modeling errands of agentic applications". "Vast majority" appears only in V3, about economy (https://arxiv.org/abs/2506.02153, https://arxiv.org/html/2506.02153).
- **Fix:** "Belcak és társai [39] álláspontja szerint a kis nyelvi modellek az ügynöki rendszerek sok hívására elég erősek, üzemeltetésre alkalmasabbak és olcsóbbak is, mint a nagy modellek."

### B6. OVERCLAIM: "at this size general training gives no usable math" (3.1.5)
- **Phrase:** "Ekkora méretnél tehát az általános célú tanítás még nem ad használható matematikai képességet."
- **Evidence:** This generalises from a single model. MobileLLM-R1-140M-base (140M) scores GSM8K 16.3 (8-shot) against 1.8 for SmolLM2-135M-base (https://huggingface.co/facebook/MobileLLM-R1-140M-base). The paper (ICLR 2026, arXiv 2509.24945) says "Models are initialized from scratch". Its mid-training applies KD "with Llama-3.1-8B-Instruct model as the teacher".
- **Fix:** "A SmolLM2-135M esetében tehát az általános célú tanítás még nem adott használható matematikai képességet. A kifejezetten érvelési adatokra összeállított, tanári desztillációt is használó, 140 millió paraméteres MobileLLM-R1 alapmodell [w] a modellkártyája szerint 16,3%-ot ért el, ami egy feladatot önállóan megoldó ügynökhöz még mindig kevés."
  - [w] C. Zhao et al., "MobileLLM-R1: Exploring the limits of sub-billion language model reasoners with open training recipes," in Proc. Int. Conf. Learn. Represent. (ICLR), 2026. [Online]. Available: https://arxiv.org/abs/2509.24945

  The 16.3 figure is from the model card. Cite the card as well if the number stays.

### B7. OVERCLAIM (outdated): EasyDistill "supports" logits (3.1.6, dated "2026. november 3-i állapot")
- **Phrase:** "az EasyDistill [50] pedig a tanár szövegén és logitjain alapuló desztillációt is támogat"
- **Evidence:**
  - The current README says "EasyDistill 2 is a config-driven knowledge distillation toolkit for turning black-box teacher models into high-quality training data". It adds "EasyDistill 2 is a rewrite and is not backward compatible … The 1.0 implementation remains available on the 1.0 branch" (https://github.com/modelscope/easydistill).
  - White-box KD (stored top-10 logits, `"top_logits_num": 10`) exists only on the 1.0 branch.
- **Fix:** "az EasyDistill [50] 1.0-s változata pedig a tanár szövegén és elmentett logitjain alapuló desztillációt is támogatta, a 2026-os 2.0-s változat viszont már csak a tanár szövegéből állít elő tanítóadatot."

### B8. MINOR / incomplete: DistillKit offline only (3.1.6)
- **Phrase:** "Az Arcee DistillKit [48] előre elmentett (offline) tanárlogitokkal dolgozik"
- **Evidence:** The README lists "Online Distillation: Real-time teacher inference during student training" and "Offline Distillation: Train from pre-captured teacher outputs with advanced compression" (https://github.com/arcee-ai/DistillKit).
- **Fix:** "Az Arcee DistillKit [48] a tanár futás közbeni (online) és előre elmentett (offline) logitjaival is tud tanítani"

### B9. MINOR: "the thesis's two main directions" (3.1.6)
- **Phrase:** "Az eszközök az utóbbi időben a dolgozat két fő irányát is elérték."
- **Problem:** Cross-tokenizer KD (GOLD/ULD) solves a different problem. It distils between two different tokenizers of a pretrained student. The thesis keeps the teacher's own tokens, a 1:1 subset. Calling it the thesis's "fő iránya" invites the objection that the thesis does not do cross-tokenizer KD.
- **Fix:** "Az eszközök az utóbbi időben két, a dolgozathoz közeli irányba is bővültek."

### B10. MINOR: "always" in the documented examples (3.1.6)
- **Phrase:** "A dokumentált példákban viszont a tanulómodellek mindig előtanított, legalább 0,5 milliárd paraméteres modellek"
- **Evidence:**
  - **True as far as checked:**
    - TRL GKD: Qwen2-0.5B-Instruct.
    - TRL DistillationTrainer: Qwen2.5-0.5B-Instruct.
    - TRL GOLD: Llama-3.2-1B-Instruct, Qwen3-VL-2B, LFM2.5-VL-1.6B.
    - DistillKit: Qwen3-8B.
    - torchtune: `qwen2/1.5B_to_0.5B_KD`.
    - EasyDistill 1.0: Qwen2.5-0.5B-Instruct.
  - **Contradicted by** TextBrewer (B3) if the sentence is read as covering all reviewed tools.
- **Fix:** "A vizsgált mai LLM-desztillációs eszközök dokumentált példáiban viszont a tanulómodellek előtanított, legalább 0,5 milliárd paraméteres modellek …"

---

## C. MISCITED: reference-list entries

| # | Problem | Evidence | Fix |
|---|---|---|---|
| [46] | Title wrong ("Transformer" vs "Transformers") | README BibTeX: `@software{vonwerra2020trl, title = {{TRL: Transformers Reinforcement Learning}}, … year = {2020}}` (https://github.com/huggingface/trl) | "TRL: Transformers reinforcement learning". The dated claims (GOLD 2025-11-06, tool calling 2026-08-26) come from the release notes (https://github.com/huggingface/trl/releases/tag/v0.25.0, …/v1.11.0), so cite the docs or release pages for them, or add "release notes" to the access note |
| [48] | Title incomplete | README BibTeX: `@software{distillkit2024, title = {DistillKit: Flexible Knowledge Distillation for Large Language Models}, author = {Goddard, Charles and Atkins, Lucas}, year = {2024}, publisher = {Arcee AI}}` | C. Goddard and L. Atkins, "DistillKit: Flexible knowledge distillation for large language models," Arcee AI, GitHub repository, 2024. … |
| [50] | Peer-reviewed venue missing | https://aclanthology.org/2025.emnlp-demos.60/: Proc. 2025 Conf. EMNLP: System Demonstrations, Suzhou, China, Nov. 2025, pp. 787–795, doi:10.18653/v1/2025.emnlp-demos.60 | C. Wang, J. Yan, W. Cai, Y. Yue, and J. Huang, "EasyDistill: A comprehensive toolkit for effective knowledge distillation of large language models," in Proc. 2025 Conf. Empirical Methods Natural Lang. Process.: Syst. Demonstrations, Suzhou, China, 2025, pp. 787–795, doi: 10.18653/v1/2025.emnlp-demos.60. (All five authors are now listed, since IEEE spells out up to six.) |
| [52] | Incomplete | https://aclanthology.org/2025.acl-long.885/: Vol. 1 Long Papers, Vienna, Austria, July 2025, pp. 18085–18108, doi:10.18653/v1/2025.acl-long.885 | "… in Proc. 63rd Annu. Meeting Assoc. Comput. Linguistics (Vol. 1: Long Papers), Vienna, Austria, 2025, pp. 18085–18108, doi: 10.18653/v1/2025.acl-long.885." |
| [53] | DOI missing (other ACL entries carry DOIs) | https://aclanthology.org/2020.acl-demos.2/: Online, July 2020, pp. 9–16, doi:10.18653/v1/2020.acl-demos.2 | Append "Online, 2020, pp. 9–16, doi: 10.18653/v1/2020.acl-demos.2." |
| [41] | Used for a number that is only on the model card | see A6 | add the model-card reference |
| [49] | OK as is | README BibTeX: `@software{torchtune, title = {torchtune: PyTorch's finetuning library}, author = {torchtune maintainers and contributors}, … year = {2024}}` | Optional: the README's own URL has a typo (`https//github.com/pytorch/torchtune`); the thesis's meta-pytorch URL is the right one |

**Entries [28], [38]–[40], [42]–[45], [47], [51] are all correct.** Each was checked against its primary page; details in section F.

**Renumbering check:** old [28]–[41] map to new [29]–[37] and [54]–[58] with byte-identical text, checked by script on the diff. The in-text numbers outside chapter 3 were also checked: [54] Gemma 3 270M, [55] Qwen3, [56] Llama 3.2, [57] Octoverse, [58] HUN-REN. First citations appear in ascending order.

---

## D. MISSING (a reviewer would expect)

**D1. Teacher annealing (KL→CE schedule).** 3.1.6 and 3.3 cite only a toolkit, TextBrewer, for "arányuk tanítás közbeni változtatása sem új". The method papers are missing:
- **BAM!** (Clark et al., ACL 2019, arXiv 1907.04829): "teacher annealing … gradually transitions the model from distillation to supervised learning". This is exactly the thesis's kl-to-ce arm (KL share 1 → 0).
- **Peng et al.**, "Pre-training distillation for large language models: A design space exploration" (arXiv 2410.16215; now ACL 2025 long, Anthology 2025.acl-long.181 per search; that Anthology entry was not opened). Linear Dec α 1→0 scored "39.2, ↑4.1%" and WSD-α + WSD-LR "↑8.0%" for a 1.9B student distilled "from scratch" from GLM-4-9B.
- **Fix:** "A KL- és a CE-veszteség arányának tanítás közbeni változtatása sem új. A BAM-módszer [a] már 2019-ben a desztillációtól a helyes válaszokon történő tanítás felé lineárisan átúsztatott tanítást javasolt, Peng és társai [b] pedig nagy nyelvi modellek előtanító desztillációjában a desztillációs arány fokozatos csökkentésével értek el jobb eredményt. Eszközszinten a TextBrewer [53] már 2020-ban támogatta …"

**D2. LLM-KD objectives.** The thesis uses forward KL and offers reverse KL, JS and TVD, but cites no LLM-KD objective paper:
- MiniLLM (Gu et al., ICLR 2024, arXiv 2306.08543): "replace the forward KLD … with reverse KLD, which is more suitable for KD on generative language models". The current arXiv title is "MiniLLM: On-Policy Distillation of Large Language Models"; the ICLR 2024 title was "MiniLLM: Knowledge distillation of large language models" (that ICLR title was not re-opened, UNVERIFIED).
- DistiLLM (Ko et al., ICML 2024, arXiv 2402.03898): skew KL.

A reviewer will ask why forward KL. One sentence plus a citation is enough.

**D3. From-scratch students with an LLM teacher's distribution exist at about 140M.** This bears on 3.3's "legalább néhány száz millió" and on the positioning.
- MobileLLM-R1 (ICLR 2026): 140.2M / 359.4M / 949.2M, "initialized from scratch", KL to Llama-3.1-8B-Instruct logits in mid-training.
- Busbridge [31] (cited): students from 143M.
- Anshumann [52] (cited): students 300M–3B.
- **Fix:** add one sentence to 3.1.5 or 3.1.6, e.g. "A tanár eloszlásán, alapoktól tanított általános célú tanulómodellek már 140 millió paraméter körül is léteznek [w], [31], ezek azonban általános előtanításon esnek át, és a tanár teljes szótárát használják."

**D4. Reduced or task-subset vocabularies besides TinyStories / VocabTailor.**
- Ushio et al. 2023 (above).
- VocabTrim (Goel et al., ICML 2025 ES-FoMo workshop, arXiv 2506.22694): a drafter LM head restricted to tokens "most frequently sampled from the vocabulary of the target model", "training-free". This is the closest in spirit to a vocabulary derived from the teacher's own outputs.
- Reduced-vocabulary students under KD:
  - Zhao et al., "Extremely small BERT models from mixed-vocabulary training", EACL 2021 (arXiv 1909.11687): "existing distillation methods can't directly train students with reduced vocabularies".
  - Kolesnikova et al. 2022 (arXiv 2205.02340, preprint): "it is impossible to directly apply KL-based knowledge distillation"; 17×–49× compression.
- These strengthen the thesis's point that a strict subset of the teacher's ids avoids the alignment problem these papers had to solve.

**D5. Cross-tokenizer KD beyond ULD.** Low priority; one clause is enough. TRL v1.17.0 (2026-10-29) added X-Token to GOLD; the docs show `xtoken_loss_type` "p_kl" / "h_kl". Also DSKD (EMNLP 2024) and MinED (FuseLLM, ICLR 2024); these two are from the research notes and were not re-opened (UNVERIFIED).

**D6. Capacity gap.** Cho & Hariharan (ICCV 2019) and Mirzadeh et al. (AAAI 2020); see A5. These are needed if the chapter says KL > CE is "expected" at a 700x ratio.

**D7. Small agent models below the 0.5B line.**
- FunctionGemma (https://huggingface.co/google/functiongemma-270m-it): built on Gemma 3 270M, "a foundation for creating your own specialized function calling models"; Mobile Actions 58% base and 85% fine-tuned. The release date is not on the card (UNVERIFIED).
- Grey literature: Needle (https://huggingface.co/Cactus-Compute/needle), "26m parameter", own SentencePiece vocabulary of 8192, own 200B-token pre-training, function-call data via Gemini, no accuracy numbers. A reviewer who follows on-device agents may raise it. It is text-level and general-pretrained, so it does not break the positioning, but naming it pre-empts the objection.
- EasyDistill 1.0 also has agent distillation (AgentKD, `configs/agentkd_distill.json` on the 1.0 branch).
- **Fix:** "Előtanított, 270 millió paraméteres függvényhívó alapmodell is elérhető [FunctionGemma], és nem lektorált modellkártyák szerint 2026-ban már néhány tízmillió paraméteres, saját előtanítású függvényhívó modell is megjelent [Needle], ezek azonban a tanár szövegéből tanulnak, és pontossági eredményt nem közölnek."

**D8. Missing survey.** Xu et al., "A survey on knowledge distillation of large language models", arXiv 2402.13116 (v4, Oct 2024). A literature review chapter usually anchors to one survey. Low priority.

---

## E. MISSING citation for a factual claim

| Phrase | Fix |
|---|---|
| "Ez utóbbi a szakirodalom alapján várható irány" | cite [16], [27], [29], [31] (see A5) |
| "több eszköz pedig a tanár legvalószínűbb tokenjeire szűkíti a veszteséget, miközben a tanulómodell kimeneti rétege a teljes szótárat megtartja" | add [48], [50]. DistillKit README: "Select top-k logits from teacher output"; EasyDistill 1.0 `"top_logits_num": 10` |
| "A TRL például csak a tanár- és a tanulómodell szótárméretét veti össze." | add [46]. Source: `gkd_trainer.py` lines 177–183 and `distillation_trainer.py` lines 740–746, both comparing `config.get_text_config().vocab_size`. Note that these are live-model checks; TRL's trainers use an online teacher, not stored data |
| "A legtöbb kis nyelvi modell általános célú, nagy szövegkorpuszon előtanított modell." | add [40] |
| Table 3.1, "Korai feladatspecifikus desztilláció … előtanítás nélkül" | Tang [28] uses "the traditional 300-dimensional word2vec embeddings trained on Google News" (paper §4.2). Cell fix: "előtanítás nélkül (Tang és társainál előtanított szóbeágyazásokkal)" |

---

## F. MINOR

- **Tang 0.96M** (3.1.1, "egy mindössze 0,96 millió paraméteres … BiLSTM"):
  - Table 2 gives "BiLSTMSOFT 0.96" and "BERTLARGE 335 (349×)".
  - The 0.96M matches the BiLSTM plus classifier only. With a 600-d multichannel input, 150 hidden units per direction and a 200-unit ReLU layer: 2·4·150·(600+150+1) + 300·200+200 + 200·2+2 ≈ 0.96M. The word-embedding table is excluded. This is my own calculation; the paper does not say whether embeddings are counted.
  - The 90.7 also used an augmented transfer set, while the 86.7 baseline did not ("We apply our distillation approach of matching logits using the augmented training dataset").
  - Suggested: "A modell a BERT-large-nál 349-szer kevesebb paraméterrel (a szóbeágyazások nélkül számolva) …"
- **VocabTailor:** the head is "a hybrid static-dynamic vocabulary selection strategy for LM Head". Math is "5,135 + [14]" (static + average dynamic input tokens), with the "embedding layer … offloaded" (https://arxiv.org/html/2508.15229v2). "rögzített tokenkészletet" is accurate for the static part. Optional addition: "amelyet futás közben a bemenet tokenjei egészítenek ki".
- **Gemma 3 vs Anshumann (3.1.6):** Gemma 3 samples 256 logits "weighted by teacher probabilities". Anshumann et al. show that random sampling of this kind is the unbiased route, while fixed Top-K is the biased one. The thesis's fixed subset belongs to the Top-K family. The current text lists both as plain "subset" cases. Optional: "A Gemma 3 mintavételezése a torzítatlan, a dolgozat rögzített részhalmaza a Top-K-hoz hasonló, torzított esetbe tartozik."
- **Busbridge, old user-written sentence:** "és ha a modell általános képességeit nézzük, akkor ez a tömörítés veszteséges". The paper also shows "a student can also outperform its teacher" (weak-to-strong, mainly when the student is larger), and the law depends on the teacher's cross-entropy, not only on student size and compute. Suggest "kisebb tanulómodellnél ez a tömörítés jellemzően veszteséges".
- **Hinton, old text:** "A klasszikus tudásdesztilláció alapötletét Hinton és társai [16] fogalmazták meg". Buciluǎ, Caruana and Niculescu-Mizil, "Model compression", KDD 2006, doi:10.1145/1150402.1150464, preceded it. Hinton et al. cite this line of work themselves. Suggest "a ma használt formájában Hinton és társai [16] fogalmazták meg".
- **Table row 1, "Cél: általános képességek":** R1-Distill targets reasoning (3.1.1 says "kifejezetten az érvelési képességek átadására"). Use "általános vagy érvelési képességek".
- **Lu et al. "sorolja ide":** the abstract reads "Focusing on transformer-based, decoder-only language models with 100M-5B parameters". This is a survey scope, not a definition. Use "vizsgálja".
- **TRL GKD** now lives in `trl.experimental.gkd`, and `trl.DistillationTrainer` is the stable successor (v1.10.0, 2026-08-13; v1.11.0 adds "Suggest migrating to DistillationTrainer when the GKD config is fully covered"). Optional mention.
- **3.3 "néhány millió paraméteres"** vs the working 6.1–54M students and the 16.5M main student. "néhány millió – néhány tízmillió" is more exact. This is the other reviewer's area; noted only because it sits in the positioning sentence.

---

## G. Positioning check (3.3 contribution and the "not documented" claims)

- **Contribution sentence** ("practical combination … in a setting not found in the reviewed literature"):
  - This holds. No counterexample was found for an LLM teacher distribution + a from-scratch generative student under 100M + one task or agent + a fixed task-derived subset of the teacher's tokens as the whole head.
  - This pass ran two extra extended searches on top of the research notes' ~28 searches and 3 arXiv-API passes.
  - The nearest items each miss at least one part:
    - Fu 2023: pretrained, ≥250M.
    - GKD: pretrained T5-small 77M.
    - MobileLLM-R1: 140M, general.
    - Kim & Rush: 8–84M, non-LLM teacher.
    - TextBrewer: 14–17M random-init encoders, BERT teacher.
    - Needle: 26M, text only, general pre-training first.
    - VocabTailor and VocabTrim: subset heads, no training.
  - Keep the hedge "a feldolgozott szakirodalomban". Add Fu et al. (A1) so the specialisation idea is credited, not implied as the thesis's own.
- **3.1.6 last paragraph:**
  - **Holds as worded (dated, docs-level):** the "not documented" items on the task-subset head with dense subset logits, and on the vocabulary fingerprint. In the DistillKit source, the only hash is a dataset cache key over `{"path", "seed", "num_samples"}` (`main.py`). The README asks the user to keep capture and training configs consistent by hand.
  - **Needs fixes:** "néhány millió paraméteres tanulómodellek" (B3) and "rögzített súlyokkal" (B4).

---

## H. Verified as correct (primary page opened in this pass)

**Distillation papers**
- Kim & Rush [27]:
  - Students 84M / 49M (En→De) and 8M (Th→En), task-only.
  - Word-KD, Seq-KD and their combination are all defined and tested.
  - "word embeddings account for approximately 63% (50m out of 84m)".
- Tang et al. [28]:
  - 0.96M BiLSTM, word2vec, logits of fine-tuned BERT-large.
  - SST-2 90.7 vs 86.7; "349 times fewer parameters than … BERTLARGE".
  - The reference entry is exact (arXiv-only paper).
- Gemma 2 [29]: 2B and 9B distilled, 27B "trained from scratch"; Table 6 60.3 vs 67.7, 500B tokens, 7B teacher.
- Gemma 3 [12]:
  - "All Gemma 3 models are trained with knowledge distillation".
  - "We sample 256 logits per token, weighted by teacher probabilities".
  - "set to zero probability for non-sampled logits, and renormalized"; the student uses CE.
- DeepSeek-R1-Distill [30]: 800k samples, 1.5B–70B Qwen/Llama bases, fine-tuned on R1 text.
- Busbridge [31]: distillation beats supervised learning with an existing teacher "up to a compute level that scales predictably with student size"; ICML 2025.
- GKD [47]: authors, title, "Accepted at ICLR 2024".
- ULD [51]: authors, title, TMLR January 2025.
- Anshumann [52]: "caching Top-K probabilities … provide biased estimates"; authors as listed (entry incomplete, see C).

**Agents and small models**
- Kang et al. [37]: teacher Qwen2.5-32B-Instruct; agent-distilled students 0.5B, 1.5B, 3B and 7B; 0.5B competitive with the next-tier CoT-distilled model.
- TinyAgent [38]:
  - TinyLlama-1.1B base, GPT-4-Turbo-generated plans.
  - Own 1K test set: 80.06% (16-bit) vs GPT-4-Turbo 79.08%.
  - Anthology entry exact (authors, pp. 80–88, DOI).
- Belcak [39]: SLM = "can fit onto a common consumer electronic device"; "most models below 10bn parameters"; the eight authors as listed.
- Lu et al. [40]: 100M–5B scope; the eight authors as listed.
- SmolLM2 [41]: 1.7B on "~11 trillion tokens"; title and first authors.
- TinyStories [42]: "below 10 million total parameters"; GPT-3.5 and GPT-4 data; "only keep the top 10K most common tokens".
- MobileLLM [43]: "embedding layers account for more than 20% of the total parameters of a 125M-parameter model"; ICML 2024.
- Tao et al. [44]: optimal vocabulary grows with compute and model size; NeurIPS 2024; the eight authors as listed.
- VocabTailor [45]:
  - Math 5,135 static of 32,000 (rho-math-1b); LM-head subset; no retraining.
  - Anthology entry exact (authors, Findings ACL 2026, pp. 28453–28471, DOI).

**Toolkits**
- TRL GKD: student-generated outputs scored by the teacher (`lmbda`), teacher-generated (`seq_kd`), gold outputs (`lmbda=0`); generalized JSD via `beta`.
- TRL GOLD: "an extension of Universal Logit Distillation (ULD)"; first in v0.25.0, released 2025-11-06.
- TRL DistillationTrainer:
  - "DistillationTrainer supports tool calling" in v1.11.0, 2026-08-26; "Tool-result tokens are masked out of the loss".
  - The teacher "must share the student's vocabulary"; checked only by `vocab_size`.
- TRL documented student examples are all pretrained and ≥0.5B (Qwen2-0.5B, Qwen2.5-0.5B, Llama-3.2-1B, Qwen3-VL-2B, LFM2.5-VL-1.6B).
- DistillKit: offline capture with top-k compression, per-loss fixed weight and temperature, kl/jsd/tvd, consistency rule stated only in the README; examples Qwen3-8B and 70B→8B.
- torchtune:
  - "no longer actively maintained … wound down in 2025".
  - Fixed `kd_ratio` (default 0.5); Forward, Reverse and Symmetric KL losses; KD example 1.5B→0.5B.
  - Reference [49] content OK.
- EasyDistill 1.0: white-box KD from stored top-10 logits, fixed `kd_ratio`, forward/reverse KLD, student Qwen2.5-0.5B; the abstract says "black-box and white-box knowledge distillation".
- TextBrewer [53]: the 2020 paper says "Schedulers are used to adjust loss weight or temperature dynamically"; ACL 2020 demos, pp. 9–16; authors as listed.

**Old user-written 3.1.2–3.1.4 claims, not re-opened in this pass** (unchanged text: GPTQ, Dettmers & Zettlemoyer, LoRA, QLoRA, Biderman, ReAct, Claude Code): no change was made there and these were outside the new material.
