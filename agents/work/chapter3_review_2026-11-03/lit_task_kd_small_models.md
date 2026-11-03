# Fact sheet: task-specific and from-scratch distillation, small LMs and small agents

Checked 2026-11-03 for chapter 3 (literature review). Scope: prior work on task-specific or from-scratch
distillation, plus small language models and small agents. Distillation toolkits and cross-tokenizer
distillation are covered by another agent.

Method: arXiv abstract and HTML pages, ACL Anthology entries, and model cards, opened one by one. Where a
PDF was the only source, I ran pdftotext locally and read the tables. All numbers below come from the pages
themselves unless an item says UNVERIFIED. These are fact sheets, not thesis prose.

Not re-checked (verified 2026-11-01): TinyStories, phi-1, TinyGSM, MobileLLM, SmolLM2, Tao et al. 2024,
the Gemma 3 270M blog, Kang et al. 2025. I re-opened Kang et al. only to confirm student sizes (see 3.4).

---

## 1. Pre-LLM task-specific distillation (from-scratch students already exist)

### 1.1 Hinton, Vinyals, Dean 2015
- **IEEE:** G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," presented
  at the NIPS 2014 Deep Learning Workshop, 2015, arXiv:1503.02531.
- **Student:** MNIST: a small net with 2 hidden layers of 800 ReLU units and no regularisation, trained from
  scratch. Teacher: 2x1200 ReLU, heavily regularised (dropout, jittered inputs). Speech: the distilled model
  has the *same* architecture as the baseline (8 hidden layers x 2560 ReLU, 14,000 HMM-state softmax,
  about 85M parameters). It is distilled from an ensemble of 10 such models, so this is ensemble compression,
  not size compression.
- **Learns from:** the teacher's softened distribution (T = 20 on MNIST; T in {1, 2, 5, 10} for speech), plus
  hard-label CE with a lower weight (0.5 relative weight on the hard targets for speech).
- **Key numbers:**
  - MNIST test errors: teacher 67; small net without distillation 146; small net with distillation 74.
  - Speech frame accuracy / WER: baseline 58.9% / 10.9%; 10x ensemble 61.1% / 10.7%; distilled single
    model 60.8% / 10.7%. "More than 80%" of the ensemble's frame-accuracy gain transfers.
  - **Soft targets as a regulariser (Table 5):** with 3% of the speech data, hard targets give 44.5% test
    frame accuracy (with early stopping), soft targets 57.0% (no early stopping needed). This is a direct
    precedent for "KL helps most when data is limited".
  - MNIST, 2x30 units: temperatures of 2.5 to 4 worked best. With 300 or more units, any T above 8 was similar.

### 1.2 Kim & Rush 2016: sequence-level KD (the closest method precedent)
- **IEEE:** Y. Kim and A. M. Rush, "Sequence-level knowledge distillation," in *Proc. 2016 Conf. Empirical
  Methods in Natural Language Processing (EMNLP)*, Austin, TX, USA, Nov. 2016, pp. 1317–1327,
  doi: 10.18653/v1/D16-1139.
- **Task:** NMT. WMT 2014 En→De (4M sentences, top-50k word vocabulary, rest UNK) and IWSLT 2015 Th→En
  (90k sentences, 25k vocabulary).
- **Teacher / students:**
  - En→De: teacher 4x1000 LSTM (221M); students 2x500 (84M) and 2x300 (49M).
  - Th→En: teacher 2x500 (47M); student 2x100 (8M).
  - Students are task-only models with no general pre-training. The paper does not state "random init" in
    words; training "mirrors Luong et al. (2015)" (see UNVERIFIED).
  - Teacher and student share the task's word vocabulary.
- **Three variants, defined in the paper:**
  - **Word-KD:** the student is trained on the original data and also minimises CE to the teacher's
    per-token distribution, mixed with NLL (α = 0.5 worked better than 0.9).
  - **Seq-KD:** the student is trained on the teacher's beam-search output (beam 5). These are hard
    sequences, so only text is used, not the distribution.
  - **Seq-Inter:** the student is fine-tuned on the beam hypothesis closest to gold by BLEU (beam 35).
  - Combinations are allowed. "Seq-KD + Word-KD" means training on teacher-generated text while still
    matching the teacher's token distribution. That is structurally the thesis's *generated* mode (teacher
    writes, plus stored logits).
- **Key numbers (En→De 2x500, BLEU greedy / beam 5):**
  - Baseline 14.7 / 17.6; Word-KD 15.4 / 17.7; Seq-KD 18.9 / 19.0; Seq-KD + Seq-Inter + Word-KD
    18.7 / 18.9; teacher 17.7 / 19.5.
  - Speed: the student with greedy decoding runs 10x faster than the teacher with beam search on GPU
    (1051.3 vs 101.9 words/s).
  - Seq-KD students have *higher* perplexity (22.7 vs 8.2) but better BLEU. This shows that perplexity does
    not track task quality.
  - **Embedding share:** in the 2x500 student, word embeddings are about 63% of parameters (50M of 84M).
    This is an early "the vocabulary dominates small models" data point, alongside MobileLLM and Gemma 3 270M.
  - Pruning: 2x500 with 90% of weights pruned has 8M parameters and reaches BLEU 18.5 (26x fewer than the teacher).

### 1.3 Tang et al. 2019: BERT into a BiLSTM
- **IEEE:** R. Tang, Y. Lu, L. Liu, L. Mou, O. Vechtomova, and J. Lin, "Distilling task-specific knowledge
  from BERT into simple neural networks," arXiv:1903.12136, Mar. 2019.
- **Student:** a single-layer BiLSTM (a siamese version for sentence pairs) with 0.96M parameters
  (Table 2). A 300-unit variant has 2.2M. It is trained from scratch, **except** that it uses pre-trained
  300-d word2vec embeddings (Google News, multichannel). It is a classifier, not a generator.
- **Teacher:** BERT-large (335M in Table 2), fine-tuned on each task.
- **Learns from:** MSE between student and teacher **logits**. MSE "performed slightly better" than soft
  CE. The best runs use only the distillation loss (α = 0).
- **Transfer set:** the original data plus augmentation (masking, POS-guided word replacement, n-gram
  sampling; p_mask = p_pos = 0.1, p_ng = 0.25; n_iter = 20 for SST-2, 10 for MNLI/QQP), labelled by the teacher.
- **Key numbers (GLUE test):**
  - BiLSTM_SOFT: SST-2 90.7, QQP 68.2/88.1, MNLI-m/mm 73.0/72.6.
  - Same BiLSTM without distillation: 86.7, 63.7/86.2, 68.7/68.3.
  - BERT-large: 94.9, 72.1/89.3, 86.7/85.9.
  - Size and speed: 349x fewer parameters than BERT-large and 434x faster; 98x fewer than ELMo and 15x
    faster.

### 1.4 Counter-cases: students initialised from, or pre-trained like, the teacher
- **DistilBERT.** V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT:
  smaller, faster, cheaper and lighter," presented at the 5th Workshop on Energy Efficient Machine Learning
  and Cognitive Computing at NeurIPS 2019, arXiv:1910.01108.
  - Student: 6 layers, 66M parameters (BERT-base 110M). It is **initialised from the teacher by taking one
    layer out of two**, then distilled during general pre-training (Wikipedia + Toronto Books) with a triple
    loss (MLM + soft-target CE + cosine).
  - GLUE dev macro-score 77.0 vs BERT-base 79.5 ("97%"); 40% fewer parameters, 60% faster.
  - **Ablation:** replacing teacher initialisation with random initialisation costs **−3.69** GLUE macro-score.
- **TinyBERT.** X. Jiao, Y. Yin, L. Shang, X. Jiang, X. Chen, L. Li, F. Wang, and Q. Liu, "TinyBERT:
  Distilling BERT for natural language understanding," in *Findings of the Association for Computational
  Linguistics: EMNLP 2020*, Online, Nov. 2020, pp. 4163–4174, doi: 10.18653/v1/2020.findings-emnlp.372.
  - Student TinyBERT4: M = 4 layers, d = 312, 14.5M parameters, the BERT tokenizer. Training has two stages:
    general distillation on English Wikipedia (the student is distilled from BERT's layers on general text),
    then task-specific distillation with data augmentation.
  - GLUE test average 77.0 vs teacher BERT-base 79.5; 7.5x smaller, 9.4x faster.
  - **Ablation (dev, 4 tasks):** full 75.6; without general distillation 72.5; without task distillation 68.5;
    without augmentation 68.4. The "without general distillation" row is the nearest BERT-era analogue of a
    task-only student, and it loses about 3 points.
- **Take-away:** BERT-era compression mostly used students that were initialised from the teacher or
  pre-trained on general text. Task-only students (Tang; Kim & Rush; pQRNN below) existed but were
  classifiers, or NMT models whose teachers were the same kind of model.

### 1.5 Other small task-specific students (brief)
- **pQRNN.** P. Kaliamoorthi, A. Siddhant, E. Li, and M. Johnson, "Distilling large language models into
  tiny and effective students using pQRNN," arXiv:2101.08890, Jan. 2021.
  - An embedding-free projection encoder distilled from mBERT for multilingual semantic parsing.
  - 350x smaller than mBERT, reaching 95.9% (MTOP) and 97.1% (mATIS) of the teacher's performance.
  - The abstract gives no absolute parameter count.
- **Generation-distillation.** L. Melas-Kyriazi, G. Han, and C. Liang, "Generation-distillation for efficient
  natural language understanding in low-data settings," EMNLP 2019 Workshop on Deep Learning for
  Low-resource NLP, arXiv:2002.00733.
  - A large fine-tuned LM generates unlabelled examples, which are used to distil into a small classifier.
  - Comparable to BERT with 300x fewer parameters.
- **ZeroGen.** J. Ye, J. Gao, Q. Li, H. Xu, J. Feng, Z. Wu, T. Yu, and L. Kong, "ZeroGen: Efficient
  zero-shot learning via dataset generation," in *Proc. EMNLP 2022*, arXiv:2202.07922.
  - A PLM synthesises a dataset, and a tiny task model (for example an LSTM) is trained on it, for
    classification, QA and NLI.
  - The student learns from text only and is not generative.

---

## 2. LLM-era task-specific distillation into small models

### 2.1 Distilling step-by-step (Hsieh et al. 2023)
- **IEEE:** C.-Y. Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y. Fujii, A. Ratner, R. Krishna, C.-Y. Lee, and
  T. Pfister, "Distilling step-by-step! Outperforming larger language models with less training data and
  smaller model sizes," in *Findings of the Association for Computational Linguistics: ACL 2023*, Toronto,
  Canada, Jul. 2023, pp. 8003–8017, doi: 10.18653/v1/2023.findings-acl.507.
- **Student:** T5-Base 220M, T5-Large 770M, T5-XXL 11B, "initialize[d] … with pretrained weights", with the
  T5 vocabulary.
- **Teacher:** PaLM 540B (few-shot CoT). The student learns from **text only**: the teacher's rationales
  plus labels, trained multi-task (one task predicts the label, the other the rationale).
- **Tasks:** e-SNLI and ANLI (NLI), CQA (commonsense QA), SVAMP (math word problems; also used as the
  thesis's human test set).
- **Key numbers:**
  - On ANLI, a 770M T5 beats 540B PaLM few-shot CoT using 80% of the training set; standard fine-tuning
    fails even at 100%.
  - The paper reports models "up to 2000×" smaller than the LLM. With unlabelled data only, an 11B T5 beats
    PaLM on 3 of 4 datasets.
  - The exception is SVAMP (only about 800 training points); augmenting with ASDiv closes the gap.

### 2.2 GKD / on-policy distillation (Agarwal et al. 2024): the only ≤100M generative student found in the LLM-KD mainstream
- **IEEE:** R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos, M. Geist, and O. Bachem, "On-policy
  distillation of language models: Learning from self-generated mistakes," in *Proc. Int. Conf. Learning
  Representations (ICLR)*, 2024, arXiv:2306.13649.
- **Student:** T5-small (**77M**), T5-base (250M), T5-large (800M). All are **pre-trained** (LM-adapted
  T5 v1.1; FLAN-T5 for GSM8K) and then supervised fine-tuned before distillation. They share the T5
  vocabulary.
- **Teacher:** supervised fine-tuned T5-XL (about 3B). This is not a modern LLM.
- **Tasks:** XSum summarisation, WMT14 en-de translation, GSM8K, plus FLAN instruction tuning.
- **Taxonomy (useful for the thesis's two data modes):**
  - "Supervised KD" means ground-truth outputs + the teacher's token-level distribution. This is the
    thesis's *teacher-forced* mode.
  - "SeqKD" means supervised fine-tuning on teacher-generated outputs, with no distribution.
  - On-policy GKD means the student generates and the teacher scores.
- **Key numbers:** averaged over student sizes, on-policy GKD improves over the initial students by about
  2.1x (summarisation), 1.7x (translation) and 1.9x (GSM8K), relative to the baseline KD methods.

### 2.3 Tiny classification student from an LLM (education)
- E. Latif, L. Fang, P. Ma, and X. Zhai, "Knowledge distillation of LLM for automatic scoring of science
  education assessments," accepted to AIED 2024, arXiv:2312.15842.
  - A fine-tuned LLM's prediction probabilities are used as soft labels for a student of about **0.03M**
    parameters (roughly 4,000x smaller).
  - Task: scoring student answers (the 7T dataset has 6,684 responses).
  - Scoring accuracy is 3% above an ANN and 2% above TinyBERT; inference is about 10x faster.
  - It is a classifier, not a generator.

### 2.4 From-scratch students trained on an LLM teacher's token distribution (general pre-training, not one task)
These are the nearest *method* matches (from scratch + teacher distribution). All of them are
general-purpose, use the full vocabulary, and have at least 140M parameters, except Baby Llama and TF3, whose
teachers are small models.
- **Gemma 3.** Gemma Team, Google DeepMind, "Gemma 3 technical report," arXiv:2503.19786, 2025.
  - "All Gemma 3 models are trained with knowledge distillation."
  - Section 2.2 quotes: "We sample 256 logits per token, weighted by teacher probabilities." and "The
    teacher's target distribution is set to zero probability for non-sampled logits, and renormalized."
  - The student learns the distribution inside the sample with CE.
  - Sizes 1B (2T tokens), 4B (4T), 12B, 27B.
  - Vocabulary is "262k entries" in the text but "256k" in the Table 1 caption; the report is internally
    inconsistent.
  - **Relevance:** renormalising the teacher's distribution over a subset of tokens is Google's own
    production recipe for the thesis's teacher family. In Gemma 3 the subset is sampled *per token* and the
    student keeps the *full* head. The thesis uses a *fixed, task-derived* subset and shrinks the student's
    head to it.
- **Pre-training distillation.** H. Peng, X. Lv, Y. Bai, Z. Yao, J. Zhang, L. Hou, and J. Li, "Pre-training
  distillation for large language models: A design space exploration," arXiv:2410.16215, Oct. 2024.
  - Teacher GLM-4-9B; a 1.9B student "from scratch"; students of 330M, 670M, 1.9B, 3.8B, 6.8B.
  - Logits truncated with top-p 0.95 then top-k 100, which cuts storage 4,000x (about 15 TB for 100B tokens
    vs about 58.6 PB for full float32).
  - The 1.9B student averages 38.3 vs 37.7 for the LM-loss baseline (+1.6%).
  - "Larger student LLMs generally benefit more from pre-training distillation."
- **Sparse Logit Sampling.** Anshumann, M. A. Zaidi, A. Kedia, J. Ahn, T. Kwon, K. Lee, H. Lee, and J. Lee,
  "Sparse logit sampling: Accelerating knowledge distillation in LLMs," ACL 2025 (oral), arXiv:2503.16870.
  - Caching only the **top-K** teacher probabilities gives biased estimates of the teacher's distribution,
    with worse performance and calibration. Importance-sampled sparse logits are unbiased.
  - Overhead is under 10% vs CE; models from 300M to 3B.
  - **Relevance:** this is the main argument a reviewer could raise against renormalising over a fixed
    subset. It is a reason to report the teacher mass that falls inside the vocabulary (see the retained-mass
    reminder in memory).
- **MobileLLM-R1.** C. Zhao, E. Chang, Z. Liu, C.-J. Chang, W. Wen, C. Lai, S. Cao, Y. Tian,
  R. Krishnamoorthi, Y. Shi, and V. Chandra, "MobileLLM-R1: Exploring the limits of sub-billion language
  model reasoners with open training recipes," ICLR 2026, arXiv:2509.24945.
  - Sizes 140.2M, 359.4M, 949.2M. "Models are initialized from scratch."
  - The **mid-training** stage uses KD with "Llama-3.1-8B-Instruct model as the teacher", minimising KL
    between student and teacher logits. Section 3 calls the teacher "LLaMa3-8B", an inconsistency.
  - LLaMA 3.2 tokenizer with a 128k vocabulary; tied embeddings.
  - The 950M model scores AIME 15.5 vs 0.6 (OLMo-2-1.48B) and 0.3 (SmolLM2-1.7B).
- **Strong teacher not needed?** T. Lu and Z. Liu, "Strong teacher not needed? On distillation in LLM
  pretraining," arXiv:2605.23857, 2026 (arXiv comment: NeurIPS 2026).
  - Llama-3 architectures of 0.7B, 1.7B, 3.8B, 8.0B; 128,256 vocabulary; forward KL; FineWeb-Edu.
  - Weak or undertrained teachers can help; scaling the teacher up "can saturate or even reverse" the gains.
- **Baby Llama.** I. Timiryasov and J.-L. Tastet, "Baby Llama: knowledge distillation from an ensemble of
  teachers trained on a small dataset with no performance penalty," in *Proc. BabyLM Challenge at the 27th
  Conf. Computational Natural Language Learning*, Singapore, Dec. 2023, pp. 279–289,
  doi: 10.18653/v1/2023.conll-babylm.24.
  - A 58M LLaMA student distilled from an ensemble of GPT-2 and small LLaMA teachers, all trained on the
    10M-word BabyLM data.
  - The student "exceeds in performance both of its teachers". The teachers are small, not LLMs.
- **BabyLM 2025 interaction track (rules).** L. Charpentier et al., "BabyLM turns 3: Call for papers for the
  2025 BabyLM workshop," arXiv:2502.10645, Feb. 2025.
  - Pre-trained LMs may act as teachers (from a fixed list).
  - The student sees at most 100M words, including teacher-generated text.
  - **The teacher's "weights, hidden states, or output distribution cannot be revealed to the submission
    model."** So the main from-scratch small-LM benchmark with LLM teachers rules out distribution-level KD
    by design.
- **TF3-RO-50M.** M. D. Nadas, L. Diosan, A. Tomescu, and A. Piscoran, "TF3-RO-50M: Training compact
  Romanian language models from scratch on synthetic moral microfiction," arXiv:2601.10410, Jan. 2026.
  - A 51.65M LLaMA-style model trained from scratch, then pruned and distilled with logit-based KD into a
    26.45M student. The teacher is itself small and from scratch.
  - General story language model; Romanian-specific tokenizer.
- **FBI-LLM.** L. Ma, M. Sun, and Z. Shen, "FBI-LLM: Scaling up fully binarized LLMs from scratch via
  autoregressive distillation," arXiv:2407.07093, Jul. 2024.
  - 130M, 1.3B, 7B fully binarised (1-bit) models trained from scratch with an "autoregressive distillation"
    loss. The teacher is not named in the abstract.
  - Side note: relevant to the ternary-student idea.

### 2.5 Searched for and not found
No paper found in which an LLM is distilled into a ≤100M **generative** student, trained from scratch,
for one task. The ≤100M students found are classifiers (Tang 0.96M, Latif 0.03M, pQRNN, ZeroGen) or
pre-trained T5-small (GKD 77M).

---

## 3. Small models as agents

### 3.1 Belcak et al. 2025 (NVIDIA): position paper
- **IEEE:** P. Belcak, G. Heinrich, S. Diao, Y. Fu, X. Dong, S. Muralidharan, Y. C. Lin, and P. Molchanov,
  "Small language models are the future of agentic AI," arXiv:2506.02153, Jun. 2025 (v3, Sep. 2026).
- **Definition (section 2.1):**
  - "A SLM is a LM that can fit onto a common consumer electronic device", with inference latency low enough
    to serve one user's agentic requests. "An LLM is a LM that is not a SLM."
  - As of 2025 they "would be comfortable with considering most models below 10bn parameters in size to be
    SLMs".
- **Argument:**
  - V1: SLMs are "principally sufficiently powerful" for agentic errands.
  - V2: they are "inherently more operationally suitable".
  - V3: they are "necessarily more economical".
  - Hence SLMs should be the default, in heterogeneous systems that call LLMs selectively.
- **LLM-to-SLM conversion algorithm (S1–S6):**
  - S1 log agent calls; S2 curate and filter (10k–100k examples as a rule of thumb); S3 cluster tasks.
  - S4 pick an SLM; S5 fine-tune with LoRA/QLoRA, optionally with KD from the LLM; S6 iterate.
- **Case-study estimates (not measured):** share of LLM queries that SLMs could take over: MetaGPT about
  60%, Open Operator about 40%, Cradle about 70%.
- **Example SLMs** range from SmolLM2 125M–1.7B to 9B. All are pre-trained; nothing from scratch or under
  100M.
- **Relevance:** their S1–S5 pipeline is close to what the thesis does. The thesis students are 2–4 orders of
  magnitude smaller than their SLM examples and are not fine-tuned from a pre-trained model.

### 3.2 TinyAgent (Erdogan et al. 2024)
- **IEEE:** L. E. Erdogan, N. Lee, S. Jha, S. Kim, R. Tabrizi, S. Moon, C. R. C. Hooper, G. Anumanchipalli,
  K. Keutzer, and A. Gholami, "TinyAgent: Function calling at the edge," in *Proc. 2024 Conf. Empirical Methods
  in Natural Language Processing: System Demonstrations*, Miami, FL, USA, Nov. 2024, pp. 80–88,
  doi: 10.18653/v1/2024.emnlp-demo.9.
- **Student:** TinyAgent-1.1B (from TinyLlama-1.1B instruct-32k) and TinyAgent-7B (from Wizard-2-7B), both
  pre-trained and fine-tuned.
- **Data:** synthetic function-calling plans from GPT-4-Turbo (80K train / 1K val / 1K test, about $500), so
  text only.
- **Harness:** 16 macOS functions, LLMCompiler planning, Tool RAG (a DeBERTa-v3-small multi-label tool
  classifier, 3.97 tools per query on average vs 6 for basic RAG), 4-bit quantisation (group size 32).
- **Success rate:**

  | Model | Success rate |
  |---|---|
  | GPT-3.5 | 65.04% |
  | GPT-4-Turbo | 79.08% |
  | TinyAgent-1.1B | 80.06% (16-bit) / 80.35% (4-bit) |
  | TinyAgent-7B | 84.95% / 85.14% |

### 3.3 Octopus v2 (Chen & Li 2024)
- **IEEE:** W. Chen and Z. Li, "Octopus v2: On-device language model for super agent," arXiv:2404.01744,
  Apr. 2024.
- **Student:** Gemma-2B ("as the pretrained model"), full fine-tune or LoRA.
- **Functional tokens:** each of N functions gets a special token `<nexa_0>`…`<nexa_N-1>` plus `<nexa_end>`,
  and the LM head is expanded by N. Choosing a function becomes one-token classification. This is a
  precedent for task-specific output tokens, though the head is *added to* the vocabulary rather than cut down.
- **Data:** 20 Android APIs; queries generated and verified by Gemini; 1,000 points per API plus an equal
  number of negatives.
- **Key numbers (authors' own evaluation set):** Octopus-0 accuracy 99.524%, 0.38 s; GPT-4 98.571%, 1.02 s;
  Llama-7B + RAG 68.095%, 13.46 s. The abstract claims 35x lower latency than Llama-7B with RAG and a 95%
  shorter context.

### 3.4 Kang et al. 2025 (already verified)
- M. Kang, J. Jeong, S. Lee, J. Cho, and S. J. Hwang, "Distilling LLM agent into small models with retrieval
  and code tools," NeurIPS 2025 (spotlight), arXiv:2505.17612.
- The distilled agents of 0.5B, 1.5B and 3B are competitive with CoT-distilled models of 1.5B, 3B and 7B.
  Students are pre-trained.

### 3.5 Other small tool-use models (pre-trained bases)
- **FunctionGemma** (Google DeepMind), model card "google/functiongemma-270m-it", Hugging Face.
  - Built on Gemma 3 270M (listed as 0.3B, BF16); a base to be fine-tuned for specific function-calling
    tasks, "not intended for use as a direct dialogue model".
  - Mobile Actions accuracy is 58% for the base and 85% after fine-tuning; BFCL 0-shot Simple 61.6.
  - Release date is not on the card. The search snippet says 22 Dec 2025 (see UNVERIFIED).
- **Jhandi et al.** P. Jhandi, O. Kazi, S. Subramanian, and N. Sendas, "Small language models for efficient
  agentic tool calling: Outperforming large models with targeted fine-tuning," AAAI 2026 Workshop on Agentic
  AI Benchmarks and Applications for Enterprise Tasks, arXiv:2512.15943.
  - OPT-350M, fine-tuned (pre-trained base) with one epoch of SFT.
  - ToolBench pass rate 77.55% vs ChatGPT-CoT 26.00% and ToolLLaMA-DFS 30.18%.

### 3.6 ≤100M from-scratch agents: grey literature only (2026 model cards and repos, not peer-reviewed)
- **Needle (Cactus Compute).** https://huggingface.co/Cactus-Compute/needle
  - **26M** encoder-decoder with "pure attention" and no FFN: 12 encoder layers, 8 decoder layers,
    d_model 512.
  - Own SentencePiece BPE with an **8192** vocabulary.
  - Pre-trained from scratch on 200B tokens (16x TPU v6e, about 27 h), then post-trained on 2B tokens of
    function-call data (about 45 min).
  - The data is generated via Gemini (Show HN: "Dataset synthesized via Gemini with 15 tool categories"; a
    commenter notes the weights were not accessed). This is **text-level** distillation from an API LLM.
  - Single-shot function calling. **No evaluation numbers published.** The authors claim it beats
    FunctionGemma-270M, Qwen-0.6B, Granite-350M and LFM2.5-350M.
- **OpenLocalAgent.** https://github.com/sangbumchoi/openlocalagent, 2026, MIT.
  - Seven from-scratch sizes (la-10m 10.5M … la-700m), hybrid conv/attention, 16,384 BPE.
  - Pretrain on 460.7M tokens (FineWeb-Edu, Cosmopedia v2, code), then midtrain and posttrain. All stages
    use plain next-token CE; **no teacher named**.
  - Ten public agent benchmarks; mean step success: la-45m (45.1M) 25.61; "fcall-96m" (previous best from
    scratch, 95.3M) 23.07; fine-tuned Qwen3-0.6B 34.83.
- **LocalAgent tiny-30m-byte.** https://huggingface.co/danelcsb/localagent-tiny-30m-byte
  - 28.3M, byte-level, from scratch.
  - Route, selector and pointer heads, so it is *not* a generative LM.
  - 50 tools; 53% out-of-distribution call-name accuracy.
- **tiny-agent-112m.** https://huggingface.co/darioooooo0o/tiny-agent-112m
  - 112M backbone plus 50M embedding (plus Engram tables), from scratch on 2.51B tokens; 32k BPE.
  - Then **text** trajectories from a 27B Qwen teacher (9.3k episodes), then GRPO.
  - About 91% pass@1 on its 14 synthetic task kinds, 68–74% on rewordings, about 0% on new task kinds.

### 3.7 SLM size definitions (for the "how small is small" line)
- Z. Lu, X. Li, D. Cai, R. Yi, F. Liu, X. Zhang, N. D. Lane, and M. Xu, "Small language models: Survey,
  measurements, and insights," arXiv:2409.15790, 2024 (v3 2025).
  - "transformer-based, decoder-only language models with 100M-5B parameters"; 70 SLMs surveyed.
  - The thesis's 1M–53M students are below that survey's lower bound.
- Belcak et al.: below 10B counts as an SLM (3.1).
- D. Hillier, L. Guertler, C. Tan, P. Agrawal, C. Ruirui, and B. Cheng, "Super tiny language models,"
  arXiv:2405.14159, 2024.
  - Targets 10M–100M (byte-level tokenisation with pooling, weight tying). No distillation; the abstract
    reports no results.

---

## 4. Recency check: does anyone do what the thesis does?

**Thesis setting:**
- (a) the LLM teacher's token distribution (forward KL + CE), with stored bf16 logits;
- (b) a randomly initialised *generative* transformer of about 1M–53M parameters, with **no general
  pre-training at all**;
- (c) one task or one agent, run inside a checking harness;
- (d) an output vocabulary that is a task-derived subset of the teacher's tokenizer, with the teacher's
  logits renormalised over it.

### 4.1 Closest matches and the exact difference

| Work | (a) teacher distribution | (b) from scratch, ≤100M, generative | (c) one task / agent | (d) teacher-tokenizer subset | Difference from the thesis |
|---|---|---|---|---|---|
| Kim & Rush 2016 | yes (Word-KD); teacher is a 221M/47M LSTM, not an LLM | yes: 8M–84M seq2seq, task-only | yes (NMT) | no; shared task word vocabulary | Same loss structure ("Seq-KD + Word-KD"); the teacher is a same-type task model, not a general LLM |
| Tang 2019 | logits (MSE); teacher BERT-large | 0.96M, but word2vec embeddings and a classifier | yes | n/a (label space) | Not generative; teacher not generative |
| MobileLLM-R1 2026 | yes (KL to Llama-3.1-8B-Instruct, mid-training) | from scratch, but **140M+** and 4.2T-token general pre-training | no (general reasoning) | no (full 128k) | General-purpose pre-training; full vocabulary |
| Gemma 3 2025 | yes; 256 sampled logits, **renormalised** | from scratch, but ≥1B, general | no | per-token sampled subset; student keeps the full head | The subset is per token and random, not fixed and task-derived; no head reduction |
| Needle 2026 (grey) | **no**: Gemini text | yes: 26M, own pre-training | yes (single-shot function calling) | no (own 8192 BPE) | Text only; general pre-training first; no harness; no published evaluation |
| tiny-agent-112m (grey) | no: teacher text trajectories | 112M + 50M embedding, own pre-training | narrow synthetic tasks | no (32k BPE) | Text only; larger; general pre-training |
| OpenLocalAgent (grey) | no teacher; CE only | 10M–700M, own pre-training | multi-benchmark tool calling | no (16k BPE) | No distillation |
| GKD 2024 | yes (token-level) | 77M, but **pre-trained** T5 | yes (XSum, WMT, GSM8K) | no (shared T5 vocabulary) | Pre-trained student; teacher is a 3B T5 |
| Baby Llama 2023 / TF3-RO 2026 | yes | 58M / 26M from scratch | no (general LM) | no | Teachers are small from-scratch models, not LLMs |
| VocabTailor 2026 | — | no (pre-trained 0.6B–3B SLMs, no retraining) | yes (per-task LM-head set) | **yes**: static task-specific LM-head set (math 5,135 of 32,000; code 3,521 of 32,000) | No distillation, no training; inference-time memory saving only |

### 4.2 Verdict
I found no published work, up to 2026-11-03, that combines (a), (b) and (c). Adding (d) makes the match
even less likely. Each ingredient has close precedent:
- (a)+(b) for a task: Kim & Rush 2016 and Tang 2019, with non-LLM teachers.
- (a)+(b) with an LLM teacher: from-scratch pre-training distillation (Gemma 3, MobileLLM-R1, Peng et al.),
  but general-purpose and with at least 140M parameters.
- (b)+(c) for an agent: 2026 grey-literature tiny tool-calling models (Needle 26M), with text-only signal and
  general pre-training first.
- (d): task-specific LM-head sets on pre-trained SLMs (VocabTailor), multilingual vocabulary trimming,
  per-token renormalised logit subsets (Gemma 3), and MT run-time vocabulary selection.

One structural reason the combination is rare: API teachers (Gemini, GPT) do not expose distributions, and
the BabyLM interaction track explicitly forbids exposing them. An open-weights teacher (Gemma) run locally
is what makes (a) possible at all.

### 4.3 How hard I searched
**About 28 web searches**, mostly in extended mode. Queries included:
- "distilling large language model into tiny transformer trained from scratch task-specific generation"
- "knowledge distillation LLM teacher logits student 'from scratch' under 100M parameters single task arXiv 2025"
- "vocabulary trimming knowledge distillation reduced output vocabulary small student language model"
- "tiny language model agent tool calling trained from scratch million parameters distilled"
- "domain-specific tiny transformer distilled from LLM teacher token distribution from scratch arXiv 2026"
- "'from scratch' small language model knowledge distillation teacher logits restricted vocabulary task-specific generative 2026"
- "task-specific vocabulary pruning LM head small language model domain tokens subset of tokenizer distillation"
- "tiny transformer few million parameters trained from scratch on LLM-generated data single task …"
- "'trained from scratch' student 'knowledge distillation' LLM teacher 'subset of the vocabulary' renormalized logits"
- "specialized small language model trained from scratch on synthetic data from LLM for agentic task harness on-device"
- "tiny language model distillation from scratch narrow domain generative task '10M parameters' OR '20M parameters' …"
- "distill LLM into 'small transformer' 'trained from scratch' structured output JSON …"
- "code localization agent small model distillation …"
- "math tutoring dialogue knowledge distillation small model …"
- "'output vocabulary' reduced student language model distillation from LLM teacher logits renormalised task tokens"
- "from-scratch small specialist LM vs fine-tuned pretrained small model distillation …"
- "'knowledge distillation' 'randomly initialized' student generative Gemma OR Llama teacher 'domain-specific' …"
- "BabyLM 2025 interaction track …"
- "TinyStories … teacher logits KL … from scratch"
- "arXiv 2026 sub-100M parameter language model tool use function calling agent … from scratch"
- "fcall-96m …"
- "Cactus Needle 26M …"

**Three arXiv API full-abstract searches** (newest first):
1. `abs:"from scratch" AND abs:distillation AND abs:"small language model"` (6 hits)
2. `abs:distillation AND abs:vocabulary AND (reduced OR trimmed OR restricted OR pruned)` (95 hits; the 39
   most recent were screened)
3. `abs:distillation AND abs:"from scratch" AND (tiny OR agent OR "task-specific") AND (LLM OR "language model")`
   (17 hits)

None matched (a)+(b)+(c).

**Limits:**
- No direct Google Scholar or Semantic Scholar access.
- Only abstracts were screened in the API passes.
- Non-English work was not searched.
- Industry systems that were never written up cannot be excluded.

---

## 5. Optional: small or reduced vocabularies (besides Tao et al. and MobileLLM)
- **VocabTailor.** H. Zhang, Y. Zhou, T. Fang, Z. Yuan, G. Dai, W. Ouyang, and Y. Wang, "VocabTailor: Dynamic
  vocabulary selection for downstream tasks in small language models," in *Findings of the Association for
  Computational Linguistics: ACL 2026*, San Diego, CA, USA, Jul. 2026, pp. 28453–28471,
  doi: 10.18653/v1/2026.findings-acl.1418.
  - Static task token sets for the LM head, built from task profiling corpora (by frequency with a
    tolerance τ = 0.01), plus dynamic loading of input tokens.
  - Static sets per task:

    | Task | Model and vocabulary | Static set |
    |---|---|---|
    | EN→ZH translation | Qwen3-1.7B, 151,643 | 18,874 |
    | Summarisation | Llama 3.2 3B, 128,000 | 36,332 |
    | Code | deepseek-coder-1.3b, 32,000 | 3,521 |
    | Math | rho-math-1b, 32,000 | 5,135 |

  - For Llama 3.2 1B, "the embedding and LM head account for over 20% of the total memory usage".
  - Up to 99% memory saving on vocabulary components; end-to-end peak VRAM falls about 22% (1.17 → 0.91 GB).
- **Vocabulary trimming.** A. Ushio, Y. Zhou, and J. Camacho-Collados, "An efficient multilingual language
  model compression through vocabulary trimming," in *Findings of the Association for Computational
  Linguistics: EMNLP 2023*, Singapore, Dec. 2023, pp. 14725–14739, doi: 10.18653/v1/2023.findings-emnlp.981
  (pages and DOI from a search snippet; venue confirmed on arXiv:2305.15020).
  - Trims a multilingual model's vocabulary to one language without retraining; "around 50%" of the
    vocabulary is enough.
- **Vocabulary reduction for SLMs.** Y. Nozaki, D. Nakashima, R. Sato, N. Asaba, and S. Kawamura, "Efficient
  vocabulary reduction for small language models," in *Proc. 31st Int. Conf. Computational Linguistics:
  Industry Track*, Abu Dhabi, UAE, Jan. 2025, pp. 771–783.
  - Performance generally drops after reduction, and fine-tuning recovers much of it.
- **KD with a reduced vocabulary.** A. Kolesnikova, Y. Kuratov, V. Konovalov, and M. Burtsev, "Knowledge
  distillation of Russian language models with reduction of vocabulary," arXiv:2205.02340, May 2022.
  - A reduced student vocabulary breaks plain KL distillation, so they propose two alignment techniques.
  - 17x–49x compression.
- **Run-time vocabulary selection in MT.** X. Shi and K. Knight, "Speeding up neural machine translation
  decoding by shrinking run-time vocabulary," in *Proc. 55th Annu. Meeting of the Association for
  Computational Linguistics (Vol. 2: Short Papers)*, Vancouver, Canada, Jul. 2017, pp. 574–579,
  doi: 10.18653/v1/P17-2091.
  - Word-alignment candidate lists give a 2x decoding speed-up with no BLEU loss.

---

## 6. UNVERIFIED or partly verified
- Kim & Rush: that the students were *randomly initialised* is inferred (no general pre-training existed;
  training "mirrors Luong et al. (2015)"). The paper does not say it in words.
- "fcall-96m" (named in the OpenLocalAgent README as the previous best from-scratch model, 95.3M): no
  primary source found.
- Needle: no published evaluation. The "beats FunctionGemma-270M …" claim is from the authors' Show HN post.
  The release date is unknown (model card shows "6 months ago").
- FunctionGemma release date of 22 Dec 2025: from secondary coverage only. The Google blog post was not opened.
- tiny-agent-112m and OpenLocalAgent: self-reported, not reviewed. The teacher name "Qwen3.8-27B" is as
  written on the card.
- Ushio et al.: Anthology pages and DOI come from a search snippet; the Anthology page was not opened.
- Sparse Logit Sampling: ACL 2025 Anthology pages not opened; which models are students was not checked.
- Latif et al.: AIED 2024 proceedings pages not opened; the teacher LLM is not named in the abstract.
- Lu & Liu: "NeurIPS 2026" comes only from the arXiv comment.
- Peng et al. (2410.16215): no venue on the arXiv page.
- MobileLLM-R1: the paper itself names the teacher two ways (Llama-3.1-8B-Instruct vs "LLaMa3-8B").
- Gemma 3: the vocabulary is 262k vs 256k inside the same report.
- Generation-distillation (Melas-Kyriazi et al.) and ZeroGen: the generator and teacher models, and the
  student size and initialisation, are not stated in the abstracts.
- pQRNN: no absolute parameter count in the abstract.
- Kolesnikova et al.: vocabulary sizes and models not checked.
- FBI-LLM: the distillation teacher is not named on the abstract page.
- Seen only in search or API summaries, not opened; do not cite without opening:
  - Liang et al., "Let GPT be a Math Tutor" (EMNLP 2023): student details unknown.
  - arXiv 2503.06424 (training LLM tutors).
  - arXiv 2605.07990 (the claim that a 270M model is at chance, 27%, on a 15-tool task).
  - arXiv 2610.22093 (a quantum Born machine distilling GPT-2 distributions over "restricted vocabularies").
  - arXiv 2608.03480 (corpus-driven vocabulary pruning for NMT).
  - arXiv 2606.24747 (task-specific distillation scaling laws, finance; sizes unknown).
