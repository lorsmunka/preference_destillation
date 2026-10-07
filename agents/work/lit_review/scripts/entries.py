# Content for the literature review: one dict per reference.
# 'arxiv' entries take their author list from the arXiv API dump; 'authors' overrides it.
# bib: type + fields (booktitle / journal / pages / doi / note). 'verified' says how it was checked.

THEMES = [
    ("A", "Knowledge distillation: foundations",
     "Distillation trains a small student to match a teacher's output distribution instead of (or alongside) "
     "one-hot labels. The classic recipe (softened soft targets, a temperature, a mix with the hard-label loss) "
     "is still the core of almost every LLM method. Since 2020 the theory has asked why it helps (variance "
     "reduction, label information) and when it does not (optimisation limits, capacity gap)."),
    ("B", "Distilling autoregressive language models",
     "For generative models the field moved from word- and sequence-level KD (2016) to a debate about the "
     "divergence (forward vs reverse KL, skewed and adaptive mixes) and about the train/inference mismatch, "
     "answered with on-policy distillation (2023-2025). The newest work (2025-2026) studies why mixing hard and "
     "soft labels works, how to store teacher logits cheaply without bias, and per-token weighting by teacher "
     "entropy. Students are almost always pretrained models of 0.1-7B parameters."),
    ("C", "Pre-training distillation and from-scratch students",
     "Distillation from random initialisation is established, but at general pre-training scale: Gemma 2 and 3 "
     "replace next-token prediction with KD for their small models, and design-space studies map truncation, "
     "loss mixing and teacher size. Scaling laws now say when distillation beats plain supervised training."),
    ("D", "Small models on teacher-written data; vocabulary reduction",
     "A second line trains small models from scratch on LLM-written narrow-domain text with plain cross-entropy "
     "(TinyStories, phi-1, TinyGSM). Vocabulary size is a known lever for small models; trimming or reducing "
     "vocabularies exists for encoders, drafters and multilingual models, but not as a task-derived output "
     "space for KD."),
    ("E", "Distilling reasoning: steps, programs and verifiers",
     "Small models learn multi-step maths better from rationales, and better still when the reasoning is a "
     "program that can be executed and checked. Verifiers and process (step-level) supervision filter or rank "
     "solutions; in 2025 large-scale RL plus SFT distillation of traces became the dominant recipe for small "
     "reasoners."),
    ("F", "Agents, harnesses and agent distillation",
     "An agent is a model inside a loop (the harness or scaffold) that manages context, tools and checks. "
     "2024-2026 saw a wave of agent distillation into small models: trajectory SFT, span-aware KL, top-k KD "
     "plus RL, on-device function calling, and 'harness distillation' that moves scaffold behaviour into "
     "weights. Students are pretrained (0.5-9B); the closest from-scratch analogue distils into an RL policy."),
    ("G", "Structured output, routing and function calling",
     "Structured output is obtained either by constraining decoding (grammars, finite-state machines) or by "
     "training the model on the format. Format constraints can cost reasoning accuracy. Small distilled "
     "students for text-to-JSON tasks appear in 2026 work with pretrained 0.6B models."),
    ("H", "Math word problem datasets and evaluation",
     "Elementary word-problem benchmarks (MAWPS, ASDiv, SVAMP, GSM8K) are saturated for large models but remain "
     "informative for tiny ones. Robustness work shows that changing numbers, adding irrelevant clauses or "
     "varying structure breaks shallow solvers, and contamination studies show public test sets leak into "
     "teachers."),
    ("I", "Tutoring dialogues, explanation quality and sycophancy",
     "LLMs solve well but tutor poorly: they give wrong feedback, reveal answers or agree with a pupil's wrong "
     "answer. Recent datasets and benchmarks ground tutoring in verified solutions and measure pedagogy; "
     "sycophancy under pushback is now framed as an educational safety risk."),
    ("J", "Error compounding and multi-turn evaluation",
     "Training on gold prefixes and testing on one's own output lets errors compound (exposure bias). The same "
     "effect appears across turns: models conditioned on their own earlier replies do much worse than on gold "
     "history, and once they take a wrong turn they rarely recover. Cleaning the history helps."),
    ("K", "Distillation frameworks and toolkits",
     "Open toolkits range from encoder-era research kits (TextBrewer, torchdistill) to 2024-2026 LLM tools "
     "(TRL's GKD trainer, DistillKit, EasyDistill, KDFlow) focused on post-training pretrained models, stored "
     "or streamed logits and throughput. None targets from-scratch students with a task-derived vocabulary."),
]

ENTRIES = [
    # ---------------- A ----------------
    dict(key="hinton2015distilling", theme="A", arxiv="1503.02531", year=2015,
         venue="NIPS 2014 Deep Learning Workshop (arXiv 2015)",
         bib=dict(type="misc", note="NIPS 2014 Deep Learning Workshop"),
         summary="Introduces knowledge distillation as training a small model on the teacher's temperature-softened "
                 "output distribution, combined with the ordinary hard-label loss. Explains why soft targets carry "
                 "'dark knowledge' about which wrong answers are nearly right, and why the soft term must be scaled "
                 "by T squared.",
         relevance="The thesis loss (forward KL at temperature T, times T squared, mixed with CE) is this recipe "
                   "applied per token over a reduced vocabulary."),
    dict(key="gou2021survey", theme="A", arxiv="2006.05525", year=2021,
         venue="International Journal of Computer Vision 129, 1789-1819",
         bib=dict(type="article", journal="International Journal of Computer Vision", volume="129",
                  pages="1789--1819", doi="10.1007/s11263-021-01453-z"),
         summary="Broad survey of distillation organised by kind of knowledge (response-, feature- and "
                 "relation-based), training scheme (offline, online, self-distillation), teacher-student "
                 "architecture and applications, with open challenges.",
         relevance="Taxonomy for the background chapter: the thesis is offline, response-based (logit) "
                   "distillation into an architecturally different student."),
    dict(key="stanton2021does", theme="A", arxiv="2106.05945", year=2021,
         venue="NeurIPS 2021",
         bib=dict(type="inproceedings", booktitle="Advances in Neural Information Processing Systems 34 (NeurIPS 2021)"),
         summary="Separates fidelity (how closely the student matches the teacher) from generalisation. Students "
                 "often fail to match the teacher even when they have the capacity, mainly because of optimisation "
                 "difficulty, and closer matching does not always give better generalisation.",
         relevance="Justifies logging fidelity (top-k agreement, KL, mean target rank) separately from task "
                   "accuracy, and warns against reading 'closer to the teacher' as 'better student'."),
    dict(key="menon2021statistical", theme="A", arxiv="2005.10419", year=2021,
         title="A statistical perspective on distillation",
         venue="ICML 2021, PMLR 139:7632-7642 (arXiv title: 'Why distillation helps: a statistical perspective')",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 38th International Conference on Machine Learning (ICML 2021)",
                  series="Proceedings of Machine Learning Research", volume="139", pages="7632--7642",
                  url="https://proceedings.mlr.press/v139/menon21a.html"),
         links=[("PMLR", "https://proceedings.mlr.press/v139/menon21a.html"), ("arXiv", "https://arxiv.org/abs/2005.10419")],
         verified="arXiv API + PMLR page",
         summary="Treats the teacher as an estimate of the true (Bayes) class probabilities. A good teacher lowers "
                 "the variance of the student's training objective, and a bias-variance trade-off says how much "
                 "imperfect teacher probabilities can still help.",
         relevance="Theory behind the entropy finding: when the teacher is almost one-hot (exp(H) near 1) its soft "
                   "labels carry little beyond the hard label, so KD and CE should behave alike."),
    dict(key="zhang2025capacity", theme="A", arxiv="2311.07052", year=2025,
         venue="ACL 2025",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)"),
         summary="Studies the 'curse of capacity gap': a bigger teacher does not always give a better student. Across "
                 "small (under 3B) language models the best teacher size grows roughly linearly with student size; "
                 "the law is used to build the MiniMA models.",
         relevance="A 4-12B Gemma teacher for a 1-100M student is far outside that ratio. Acknowledge the gap and, if "
                   "affordable, add a smaller-teacher control (Gemma 1B shares the tokenizer)."),
    # ---------------- B ----------------
    dict(key="kim2016sequence", theme="B", arxiv="1606.07947", year=2016,
         venue="EMNLP 2016",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
                  doi="10.18653/v1/D16-1139"),
         summary="Applies distillation to neural machine translation and introduces sequence-level KD: training the "
                 "student on the teacher's beam-search outputs. Combined with word-level KD, students run 10x "
                 "faster with little BLEU loss and barely need beam search.",
         relevance="The 'teacher-generated' data mode is sequence-level KD; storing token logits on those sequences "
                   "adds word-level KD on top, as Kim and Rush combined them."),
    dict(key="xu2024survey", theme="B", arxiv="2402.13116", year=2024,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Survey of distillation for LLMs organised around algorithms (eliciting knowledge, divergence- and "
                 "similarity-based transfer), skills (reasoning, alignment, agents, NLP tasks) and domain "
                 "verticalisation, with a strong focus on data augmentation by the teacher.",
         relevance="Main overview reference for the LLM-KD section; places the thesis in white-box, divergence-based "
                   "KD with teacher-generated data."),
    dict(key="gu2024minillm", theme="B", arxiv="2306.08543", year=2024,
         venue="ICLR 2024",
         bib=dict(type="inproceedings", booktitle="International Conference on Learning Representations (ICLR 2024)"),
         summary="Argues that forward KL makes a small generative student spread mass over the teacher's "
                 "low-probability regions, and minimises reverse KL with an on-policy, policy-gradient method. "
                 "Reports better responses, lower exposure bias and calibration for 120M-13B students.",
         relevance="The main argument for reverse KL. The thesis's divergence set includes reverse KL but applies it "
                   "off-policy on stored teacher sequences, which is not what MiniLLM optimises; say so."),
    dict(key="agarwal2024onpolicy", theme="B", arxiv="2306.13649", year=2024,
         venue="ICLR 2024",
         bib=dict(type="inproceedings", booktitle="International Conference on Learning Representations (ICLR 2024)"),
         summary="Generalized KD (GKD) trains the student on its own sampled outputs with token-level teacher "
                 "feedback, mixing on-policy and fixed data and allowing any divergence (a generalised "
                 "Jensen-Shannon between forward and reverse KL). Tested on summarisation, translation and GSM8K.",
         relevance="The standard fix for the teacher-forced vs autoregressive gap the framework measures. The thesis "
                   "does not do on-policy KD (it needs a live teacher); a clear future-work item."),
    dict(key="ko2024distillm", theme="B", arxiv="2402.03898", year=2024,
         venue="ICML 2024",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 41st International Conference on Machine Learning (ICML 2024)"),
         summary="Proposes skew KL (mixing a little of one distribution into the other for stable gradients) and an "
                 "adaptive off-policy scheme that reuses student-generated outputs, up to 4.3x faster than earlier "
                 "on-policy KD.",
         relevance="A divergence beyond KL/JS/TVD that would be cheap to add to the framework's divergence set, and "
                   "an example of scheduling how much student data is used."),
    dict(key="wu2025rethinking", theme="B", arxiv="2404.02657", year=2025,
         venue="COLING 2025",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)"),
         summary="Shows that in LLM distillation forward and reverse KL share the same optimum and both reach it "
                 "given enough epochs; early on, forward KL fits the head of the distribution and reverse KL the "
                 "tail. Proposes an adaptive mix (AKL).",
         relevance="Supports forward KL as the default for long multi-epoch training of small students, and frames "
                   "forward-vs-reverse differences as a training-budget effect."),
    dict(key="li2023curriculum", theme="B", arxiv="2211.16231", year=2023,
         venue="AAAI 2023",
         bib=dict(type="inproceedings", booktitle="Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2023)"),
         summary="Curriculum Temperature for KD (CTKD): a learnable temperature, trained adversarially, makes the "
                 "distillation task gradually harder. A plug-in gain on CIFAR-100, ImageNet and COCO.",
         relevance="Closest prior art for changing the temperature during training; the thesis uses a fixed cosine "
                   "schedule rather than a learned one and should cite CTKD when describing it."),
    dict(key="clark2019bam", theme="B", arxiv="1907.04829", year=2019,
         venue="ACL 2019",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
                  pages="5931--5937", doi="10.18653/v1/P19-1595"),
         summary="Distils single-task BERT teachers into one multi-task student and introduces teacher annealing: the "
                 "training target mixes the teacher's prediction and the gold label, moving linearly from all-teacher "
                 "to all-gold over training.",
         relevance="Direct precedent for annealing the KD/CE ratio from KD-heavy to CE-heavy (0.9 to 0.1). Cite it; do "
                   "not present the schedule as new."),
    dict(key="anshumann2025sparse", theme="B", arxiv="2503.16870", year=2025,
         venue="ACL 2025 (oral)",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)"),
         summary="Proves that caching only the teacher's top-K probabilities gives biased estimates of the teacher "
                 "distribution, hurting accuracy and calibration, and proposes random-sampling KD (importance "
                 "sampling): unbiased and sparser storage, tested on 300M-3B students.",
         relevance="The thesis stores the teacher distribution restricted to a fixed reduced vocabulary, a different "
                   "truncation; this paper gives the terms to discuss and measure the bias of dropping mass outside it."),
    dict(key="wang2026bridge", theme="B", arxiv="2605.26246", year=2026,
         venue="ICML 2026 (accepted, per arXiv)",
         bib=dict(type="inproceedings", booktitle="International Conference on Machine Learning (ICML 2026)",
                  note="Accepted at ICML 2026 according to the arXiv record"),
         summary="Finds that mixing hard labels (teacher-sampled tokens) with soft labels (full teacher distributions) "
                 "beats either alone, and that the gain comes from reduced exposure bias, not closer teacher "
                 "matching. Splits steps into 'bridges' (one token is right) and 'gardens' (many are fine) and adapts "
                 "the mix per step.",
         relevance="The closest recent theory for the thesis's KD/CE mix and its entropy finding: near-deterministic "
                   "domains are almost all bridges, so the anneal direction should matter little there."),
    dict(key="zhang2026egad", theme="B", arxiv="2605.01732", year=2026,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Uses the teacher's per-token entropy to steer distillation: a curriculum from low- to high-entropy "
                 "tokens, an entropy-adaptive temperature, and separate treatment of confident and uncertain tokens.",
         relevance="Precedent for the framework's entropy-weighted per-token KL ratio. Claim the measured "
                   "entropy-dependence across domains, not entropy-aware weighting itself."),
    # ---------------- C ----------------
    dict(key="gemma2024gemma2", theme="C", arxiv="2408.00118", year=2024,
         authors=["{Gemma Team}"], display_authors="Gemma Team (Google DeepMind)",
         venue="arXiv technical report",
         bib=dict(type="misc", author_others=True),
         summary="The 2B and 9B Gemma 2 models are pre-trained with knowledge distillation from a larger teacher "
                 "instead of next-token prediction, on many times the compute-optimal number of tokens, and compete "
                 "with models 2-3x larger.",
         relevance="Industrial evidence that logit distillation from random initialisation works; the thesis does the "
                   "same at 1-100M parameters on one domain."),
    dict(key="gemma2025gemma3", theme="C", arxiv="2503.19786", year=2025,
         authors=["{Gemma Team}"], display_authors="Gemma Team (Google DeepMind)",
         venue="arXiv technical report",
         bib=dict(type="misc", author_others=True),
         summary="Gemma 3 (1B-27B) is pre-trained with distillation in which 256 logits per token are sampled by "
                 "teacher probability and the teacher distribution is renormalised over them; post-training also "
                 "distils from a large instruction-tuned teacher.",
         relevance="Teacher family for part 1, and a precedent for training on a renormalised teacher distribution "
                   "over a token subset, which the reduced vocabulary also does."),
    dict(key="gemma2026gemma4", theme="C", arxiv="2607.02770", year=2026,
         authors=["{Gemma Team}"], display_authors="Gemma Team (Google DeepMind)",
         venue="arXiv technical report",
         bib=dict(type="misc", author_others=True),
         summary="Introduces Gemma 4: open-weight multimodal dense and mixture-of-experts models from 2.3B to 31B "
                 "parameters, with a thinking mode and efficiency and long-context improvements.",
         relevance="Teacher of the maths tutor (Gemma 4 12B in the run notes); state the exact variant, quantisation "
                   "(NF4) and that thinking mode was off."),
    dict(key="peng2025pretraining", theme="C", arxiv="2410.16215", year=2025,
         venue="ACL 2025, pages 3603-3618",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
                  pages="3603--3618", doi="10.18653/v1/2025.acl-long.181"),
         links=[("ACL Anthology", "https://aclanthology.org/2025.acl-long.181/"), ("arXiv", "https://arxiv.org/abs/2410.16215")],
         verified="arXiv API + ACL Anthology",
         summary="Design-space study of pre-training distillation (GLM-4-9B into a 1.9B student trained from scratch, "
                 "plus other sizes): logits truncated to top-p 0.95 then top-100 (about 4000x less storage), a "
                 "warmup-stable-decay schedule for the KD share of the loss works best, offline logits beat online "
                 "ones, larger students gain more and larger teachers do not always help.",
         relevance="The closest design study to this framework (stored truncated logits, scheduled KD/LM mix, "
                   "from-scratch student), but general-domain and about 1B+. The main 'do not claim' reference for "
                   "scheduled mixing."),
    dict(key="busbridge2025scaling", theme="C", arxiv="2502.08606", year=2025,
         venue="ICML 2025",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)"),
         summary="Fits a distillation scaling law that predicts student performance from the compute split between "
                 "teacher and student. Distillation beats supervised learning only up to a compute level that grows "
                 "with student size; if a teacher must be trained for one student, supervised learning usually wins.",
         relevance="Frames when KD should beat CE for a given student; the thesis's teacher-entropy axis is a "
                   "domain-side factor this law does not model."),
    # ---------------- D ----------------
    dict(key="eldan2023tinystories", theme="D", arxiv="2305.07759", year=2023,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Builds a synthetic corpus of simple stories written by GPT-3.5/4 with a small child's vocabulary, "
                 "and shows that models under 10M parameters (even with one layer) trained from scratch write fluent, "
                 "consistent stories. Also proposes grading outputs with GPT-4.",
         relevance="Main precedent for tiny from-scratch models on teacher-written narrow-domain text; it uses only CE "
                   "on text (no teacher distributions) and keeps the top 10K tokens."),
    dict(key="gunasekar2023textbooks", theme="D", arxiv="2306.11644", year=2023,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="phi-1, a 1.3B code model trained on filtered 'textbook-quality' web data plus GPT-3.5-written "
                 "textbooks and exercises, reaches 50.6% pass@1 on HumanEval; data quality substitutes for scale.",
         relevance="Supports the data strategy (LLM-written, curated, domain-focused corpus) and the provenance "
                   "statement."),
    dict(key="liu2023tinygsm", theme="D", arxiv="2312.09241", year=2023,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="12.3M GPT-3.5-written grade-school problems with Python solutions; a 1.3B generator plus a 1.3B "
                 "verifier reaches 81.5% on GSM8K. Solutions are filtered only for being executable ('we do not "
                 "check for the correctness'), and an appendix ablation shows random initialisation costs about 10 "
                 "points (53.1 vs 63.3 at 125M).",
         relevance="Closest data-pipeline precedent for the tutor (LLM-written problems plus code). Its random-init "
                   "ablation is the baseline to discuss; like TinyGSM, the thesis cannot prove the code models the "
                   "problem text correctly."),
    dict(key="liu2024mobilellm", theme="D", arxiv="2402.14905", year=2024,
         venue="ICML 2024",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 41st International Conference on Machine Learning (ICML 2024)"),
         summary="Shows that architecture matters at sub-billion scale: deep-and-thin networks, embedding sharing, "
                 "grouped-query attention and block-wise weight sharing give better 125M/350M on-device models.",
         relevance="Design guidance for the phone-sized student and the size study (depth vs width, tied "
                   "embeddings)."),
    dict(key="tao2024scaling", theme="D", arxiv="2407.13623", year=2024,
         venue="NeurIPS 2024",
         bib=dict(type="inproceedings", booktitle="Advances in Neural Information Processing Systems 37 (NeurIPS 2024)"),
         summary="Trains 33M-3B models with different vocabulary sizes and finds the compute-optimal vocabulary grows "
                 "with model size; most large LLMs use too small a vocabulary.",
         relevance="Read in reverse, it supports small vocabularies for tiny students: the embedding and output "
                   "matrices dominate a small model's parameters, which the reduced vocabulary removes."),
    dict(key="goel2025vocabtrim", theme="D", arxiv="2506.22694", year=2025,
         venue="ICML 2025 workshop on Efficient Systems for Foundation Models",
         bib=dict(type="misc", note="ICML 2025 Workshop on Efficient Systems for Foundation Models"),
         summary="For speculative decoding, rebuilds the drafter's output head to contain only the target model's "
                 "most frequently sampled tokens, cutting drafting latency on memory-bound devices at a small "
                 "acceptance-rate cost (16% faster for Llama-3.2-3B-Instruct).",
         relevance="Closest precedent for restricting an output head to a frequency-selected subset of a larger "
                   "tokenizer; it is training-free and not derived from a task."),
    dict(key="kolesnikova2022knowledge", theme="D", arxiv="2205.02340", year=2022,
         venue="Computational Linguistics and Intellectual Technologies (Dialogue conference), 2022",
         bib=dict(type="inproceedings", booktitle="Computational Linguistics and Intellectual Technologies (Dialogue)"),
         summary="Distils Russian BERT-style models into students with a much smaller vocabulary. Because teacher and "
                 "student then tokenise differently, plain KL does not apply; two alignment techniques fix this and "
                 "reach 17-49x compression.",
         relevance="Nearest 'KD with reduced vocabulary' work. It is encoder-only and general-language; the thesis "
                   "derives the vocabulary from the task and keeps the teacher's tokenisation so KL applies directly."),
    # ---------------- E ----------------
    dict(key="wei2022chain", theme="E", arxiv="2201.11903", year=2022,
         venue="NeurIPS 2022",
         bib=dict(type="inproceedings", booktitle="Advances in Neural Information Processing Systems 35 (NeurIPS 2022)"),
         summary="Few-shot prompting with intermediate reasoning steps (chain of thought) greatly improves large "
                 "models on arithmetic, commonsense and symbolic reasoning; the gain emerges with scale.",
         relevance="Background for writing solutions as steps; the tutor's facts-first named steps are a fixed, "
                   "checkable chain-of-thought format."),
    dict(key="hsieh2023distilling", theme="E", arxiv="2305.02301", year=2023,
         venue="Findings of ACL 2023",
         bib=dict(type="inproceedings", booktitle="Findings of the Association for Computational Linguistics: ACL 2023"),
         summary="Uses LLM-written rationales as extra multi-task supervision for small T5 models; a 770M model beats "
                 "few-shot 540B PaLM on a benchmark with less training data.",
         relevance="Shows rationales help small students; the annotation study (answer / working / named / code) asks "
                   "the same question for a from-scratch student."),
    dict(key="gao2023pal", theme="E", arxiv="2211.10435", year=2023,
         venue="ICML 2023",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 40th International Conference on Machine Learning (ICML 2023)"),
         summary="Program-aided language models: the LLM writes a program as its reasoning and a Python interpreter "
                 "computes the answer, beating chain of thought on GSM8K and other tasks.",
         relevance="Basis of the 'code' annotation format, where the harness runs the student's code "
                   "(program-of-thought tool use)."),
    dict(key="zhu2024pad", theme="E", arxiv="2305.13888", year=2024,
         venue="NAACL 2024, pages 2571-2597",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
                  pages="2571--2597", doi="10.18653/v1/2024.naacl-long.142"),
         summary="Program-aided distillation (PaD): reasoning programs replace chain of thought in distillation data so "
                 "errors can be checked automatically; adds error injection, self-refinement and step-wise verified "
                 "beam search. Small students beat some much larger LLMs on arithmetic.",
         relevance="Closest precedent for program-checked distillation data in maths; the thesis checks against each "
                   "problem's own executed solve() rather than the teacher's program."),
    dict(key="cobbe2021training", theme="E", arxiv="2110.14168", year=2021,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Introduces GSM8K (8.5K grade-school word problems) and shows that sampling many solutions and "
                 "keeping the one a trained verifier ranks highest beats fine-tuning alone.",
         relevance="GSM8K is an exemplar source and an integer-subset test; the verifier idea parallels the harness "
                   "checks, which here are rule-based rather than learned."),
    dict(key="lightman2024verify", theme="E", arxiv="2305.20050", year=2024,
         venue="ICLR 2024",
         bib=dict(type="inproceedings", booktitle="International Conference on Learning Representations (ICLR 2024)"),
         summary="Compares outcome supervision with process supervision (feedback on every step) and finds "
                 "process-supervised reward models much better on MATH; releases PRM800K step labels.",
         relevance="The harness's line check (every line is a fact or a true calculation) is an exact, symbolic form "
                   "of process supervision."),
    dict(key="deepseek2025r1", theme="E", arxiv="2501.12948", year=2025,
         authors=["Daya Guo", "Dejian Yang", "Haowei Zhang"], display_authors="DeepSeek-AI (Daya Guo, Dejian Yang, Haowei Zhang et al.)",
         venue="Nature 645, 633-638 (2025)",
         bib=dict(type="article", journal="Nature", volume="645", pages="633--638", doi="10.1038/s41586-025-09422-z", author_others=True),
         summary="Shows that reasoning can be elicited by large-scale reinforcement learning without human reasoning "
                 "traces, and that the resulting reasoning patterns can be passed to smaller models.",
         relevance="Current reference point for reasoning distillation (SFT on teacher traces into small dense "
                   "models); the thesis instead distils token distributions into far smaller from-scratch students on "
                   "one narrow domain."),
    # ---------------- F ----------------
    dict(key="yao2023react", theme="F", arxiv="2210.03629", year=2023,
         venue="ICLR 2023",
         bib=dict(type="inproceedings", booktitle="International Conference on Learning Representations (ICLR 2023)"),
         summary="Prompts LLMs to interleave reasoning traces with actions (tool calls), improving question answering, "
                 "fact checking and interactive tasks (ALFWorld, WebShop).",
         relevance="The canonical reason-act loop that agent-distillation work imitates; the tutor's loop is a much "
                   "narrower, fixed version of it."),
    dict(key="sumers2024cognitive", theme="F", arxiv="2309.02427", year=2024,
         venue="Transactions on Machine Learning Research (TMLR), 2024",
         bib=dict(type="article", journal="Transactions on Machine Learning Research"),
         summary="CoALA describes language agents by their memory modules, a structured action space (internal and "
                 "external actions) and a decision procedure, and uses this to organise existing agents.",
         relevance="Vocabulary for defining 'agent' in the thesis: working memory (main line and side thread), a "
                   "small action space (solve / explain / out of scope) and a decision cycle run by the harness."),
    dict(key="belcak2025small", theme="F", arxiv="2506.02153", year=2025,
         venue="arXiv preprint (position paper)",
         bib=dict(type="misc"),
         summary="Position paper: most agent calls are narrow, repetitive tasks for which small language models are "
                 "sufficient, more suitable and cheaper; outlines an LLM-to-SLM agent conversion algorithm.",
         relevance="Motivation for part 2; the thesis is an extreme case (tens of millions of parameters, one "
                   "domain)."),
    dict(key="kang2025distilling", theme="F", arxiv="2505.17612", year=2025,
         venue="NeurIPS 2025 (spotlight)",
         bib=dict(type="inproceedings", booktitle="Advances in Neural Information Processing Systems 38 (NeurIPS 2025)"),
         summary="Agent distillation: transfers whole task-solving behaviour (reasoning plus retrieval and code tools) "
                 "from an LLM agent into instruction-tuned Qwen2.5 students of 0.5-7B by SFT on teacher trajectories, "
                 "with wrong trajectories filtered out; adds a first-thought prefix and self-consistent actions.",
         relevance="Closest agent-distillation method on maths with code tools; pretrained students and "
                   "sequence-level SFT, versus the thesis's from-scratch student with token distributions."),
    dict(key="liu2026structured", theme="F", arxiv="2505.13820", year=2026,
         venue="AAMAS 2026",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)"),
         summary="Splits ReAct trajectories into reasoning and action spans and applies span-specific KL losses to the "
                 "teacher's distributions; on ALFWorld, WebShop and HotPotQA students of 120M-6.7B keep most of the "
                 "teacher's performance. The action loss is normalised over a restricted token set.",
         relevance="Closest on signal (token-level KL inside agent trajectories) and student size; its span-restricted "
                   "normalisation should be cited next to the reduced output vocabulary."),
    dict(key="ni2026star", theme="F", arxiv="2602.03022", year=2026,
         venue="ICLR 2026",
         bib=dict(type="inproceedings", booktitle="International Conference on Learning Representations (ICLR 2026)"),
         summary="Transfers function calling to 'super-tiny' pretrained models (the 0.6B is best among open models "
                 "under 1B) with constrained KD (top-k forward KL that suppresses confidently wrong tokens) followed "
                 "by RL with a similarity-based reward.",
         relevance="Shows top-k forward KL plus a curriculum is standard for small agentic students; also a strong "
                   "pretrained baseline class the thesis does not train."),
    dict(key="erdogan2024tinyagent", theme="F", arxiv="2409.00608", year=2024,
         venue="EMNLP 2024 System Demonstrations",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations"),
         summary="End-to-end framework for task-specific function-calling agents at the edge: curated data, fine-tuned "
                 "1.1B and 7B models, tool retrieval to shorten prompts, quantisation and a local Mac assistant.",
         relevance="Full-stack precedent for a small agent plus harness on device; pretrained students, single-turn "
                   "commands, no logit distillation."),
    dict(key="chen2024octopus", theme="F", arxiv="2404.01744", year=2024,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Fine-tunes Gemma-2B for on-device function calling and adds 'functional tokens' to the vocabulary, "
                 "so choosing a function becomes a single-token decision, cutting context length and latency.",
         relevance="Precedent for the tutor's route-by-first-tag design (the decision is one output token). Octopus "
                   "expands the vocabulary for this; the thesis reduces it."),
    dict(key="zhou2024llm4teach", theme="F", arxiv="2311.13373", year=2024,
         venue="IJCAI 2024, pages 5671-5679",
         bib=dict(type="inproceedings", booktitle="Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI 2024)",
                  pages="5671--5679", doi="10.24963/ijcai.2024/627"),
         summary="An LLM's action distribution guides a small RL student (24K and 10M parameters) through a KL term "
                 "whose weight decays to zero; the student then keeps learning from environment reward and "
                 "surpasses its teacher (MiniGrid, Habitat).",
         relevance="Closest from-scratch agentic analogue and a precedent for an annealed distillation weight; its "
                   "student outputs discrete actions, not text."),
    dict(key="luo2026distilling", theme="F", arxiv="2609.36630", year=2026,
         venue="arXiv preprint (survey)",
         bib=dict(type="misc"),
         summary="Roadmap that defines agent distillation as transferring task-solving knowledge from a teacher agent "
                 "to a student agent, organised by where the knowledge ends up: model weights, artifacts, the "
                 "execution harness, or several of these; proposes judging retention by causal contribution.",
         relevance="Framing for part 2: here the language skill goes into weights while the harness is hand-written. "
                   "The novelty check found no from-scratch or sub-1B students in its coverage."),
    dict(key="ye2026harnesszero", theme="F", arxiv="2609.24974", year=2026,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Harness distillation: behaviours induced by a specialised, optimised harness are turned into "
                 "training demonstrations and fine-tuned into the model so the special harness can be removed; macro "
                 "task success rises from 23.3% to 44.3%.",
         relevance="The opposite design choice to the thesis, which keeps a deterministic harness (checks, context "
                   "rules) in code and distils only the language part; use it to justify that split."),
    # ---------------- G ----------------
    dict(key="willard2023efficient", theme="G", arxiv="2307.09702", year=2023,
         venue="arXiv preprint (Outlines library)",
         bib=dict(type="misc"),
         summary="Recasts constrained generation as moves through a finite-state machine indexed over the vocabulary, "
                 "so regular-expression or grammar constraints are enforced with little overhead.",
         relevance="Main alternative to learning the format: a reduced output vocabulary is a coarse, static "
                   "constraint, whereas grammar-constrained decoding could guarantee tag/JSON validity at inference."),
    dict(key="tam2024speak", theme="G", arxiv="2408.02442", year=2024,
         venue="EMNLP 2024 Industry Track, pages 1218-1236",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
                  pages="1218--1236"),
         summary="Forcing LLMs into strict formats such as JSON can lower reasoning performance, and stricter "
                 "constraints hurt more.",
         relevance="A caution when reporting structural validity: check whether the fixed output format costs task "
                   "accuracy (the annotation-format study can show this)."),
    dict(key="chaganti2026different", theme="G", arxiv="2607.08268", year=2026,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Distils an 8B reasoning teacher into Qwen3-0.6B (QLoRA) to map news articles to one JSON object (a "
                 "summary and five labels); the student is about 50x faster, recovers 58% of the summary-quality gap, "
                 "and which teacher is used changes which skills transfer.",
         relevance="Almost the part-1 Reddit task (text to JSON labels) with a pretrained 0.6B student; a useful "
                   "contrast for a from-scratch student of a few million parameters."),
    # ---------------- H ----------------
    dict(key="koncel2016mawps", theme="H", year=2016,
         title="MAWPS: A Math Word Problem Repository",
         authors=["Rik Koncel-Kedziorski", "Subhro Roy", "Aida Amini", "Nate Kushman", "Hannaneh Hajishirzi"],
         venue="NAACL-HLT 2016, pages 1152-1157",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
                  pages="1152--1157", doi="10.18653/v1/N16-1136", url="https://aclanthology.org/N16-1136/"),
         links=[("ACL Anthology", "https://aclanthology.org/N16-1136/")],
         verified="ACL Anthology page",
         summary="An online repository unifying 3,320 word problems from earlier datasets, with tools to control "
                 "lexical and template overlap and to filter ungrammatical problems.",
         relevance="One exemplar source and part of SVAMP's base data; cite when describing seeds and the "
                   "human-written test set."),
    dict(key="miao2020diverse", theme="H", arxiv="2106.15772", year=2020,
         venue="ACL 2020, pages 975-984",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
                  pages="975--984", doi="10.18653/v1/2020.acl-main.92"),
         summary="ASDiv: 2,305 elementary-school problems annotated with problem type and grade level, built for "
                 "lexical diversity; less diverse corpora overstate solver ability.",
         relevance="Exemplar source; its type and grade annotations match the corpus taxonomy cells."),
    dict(key="patel2021svamp", theme="H", arxiv="2103.07191", year=2021,
         venue="NAACL 2021, pages 2080-2094",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
                  pages="2080--2094", doi="10.18653/v1/2021.naacl-main.168"),
         summary="Shows word-problem solvers rely on shallow heuristics (many problems are solved without reading the "
                 "question) and introduces SVAMP, built by controlled variations of existing problems, on which "
                 "accuracy drops sharply.",
         relevance="The tutor's human-written test set; its variation types (question and structure changes) line up "
                   "with the what-if follow-ups."),
    dict(key="mirzadeh2025gsmsymbolic", theme="H", arxiv="2410.05229", year=2025,
         venue="ICLR 2025",
         bib=dict(type="inproceedings", booktitle="International Conference on Learning Representations (ICLR 2025)"),
         summary="Generates GSM8K variants from symbolic templates: accuracy varies across instantiations, drops when "
                 "only the numbers change, and falls by up to 65% when one irrelevant clause is added.",
         relevance="Supports the renumbering and held-out-shape design; report variance across renumbered copies, not "
                   "a single accuracy."),
    dict(key="shi2023distracted", theme="H", arxiv="2302.00093", year=2023,
         venue="ICML 2023",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 40th International Conference on Machine Learning (ICML 2023)",
                  pages="31210--31227"),
         summary="GSM-IC adds irrelevant sentences to grade-school problems; accuracy of prompted LLMs drops "
                 "dramatically, and self-consistency or an 'ignore irrelevant information' instruction helps.",
         relevance="The distractor slice of the corpus and the distractor follow-ups test exactly this weakness."),
    dict(key="zhang2024careful", theme="H", arxiv="2405.00332", year=2024,
         venue="NeurIPS 2024 Datasets and Benchmarks Track",
         bib=dict(type="inproceedings", booktitle="Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track"),
         summary="GSM1k: newly written problems that mirror GSM8K. Some model families drop up to 8%, and the drop "
                 "correlates with how likely a model is to generate GSM8K items (partial memorisation).",
         relevance="The teacher may have seen GSM8K/SVAMP, so its benchmark accuracy can be inflated; the from-scratch "
                   "student has not, which is worth stating."),
    # ---------------- I ----------------
    dict(key="macina2023mathdial", theme="I", arxiv="2305.14536", year=2023,
         venue="Findings of EMNLP 2023",
         bib=dict(type="inproceedings", booktitle="Findings of the Association for Computational Linguistics: EMNLP 2023"),
         summary="3K one-to-one tutoring dialogues made by pairing human teachers with an LLM that plays a student "
                 "with typical errors, grounded in multi-step maths problems. LLMs solve well but tutor poorly (wrong "
                 "feedback, revealing answers); fine-tuning on MathDial helps.",
         relevance="Closest precedent for a simulated pupil; the template pupil is a cheaper, controllable alternative "
                   "and should be compared with it."),
    dict(key="macina2025mathtutorbench", theme="I", arxiv="2502.18940", year=2025,
         venue="EMNLP 2025 (oral)",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing"),
         summary="Benchmark of datasets and metrics for tutoring ability, including a reward model for pedagogical "
                 "quality; solving skill does not translate into good teaching, and longer dialogues are harder.",
         relevance="Candidate yardstick for explanation quality, or at least the vocabulary for the limitation that the "
                   "tutor explains rather than scaffolds."),
    dict(key="daheim2024stepwise", theme="I", arxiv="2407.09136", year=2024,
         venue="EMNLP 2024, pages 8386-8411",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
                  pages="8386--8411"),
         summary="Collects 1K stepwise student solutions with the first wrong step marked by teachers; LLMs struggle to "
                 "find the error, and grounding the tutor's reply in a verifier's verdict gives more targeted, "
                 "correct feedback.",
         relevance="Supports verify-then-respond: in the thesis the executed solution plays the verifier for pushback "
                   "and what-if messages."),
    dict(key="sharma2024sycophancy", theme="I", arxiv="2310.13548", year=2024,
         venue="ICLR 2024",
         bib=dict(type="inproceedings", booktitle="International Conference on Learning Representations (ICLR 2024)"),
         summary="Five assistants consistently tailor answers to users' stated views; human preference data and "
                 "preference models sometimes favour convincing sycophantic answers over correct ones.",
         relevance="Motivates the pushback tests; a from-scratch student never sees preference training, so its "
                   "sycophancy can only come from the teacher's replies."),
    dict(key="wei2023simple", theme="I", arxiv="2308.03958", year=2023,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Scaling and instruction tuning increase sycophancy (models even agree with wrong addition "
                 "statements); a small fine-tuning set of synthetic examples whose correct answer ignores the user's "
                 "opinion reduces it.",
         relevance="Direct precedent for training on computed pushback mistakes so the tutor holds a checked answer."),
    dict(key="kasneci2026sycophancy", theme="I", arxiv="2605.14604", year=2026,
         venue="ICML 2026 position paper track (titled 'Position: Sycophancy is an Educational Safety Risk ...')",
         bib=dict(type="inproceedings", booktitle="International Conference on Machine Learning (ICML 2026), Position Paper Track"),
         links=[("arXiv", "https://arxiv.org/abs/2605.14604"), ("ICML", "https://icml.cc/virtual/2026/poster/67162")],
         verified="arXiv API + icml.cc page",
         summary="Argues that tutoring needs 'corrective friction' and that sycophancy is a safety risk; introduces "
                 "EduFrameTrap, which varies pupil confidence and pressure (context switch, authority, social "
                 "face-saving), and finds frontier models still give way under social pressure.",
         relevance="Current support for the harness check that an explanation must not agree with wrong pushback; its "
                   "pressure types suggest extra template categories."),
    # ---------------- J ----------------
    dict(key="bengio2015scheduled", theme="J", arxiv="1506.03099", year=2015,
         venue="NeurIPS 2015",
         bib=dict(type="inproceedings", booktitle="Advances in Neural Information Processing Systems 28 (NIPS 2015)"),
         summary="Names the mismatch between teacher-forced training and free-running inference, where errors "
                 "accumulate, and proposes scheduled sampling: gradually feeding the model its own predictions "
                 "during training.",
         relevance="Explains why teacher-forced accuracy overstates autoregressive accuracy, which the framework "
                   "reports side by side."),
    dict(key="ross2011dagger", theme="J", arxiv="1011.0686", year=2011,
         venue="AISTATS 2011, PMLR 15:627-635",
         bib=dict(type="inproceedings", booktitle="Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS 2011)",
                  series="Proceedings of Machine Learning Research", volume="15", pages="627--635"),
         summary="Shows that imitating an expert only on the expert's own states lets errors compound, and proposes "
                 "DAgger, which trains on the states the learner itself visits, with no-regret guarantees.",
         relevance="Theoretical root of on-policy distillation and of measuring the agent on its own history rather "
                   "than gold history."),
    dict(key="arora2022exposure", theme="J", arxiv="2204.01171", year=2022,
         venue="Findings of ACL 2022, pages 700-710",
         bib=dict(type="inproceedings", booktitle="Findings of the Association for Computational Linguistics: ACL 2022",
                  pages="700--710", doi="10.18653/v1/2022.findings-acl.58"),
         summary="Analyses exposure bias from an imitation-learning view: it causes error accumulation, perplexity "
                 "fails to capture it, and the accumulation degrades generation quality.",
         relevance="Supports logging autoregressive metrics alongside perplexity and teacher-forced accuracy."),
    dict(key="kwan2024mteval", theme="J", arxiv="2401.16745", year=2024,
         venue="EMNLP 2024",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing"),
         summary="Multi-turn benchmark (recollection, expansion, refinement, follow-up). Most models do worse than in "
                 "single-turn form; conditioning on gold instead of self-generated history raises scores sharply "
                 "(4.64 to 7.32 for Vicuna-13B on recollection), and about half of the analysed failures trace back "
                 "to errors in earlier turns.",
         relevance="Precedent and numbers for the tutor's gold-history vs own-history evaluation."),
    dict(key="laban2025lost", theme="J", arxiv="2505.06120", year=2025,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Over 200K simulated conversations where the instruction is revealed across turns: every tested LLM "
                 "drops (39% on average), mostly through unreliability; once a model takes a wrong turn it does not "
                 "recover.",
         relevance="Motivates the harness rules that drop the side thread at a new task and check content before it "
                   "enters the history."),
    dict(key="zheng2026maigo", theme="J", arxiv="2605.27186", year=2026,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Traces part of multi-turn degradation to 'self-contamination' (earlier assistant replies carry "
                 "deviations forward) and trains with history-cleaned references that remove prior replies, raising "
                 "Qwen2.5-7B's sharded accuracy from 52.8 to 66.1.",
         relevance="Independent support for cleaning what the model sees; the thesis does it in the harness instead "
                   "of in training."),
    # ---------------- K ----------------
    dict(key="yang2020textbrewer", theme="K", arxiv="2002.12620", year=2020,
         venue="ACL 2020 System Demonstrations",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
                  doi="10.18653/v1/2020.acl-demos.2"),
         links=[("arXiv", "https://arxiv.org/abs/2002.12620"), ("docs", "https://textbrewer.readthedocs.io/en/latest/Configurations.html")],
         verified="arXiv API + TextBrewer documentation",
         summary="PyTorch distillation toolkit for NLP (classification, reading comprehension, sequence labelling) "
                 "with configurable losses and intermediate-layer matching. Its documentation lists temperature "
                 "schedulers, KD-weight and hard-label-weight schedulers (linear decay / growth) and in-memory caching "
                 "of teacher logits.",
         relevance="Annealed KD/hard-label weights and temperature schedules already exist in tooling; TextBrewer "
                   "targets encoder models, not generative from-scratch students."),
    dict(key="matsubara2021torchdistill", theme="K", arxiv="2011.12913", year=2021,
         venue="RRPR 2021 (ICPR workshop), LNCS 12636, pages 24-44",
         bib=dict(type="inproceedings", booktitle="Reproducible Research in Pattern Recognition (RRPR 2021)",
                  series="Lecture Notes in Computer Science", volume="12636", pages="24--44",
                  publisher="Springer", doi="10.1007/978-3-030-76423-4_3"),
         summary="Configuration-driven (YAML) PyTorch framework for reproducible distillation experiments, "
                 "reproducing many published methods on ImageNet and COCO.",
         relevance="Precedent for 'the configuration is the experiment'; the thesis uses typed Python run objects for "
                   "the same goal."),
    dict(key="vonwerra2020trl", theme="K", year=2020,
         title="TRL: Transformers Reinforcement Learning",
         authors=["Leandro von Werra", "Younes Belkada", "Lewis Tunstall", "Edward Beeching", "Tristan Thrush",
                  "Nathan Lambert", "Shengyi Huang", "Kashif Rasul", "Quentin Gallouédec"],
         venue="Software, GitHub (Hugging Face)",
         bib=dict(type="misc", howpublished="\\url{https://github.com/huggingface/trl}", url="https://github.com/huggingface/trl",
                  note="Software; GKDTrainer documented at https://huggingface.co/docs/trl/gkd_trainer"),
         links=[("GitHub", "https://github.com/huggingface/trl"), ("GKD docs", "https://huggingface.co/docs/trl/gkd_trainer")],
         verified="GitHub repository + TRL documentation",
         summary="Hugging Face's post-training library (SFT, DPO, GRPO and more). Its experimental GKDTrainer implements "
                 "supervised and on-policy generalised-JSD distillation, with settings for the student-data fraction, "
                 "the forward/reverse interpolation and sequence-level KD.",
         relevance="The standard off-the-shelf route for distilling a pretrained Hugging Face model; it needs a live "
                   "teacher with a shared tokenizer and offers no vocabulary reduction."),
    dict(key="goddard2024distillkit", theme="K", year=2024,
         title="DistillKit: Flexible Knowledge Distillation for Large Language Models",
         authors=["Charles Goddard", "Lucas Atkins"],
         venue="Software, GitHub (Arcee AI)",
         bib=dict(type="misc", howpublished="\\url{https://github.com/arcee-ai/DistillKit}", url="https://github.com/arcee-ai/DistillKit",
                  note="Software, Arcee AI"),
         links=[("GitHub", "https://github.com/arcee-ai/DistillKit")],
         verified="GitHub repository",
         summary="Production toolkit for LLM distillation with online and offline modes. Offline teacher outputs are "
                 "stored compressed (sparse top-k or dense, with polynomial approximation, quantisation and "
                 "bit-packing); losses include KL, JSD, TVD, ranking and hidden-state alignment.",
         relevance="The closest tool for stored teacher logits; aimed at post-training pretrained models at scale."),
    dict(key="wang2025easydistill", theme="K", arxiv="2505.20888", year=2025,
         venue="EMNLP 2025 System Demonstrations, pages 787-795",
         bib=dict(type="inproceedings", booktitle="Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
                  pages="787--795"),
         summary="Alibaba's toolkit for black-box and white-box LLM distillation: data synthesis, SFT, ranking "
                 "optimisation and RL. In white-box mode the teacher's forward pass runs before student training and "
                 "only the top-k logits may be used, with forward or reverse KL.",
         relevance="Shows stored (top-k) teacher logits plus teacher data synthesis in an open toolkit; its examples "
                   "use pretrained Qwen students."),
    dict(key="zhang2026kdflow", theme="K", arxiv="2603.01875", year=2026,
         venue="arXiv preprint",
         bib=dict(type="misc"),
         summary="Decoupled LLM distillation framework: the teacher runs on SGLang and the student on FSDP2; the "
                 "teacher's final hidden states are sent instead of logits, which are recomputed on the student side. "
                 "Supports off- and on-policy and cross-tokenizer KD with 1.4-6.4x speed-ups.",
         relevance="State of KD tooling in 2026 (throughput and scale); orthogonal to the thesis's role as a research "
                   "instrument."),
]
