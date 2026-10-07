# Non-bibliography content of the review page. Reference keys in [[key]] become links.

TLDR = [
    "<b>State of the field.</b> Knowledge distillation (KD) for language models moved from word- and sequence-level KD "
    "[[kim2016sequence]] to choosing the divergence and fixing the train/inference mismatch with on-policy KD "
    "[[gu2024minillm]] [[agarwal2024onpolicy]] [[ko2024distillm]]. 2025-2026 work explains why mixing hard and soft labels "
    "works [[wang2026bridge]], how stored top-k logits bias the student [[anshumann2025sparse]], and when KD beats plain "
    "training at all [[busbridge2025scaling]]. Nearly every student is a pretrained model of 0.1-7B parameters.",
    "<b>From scratch is established, but big and general.</b> Gemma 2/3 pre-train their small models with KD "
    "[[gemma2024gemma2]] [[gemma2025gemma3]] and a design study maps truncation, KD/LM mixing and teacher size "
    "[[peng2025pretraining]]. Tiny from-scratch models on teacher-written narrow text exist [[eldan2023tinystories]] "
    "[[liu2023tinygsm]], but they learn from the text only (cross-entropy), not from teacher distributions.",
    "<b>Agent distillation is hot and uses pretrained students.</b> Trajectory SFT [[kang2025distilling]], span-aware KL "
    "[[liu2026structured]], top-k KD plus RL [[ni2026star]], on-device function calling [[erdogan2024tinyagent]] "
    "[[chen2024octopus]] and harness distillation [[ye2026harnesszero]] all start from 0.5-9B pretrained models; the only "
    "from-scratch agentic analogue distils into an RL policy, not a text model [[zhou2024llm4teach]] (survey: "
    "[[luo2026distilling]]).",
    "<b>Where this thesis sits.</b> A research framework that combines a randomly initialised 1-100M student, a "
    "task-derived output vocabulary with stored teacher logits over it, annealed KD/CE and temperature schedules, "
    "distribution analytics logged at evaluation time, and a model-plus-harness agent trained on code-verified data. "
    "Every ingredient has precedent; the combination, and the measurement of when it helps, is the contribution.",
    "<b>Defensible claims.</b> (1) The combination above, hedged with 'to our knowledge' (see the novelty check). "
    "(2) A seed-replicated finding that the KD/CE schedule is a null on near-deterministic domains and orders the arms "
    "on a high-entropy one, which agrees with theory [[menon2021statistical]] [[wang2026bridge]]. (3) A working "
    "phone-sized tutor whose harness checks every solution line against executed code.",
    "<b>Claims to avoid.</b> Annealed KD-to-CE or scheduled KD share as new [[clark2019bam]] [[peng2025pretraining]] "
    "[[zhou2024llm4teach]] [[yang2020textbrewer]]; entropy-aware per-token weighting as new [[wang2026bridge]] "
    "[[zhang2026egad]]; vocabulary reduction in general [[kolesnikova2022knowledge]] [[goel2025vocabtrim]] "
    "[[eldan2023tinystories]]; stored or truncated teacher logits [[gemma2025gemma3]] [[goddard2024distillkit]] "
    "[[wang2025easydistill]]; 'the first token is the decision' [[chen2024octopus]]; 'verified' data without saying "
    "verified against what.",
    "<b>Where a reviewer will push.</b> Teacher-forced and gold-history scores flatter (gold history lifts one model "
    "from 4.64 to 7.32 in [[kwan2024mteval]]; [[arora2022exposure]]); the 4-12B teacher is far from the capacity-gap "
    "optimum for a 10M student [[zhang2025capacity]]; random initialisation costs about 10 points on GSM8K in "
    "[[liu2023tinygsm]]; public test sets may be in the teacher [[zhang2024careful]].",
    "<b>Tools.</b> TextBrewer, torchdistill, TRL's GKD trainer, DistillKit, EasyDistill and KDFlow target pretrained "
    "(mostly Hugging Face) students and scale; none builds a task vocabulary or trains a from-scratch student by "
    "default. The framework is a research instrument for that niche, not a replacement for them.",
]

CLAIMS = [
    ("A large LLM distilled into a randomly initialised student of tens of millions of parameters, trained on one "
     "narrow domain, that works as an agent inside a harness and learns from token-level teacher distributions.",
     "defend", "Defensible as a combination, hedged ('to our knowledge'). Closest: span-aware agent KD on pretrained "
     "students, agent SFT on pretrained students, an RL-policy analogue, tiny from-scratch models without logits, "
     "from-scratch KD at general scale.",
     ["liu2026structured", "kang2025distilling", "zhou2024llm4teach", "eldan2023tinystories", "peng2025pretraining", "luo2026distilling"]),
    ("A task-derived reduced output vocabulary, with KL to the teacher's distribution restricted to it.",
     "defend", "No direct precedent found, but cite the neighbours: KD with a reduced (re-tokenised) vocabulary, a "
     "frequency-trimmed drafter head, TinyStories' top-10K vocabulary, Gemma 3's renormalised sampled subset, "
     "span-restricted normalisation in agent KD.",
     ["kolesnikova2022knowledge", "goel2025vocabtrim", "eldan2023tinystories", "gemma2025gemma3", "liu2026structured"]),
    ("Annealing the KD/CE ratio (0.9 to 0.1) and the temperature.",
     "avoid", "Present as configurable dials with precedent: teacher annealing from teacher to gold labels, a "
     "scheduled KD share in pre-training distillation, a KL weight annealed to zero, toolkit weight and temperature "
     "schedulers, curriculum temperature.",
     ["clark2019bam", "peng2025pretraining", "zhou2024llm4teach", "yang2020textbrewer", "li2023curriculum"]),
    ("Per-token KD/CE weighting by teacher entropy.",
     "avoid", "Adaptive hard/soft mixing per step and entropy-guided curricula and temperatures already exist.",
     ["wang2026bridge", "zhang2026egad"]),
    ("The benefit of distillation, and of the schedule, depends on teacher entropy.",
     "reword", "Defensible as a measured, seed-replicated confirmation across domains, not as a discovery: theory "
     "predicts little gain from a near one-hot teacher. Weak point: only two or three entropy levels so far.",
     ["menon2021statistical", "wang2026bridge", "stanton2021does", "busbridge2025scaling"]),
    ("Teacher logits are stored once (bf16) and reused across runs.",
     "reword", "An engineering choice with precedent, not a contribution. The fixed vocabulary avoids per-token top-k "
     "truncation, but teacher mass outside the vocabulary is still dropped: report how much.",
     ["peng2025pretraining", "gemma2025gemma3", "anshumann2025sparse", "goddard2024distillkit", "wang2025easydistill"]),
    ("Distribution analytics logged during evaluation (teacher-forced vs autoregressive accuracy, top-k agreement, "
     "teacher and student entropy, mean target rank).",
     "defend", "Defensible as a framework feature (no surveyed toolkit logs these by default); the metrics themselves "
     "are standard and should be credited.",
     ["stanton2021does", "arora2022exposure", "bengio2015scheduled"]),
    ("Routing by the first output tag.",
     "reword", "Claim the harness design, not the idea: functional tokens already make one token the decision.",
     ["chen2024octopus", "erdogan2024tinyagent"]),
    ("Training data is verified.",
     "reword", "Say exactly what is checked: every gold solution and follow-up is consistent with an executed "
     "solve(); whether the code models the problem text correctly is not proven (TinyGSM checked only executability).",
     ["liu2023tinygsm", "zhu2024pad", "kang2025distilling"]),
]

# Positioning table: (row label, [keys], cells...). Each cell = (status, note). status in yes/partial/no/unknown/na
COLUMNS = ["From-scratch student", "Reduced vocabulary", "Stored teacher logits", "Annealed KD/CE",
           "Built-in distribution analytics", "Agent / harness distillation", "Verified data"]

POSITIONING = [
    ("TextBrewer (2020)", ["yang2020textbrewer"], [
        ("partial", "any model; examples are pretrained encoders"), ("no", ""), ("partial", "in-memory logit cache"),
        ("yes", "KD/hard-label weight and temperature schedulers"), ("no", "losses only"), ("no", ""), ("no", "")]),
    ("torchdistill (2021)", ["matsubara2021torchdistill"], [
        ("partial", "any model via config"), ("no", ""), ("unknown", "not documented"), ("unknown", "not documented"),
        ("no", ""), ("no", ""), ("no", "")]),
    ("TRL GKDTrainer", ["vonwerra2020trl", "agarwal2024onpolicy"], [
        ("no", "post-training HF models"), ("no", "shared tokenizer"), ("no", "live teacher"), ("no", "fixed mix settings"),
        ("no", ""), ("no", ""), ("no", "")]),
    ("DistillKit (2024)", ["goddard2024distillkit"], [
        ("no", "post-training"), ("no", ""), ("yes", "compressed top-k or dense"), ("unknown", "not documented"),
        ("no", ""), ("no", ""), ("no", "")]),
    ("EasyDistill (2025)", ["wang2025easydistill"], [
        ("no", "pretrained Qwen students"), ("no", ""), ("yes", "teacher pass first, top-k"), ("unknown", "not documented"),
        ("no", ""), ("partial", "agent trajectory synthesis"), ("partial", "filtered / verified synthesis")]),
    ("KDFlow (2026)", ["zhang2026kdflow"], [
        ("no", ""), ("no", "cross-tokenizer instead"), ("no", "hidden states streamed, logits recomputed"),
        ("unknown", "not documented"), ("no", ""), ("no", ""), ("no", "")]),
    ("MiniLLM / GKD / DistiLLM", ["gu2024minillm", "agarwal2024onpolicy", "ko2024distillm"], [
        ("no", "pretrained 0.1-13B"), ("no", ""), ("no", "on-policy, live teacher"), ("no", "schedules student data, not KD/CE"),
        ("no", "analyses in papers only"), ("no", ""), ("no", "")]),
    ("Pre-training distillation (Peng 2025; Gemma 2/3)", ["peng2025pretraining", "gemma2024gemma2", "gemma2025gemma3"], [
        ("yes", "about 1B and up"), ("no", "per-token top-p/top-k or sampled subset"), ("yes", "offline, truncated"),
        ("yes", "WSD-scheduled KD share (Peng)"), ("no", ""), ("no", ""), ("no", "")]),
    ("TinyStories / TinyGSM", ["eldan2023tinystories", "liu2023tinygsm"], [
        ("yes", "TinyStories; TinyGSM as an ablation"), ("partial", "TinyStories keeps top 10K tokens"),
        ("no", "CE on teacher text"), ("no", ""), ("no", ""), ("no", ""), ("partial", "TinyGSM: executable only")]),
    ("Structured Agent Distillation (2026)", ["liu2026structured"], [
        ("unknown", "GPT-2-size students; init not stated"), ("partial", "span-restricted normalisation"),
        ("unknown", "not stated"), ("no", "fixed span weights"), ("no", ""), ("yes", "ReAct trajectories"), ("no", "")]),
    ("Agent Distillation (Kang 2025)", ["kang2025distilling"], [
        ("no", "Qwen2.5 0.5-7B"), ("no", ""), ("no", "trajectory SFT"), ("no", ""), ("no", ""),
        ("yes", "retrieval and code tools"), ("partial", "wrong trajectories filtered")]),
    ("STAR (2026)", ["ni2026star"], [
        ("no", "Qwen3 0.6B+"), ("no", ""), ("partial", "top-k forward KL"), ("partial", "KD then RL curriculum"),
        ("no", ""), ("partial", "function calling"), ("partial", "reward vs ground truth")]),
    ("TinyAgent / Octopus v2", ["erdogan2024tinyagent", "chen2024octopus"], [
        ("no", "1.1-7B / 2B pretrained"), ("no", "Octopus adds tokens"), ("no", "SFT"), ("no", ""), ("no", ""),
        ("partial", "function calling, single turn"), ("unknown", "")]),
    ("LLM4Teach (2024)", ["zhou2024llm4teach"], [
        ("unknown", "24K / 10M RL policy; init not stated"), ("na", "discrete actions"), ("no", "live LLM teacher"),
        ("yes", "KL weight annealed to zero"), ("no", ""), ("partial", "RL agent, not text"), ("partial", "environment reward")]),
    ("Harness-Zero (2026)", ["ye2026harnesszero"], [
        ("no", "pretrained base model"), ("no", ""), ("no", "SFT on corrected trajectories"), ("no", ""), ("no", ""),
        ("yes", "harness behaviour moved into weights"), ("partial", "task success")]),
    ("This thesis", [], [
        ("yes", "1-100M, random init"), ("yes", "task-derived (about 525 / 18k tokens)"),
        ("yes", "bf16 over the reduced vocabulary; teacher-generated or teacher-forced on gold"),
        ("yes", "cosine KD/CE ratio and temperature; optional entropy weighting"),
        ("yes", "logged at every evaluation"),
        ("partial", "model distilled; harness hand-written, not distilled"),
        ("yes", "each gold solution consistent with an executed solve()")]),
]

NOT_DONE = [
    "No on-policy distillation (the student never trains on its own samples with teacher feedback) and no RL "
    "[[agarwal2024onpolicy]] [[gu2024minillm]].",
    "No pretrained-student baseline and no smaller-teacher control so far [[liu2023tinygsm]] [[zhang2025capacity]].",
    "No cross-tokenizer or cross-family teacher: the vocabulary comes from the teacher's tokenizer, so teachers from "
    "other families are not comparable (repository finding F-70a).",
    "No hidden-state or attention distillation; only output distributions.",
    "The harness is hand-written; nothing of it is learned or distilled [[ye2026harnesszero]].",
    "No study with real pupils and no learning-gain measurement; the pupil is a template bank plus a 360-message "
    "free-written naturalness set [[macina2023mathdial]] [[macina2025mathtutorbench]].",
    "No constrained decoding at inference; validity is learned and checked [[willard2023efficient]].",
    "Single-GPU scale; no claim about throughput versus LLM toolkits [[zhang2026kdflow]].",
]

OUTLINE = [
    ("2.1", "Knowledge distillation: foundations", "Tudásdesztilláció: alapok",
     "Soft targets and temperature; why KD helps (variance, label information); fidelity vs generalisation; capacity gap.",
     ["hinton2015distilling", "gou2021survey", "menon2021statistical", "stanton2021does", "zhang2025capacity"]),
    ("2.2", "Distilling autoregressive language models", "Autoregresszív nyelvi modellek desztillációja",
     "Word- vs sequence-level KD; forward vs reverse KL and other divergences; on-policy KD and exposure bias; "
     "temperature; mixing and scheduling KD with CE; storing and truncating teacher logits.",
     ["kim2016sequence", "xu2024survey", "gu2024minillm", "agarwal2024onpolicy", "ko2024distillm", "wu2025rethinking",
      "li2023curriculum", "clark2019bam", "wang2026bridge", "zhang2026egad", "anshumann2025sparse"]),
    ("2.3", "Training small models from scratch", "Kis modellek tanítása nulláról",
     "Pre-training distillation at scale; distillation scaling laws; tiny models on teacher-written text; "
     "architecture for on-device models. Ends with the gap: no logit KD into tiny narrow-domain students.",
     ["gemma2024gemma2", "gemma2025gemma3", "peng2025pretraining", "busbridge2025scaling", "eldan2023tinystories",
      "gunasekar2023textbooks", "liu2023tinygsm", "liu2024mobilellm"]),
    ("2.4", "Vocabulary and output space", "Szókészlet és kimeneti tér",
     "Vocabulary size vs model size; trimmed and reduced vocabularies; subsets of the teacher distribution; special "
     "decision tokens. Motivates the task-derived output vocabulary.",
     ["tao2024scaling", "kolesnikova2022knowledge", "goel2025vocabtrim", "gemma2025gemma3", "liu2026structured", "chen2024octopus"]),
    ("2.5", "Distilling reasoning and verified data", "Érvelés desztillációja és ellenőrzött adatok",
     "Chain of thought and rationales; programs as reasoning; verifiers and step-level supervision; synthetic maths "
     "data and its checks. Motivates the four annotation formats and code-derived gold.",
     ["wei2022chain", "hsieh2023distilling", "gao2023pal", "zhu2024pad", "cobbe2021training", "lightman2024verify",
      "liu2023tinygsm", "deepseek2025r1"]),
    ("2.6", "LLM agents, harnesses and agent distillation", "Ágensek, keretrendszerek (harness) és ágensdesztilláció",
     "What an agent is (define it here); small models for agents; distilling agents into small models; harness vs "
     "weights; structured output and routing.",
     ["yao2023react", "sumers2024cognitive", "belcak2025small", "kang2025distilling", "liu2026structured", "ni2026star",
      "erdogan2024tinyagent", "chen2024octopus", "zhou2024llm4teach", "luo2026distilling", "ye2026harnesszero",
      "willard2023efficient", "tam2024speak", "chaganti2026different"]),
    ("2.7", "Maths word problems and their evaluation", "Matematikai szöveges feladatok és kiértékelésük",
     "Datasets and their artefacts; perturbation robustness; distractors; contamination. Justifies the held-out "
     "shapes, renumbering and SVAMP as the human-written test.",
     ["koncel2016mawps", "miao2020diverse", "patel2021svamp", "cobbe2021training", "mirzadeh2025gsmsymbolic",
      "shi2023distracted", "zhang2024careful"]),
    ("2.8", "Tutoring dialogues and multi-turn reliability", "Tutor-párbeszédek és többfordulós megbízhatóság",
     "LLM tutors and their failures; verify-then-respond; sycophancy under pushback; error compounding across turns "
     "and gold vs own history; cleaning the context.",
     ["macina2023mathdial", "macina2025mathtutorbench", "daheim2024stepwise", "sharma2024sycophancy", "wei2023simple",
      "kasneci2026sycophancy", "bengio2015scheduled", "ross2011dagger", "arora2022exposure", "kwan2024mteval",
      "laban2025lost", "zheng2026maigo"]),
    ("2.9", "Available tools and the tool choice", "Rendelkezésre álló eszközök és az eszközválasztás",
     "What each toolkit assumes (pretrained HF students, live or stored teacher, shared tokenizer) and why the thesis "
     "builds its own loop on PyTorch (the topic description asks for this choice to be justified).",
     ["yang2020textbrewer", "matsubara2021torchdistill", "vonwerra2020trl", "goddard2024distillkit",
      "wang2025easydistill", "zhang2026kdflow"]),
    ("2.10", "Summary: the gap this thesis fills", "Összegzés: a hiány, amelyet a dolgozat betölt",
     "The positioning table, the hedged novelty sentence and the list of what the thesis does not do.",
     ["luo2026distilling", "liu2026structured", "zhou2024llm4teach", "peng2025pretraining"]),
]

RISKS = [
    ("'Agent' may be read as 'structured intent routing'.",
     "The student picks one of three tags and writes a block; the harness keeps context, checks, retries and answers "
     "off-topic messages; the pupil is a template bank. A reviewer can say the agency lives in hand-written code.",
     "Define 'agent' up front with [[sumers2024cognitive]] and place the work in the 'knowledge in weights, harness in "
     "code' cell of [[luo2026distilling]]. Report route accuracy on the ~100 hand-written messages and an own-history "
     "multi-turn evaluation, not only per-turn scores."),
    ("The entropy claim rests on two or three points and is predicted by theory.",
     "Teacher entropy is exp(H) of about 1.004 (maths), 1.02 (Reddit) and 2.7-2.9 (post-gen); tokenizer granularity "
     "confounds per-token entropy (repository finding F-70a), and [[menon2021statistical]] and [[wang2026bridge]] "
     "already predict little KD gain from a near one-hot teacher.",
     "Phrase it as a measured confirmation. Add the free points the repository already lists: per-field entropy from "
     "stored Reddit logits, the teacher-temperature dial, tokenizer granularity (F-66a, F-69a)."),
    ("Missing baselines.",
     "No pretrained small student and no smaller teacher. Random initialisation costs about 10 GSM8K points in "
     "[[liu2023tinygsm]], and the capacity-gap law [[zhang2025capacity]] suggests the teacher is far too large. A "
     "reviewer will ask why not fine-tune a pretrained 100M model.",
     "Add one pretrained-student arm and/or a Gemma 1B teacher arm, or argue the scope explicitly (phone size, "
     "vocabulary control, clean measurement of what KD alone transfers)."),
    ("Evaluation that flatters.",
     "Teacher-forced and gold-history numbers overstate [[arora2022exposure]] [[kwan2024mteval]]; public benchmarks may "
     "be in the teacher [[zhang2024careful]]; single accuracies hide variance across renumberings "
     "[[mirzadeh2025gsmsymbolic]].",
     "Make autoregressive, own-history accuracy the headline; give seed and renumbering intervals; keep SVAMP strictly "
     "held out and say the student cannot be contaminated while the teacher can."),
    ("'Verified' data.",
     "Gold is consistent with an executed solve(), but nothing proves the code reads the problem correctly; TinyGSM "
     "did not check either [[liu2023tinygsm]].",
     "Report teacher-vs-code answer agreement (the answer_correct flag) as the evidence and a manual audit sample."),
    ("Offline-only distillation.",
     "Exposure bias is left untreated; on-policy KD [[agarwal2024onpolicy]] and hard/soft mixing [[wang2026bridge]] "
     "target exactly the autoregressive gap the framework measures.",
     "List on-policy KD as future work and connect the KD/CE results to [[wang2026bridge]]."),
]

UNVERIFIED = [
    "TinyGSM appendix B (random vs pretrained initialisation: 53.1 vs 63.3 at 125M, 55.5 vs 65.9 at 350M, 57.3 vs "
    "68.2 at 1.3B) is reported as pass@1 without the verifier by a full-text fetch summary; confirm against the table "
    "in the PDF before quoting.",
    "Student initialisation in LLM4Teach and Structured Agent Distillation is not stated in the papers (marked "
    "'not stated' in the table).",
    "Gemma 2: the exact teacher of the 2B model was not confirmed; the entry only says 'a larger teacher'.",
    "torchdistill, DistillKit and KDFlow: whether they offer loss-weight or temperature schedules is not documented in "
    "what was read (marked 'not documented').",
    "Structured Agent Distillation: a DOI reported by Semantic Scholar (10.65109/OLHJ8062) was not opened, so it is "
    "left out of the BibTeX.",
    "Bridge-Garden (ICML 2026) and STAR (ICLR 2026) acceptance comes from the authors' arXiv comments; final "
    "proceedings pages were not available.",
    "A claim seen only in a blog snippet, that the harness explains about eight times more variance in agent "
    "performance than the model, was not traced to a paper and is not cited.",
    "Considered but not included (verified, available if needed): DistilBERT (1910.01108), TinyBERT (1909.10351), "
    "f-divergence KD (2307.15190), DistiLLM-2 (2503.07067), ABKD (2505.04560), Speculative KD (2410.11325), "
    "MiniPLM (2410.17215), Minitron (2407.14679), on-policy distillation survey (2604.00626), phi-1.5 (2309.05463), "
    "TinyDialogues (2408.03617), BabyLM 2025 call (2502.10645), vocabulary trimming (2305.15020), ULD cross-tokenizer "
    "KD (2402.12030), MT-Bench-101 (2402.14762), 'What if LLMs ate their words' (2609.05882), GSM-Plus (2402.19255), "
    "Bridge tutoring (2310.10648), Toolformer (2302.04761), CodeAct (2402.01030), SWE-agent (2405.15793), DeDer "
    "(2412.11499), KD-Lib (2011.14691).",
]
