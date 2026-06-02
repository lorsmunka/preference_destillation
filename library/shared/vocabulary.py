"""Reduced output vocabulary — the domain-agnostic half.

The student predicts over a small, ordered vocabulary built from four sections:

1. example     (0  .. n1): tokens that appear in the domain's example responses — the
                            "hardest" labels the student must get exactly right.
2. whitespace  (n1 .. n2): every whitespace variation (shared across domains).
3. prompt      (n2 .. n3): tokens from the domain's evaluation prompt.
4. auxiliary   (n3 .. n4): common English words — the "softest" labels (shared base,
                            optionally prefixed by domain-specific extras).

Only the whitespace and base-auxiliary banks live here; the example responses, prompt
tokens and any extra auxiliary tokens are owned by each `Domain` (see `library.domain`).
`build_vocabulary` takes a `Domain` object and assembles the sections — no string dispatch.
"""

from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle; we only duck-type on the Domain below
    from library.domain.base import Domain


WHITESPACE_TOKENS = [
    "\n", "\n\n", "\n\n\n",
    "▁",
    "▁▁",
    "▁▁▁",
    "▁▁▁▁",
    "▁▁▁▁▁",
    "▁▁▁▁▁▁",
    "▁▁▁▁▁▁▁",
    "▁▁▁▁▁▁▁▁",
    "▁▁▁▁▁▁▁▁▁",
    "▁▁▁▁▁▁▁▁▁▁",
    "▁▁▁▁▁▁▁▁▁▁▁",
    "▁▁▁▁▁▁▁▁▁▁▁▁",
    "\t", "\t\t",
    " ",
    "    ",
    "        ",
]

# Common English words shared by every domain — the "soft" end of the vocabulary.
AUXILIARY_TOKENS = [
    "the", "a", "an", "is", "of", "and", "to", "in", "that", "it", "you",
    "this", "these", "those", "my", "your", "his", "her", "its",
    "our", "their", "we", "they", "he", "she", "me", "him", "them", "us",
    "who", "what", "which", "where", "when", "why", "how", "all", "some",
    "any", "no", "every", "each", "both", "few", "many", "much", "most",
    "very", "quite", "somewhat", "extremely", "slightly", "moderately",
    "be", "have", "do", "say", "get", "make", "go", "know", "take", "see",
    "come", "think", "look", "want", "give", "use", "find", "tell", "ask",
    "work", "seem", "feel", "try", "leave", "call", "need", "become", "put",
    "mean", "keep", "let", "begin", "help", "show", "hear", "play", "run",
    "move", "live", "believe", "hold", "bring", "happen", "write", "provide",
    "sit", "stand", "lose", "pay", "meet", "include", "continue", "set",
    "learn", "change", "lead", "understand", "watch", "follow", "stop",
    "create", "speak", "read", "allow", "add", "spend", "grow", "open",
    "walk", "win", "offer", "remember", "love", "consider", "appear", "buy",
    "wait", "serve", "die", "send", "expect", "build", "stay", "fall",
    "good", "new", "first", "last", "long", "great", "little", "own", "other",
    "old", "right", "big", "high", "different", "small", "large", "next",
    "early", "young", "important", "public", "bad", "same", "able", "best",
    "better", "worse", "worst", "real", "sure", "free", "true", "full",
    "special", "easy", "clear", "recent", "certain", "personal",
    "nice", "wrong", "hard", "possible", "whole", "simple", "strong",
    "happy", "serious", "ready", "main", "major", "local", "low", "short",
    "yeah", "yep", "nope", "ok", "okay", "lol", "lmao", "tbh", "imo", "imho",
    "btw", "idk", "afaik", "iirc", "fwiw", "til", "eli5", "ama", "tldr",
    "literally", "actually", "basically", "honestly", "seriously", "really",
    "definitely", "probably", "maybe", "perhaps", "obviously", "clearly",
    "apparently", "supposedly", "allegedly", "essentially", "technically",
    "agree", "disagree", "like", "dislike", "hate", "enjoy", "prefer",
    "support", "oppose", "approve", "reject", "accept", "deny",
    "doubt", "trust", "suspect", "wonder", "guess", "assume", "hope", "wish",
    "amazing", "awesome", "terrible", "horrible", "fantastic", "wonderful",
    "awful", "excellent", "perfect", "stupid", "smart", "brilliant", "dumb",
    "ridiculous", "absurd", "reasonable", "fair", "unfair", "valid", "invalid",
    "but", "however", "although", "though", "still", "yet", "also", "too",
    "even", "just", "only", "already", "never", "always", "sometimes",
    "often", "usually", "rarely", "seldom", "ever", "again", "once",
    "therefore", "thus", "hence", "so", "because", "since", "while",
    "whereas", "unless", "until", "before", "after", "if", "then",
    "second", "third", "finally", "lastly", "meanwhile",
    "furthermore", "moreover", "besides", "instead", "otherwise",
    "people", "time", "year", "way", "day", "man", "thing", "woman", "life",
    "child", "world", "school", "state", "family", "student", "group",
    "country", "problem", "hand", "part", "place", "case", "week", "company",
    "system", "program", "question", "government", "number", "night",
    "point", "home", "water", "room", "mother", "area", "money", "story",
    "fact", "month", "lot", "study", "book", "eye", "job", "word",
    "business", "issue", "side", "kind", "head", "house", "service", "friend",
    "father", "power", "hour", "game", "line", "end", "member", "law", "car",
    "city", "community", "name", "president", "team", "minute", "idea",
    "with", "at", "by", "from", "up", "about", "into", "over",
    "beneath", "under", "above", "below", "between", "among", "through",
    "during", "without", "behind", "beyond", "against", "within",
    "along", "following", "across", "around", "toward", "upon", "onto",
    "or", "nor", "as", "than", "whether", "either", "neither",
    "not", "more", "now", "here", "there", "well", "back", "then",
    "away", "down", "off", "out", "almost", "enough", "together",
    "especially", "particularly", "simply", "certainly",
    "completely", "entirely", "generally", "largely", "mainly",
    "mostly", "nearly", "partly", "primarily", "purely", "relatively",
    "roughly", "significantly", "solely", "specifically",
    "strongly", "totally", "typically", "ultimately", "virtually", "widely",
    "stuff", "things", "gonna", "wanna", "gotta", "kinda", "sorta", "dunno",
    "nah", "yup", "meh", "wow", "whoa", "damn", "dude", "guy", "guys",
    "bro", "buddy", "mate", "folks", "hey", "hi", "hello", "bye",
    "thanks", "thank", "please", "sorry", "excuse", "cool", "neat", "sick",
    "crazy", "insane", "wild", "weird", "odd", "strange", "funny", "hilarious",
    "false", "correct", "incorrect", "yes",
    "precisely", "indeed", "absolutely",
    "partially", "entirely",
    "agreed", "disagreed", "confirmed", "denied", "verified", "disputed",
    "might", "could", "would", "should", "must", "can", "will", "may",
    "suggest", "indicate", "imply", "tend",
    "likely", "unlikely", "impossible", "uncertain",
    "definite", "indefinite", "unsure", "confident", "doubtful",
]


_vocabulary_cache: Dict[str, dict] = {}


def _example_tokens(tokenizer, domain: "Domain") -> List[str]:
    """Unique tokens (order preserved) across the domain's example responses."""
    seen = set()
    unique = []
    for response in domain.example_responses:
        for token in tokenizer.tokenize(response):
            if token not in seen:
                seen.add(token)
                unique.append(token)
    return unique


def build_vocabulary(tokenizer, domain: "Domain",
                     auxiliary_token_percentage: float = 1.0) -> dict:
    """Assemble the ordered reduced vocabulary for a domain.

    Sections: example (from `domain.example_responses`), whitespace (shared), prompt
    (`domain.prompt_tokens`), auxiliary (`domain.extra_auxiliary_tokens` + the shared
    base). Returns a dict with `token_list`, `token_to_id`, `positions`, `vocab_size`.
    """
    tokenizer_name = getattr(tokenizer, "name_or_path", str(id(tokenizer)))
    cache_key = f"{tokenizer_name}_{domain.name}_{auxiliary_token_percentage}"
    if cache_key in _vocabulary_cache:
        return _vocabulary_cache[cache_key]

    extra_auxiliary = list(domain.extra_auxiliary_tokens)
    seen_extra = set(extra_auxiliary)
    auxiliary_tokens = extra_auxiliary + [t for t in AUXILIARY_TOKENS if t not in seen_extra]

    auxiliary_count = int(len(auxiliary_tokens) * auxiliary_token_percentage)
    auxiliary_tokens = auxiliary_tokens[:auxiliary_count]

    section_tokens = [
        ("example", _example_tokens(tokenizer, domain)),
        ("whitespace", WHITESPACE_TOKENS),
        ("prompt", domain.prompt_tokens),
        ("auxiliary", auxiliary_tokens),
    ]

    seen = set()
    token_list: List[str] = []
    positions: Dict[str, Tuple[int, int]] = {}

    for section_name, tokens in section_tokens:
        start = len(token_list)
        for token in tokens:
            if token not in seen:
                token_ids = tokenizer.convert_tokens_to_ids([token])
                if token_ids[0] != tokenizer.unk_token_id:
                    seen.add(token)
                    token_list.append(token)
        positions[section_name] = (start, len(token_list))

    positions["total"] = (0, len(token_list))

    token_to_id = {
        token: tokenizer.convert_tokens_to_ids([token])[0]
        for token in token_list
    }

    vocabulary = {
        "token_list": token_list,
        "token_to_id": token_to_id,
        "positions": positions,
        "vocab_size": len(token_list),
    }
    _vocabulary_cache[cache_key] = vocabulary

    print(f"Built vocabulary: {len(token_list)} tokens")
    for section_name in ("example", "whitespace", "prompt", "auxiliary"):
        start, end = positions[section_name]
        print(f"  {section_name.capitalize()} tokens: {start}-{end} ({end - start} tokens)")

    return vocabulary


def get_response_tokens(tokenizer, domain: "Domain") -> List[str]:
    """The example + whitespace tokens — what a valid generated response is built from."""
    vocabulary = build_vocabulary(tokenizer, domain)
    response_end = vocabulary["positions"]["whitespace"][1]
    return vocabulary["token_list"][:response_end]


def extract_logits_as_vector(logits, vocabulary: dict) -> List[float]:
    """Project a full teacher logit row down to the reduced vocabulary, in order."""
    token_to_id = vocabulary["token_to_id"]
    return [logits[token_to_id[token]].item() for token in vocabulary["token_list"]]
