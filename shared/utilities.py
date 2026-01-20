import os
import json
from typing import Dict, List, Tuple, Optional

DEBUG_TOKEN_MAP_PATH = "debug_token_map.json"


class Utilities:
    """
    Manages vocabulary for distillation with ordered token sections.

    Token order in logit vectors:
    1. EXAMPLE_TOKENS (0 to n1): Minimum tokens for JSON output - "hardest" label
    2. WHITESPACE_TOKENS (n1 to n2): All whitespace variations
    3. PROMPT_TOKENS (n2 to n3): Tokens from evaluation prompt
    4. AUXILIARY_TOKENS (n3 to n4): Common English words - "softest" label
    """

    EXAMPLE_JSON_RESPONSES = [
        '```json\n{\n    "tone": "aggressive",\n    "sentiment": "negative",\n    "safety": "harmful",\n    "toxicity": "toxic"\n}',
        '```json\n{\n    "tone": "rude",\n    "sentiment": "neutral",\n    "safety": "safe",\n    "toxicity": "respectful"\n}',
        '```json\n{\n    "tone": "neutral",\n    "sentiment": "positive",\n    "safety": "harmful",\n    "toxicity": "toxic"\n}',
        '```json\n{\n    "tone": "polite",\n    "sentiment": "negative",\n    "safety": "safe",\n    "toxicity": "respectful"\n}',
        '```json\n{\n    "tone": "friendly",\n    "sentiment": "positive",\n    "safety": "harmful",\n    "toxicity": "toxic"\n}',
    ]

    WHITESPACE_TOKENS = [
        "\n", "\n\n", "\n\n\n",
        "\u2581",
        "\u2581\u2581",
        "\u2581\u2581\u2581",
        "\u2581\u2581\u2581\u2581",
        "\u2581\u2581\u2581\u2581\u2581",
        "\u2581\u2581\u2581\u2581\u2581\u2581",
        "\u2581\u2581\u2581\u2581\u2581\u2581\u2581",
        "\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581",
        "\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581",
        "\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581",
        "\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581",
        "\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581\u2581",
        "\t", "\t\t",
        " ",
        "    ",
        "        ",
    ]

    PROMPT_TOKENS = [
        "Analyze", "sentence", "return", "evaluation", "JSON",
        "Sentence", "Provide", "exactly", "one", "value", "for", "each",
        "field", "based", "on", "the", "content",
        "tone", "sentiment", "safety", "toxicity",
        "-", "(", ")", ".", ":",
    ]

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

    @classmethod
    def create_evaluation_prompt(cls, sentence: str) -> str:
        prompt = f"""Analyze this sentence and return your evaluation as JSON:


Sentence: "{sentence}"

Provide exactly one value for each field based on the sentence content:
    - tone: aggressive, rude, neutral, polite, friendly
    - sentiment: negative, neutral, positive
    - safety: harmful, safe
    - toxicity: toxic, respectful

JSON:
"""
        return prompt

    @classmethod
    def _get_example_tokens(cls, tokenizer) -> List[str]:
        all_tokens = []
        for response in cls.EXAMPLE_JSON_RESPONSES:
            tokens = tokenizer.tokenize(response)
            all_tokens.extend(tokens)
        seen = set()
        unique = []
        for t in all_tokens:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    @classmethod
    def build_vocabulary(cls, tokenizer) -> dict:
        tokenizer_name = getattr(tokenizer, 'name_or_path', str(id(tokenizer)))

        if tokenizer_name in cls._vocabulary_cache:
            return cls._vocabulary_cache[tokenizer_name]

        example_tokens = cls._get_example_tokens(tokenizer)

        seen = set()
        token_list = []
        section_boundaries = {
            'example': [0, 0],
            'whitespace': [0, 0],
            'prompt': [0, 0],
            'auxiliary': [0, 0],
        }

        section_tokens = [
            ('example', example_tokens),
            ('whitespace', cls.WHITESPACE_TOKENS),
            ('prompt', cls.PROMPT_TOKENS),
            ('auxiliary', cls.AUXILIARY_TOKENS),
        ]

        for section_name, tokens in section_tokens:
            section_boundaries[section_name][0] = len(token_list)
            for token in tokens:
                if token not in seen:
                    token_ids = tokenizer.convert_tokens_to_ids([token])
                    if token_ids[0] != tokenizer.unk_token_id:
                        seen.add(token)
                        token_list.append(token)
            section_boundaries[section_name][1] = len(token_list)

        token_to_id = {}
        token_to_index = {}
        id_to_index = {}

        for idx, token in enumerate(token_list):
            token_ids = tokenizer.convert_tokens_to_ids([token])
            token_id = token_ids[0]
            token_to_id[token] = token_id
            token_to_index[token] = idx
            id_to_index[token_id] = idx

        positions = {
            section_name: (start_end[0], start_end[1])
            for section_name, start_end in section_boundaries.items()
        }
        positions['total'] = (0, len(token_list))

        vocabulary = {
            'token_list': token_list,
            'token_to_id': token_to_id,
            'token_to_index': token_to_index,
            'id_to_index': id_to_index,
            'positions': positions,
            'vocab_size': len(token_list),
        }

        cls._vocabulary_cache[tokenizer_name] = vocabulary

        cls._write_debug_file(vocabulary, tokenizer_name)

        print(f"Built vocabulary: {len(token_list)} tokens")
        print(f"  Example tokens: {positions['example'][0]}-{positions['example'][1]} ({positions['example'][1] - positions['example'][0]} tokens)")
        print(f"  Whitespace tokens: {positions['whitespace'][0]}-{positions['whitespace'][1]} ({positions['whitespace'][1] - positions['whitespace'][0]} tokens)")
        print(f"  Prompt tokens: {positions['prompt'][0]}-{positions['prompt'][1]} ({positions['prompt'][1] - positions['prompt'][0]} tokens)")
        print(f"  Auxiliary tokens: {positions['auxiliary'][0]}-{positions['auxiliary'][1]} ({positions['auxiliary'][1] - positions['auxiliary'][0]} tokens)")

        return vocabulary

    @classmethod
    def _write_debug_file(cls, vocabulary: dict, tokenizer_name: str) -> None:
        if os.path.exists(DEBUG_TOKEN_MAP_PATH):
            return

        debug_data = {
            'tokenizer': tokenizer_name,
            'vocab_size': vocabulary['vocab_size'],
            'positions': vocabulary['positions'],
            'tokens': [
                {
                    'index': idx,
                    'token': token,
                    'token_id': vocabulary['token_to_id'][token],
                    'section': cls._get_section_for_index(idx, vocabulary['positions']),
                }
                for idx, token in enumerate(vocabulary['token_list'])
            ],
        }

        with open(DEBUG_TOKEN_MAP_PATH, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)

        print(f"Debug token map written to: {DEBUG_TOKEN_MAP_PATH}")

    @classmethod
    def _get_section_for_index(cls, idx: int, positions: dict) -> str:
        for section in ['example', 'whitespace', 'prompt', 'auxiliary']:
            start, end = positions[section]
            if start <= idx < end:
                return section
        return 'unknown'

    @classmethod
    def get_example_token_positions(cls, vocabulary: dict) -> Tuple[int, int]:
        return vocabulary['positions']['example']

    @classmethod
    def get_whitespace_token_positions(cls, vocabulary: dict) -> Tuple[int, int]:
        return vocabulary['positions']['whitespace']

    @classmethod
    def get_prompt_token_positions(cls, vocabulary: dict) -> Tuple[int, int]:
        return vocabulary['positions']['prompt']

    @classmethod
    def get_auxiliary_token_positions(cls, vocabulary: dict) -> Tuple[int, int]:
        return vocabulary['positions']['auxiliary']

    @classmethod
    def get_label_size_for_softness(cls, vocabulary: dict, softness: str) -> int:
        positions = vocabulary['positions']
        if softness == 'hard':
            return positions['example'][1]
        elif softness == 'medium':
            return positions['prompt'][1]
        elif softness == 'soft':
            return positions['auxiliary'][1]
        else:
            raise ValueError(f"Unknown softness level: {softness}")

    @classmethod
    def extract_logits_as_vector(cls, logits, vocabulary: dict) -> List[float]:
        token_list = vocabulary['token_list']
        token_to_id = vocabulary['token_to_id']

        result = []
        for token in token_list:
            token_id = token_to_id[token]
            result.append(logits[token_id].item())

        return result

    @classmethod
    def get_json_response_tokens(cls, tokenizer) -> List[str]:
        vocabulary = cls.build_vocabulary(tokenizer)
        positions = vocabulary['positions']
        example_end = positions['whitespace'][1]  # Include whitespace
        return vocabulary['token_list'][:example_end]
