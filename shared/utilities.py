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

    REDDIT_SENTIMENT_EXAMPLE_RESPONSES = [
        '```json\n{\n    "tone": "aggressive",\n    "sentiment": "negative",\n    "safety": "harmful",\n    "toxicity": "toxic"\n}',
        '```json\n{\n    "tone": "rude",\n    "sentiment": "neutral",\n    "safety": "safe",\n    "toxicity": "respectful"\n}',
        '```json\n{\n    "tone": "neutral",\n    "sentiment": "positive",\n    "safety": "harmful",\n    "toxicity": "toxic"\n}',
        '```json\n{\n    "tone": "polite",\n    "sentiment": "negative",\n    "safety": "safe",\n    "toxicity": "respectful"\n}',
        '```json\n{\n    "tone": "friendly",\n    "sentiment": "positive",\n    "safety": "harmful",\n    "toxicity": "toxic"\n}',
    ]

    # Math vocabulary — populated from empirical test (200 examples, 0 failures)
    # Observed 31 unique tokens. F, G added for robustness (+2 beyond max scaffold var E).
    MATH_WORD_PROBLEM_EXAMPLE_RESPONSES = [
        "100\nB=50\nC=A+B=150\nSolution: 150;",
        "87\nB=43\nC=A-B=44\nSolution: 44;",
        "6\nB=9\nC=A*B=54\nSolution: 54;",
        "900\nB=300\nC=A/B=3\nSolution: 3;",
        "4\nB=7\nC=10\nD=A*B=28\nE=D+C=38\nSolution: 38;",
        "3\nB=5\nC=2\nD=A*B=15\nE=D-C=13\nSolution: 13;",
        "93\nB=67\nC=63\nD=A+B=160\nE=D*C=10080\nSolution: 10080;",
        "20\nB=8\nC=5\nD=A-B=12\nE=D*C=60\nSolution: 60;",
        "933\nB=844\nC=208\nD=A+B=1777\nE=D-C=1569\nSolution: 1569;",
        "939\nB=313\nC=110\nD=A/B=3\nE=D+C=113\nSolution: 113;",
        "50\nB=30\nC=A>B=True\nSolution: True;",
        "20\nB=70\nC=A>B=False\nSolution: False;",
        "15\nB=7\nC=12\nD=A-B=8\nE=D>C=False\nSolution: False;",
        "90\nB=10\nC=30\nD=A-B=80\nE=D>C=True\nSolution: True;",
        "8\nB=4\nC=50\nD=A*B=32\nE=D>C=False\nSolution: False;",
        "7\nB=9\nC=40\nD=A*B=63\nE=D>C=True\nSolution: True;",
        "30\nB=50\nC=A>B=False\nSolution: False;",
        "80\nB=20\nC=A>B=True\nSolution: True;",
    ]

    MATH_WORD_PROBLEM_PROMPT_TOKENS = [
        "Follow", "this", "example", "exactly",
        "Problem", "Solution", "Calculations",
        "True", "False", "▁True", "▁False",
    ]

    MATH_WORD_PROBLEM_AUXILIARY_TOKENS = [
        "F", "G", ".",
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

    MATH_WORD_PROBLEM_MASTER_PROMPT = """Follow this example exactly:

Problem: "Lisa has 3 bags with 5 apples each. She eats 2 apples. How many are left?"
A=?
B=?
C=?
D=A*B=?
E=D-C=?
Solution: ?;

Calculations:
A=3
B=5
C=2
D=A*B=15
E=D-C=13
Solution: 13;

Problem: "Tom has 15 stickers. He gives away 7. Sarah has 12 stickers. Does Tom have more stickers than Sarah?"
A=?
B=?
C=?
D=A-B=?
E=D>C=?
Solution: ?;

Calculations:
A=15
B=7
C=12
D=A-B=8
E=D>C=False
Solution: False;

"""

    _vocabulary_cache: Dict[str, dict] = {}

    @classmethod
    def create_math_word_problem_prompt(cls, text: str) -> str:
        # text includes full scaffold: Problem: "..."\nA=?\nB=?\n...Solution: ?;
        # Append "Calculations:\nA=" so Gemma sees the scaffold then is forced
        # to start filling in values immediately.
        return cls.MATH_WORD_PROBLEM_MASTER_PROMPT + text + "\n\nCalculations:\nA="

    @classmethod
    def create_reddit_sentiment_prompt(cls, sentence: str) -> str:
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
    def _get_example_tokens(cls, tokenizer, domain: str = "reddit_comment_sentiment") -> List[str]:
        if domain == "math_word_problem":
            responses = cls.MATH_WORD_PROBLEM_EXAMPLE_RESPONSES
        else:
            responses = cls.REDDIT_SENTIMENT_EXAMPLE_RESPONSES

        all_tokens = []
        for response in responses:
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
    def build_vocabulary(cls, tokenizer, domain: str = "reddit_comment_sentiment") -> dict:
        tokenizer_name = getattr(tokenizer, 'name_or_path', str(id(tokenizer)))
        cache_key = f"{tokenizer_name}_{domain}"

        if cache_key in cls._vocabulary_cache:
            return cls._vocabulary_cache[cache_key]

        example_tokens = cls._get_example_tokens(tokenizer, domain)

        if domain == "math_word_problem":
            prompt_tokens = cls.MATH_WORD_PROBLEM_PROMPT_TOKENS
            auxiliary_tokens = cls.MATH_WORD_PROBLEM_AUXILIARY_TOKENS
        else:
            prompt_tokens = cls.PROMPT_TOKENS
            auxiliary_tokens = cls.AUXILIARY_TOKENS

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
            ('prompt', prompt_tokens),
            ('auxiliary', auxiliary_tokens),
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

        for idx, token in enumerate(token_list):
            token_ids = tokenizer.convert_tokens_to_ids([token])
            token_to_id[token] = token_ids[0]

        positions = {
            section_name: (start_end[0], start_end[1])
            for section_name, start_end in section_boundaries.items()
        }
        positions['total'] = (0, len(token_list))

        vocabulary = {
            'token_list': token_list,
            'token_to_id': token_to_id,
            'positions': positions,
            'vocab_size': len(token_list),
        }

        cls._vocabulary_cache[cache_key] = vocabulary

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
    def get_response_tokens(cls, tokenizer, domain: str = "reddit_comment_sentiment") -> List[str]:
        vocabulary = cls.build_vocabulary(tokenizer, domain)
        positions = vocabulary['positions']
        example_end = positions['whitespace'][1]  # Include whitespace
        return vocabulary['token_list'][:example_end]

    @classmethod
    def get_json_response_tokens(cls, tokenizer) -> List[str]:
        return cls.get_response_tokens(tokenizer, "reddit_comment_sentiment")
