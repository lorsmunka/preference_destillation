from typing import Dict, List, Tuple

EXAMPLE_RESULT = """
```json
{
    "tone": "aggressive",
    "tone": "rude",
    "tone": "neutral",
    "tone": "polite",
    "tone": "friendly",
    "sentiment": "negative",
    "sentiment": "neutral",
    "sentiment": "positive",
    "safety": "harmful",
    "safety": "safe",
    "toxicity": "toxic",
    "toxicity": "respectful"
}
```"""

WHITESPACE_VARIATIONS = "\n \n  \n   \n    \n     \n      \n       \n        \n         \n          "


class Utilities:
    @staticmethod
    def create_evaluation_prompt(sentence: str) -> str:
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

    @staticmethod
    def build_vocabulary_map(tokenizer) -> Dict[str, int]:

        auxiliary_tokens = [
            # Original tokens
            "the", "a", "is", "of", "and", "to", "in", "that", "it", "you",
            "very", "quite", "somewhat", "extremely", "slightly", "moderately",

            # Common articles, pronouns, determiners
            "an", "this", "these", "those", "my", "your", "his", "her", "its",
            "our", "their", "we", "they", "he", "she", "me", "him", "them", "us",
            "who", "what", "which", "where", "when", "why", "how", "all", "some",
            "any", "no", "every", "each", "both", "few", "many", "much", "most",

            # Common verbs
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

            # Common adjectives
            "good", "new", "first", "last", "long", "great", "little", "own", "other",
            "old", "right", "big", "high", "different", "small", "large", "next",
            "early", "young", "important", "public", "bad", "same", "able", "best",
            "better", "worse", "worst", "real", "sure", "free", "true", "full",
            "special", "easy", "clear", "recent", "certain", "personal", "open",
            "nice", "wrong", "hard", "possible", "whole", "simple", "strong",
            "happy", "serious", "ready", "main", "major", "local", "low", "short",

            # Reddit/internet common words
            "yeah", "yep", "nope", "ok", "okay", "lol", "lmao", "tbh", "imo", "imho",
            "btw", "idk", "afaik", "iirc", "fwiw", "til", "eli5", "ama", "tldr",
            "literally", "actually", "basically", "honestly", "seriously", "really",
            "definitely", "probably", "maybe", "perhaps", "obviously", "clearly",
            "apparently", "supposedly", "allegedly", "essentially", "technically",

            # Opinion and sentiment words
            "agree", "disagree", "like", "dislike", "hate", "love", "enjoy", "prefer",
            "support", "oppose", "approve", "reject", "accept", "deny", "believe",
            "doubt", "trust", "suspect", "wonder", "guess", "assume", "hope", "wish",
            "amazing", "awesome", "terrible", "horrible", "fantastic", "wonderful",
            "awful", "excellent", "perfect", "stupid", "smart", "brilliant", "dumb",
            "ridiculous", "absurd", "reasonable", "fair", "unfair", "valid", "invalid",

            # Discourse markers
            "but", "however", "although", "though", "still", "yet", "also", "too",
            "even", "just", "only", "already", "never", "always", "sometimes",
            "often", "usually", "rarely", "seldom", "ever", "again", "once",
            "therefore", "thus", "hence", "so", "because", "since", "while",
            "whereas", "unless", "until", "before", "after", "if", "then",
            "first", "second", "third", "finally", "lastly", "meanwhile",
            "furthermore", "moreover", "besides", "instead", "otherwise",

            # Common nouns
            "people", "time", "year", "way", "day", "man", "thing", "woman", "life",
            "child", "world", "school", "state", "family", "student", "group",
            "country", "problem", "hand", "part", "place", "case", "week", "company",
            "system", "program", "question", "work", "government", "number", "night",
            "point", "home", "water", "room", "mother", "area", "money", "story",
            "fact", "month", "lot", "right", "study", "book", "eye", "job", "word",
            "business", "issue", "side", "kind", "head", "house", "service", "friend",
            "father", "power", "hour", "game", "line", "end", "member", "law", "car",
            "city", "community", "name", "president", "team", "minute", "idea",

            # Prepositions and conjunctions
            "with", "at", "by", "for", "from", "up", "about", "into", "over", "after",
            "beneath", "under", "above", "below", "between", "among", "through",
            "during", "without", "before", "behind", "beyond", "against", "within",
            "along", "following", "across", "around", "toward", "upon", "onto",
            "or", "nor", "as", "than", "whether", "either", "neither",

            # Common adverbs
            "not", "more", "now", "here", "there", "well", "back", "much", "then",
            "away", "down", "off", "out", "over", "almost", "enough", "together",
            "perhaps", "especially", "particularly", "simply", "certainly",
            "completely", "entirely", "exactly", "generally", "largely", "mainly",
            "mostly", "nearly", "partly", "primarily", "purely", "relatively",
            "roughly", "significantly", "simply", "solely", "specifically",
            "strongly", "totally", "typically", "ultimately", "virtually", "widely",

            # Informal/casual expressions
            "stuff", "things", "gonna", "wanna", "gotta", "kinda", "sorta", "dunno",
            "yeah", "nah", "yup", "meh", "wow", "whoa", "damn", "dude", "guy", "guys",
            "bro", "man", "buddy", "mate", "folks", "hey", "hi", "hello", "bye",
            "thanks", "thank", "please", "sorry", "excuse", "cool", "neat", "sick",
            "crazy", "insane", "wild", "weird", "odd", "strange", "funny", "hilarious",

            # Agreement/disagreement phrases
            "true", "false", "correct", "incorrect", "right", "wrong", "yes", "no",
            "exactly", "precisely", "indeed", "absolutely", "totally", "completely",
            "partially", "somewhat", "partly", "mostly", "largely", "entirely",
            "agreed", "disagreed", "confirmed", "denied", "verified", "disputed",

            # Hedging and certainty
            "might", "could", "would", "should", "must", "can", "will", "may",
            "seem", "appear", "look", "suggest", "indicate", "imply", "tend",
            "likely", "unlikely", "possible", "impossible", "certain", "uncertain",
            "definite", "indefinite", "sure", "unsure", "confident", "doubtful"
        ]

        prompt_text = Utilities.create_evaluation_prompt('')

        combined_text = f"{EXAMPLE_RESULT} {WHITESPACE_VARIATIONS} {' '.join(auxiliary_tokens)} {prompt_text}"

        tokens = tokenizer.tokenize(combined_text)
        unique_tokens = sorted(set(tokens))

        token_to_id = {}
        for token_text in unique_tokens:
            token_ids = tokenizer.convert_tokens_to_ids([token_text])
            if token_ids[0] != tokenizer.unk_token_id:
                token_to_id[token_text] = token_ids[0]

        print(
            f"Pre-computed {len(token_to_id)} unique tokens for vocabulary map.")
        return token_to_id

    @staticmethod
    def get_json_response_tokens(tokenizer) -> List[str]:
        json_response = EXAMPLE_RESULT + WHITESPACE_VARIATIONS
        tokens = tokenizer.tokenize(json_response)
        unique_tokens = sorted(set(tokens))
        return unique_tokens

    @staticmethod
    def extract_logits(logits, token_to_id_map) -> Tuple[List[str], List[float]]:
        token_names = []
        token_logits = []

        for token_text, token_id in token_to_id_map.items():
            token_logits.append(logits[token_id].item())
            token_names.append(token_text)

        return token_names, token_logits
