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

        auxiliary_tokens = ["the", "a", "is", "of", "and", "to", "in", "that", "it", "you",
                            "very", "quite", "somewhat", "extremely", "slightly", "moderately"]

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
