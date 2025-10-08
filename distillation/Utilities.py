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
