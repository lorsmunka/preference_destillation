import json
from time import time


class SentenceHandler:

    def __init__(self):
        self.sentences = []
        self.sentence_count = 0
        self.load_sentences()

    def load_sentences(self):
        start_time = time()
        print("Loading sentences...")
        sentences = []
        with open("./sentences/sentences.jsonl", "r", encoding="utf-8") as file:
            print("Loading sentences...")
            for line in file:
                data = json.loads(line.strip())
                sentences.append(data["text"])

        self.sentences = sentences
        self.sentence_count = len(sentences)

        elapsed_time = time() - start_time
        print(f"Loaded {len(sentences):,} -> took {elapsed_time:.2f} seconds.")

    def get_sentence(self, index):
        return self.sentences[index]
