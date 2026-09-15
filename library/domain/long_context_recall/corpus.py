"""Long-context recall corpus — a page of facts about Virág (and distractor people), then
one question whose answer is a single fact from the page. Every record is generated, so
the true answer is known; it is stored beside the text as {"text": ..., "answer": ...}.

    python -m library.domain.long_context_recall.corpus        # writes corpus.jsonl here

The point of the domain: the answer is always in the context, so the task is retrieval
over a long input rather than knowledge — a test of whether a tiny student's attention
can find the right sentence.
"""

import json
import random
from typing import Dict, List, Optional, Tuple

SUBJECT = "Virág"
DISTRACTOR_NAMES = ["Bence", "Dóra", "Levente", "Zsófi", "Márk", "Eszter", "Áron", "Lilla", "Gergő", "Hanna"]

# attribute -> (sentence template, question, answer choices)
FACTS: Dict[str, Tuple[str, str, List[str]]] = {
    "city": ("{name} lives in {value}.", "Where does {name} live?",
             ["Budapest", "Debrecen", "Szeged", "Pécs", "Győr", "Miskolc", "Sopron", "Eger", "Veszprém", "Kecskemét"]),
    "pet": ("{name} has a {value}.", "What pet does {name} have?",
            ["cat", "dog", "rabbit", "parrot", "hamster", "turtle", "goldfish", "horse"]),
    "food": ("{name} likes to eat {value}.", "What does {name} like to eat?",
             ["bread", "soup", "pasta", "rice", "cheese", "apples", "chicken", "pancakes", "salad", "fish"]),
    "job": ("{name} works as a {value}.", "What is {name}'s job?",
            ["teacher", "doctor", "baker", "engineer", "nurse", "farmer", "pilot", "painter", "chef", "lawyer"]),
    "color": ("{name}'s favorite color is {value}.", "What is {name}'s favorite color?",
              ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown"]),
    "sport": ("{name} plays {value} on weekends.", "What sport does {name} play?",
              ["football", "tennis", "chess", "basketball", "volleyball", "hockey", "golf", "swimming"]),
    "instrument": ("{name} plays the {value}.", "What instrument does {name} play?",
                   ["piano", "guitar", "violin", "drums", "flute", "cello", "trumpet"]),
    "vehicle": ("{name} drives a {value}.", "What does {name} drive?",
                ["bicycle", "car", "scooter", "van", "truck", "motorcycle", "bus"]),
    "month": ("{name} was born in {value}.", "In which month was {name} born?",
              ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]),
    "language": ("{name} speaks {value}.", "Which language does {name} speak?",
                 ["English", "German", "French", "Spanish", "Italian", "Polish", "Czech", "Dutch"]),
}

ALL_ANSWERS = sorted({value for _, _, values in FACTS.values() for value in values})


def build_page(rng: random.Random, subject_fact_count: int, distractor_count: int):
    """Sentences about the subject plus sentences about other people, shuffled together.
    Returns (page text, {attribute: value} for the subject)."""
    subject_attributes = rng.sample(list(FACTS), subject_fact_count)
    subject_values = {attribute: rng.choice(FACTS[attribute][2]) for attribute in subject_attributes}
    sentences = [FACTS[attribute][0].format(name=SUBJECT, value=value)
                 for attribute, value in subject_values.items()]

    distractors = rng.sample(DISTRACTOR_NAMES, distractor_count)
    for name in distractors:
        for attribute in rng.sample(list(FACTS), rng.randint(2, 4)):
            sentences.append(FACTS[attribute][0].format(name=name, value=rng.choice(FACTS[attribute][2])))

    rng.shuffle(sentences)
    return " ".join(sentences), subject_values


def make_record(rng: random.Random) -> Dict:
    page, subject_values = build_page(rng, subject_fact_count=rng.randint(4, 7),
                                      distractor_count=rng.randint(6, 10))
    attribute = rng.choice(list(subject_values))
    question = FACTS[attribute][1].format(name=SUBJECT)
    return {"text": f"{page}\n\nQuestion: {question}", "answer": subject_values[attribute],
            "attribute": attribute}


def build_corpus(path: str, count: Optional[int] = None, seed: int = 0) -> None:
    count = count or 10000
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as file:
        for _ in range(count):
            file.write(json.dumps(make_record(rng), ensure_ascii=False) + "\n")
    print(f"Wrote {count} long-context recall records -> {path}")


if __name__ == "__main__":
    import os
    build_corpus(os.path.join(os.path.dirname(__file__), "corpus.jsonl"))
