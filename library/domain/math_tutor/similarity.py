"""How alike two word problems are, after the parts that do not change the problem are
taken out: numbers become N, names become NAME, case and punctuation are dropped. What is
left is compared as sets of word trigrams (Jaccard). The same words in a new order share
few trigrams, so a reworded problem survives; a copy with new numbers or a new name does not.
"""

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "dozen",
}

# Capitalised words that are not names, even when they appear mid-sentence.
NOT_NAMES = {
    "I", "Mr", "Mrs", "Ms", "Dr", "Miss", "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December", "Christmas",
    "Halloween", "Easter", "Thanksgiving", "Valentine", "English", "American", "Spanish",
}


def collect_name_words(texts: Iterable[str]) -> Set[str]:
    """Words that appear capitalised in the middle of a sentence somewhere in the
    collection: names, as far as a word list can tell. Collected over the whole
    collection, so a name that only starts sentences in one problem is still caught."""
    names = set()
    for text in texts:
        for sentence in re.split(r"[.?!]\s+", text):
            words = re.findall(r"[A-Za-z]+", sentence)
            for word in words[1:]:
                if word[0].isupper() and word not in NOT_NAMES:
                    names.add(word)
    return names


def normalise(text: str, name_words: Set[str]) -> List[str]:
    """Problem text -> the word list that similarity is measured on."""
    words = re.findall(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*", text)
    normalised = []
    for word in words:
        if word[0].isdigit() or word.lower() in NUMBER_WORDS:
            if normalised[-1:] != ["N"]:  # "seventy-eight" and "78" are both one N
                normalised.append("N")
        elif word in name_words:
            normalised.append("NAME")
        else:
            normalised.append(word.lower())
    return normalised


def word_trigrams(words: List[str]) -> Set[Tuple[str, ...]]:
    if len(words) < 3:
        return {tuple(words)}
    return {tuple(words[index:index + 3]) for index in range(len(words) - 2)}


def jaccard(first: Set, second: Set) -> float:
    if not first and not second:
        return 1.0
    shared = len(first & second)
    return shared / (len(first) + len(second) - shared)


class TrigramIndex:
    """Finds the most similar stored problem without comparing against every one: only
    problems that share at least one trigram are candidates, then the exact Jaccard."""

    def __init__(self):
        self.trigrams_by_id: Dict[str, Set[Tuple[str, ...]]] = {}
        self.ids_by_trigram: Dict[Tuple[str, ...], List[str]] = {}

    def add(self, item_id: str, trigrams: Set[Tuple[str, ...]]) -> None:
        self.trigrams_by_id[item_id] = trigrams
        for trigram in trigrams:
            self.ids_by_trigram.setdefault(trigram, []).append(item_id)

    def most_similar(self, trigrams: Set[Tuple[str, ...]]) -> Tuple[Optional[str], float]:
        shared_counts = Counter()
        for trigram in trigrams:
            shared_counts.update(self.ids_by_trigram.get(trigram, ()))
        best_id, best_score = None, 0.0
        for candidate_id, shared in shared_counts.items():
            candidate = self.trigrams_by_id[candidate_id]
            score = shared / (len(trigrams) + len(candidate) - shared)
            if score > best_score:
                best_id, best_score = candidate_id, score
        return best_id, best_score
