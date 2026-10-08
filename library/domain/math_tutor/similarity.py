"""How alike two word problems are, after the parts that do not change the problem are
taken out: numbers become N, names become NAME, case and punctuation are dropped.

Two measures on what is left:
- word-sequence ratio (difflib): the share of words that match in the same order. One or
  two words swapped in a 17-word problem gives 0.83-0.94; the same words reordered ~0.4.
  This is the near-duplicate test: "80% the same, in order".
- word-trigram Jaccard: shared three-word runs. Used as an extra, stricter test-leak check.
"""

import difflib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Two problems whose word-sequence ratio is above this are near-duplicates (pruning, seeds,
# rewording self-check all use it).
NEAR_DUPLICATE = 0.8

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
        elif word.endswith("'s") and word[:-2] in name_words:
            normalised.append("NAME's")   # "Mia's" and "Tom's" are the same word
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
        self.order_by_id: Dict[str, int] = {}

    def add(self, item_id: str, trigrams: Set[Tuple[str, ...]]) -> None:
        self.trigrams_by_id[item_id] = trigrams
        self.order_by_id[item_id] = len(self.order_by_id)
        for trigram in trigrams:
            self.ids_by_trigram.setdefault(trigram, []).append(item_id)

    def most_similar(self, trigrams: Set[Tuple[str, ...]], exclude: Optional[str] = None) -> Tuple[Optional[str], float]:
        """The stored problem closest to these trigrams and its Jaccard; ties go to the one
        stored first, so results do not depend on set iteration order."""
        shared_counts = Counter()
        for trigram in trigrams:
            shared_counts.update(self.ids_by_trigram.get(trigram, ()))
        best_id, best_score = None, 0.0
        for candidate_id, shared in shared_counts.items():
            if candidate_id == exclude:
                continue
            candidate = self.trigrams_by_id[candidate_id]
            score = shared / (len(trigrams) + len(candidate) - shared)
            if score > best_score or (score == best_score and best_id is not None
                                      and self.order_by_id[candidate_id] < self.order_by_id[best_id]):
                best_id, best_score = candidate_id, score
        return best_id, best_score


def sequence_ratio(first: List[str], second: List[str]) -> float:
    return difflib.SequenceMatcher(None, first, second, autojunk=False).ratio()


class WordSequenceIndex:
    """Finds the stored problem whose word sequence is most like a new one, comparing only
    problems that could reach `threshold`. A sequence ratio of r needs a shared word share
    (multiset Jaccard) of at least r / (2 - r), and two word sets that share that much must
    share one of their rarest words (prefix filtering, Bayardo et al. 2007). So each
    problem is filed under its few rarest words only, and a common run like "how many N"
    never makes everything a candidate. Word rarity is counted once over all `vocabulary`."""

    def __init__(self, threshold: float, vocabulary: Iterable[List[str]]):
        self.threshold = threshold
        self.word_share = threshold / (2 - threshold)
        self.frequency = Counter(token for words in vocabulary for token in self.tokens_unsorted(words))
        self.words_by_id: Dict[str, List[str]] = {}
        self.order_by_id: Dict[str, int] = {}
        self.ids_by_token: Dict[Tuple[str, int], List[str]] = {}

    @staticmethod
    def tokens_unsorted(words: List[str]) -> List[Tuple[str, int]]:
        seen = Counter()
        tokens = []
        for word in words:
            tokens.append((word, seen[word]))
            seen[word] += 1
        return tokens

    def prefix(self, words: List[str]) -> List[Tuple[str, int]]:
        tokens = sorted(self.tokens_unsorted(words), key=lambda token: (self.frequency[token], token))
        length = len(tokens) - math.ceil(self.word_share * len(tokens)) + 1
        return tokens[:max(length, 1)]

    def add(self, item_id: str, words: List[str]) -> None:
        self.words_by_id[item_id] = words
        self.order_by_id[item_id] = len(self.order_by_id)
        for token in self.prefix(words):
            self.ids_by_token.setdefault(token, []).append(item_id)

    def most_similar(self, words: List[str], exclude: Optional[str] = None) -> Tuple[Optional[str], float]:
        """The closest stored problem among those that could reach the threshold, and its
        ratio (0.0 and None when no stored problem could). Ties go to the one stored first."""
        candidates = {candidate for token in self.prefix(words) for candidate in self.ids_by_token.get(token, ())}
        candidates.discard(exclude)
        best_id, best_score = None, 0.0
        for candidate in sorted(candidates, key=self.order_by_id.get):
            other = self.words_by_id[candidate]
            if min(len(words), len(other)) * 2 / (len(words) + len(other)) < self.threshold:
                continue  # too different in length to reach the threshold
            score = sequence_ratio(words, other)
            if score > best_score:
                best_id, best_score = candidate, score
        return best_id, best_score
