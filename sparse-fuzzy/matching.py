"""Exact inside-document character matching with alternative candidate indexes."""

from collections import defaultdict
from functools import lru_cache


@lru_cache(maxsize=8192)
def signature(grams):
    """Collisions allow extra matches. Only a disjoint signature rules out a pair."""
    bits = 0
    for gram in grams:
        bits |= 1 << (hash(gram) & 1023)
    return bits


class CharacterMatcher:
    """The document lifetime bounds the word, index and pair caches."""

    def __init__(self, document, gram_function, implementation, collect=False):
        self.document = document
        self.grams = gram_function
        self.implementation = implementation
        self.postings = {}
        self.words_by_line = {}
        self.similarities = {}
        self.counts = dict(queries=0, exhaustive_pairs=0, candidates=0, pair_evaluations=0,
                           length_skips=0, cache_hits=0, indexes=0) if collect else None

    def _words(self, line):
        if line not in self.words_by_line:
            words = set(self.document.sentences[line])
            eligible = tuple(sorted(word for word in words if len(word) >= 4))
            self.words_by_line[line] = words, eligible
        return self.words_by_line[line]

    def _index(self, line, words):
        if line not in self.postings:
            if self.implementation=="sparse":
                index = defaultdict(set)
                for word in words:
                    for gram in self.grams(word):
                        index[gram].add(word)
            elif self.implementation=="bitmask":
                index = {}
                for position, word in enumerate(words):
                    mask = 1 << position
                    for gram in self.grams(word):
                        index[gram] = index.get(gram, 0) | mask
            else:
                index = [(word, signature(frozenset(self.grams(word)))) for word in words]
            if self.counts is not None:
                self.counts["indexes"] += 1
            self.postings[line] = index
        return self.postings[line]

    def _candidates(self, grams, eligible, index):
        if index is None:
            return eligible
        if self.implementation=="sparse":
            return set().union(*(index.get(gram, ()) for gram in grams))
        if self.implementation=="bitmask":
            mask = 0
            for gram in grams:
                mask |= index.get(gram, 0)
            found = []
            while mask:
                low_bit = mask & -mask
                found.append(eligible[low_bit.bit_length()-1])
                mask ^= low_bit
            return found
        fingerprint = signature(frozenset(grams))
        return [word for word, target in index if fingerprint & target]

    def _max_similarity(self, source, grams, candidates):
        best = 0.0
        count = len(grams)
        if self.counts is not None:
            self.counts["candidates"] += len(candidates)
        for target in candidates:
            key = source, target
            score = self.similarities.get(key)
            if score is not None:
                if self.counts is not None:
                    self.counts["cache_hits"] += 1
            else:
                other = self.grams(target)
                if min(count, len(other)) <= best * max(count, len(other)) - 1e-12:
                    if self.counts is not None:
                        self.counts["length_skips"] += 1
                    continue
                common = len(grams & other)
                score = common / max(1, count + len(other) - common)
                self.similarities[key] = score
                if self.counts is not None:
                    self.counts["pair_evaluations"] += 1
            best = max(best, score)
            if best == 1:
                break
        return best

    def score(self, atom, top3):
        term_grams = {term: self.grams(term) for term in atom.weighted if len(term) >= 4}
        denominator = max(1e-12, sum(atom.weighted[term] for term in term_grams))
        hard = []
        soft = []
        if self.counts is not None:
            self.counts["queries"] += 1
        for line in atom.chosen:
            words, eligible = self._words(line)
            absent = [word for word in term_grams if word not in words]
            if self.counts is not None:
                self.counts["exhaustive_pairs"] += len(absent) * len(eligible)
            index = self._index(line, eligible) if self.implementation!="dense" and absent else None
            similarities = {
                source:self._max_similarity(source, term_grams[source],
                                            self._candidates(term_grams[source], eligible, index))
                for source in absent
            }
            hard.append(sum(atom.weighted[t] * (1.0 if t in words else similarities[t] >= .65)
                            for t in term_grams) / denominator)
            soft.append(sum(atom.weighted[t] * (1.0 if t in words else similarities[t])
                            for t in term_grams) / denominator)
        return (*top3(hard), *top3(soft))
