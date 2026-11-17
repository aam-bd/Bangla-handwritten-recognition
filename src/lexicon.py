"""
Bangla Lexicon and Spell-Correction Engine for Whole-Word Reconstruction.
Enables matching segmented characters against valid vocabulary words like 'মানুষ', 'জল'.
"""

from typing import List, Tuple, Set, Optional


# Curated high-frequency Bangla vocabulary
DEFAULT_VOCABULARY = [
    # Explicit assignment target words
    "মানুষ", "জল",
    # Common everyday nouns, nature, places
    "বাংলাদেশ", "ঢাকা", "চট্টগ্রাম", "সিলেট", "খুলনা", "রাজশাহী", "বরিশাল", "রংপুর", "ময়মনসিংহ",
    "বই", "কলম", "খাতা", "ঘর", "বাড়ি", "নদী", "গাছ", "ফুল", "ফল", "পাখি", "আকাশ", "বাতাস",
    "আলো", "সূর্য", "চাঁদ", "তারা", "মাটি", "মেঘ", "বৃষ্টি", "শহর", "গ্রাম", "দেশ", "ভাষা",
    "জীবন", "সময়", "কাজ", "পথ", "মন", "চোখ", "হাত", "মাথা", "গান", "সুর", "কথা", "কাব্য",
    "কবিতা", "খবর", "নাম", "দিন", "রাত", "সকাল", "সন্ধ্যা", "বিদ্যালয়", "ছাত্র", "শিক্ষক",
    "বন্ধু", "ভাই", "বোন", "মা", "বাবা", "দাদা", "দাদি", "নানা", "নানি", "শিশু", "সন্তান",
    # Common verbs & adjectives
    "ভালো", "সুন্দর", "বড়", "ছোট", "নতুন", "পুরাতন", "সহজ", "কঠিন", "সত্য", "শান্তি",
    "খুশি", "আনন্দ", "সাহস", "শক্তি", "আশা", "স্বপ্ন", "প্রেম", "ভালোবাসা", "স্নেহ",
    "পড়া", "লেখা", "দেখা", "বলা", "শোনা", "চলা", "যাওয়া", "আসা", "করা", "খাওয়া",
    # Compound letter vocabulary
    "বিজ্ঞান", "শিক্ষা", "জ্ঞান", "বন্ধুত্ব", "শক্তি", "মুক্তি", "শান্তি", "বৃষ্টি", "সৃষ্টি",
    "স্পষ্ট", "প্রশ্ন", "পরীক্ষা", "অঙ্ক", "সঙ্গ", "রক্ত", "যুক্ত", "ভক্তি", "সন্ধ্যা"
]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate character-level edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class BanglaLexicon:
    """Dictionary and fuzzy search engine for valid Bangla words."""

    def __init__(self, vocabulary: Optional[List[str]] = None):
        self.words: Set[str] = set(vocabulary or DEFAULT_VOCABULARY)

    def add_word(self, word: str):
        if word:
            self.words.add(word.strip())

    def find_closest_word(
        self,
        query: str,
        max_distance: int = 3
    ) -> Tuple[str, float]:
        """
        Find closest vocabulary word for a reconstructed character sequence.
        Returns:
            (best_word, similarity_score_in_0_to_1)
        """
        query = query.strip()
        if not query:
            return "", 0.0

        if query in self.words:
            return query, 1.0

        best_word = query
        min_dist = 999

        for word in self.words:
            # Quick length filter
            if abs(len(word) - len(query)) > max_distance:
                continue

            dist = levenshtein_distance(query, word)
            if dist < min_dist:
                min_dist = dist
                best_word = word

        if min_dist <= max_distance:
            max_len = max(len(query), len(best_word))
            similarity = 1.0 - (min_dist / max(max_len, 1))
            return best_word, max(0.0, float(similarity))

        # Fallback to query itself if no close dictionary word exists
        return query, 0.40
