"""
Bangla Lexicon and Spell-Correction Engine with Consonant Skeleton Matching.
Enables robust whole-word reconstruction (e.g. 'মানুষ', 'জল') from isolated OCR predictions.
"""

import re
from typing import List, Tuple, Set, Optional, Dict


# Bengali vowel diacritics (Kar signs) to strip when generating consonant skeletons
BENGALI_KAR_SIGNS = set([
    '\u09be',  # া akar
    '\u09bf',  # ি hrossho-ikar
    '\u09c0',  # ী dirgho-ikar
    '\u09c1',  # ু hrossho-ukar
    '\u09c2',  # ূ dirgho-ukar
    '\u09c3',  # ৃ ri-kar
    '\u09c4',  # ৄ
    '\u09c7',  # ে e-kar
    '\u09c8',  # ৈ oi-kar
    '\u09cb',  # ো o-kar
    '\u09cc',  # ৌ ou-kar
    '\u09cd',  # ্ virama / hasant
    '\u09d7',  # ৗ length mark
])


def get_consonant_skeleton(word: str) -> str:
    """
    Extract the consonant/vowel skeleton by stripping vowel diacritics (kars).
    Examples:
        'মানুষ' -> 'মনষ'
        'জল' -> 'জল'
        'বাংলাদেশ' -> 'বংলদশ'
        'ঢাকা' -> 'ঢক'
    """
    chars = [c for c in word if c not in BENGALI_KAR_SIGNS]
    return "".join(chars)


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
    """
    Dictionary and fuzzy search engine with Consonant Skeleton Hashing.
    Enables recognizing words like 'মানুষ' when segmented as 'মনষ' or 'ম১নষ'.
    """

    def __init__(self, vocabulary: Optional[List[str]] = None):
        self.words: Set[str] = set(vocabulary or DEFAULT_VOCABULARY)
        self.skeleton_map: Dict[str, List[str]] = {}
        self._rebuild_skeleton_index()

    def _rebuild_skeleton_index(self):
        """Index all words by their consonant skeleton."""
        self.skeleton_map.clear()
        for word in self.words:
            skel = get_consonant_skeleton(word)
            if skel:
                self.skeleton_map.setdefault(skel, []).append(word)

    def add_word(self, word: str):
        if word:
            w = word.strip()
            self.words.add(w)
            skel = get_consonant_skeleton(w)
            if skel:
                self.skeleton_map.setdefault(skel, []).append(w)

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

        # 1. Exact full-word match
        if query in self.words:
            return query, 1.0

        # 2. Consonant skeleton match (solves Kar omission in isolated character OCR)
        query_skel = get_consonant_skeleton(query)
        if query_skel in self.skeleton_map:
            candidates = self.skeleton_map[query_skel]
            # Select candidate with minimal edit distance
            best_cand = min(candidates, key=lambda w: levenshtein_distance(query, w))
            return best_cand, 0.95

        # 3. Digit-stripped consonant skeleton match (e.g. 'ম১নষ' where akar is misread as '১')
        query_no_digits = re.sub(r'[\d০-৯]', '', query)
        if query_no_digits:
            skel_no_digits = get_consonant_skeleton(query_no_digits)
            if skel_no_digits in self.skeleton_map:
                candidates = self.skeleton_map[skel_no_digits]
                best_cand = min(candidates, key=lambda w: levenshtein_distance(query_no_digits, w))
                return best_cand, 0.90

        # 4. Standard Levenshtein distance fallback
        best_word = query
        min_dist = 999

        for word in self.words:
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

        # Fallback to query itself
        return query, 0.40
