"""
Vocabulary index for out-of-vocabulary (OOV) word stress prediction.

The module implements a nearest-neighbour lookup over a sorted list of
normalised word forms.  For an unknown word, it finds morphologically similar
entries and, if their suffix pattern is known, extrapolates the stress position.
"""
from __future__ import annotations

import bisect
import os
import sys

from typing import Optional
import pyarrow.parquet as pq

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _longest_common_prefix(left: str, right: str) -> int:
    """Return the length of the longest common prefix of *left* and *right*."""
    limit = min(len(left), len(right))
    for i in range(limit):
        if left[i] != right[i]:
            return i
    return limit


def _collect_neighbors(
    vocab: list,
    key: str,
    pos: int,
    min_prefix_len: int = 3,
) -> list[int]:
    """Collect indices of vocabulary entries near *pos* that share a long prefix with *key*.

    The algorithm starts with *pos* and greedily expands to adjacent entries
    while the common-prefix length does not drop below the maximum seen so far.
    This effectively finds a cluster of morphologically similar forms.
    """
    result = [pos]

    prefix_here = _longest_common_prefix(key, vocab[pos][0])
    max_prefix = prefix_here

    # Immediate neighbours
    left_prefix = 0
    if pos - 1 >= 0:
        left_prefix = _longest_common_prefix(key, vocab[pos - 1][0])
        max_prefix = max(max_prefix, left_prefix)
        result.append(pos - 1)

    right_prefix = 0
    if pos + 1 < len(vocab):
        right_prefix = _longest_common_prefix(key, vocab[pos + 1][0])
        max_prefix = max(max_prefix, right_prefix)
        result.append(pos + 1)

    if max_prefix < min_prefix_len:
        return result

    # Expand left while prefix length stays at the maximum
    if left_prefix >= max_prefix:
        i = pos - 2
        while i >= 0:
            if _longest_common_prefix(key, vocab[i][0]) < max_prefix:
                break
            result.append(i)
            i -= 1

    # Expand right while prefix length stays at the maximum
    if right_prefix >= max_prefix:
        i = pos + 2
        while i < len(vocab):
            if _longest_common_prefix(key, vocab[i][0]) < max_prefix:
                break
            result.append(i)
            i += 1

    return result


def _match_form_to_norm(
    form: str,
    vocab_entry: tuple[str, str],
    tail_index: set,
) -> Optional[tuple[int, tuple[str, str]]]:
    """Try to match *form* against a single vocabulary entry.

    Returns a ``(common_prefix_length, vocab_entry)`` pair if the suffix
    pattern of the divergence is present in *tail_index*, otherwise ``None``.
    """
    # print('vocab_entry', vocab_entry)
    norm_form, stress_positions = vocab_entry
    prefix_len = _longest_common_prefix(form, norm_form)
    tail_key = (form[prefix_len:], norm_form[prefix_len:])

    if tail_key in tail_index:
        return prefix_len, vocab_entry
    return None


# =============================================================================
# VOCABULARY
# =============================================================================

class OovVocabulary:
    """Nearest-neighbour vocabulary for extrapolating stress in unknown words.

    The underlying data is loaded from a two files that contain two items:

    1. ``acc_vocab`` — a list of ``(normalized_form, "stress_pos1,stress_pos2")``
       tuples sorted by ``normalized_form``.
    2. ``all_tails`` — a ``list`` of ``(word_suffix, norm_suffix)`` pairs that
       records which suffix divergences have been observed in the training data.

    Given an unknown word, the class finds the closest entries in the sorted
    list and, if the point where the word diverges from the normalised form
    is a known tail pattern, returns the stress position from that entry.
    """

    def __init__(self, data_path: str) -> None:
        """Load the pickled vocabulary from *data_path* / ``unk_vocab.pickle``.

        Args:
            data_path: Directory that contains ``unk_vocab.pickle``.
        """

        vocab_file = os.path.join(data_path, "oov_tails.pq")


        table = pq.read_table(vocab_file)
    
        word_suffix = table.column(0).to_pylist()
        norm_suffix = table.column(1).to_pylist()
        #tail_list1 = list(zip(word_suffix, norm_suffix))
        
        self._tail_index = set(zip(word_suffix, norm_suffix))

        #self._tail_index = set(zip(word_suffix, norm_suffix))
        #print('tail_list', word_suffix[:5], norm_suffix[:5])
        vocab_file = os.path.join(data_path, "oov_forms.pq")

        table = pq.read_table(vocab_file)
    
        normalized_form = table.column(0).to_pylist()
        stress_pos_list = table.column(1).to_pylist()
        
        #zzz = list(zip(normalized_form, stress_pos_list))
        self._vocab = list(zip(normalized_form, stress_pos_list))

        '''
        vocab_file = os.path.join(data_path, "oov_tails.csv")
        tail_list = []
        errors = 0
        with open(vocab_file, "r", encoding="windows-1251") as finp:
            
            for line in finp:
                parts = line.strip().split('|')
                if len(parts) != 2:
                    errors += 1
                    continue

                word_suffix, norm_suffix = parts
                tail_list.append((word_suffix, norm_suffix))
            
        self._tail_index = set(tail_list)
        print('diff', len(tail_list), len(tail_list1))
        
        zzz = list(zip(tail_list, tail_list1))
        print('tail_list, tail_list1', zzz[:10])

        diff = []
        for x, y in zzz:
            if x != y:
                print('diff', x, y)
              
        vocab_file = os.path.join(data_path, "oov_forms.csv")
        self._vocab = []

        with open(vocab_file, "r", encoding="windows-1251") as finp:
            
            for line in finp:
                parts = line.strip().split('|')
                if len(parts) != 2:
                    errors += 1
                    continue

                normalized_form, stress_pos_list = parts
                self._vocab.append((normalized_form, stress_pos_list))
            
        # self._vocab = []
        if errors > 0:
            print("OovVocabulary load errors", errors, file=sys.stderr)
            
        print('diff1', len(self._vocab), len(self._vocab1))
      
        zzz = list(zip(self._vocab, self._vocab1))
        print('_vocab, _vocab1', zzz[:10])

        diff = []
        for x, y in zzz:
            if x != y:
                print('diff1', x, y)
        '''

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _search_neighbour_indices(self, text: str) -> list[int]:
        """Return indices of vocabulary entries near *text* in the sorted list."""
        if not self._vocab:
            return []

        # bisect_left needs a comparable key; we use (text, "") because
        # the vocabulary entries are (form, stress_positions) tuples.
        key = (text, "")
        if key < self._vocab[0]:
            return _collect_neighbors(self._vocab, text, 0)

        try_pos = bisect.bisect_left(self._vocab, key)
        return _collect_neighbors(self._vocab, text, try_pos)

    def _find_matches(self, text: str) -> list[tuple[int, tuple[str, str]]]:
        """Find all vocabulary entries whose tail pattern matches *text*."""
        matches: list[tuple[int, tuple[str, str]]] = []
        for idx in self._search_neighbour_indices(text):
            result = _match_form_to_norm(text, self._vocab[idx], self._tail_index)
            if result is not None:
                matches.append(result)
        return matches

    def predict_stress(self, text: str) -> int:
        """Predict the stress position for an unknown word.

        The method looks for the most similar vocabulary entry whose suffix
        divergence pattern is known, and returns the stress position recorded
        for that entry.

        Args:
            text: The unknown word (should already be normalised, e.g.
                lower-cased and with ё → е).

        Returns:
            The predicted 0-based stress position, or ``-1`` if no suitable
            neighbour was found.
        """
        matches = self._find_matches(text)
        if not matches:
            return -1

        # Pick the entry with the longest common prefix (most similar form)
        matches.sort()
        prefix_len, vocab_entry = matches[-1]

        stress_positions = [int(p) for p in vocab_entry[1].split(",")]
        if not stress_positions:
            return -1

        # The last recorded position is the primary stress
        acc_pos = stress_positions[-1]
        if acc_pos < 0 or acc_pos >= prefix_len:
            return -1
        return acc_pos


__all__ = [
    "OovVocabulary",
]
