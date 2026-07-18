"""
Batching and encoding utilities for out-of-vocabulary (OOV) word stress prediction.

This module handles character-level tokenization of Russian words and groups them
into padded batches suitable for inference with a small BERT-like model.
"""
from __future__ import annotations

import enum
from typing import NamedTuple

import torch


# =============================================================================
# CONSTANTS
# =============================================================================

class SpecialToken(enum.IntEnum):
    """Special token IDs for the character-level vocabulary.

    The vocabulary reserves IDs 0–2 for structural tokens.  Real characters
    are mapped to ``ALPHABET_OFFSET + alphabet_index``.
    """
    PAD = 0   # padding / unknown character fallback
    BOS = 1   # beginning of sequence
    EOS = 2   # end of sequence
    ALPHABET_OFFSET = 3  # first real character maps to this + alphabet index


RUSSIAN_ALPHABET: str = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
MAX_BATCH_TOKENS: int = 6000  # heuristic: ~250 chars × 24 words


# =============================================================================
# DATA TYPES
# =============================================================================

class EncodedWord(NamedTuple):
    """A word encoded as token IDs alongside its original text."""
    token_ids: list[int]
    original: str


class Batch(NamedTuple):
    """A padded batch ready for model inference.

    Attributes:
        input_ids:       (batch_size, seq_len)  token IDs
        attention_mask:  (batch_size, seq_len)  1 for real tokens, 0 for padding
        texts:           original texts, aligned with the batch dimension
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    texts: list[str]


# =============================================================================
# ENCODING
# =============================================================================

def encode_word(word: str) -> tuple[list[int], int]:
    """Encode a single word into character-level token IDs.

    The function strips any legacy ``'+'`` stress-mark characters, lower-cases
    the word, and maps each remaining character to its position in
    :data:`RUSSIAN_ALPHABET`.  Characters not present in the alphabet are
    mapped to :attr:`SpecialToken.PAD` and counted as errors.

    The resulting sequence is wrapped with :attr:`SpecialToken.BOS` and
    :attr:`SpecialToken.EOS`.

    Args:
        word: Input word. May contain ``'+'`` stress marks which are removed.

    Returns:
        ``(token_ids, error_count)``.  ``error_count > 0`` means the word
        contained characters outside the Russian alphabet.
    """
    token_ids: list[int] = [SpecialToken.BOS]
    error_count = 0

    for ch in word.casefold():
        if ch == "+":
            continue
        idx = RUSSIAN_ALPHABET.find(ch)
        if idx < 0:
            error_count += 1
            token_ids.append(SpecialToken.PAD)
        else:
            token_ids.append(SpecialToken.ALPHABET_OFFSET + idx)

    token_ids.append(SpecialToken.EOS)
    return token_ids, error_count


# =============================================================================
# BATCHER
# =============================================================================

class OovBatcher:
    """Iterate over a list of words, producing padded batches for inference.

    Each batch is sized so that ``max_sequence_length × batch_size`` does not
    exceed :data:`MAX_BATCH_TOKENS`.  This keeps GPU memory usage predictable
    regardless of input length.

    Usage::

        batcher = OovBatcher(["квазисублимирующие", "некомпетентность"])
        for batch in batcher:
            logits = model(batch.input_ids, batch.attention_mask)
            ...
    """

    def __init__(self, words: list[str]) -> None:
        """Build the internal list of encoded words.

        Words that contain characters outside the Russian alphabet are silently
        dropped (they cannot be processed by the model anyway).

        Args:
            words: List of raw words to batch.

        Raises:
            ValueError: If *no* word survives filtering.
        """
        self._entries: list[EncodedWord] = []
        for word in words:
            token_ids, err_cnt = encode_word(word)
            if err_cnt == 0:
                self._entries.append(EncodedWord(token_ids, word))

        if not self._entries:
            raise ValueError("No valid words to batch (all contained non-Russian characters).")

        self._pos: int = -1
        self._epoch: int = 0

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def _advance(self) -> None:
        """Move the internal cursor forward, wrapping to the next epoch."""
        self._pos += 1
        if self._pos >= len(self._entries):
            self._epoch += 1
            self._pos = 0

    def _has_more(self, single_pass: bool) -> bool:
        """Return whether there are still items to yield."""
        if not single_pass:
            return True
        return self._epoch == 0 and (self._pos + 1) < len(self._entries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __iter__(self) -> OovBatcher:
        return self

    def __next__(self) -> Batch:
        """Build and return the next padded batch.

        Raises:
            StopIteration: When all entries have been consumed (in single-pass
                mode) or the batcher is exhausted.
        """
        if not self._has_more(single_pass=True):
            raise StopIteration

        max_length = 1
        batch_entries: list[EncodedWord] = []

        while True:
            self._advance()
            if self._epoch > 0:
                break

            entry = self._entries[self._pos]
            candidate_len = max(max_length, len(entry.token_ids))
            projected_tokens = candidate_len * (1 + len(batch_entries))
            if projected_tokens > MAX_BATCH_TOKENS and batch_entries:
                # Roll back so the word can start the next batch
                self._pos -= 1
                if self._pos < 0:
                    self._pos = len(self._entries) - 1
                    self._epoch -= 1
                break

            max_length = candidate_len
            batch_entries.append(entry)

        if not batch_entries:
            raise StopIteration

        return self._pad_batch(batch_entries, max_length)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_batch(entries: list[EncodedWord], max_length: int) -> Batch:
        """Pad a list of :class:`EncodedWord` to a uniform :class:`Batch`."""
        all_input_ids: list[torch.Tensor] = []
        all_attention_mask: list[torch.Tensor] = []
        texts: list[str] = []

        for entry in entries:
            ids = entry.token_ids
            length = len(ids)
            attention = [1] * length

            if length < max_length:
                pad = [SpecialToken.PAD] * (max_length - length)
                ids = ids + pad
                attention = attention + [0] * (max_length - length)

            all_input_ids.append(torch.tensor(ids, dtype=torch.long))
            all_attention_mask.append(torch.tensor(attention, dtype=torch.long))
            texts.append(entry.original)

        return Batch(
            input_ids=torch.stack(all_input_ids),
            attention_mask=torch.stack(all_attention_mask),
            texts=texts,
        )


__all__ = [
    "SpecialToken",
    "RUSSIAN_ALPHABET",
    "MAX_BATCH_TOKENS",
    "EncodedWord",
    "Batch",
    "encode_word",
    "OovBatcher",
]
