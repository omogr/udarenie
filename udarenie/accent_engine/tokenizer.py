"""
BERT tokenizer wrapper for accentuation.
"""
from __future__ import annotations

from typing import Optional
from pathlib import Path

from .bert_tokenizer import BertTokenizer

from .core import (
    RUSSIAN_VOWELS_LOWER,
    MAX_SENTENCE_LEN,
    BERT_BOS_ID,
    BERT_SEP_ID,
    BERT_PAD_ID,
    ModelLoadError,
)


class AccentTokenizer:
    """
    Wraps BertTokenizer and provides vowel position mapping for tokens.
    """

    def __init__(self, vocab_path: Path):
        if not vocab_path.exists():
            raise ModelLoadError(f"Vocab file not found: {vocab_path}")

        self.tokenizer = BertTokenizer(str(vocab_path), do_lower_case=False)
        self.pad_token_id = BERT_PAD_ID
        self._token_vowel_pos: dict[int, tuple[int, list[int]]] = {}
        self._build_vowel_cache()

    def _build_vowel_cache(self) -> None:
        """Build cache of vowel positions for each token."""
        for token_text, token_id in self.tokenizer.vocab.items():
            clean = token_text[2:] if token_text.startswith('##') else token_text
            vowel_pos = [i for i, ch in enumerate(clean) if ch in RUSSIAN_VOWELS_LOWER]
            self._token_vowel_pos[token_id] = (len(clean), vowel_pos)

    def encode(self, text: str) -> list[int]:
        """Tokenize text to token IDs."""
        return self.tokenizer.encode(text) # , add_special_tokens=False)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text to tokens."""
        return self.tokenizer.tokenize(text)

    def get_token_vowel_info(self, token_id: int) -> Optional[tuple[int, list[int]]]:
        """Get (token_length, vowel_positions) for a token ID."""
        return self._token_vowel_pos.get(token_id)
