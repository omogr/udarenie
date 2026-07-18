"""
OOV stress predictor using a small character-level BERT model.

The model operates on individual Russian words encoded as sequences of
character IDs.  It is intended for words that are not present in the main
accentuation dictionary.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from .bert import BertForTokenClassification
from .oov_reader import OovBatcher, SpecialToken
from .oov_vocab import OovVocabulary


# =============================================================================
# UTILITIES
# =============================================================================

def _find_yo_stress(word: str) -> int:
    """Return the position of ``ё``/``Ё`` in *word*, or ``-1`` if absent.

    In Russian, the letter ``ё`` is always stressed, so its presence
    immediately determines the stress position without needing a model.
    """
    for case in ("ё", "Ё"):
        pos = word.find(case)
        if pos >= 0:
            return pos
    return -1


# =============================================================================
# MODEL
# =============================================================================

class OovStressPredictor:
    """Predict stress position for out-of-vocabulary Russian words.

    The predictor works in two stages:

    1. **Vocabulary lookup** — tries to extrapolate stress from morphologically
       similar known words using :class:`OovVocabulary`.
    2. **Neural fallback** — if the vocabulary lookup fails, runs a small
       character-level BERT model to predict the stress position from the
       word's orthographic form.

    Args:
        data_path: Directory that contains the ``oov_model`` sub-directory
            (PyTorch model files) and ``unk_vocab.pickle``.
        device: PyTorch device.  ``"auto"`` selects CUDA when available,
            otherwise CPU.
    """

    def __init__(self, data_path: str, device: str = "auto") -> None:
        model_dir = os.path.join(data_path, "unk_model")

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model = BertForTokenClassification.from_pretrained(
            model_dir,
            num_labels=1,
            cache_dir=None,
        )
        self._model.eval()
        self._model.to(self._device)

        self._vocab = OovVocabulary(data_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, word: str) -> int:
        """Predict the stress position for a single unknown word.

        Args:
            word: The unknown word.  May contain ``ё``; may be any case.

        Returns:
            0-based character index of the stressed vowel, or ``-1`` if the
            prediction could not be made.
        """
        # Fast path: ё is always stressed
        yo_pos = _find_yo_stress(word)
        if yo_pos >= 0:
            return yo_pos

        # Try vocabulary-based extrapolation first
        vocab_pos = self._vocab.predict_stress(word)
        if vocab_pos >= 0:
            return vocab_pos

        # Fall back to the neural model
        return self._predict_with_model(word)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict_with_model(self, word: str) -> int:
        """Run the BERT model on a single word and return the predicted stress position."""
        batcher = OovBatcher([word])
        batch = next(iter(batcher))

        with torch.no_grad():
            input_ids = batch.input_ids.to(self._device)
            attention_mask = batch.attention_mask.to(self._device)

            logits = self._model(input_ids, attention_mask=attention_mask)
            logits = logits.squeeze(-1)  # (batch_size, seq_len)

        # logits[0] corresponds to the single word in the batch
        word_logits = logits[0].cpu().tolist()

        # The model outputs one logit per character position.
        # Position 0 is BOS, so real characters start at index 1.
        if len(word) < 2:
            return 0 if len(word) == 1 else -1

        max_pos = 1
        max_logit = word_logits[max_pos]
        for token_idx in range(len(word)):
            # token_idx+1 because of the leading BOS token
            if word_logits[token_idx + 1] > max_logit:
                max_pos = token_idx + 1
                max_logit = word_logits[token_idx + 1]

        # Convert from token position (including BOS) to character position
        return max_pos - 1


__all__ = [
    "OovStressPredictor",
    "_find_yo_stress",
]
