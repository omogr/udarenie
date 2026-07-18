"""
Stress resolvers using Strategy pattern.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import pyarrow.parquet as pq

import torch
import torch.nn.functional as F

LOCAL_BERT = True

if LOCAL_BERT:
    # without transformers
    from .bert import BertForTokenClassification
else:
    from transformers import BertForTokenClassification

from .core import (
    RUSSIAN_VOWELS,
    RUSSIAN_VOWELS_LOWER,
    StressMethod,
    StressPosition,
    WordInfo,
    SentenceResult,
    ModelLoadError,
    ResolutionError,
)

from .tokenizer import AccentTokenizer
from .oov_model import OovStressPredictor

def count_vowels(text: str) -> int:
    """Count vowels in text."""
    return sum(1 for ch in text if ch in RUSSIAN_VOWELS)


def vowel_positions(text: str) -> list[int]:
    """Return indices of all vowels in text."""
    return [i for i, ch in enumerate(text) if ch in RUSSIAN_VOWELS]


def normalize_word(word: str) -> str:
    """Normalize word for dictionary lookup."""
    return word.casefold().replace('ё', 'е')


# =============================================================================
# BASE CLASS
# =============================================================================

class StressResolver(ABC):
    """Base class for all stress resolvers."""

    @property
    @abstractmethod
    def method(self) -> StressMethod:
        """Method identifier."""
        pass

    @abstractmethod
    def can_resolve(self, word: WordInfo) -> bool:
        """Check if this resolver can handle the word."""
        pass

    @abstractmethod
    def resolve(
        self,
        word: WordInfo,
        context: SentenceResult,
    ) -> Optional[StressPosition]:
        """Determine stress position. Returns None if failed."""
        pass


# =============================================================================
# MONO-SYLLABLE RESOLVER
# =============================================================================

class MonosyllableResolver(StressResolver):
    """Resolver for monosyllabic words."""

    method = StressMethod.MONO

    def can_resolve(self, word: WordInfo) -> bool:
        return count_vowels(word.text) <= 1 and word.is_russian_word

    def resolve(self, word: WordInfo, context: SentenceResult) -> Optional[StressPosition]:
        vowels = vowel_positions(word.text)
        if not vowels:
            return None
        return StressPosition(vowel_index=0, char_index=vowels[0])


# =============================================================================
# YO RESOLVER
# =============================================================================

class YoResolver(StressResolver):
    """Resolver for words containing letter ё."""

    method = StressMethod.YO

    def can_resolve(self, word: WordInfo) -> bool:
        return 'ё' in word.text or 'Ё' in word.text

    def resolve(self, word: WordInfo, context: SentenceResult) -> Optional[StressPosition]:
        text = word.text.casefold()
        yo_pos = text.find('ё')
        if yo_pos < 0:
            return None

        vowels = vowel_positions(text)
        try:
            vowel_index = vowels.index(yo_pos)
        except ValueError:
            return None

        return StressPosition(vowel_index=vowel_index, char_index=yo_pos)


# =============================================================================
# DICTIONARY
# =============================================================================

@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    """Dictionary entry with stress information."""
    word: str                      # normalized word
    stress_positions: tuple[int, ...]  # absolute character positions
    stress_vowels: tuple[int, ...]     # vowel indices (0-based)
    variants: dict[str, float]         # original variants with weights


class AccentDictionary:
    """Stress dictionary loaded from wav2vec_words2.vcb format."""

    def __init__(self, data_path: Path):
        self._entries: dict[str, str] = {}
        self._load(data_path)

    def _load(self, data_path: Path) -> None:
        vcb_file = data_path / 'vec_words.pq'
        
        #print('AccentDictionary vcb_file', vcb_file)
        if vcb_file.exists():
            self._load_from_vcb(vcb_file)
        
        else:
            raise ModelLoadError(f"No dictionary file found in {vcb_file}")

    def _load_from_vcb(self, path: Path) -> None:
        table = pq.read_table(path)
    
        keys = table.column(0).to_pylist()
        values = table.column(1).to_pylist()
        self._entries = dict(zip(keys, values))

    def _load0(self, data_path: Path) -> None:
        """Load dictionary from vcb file or pickle."""
        vcb_file = data_path / 'wav2vec_words2.vcb'
        pickle_file = data_path / 'wv_word_acc.pickle'

        if vcb_file.exists():
            self._load_from_vcb(vcb_file)
        elif pickle_file.exists():
            self._load_from_pickle(pickle_file)
        else:
            raise ModelLoadError(f"No dictionary file found in {data_path}")

    def _load_from_vcb0(self, path: Path) -> None:
        """Load from text vcb file."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                word = parts[1]
                try:
                    variants = json.loads(parts[2])
                except json.JSONDecodeError:
                    continue

                stress_positions = []
                stress_vowels = []
                for variant, weight in variants.items():
                    plus_pos = variant.find('+')
                    if plus_pos >= 0:
                        stress_positions.append(plus_pos)
                        # Count vowels before this position
                        vowel_idx = sum(1 for i in range(plus_pos) 
                                       if variant[i] in RUSSIAN_VOWELS)
                        stress_vowels.append(vowel_idx)

                if stress_positions:
                    normalized = normalize_word(word)
                    self._entries[normalized] = DictionaryEntry(
                        word=normalized,
                        stress_positions=tuple(sorted(set(stress_positions))),
                        stress_vowels=tuple(sorted(set(stress_vowels))),
                        variants=variants,
                    )

    def _load_from_pickle(self, path: Path) -> None:
        """Load from legacy pickle format."""
        import pickle
        with open(path, 'rb') as f:
            vocab, vocab_index = pickle.load(f)

        for word, word_id in vocab.items():
            positions = vocab_index[word_id] if word_id < len(vocab_index) else []
            if not positions:
                continue

            normalized = normalize_word(word)
            vowels = vowel_positions(normalized)
            stress_vowels = []
            for pos in positions:
                # Find vowel index for this position
                for i, vpos in enumerate(vowels):
                    if vpos == pos:
                        stress_vowels.append(i)
                        break

            self._entries[normalized] = DictionaryEntry(
                word=normalized,
                stress_positions=tuple(positions),
                stress_vowels=tuple(stress_vowels),
                variants={},
            )

    def lookup0(self, word: str) -> Optional[DictionaryEntry]:
        """Look up word in dictionary."""
        return self._entries.get(normalize_word(word))

    def lookup(self, word: str) -> Optional[DictionaryEntry]:
        normalized = normalize_word(word)
        json_str = self._entries.get(normalized)
        if not json_str:
            return None
        try:
            variants = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        stress_positions = []
        stress_vowels = []
        for variant, weight in variants.items():
            plus_pos = variant.find('+')
            if plus_pos >= 0:
                stress_positions.append(plus_pos)
                vowel_idx = sum(1 for i in range(plus_pos) 
                               if variant[i] in RUSSIAN_VOWELS)
                stress_vowels.append(vowel_idx)

        if stress_positions:
            
            return DictionaryEntry(
                word=normalized,
                stress_positions=tuple(sorted(set(stress_positions))),
                stress_vowels=tuple(sorted(set(stress_vowels))),
                variants=variants,
            )
        return None

    def has_single_stress(self, word: str) -> bool:
        """Check if word has unambiguous stress in dictionary."""
        entry = self.lookup(word)
        return entry is not None and len(entry.stress_positions) == 1

    def has_multiple_stresses(self, word: str) -> bool:
        """Check if word has ambiguous stress (needs BERT)."""
        entry = self.lookup(word)
        return entry is not None and len(entry.stress_positions) > 1


# =============================================================================
# DICTIONARY RESOLVER
# =============================================================================

class DictionaryResolver(StressResolver):
    """Resolver using accent dictionary."""

    method = StressMethod.DICT_SINGLE

    def __init__(self, dictionary: AccentDictionary):
        self.dictionary = dictionary

    def can_resolve(self, word: WordInfo) -> bool:
        return self.dictionary.has_single_stress(word.text)

    def resolve(self, word: WordInfo, context: SentenceResult) -> Optional[StressPosition]:
        entry = self.dictionary.lookup(word.text)
        if not entry or not entry.stress_positions:
            return None

        pos = entry.stress_positions[0]
        vowels = vowel_positions(word.text.casefold())
        try:
            vowel_index = vowels.index(pos)
        except ValueError:
            return None

        return StressPosition(vowel_index=vowel_index, char_index=pos)
# =============================================================================
# HEURISTIC RESOLVER (OOV words)
# =============================================================================

class HeuristicResolver(StressResolver):
    """Heuristic resolver for out-of-vocabulary words."""

    method = StressMethod.HEURISTIC

    def __init__(self, data_path: Path, device: torch.device):
        self.data_path = data_path
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load heuristic model (UnkModel equivalent)."""
        # Placeholder — actual implementation depends on oov_model.py
        try:
            
            self._model = OovStressPredictor(self.data_path, device=str(self.device))
        except ImportError:
            self._model = None

    def can_resolve(self, word: WordInfo) -> bool:
        return word.is_russian_word and count_vowels(word.text) > 0

    def resolve(self, word: WordInfo, context: SentenceResult) -> Optional[StressPosition]:
        if self._model is None:
            # Fallback: stress on first vowel
            vowels = vowel_positions(word.text.casefold())
            if not vowels:
                return None
            return StressPosition(vowel_index=0, char_index=vowels[0])

        try:
            acc_pos = self._model.predict(word.text.casefold())
            if acc_pos < 0:
                return None

            vowels = vowel_positions(word.text.casefold())
            try:
                vowel_index = vowels.index(acc_pos)
            except ValueError:
                return None

            return StressPosition(vowel_index=vowel_index, char_index=acc_pos)
        except Exception:
            return None


# =============================================================================
# BERT RESOLVER
# =============================================================================

class BERTResolver(StressResolver):
    """BERT-based resolver for ambiguous words."""

    method = StressMethod.BERT

    def __init__(
        self,
        model_path: Path,
        tokenizer: AccentTokenizer,
        dictionary: AccentDictionary,
        device: torch.device,
    ):
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.device = device

        if not model_path.exists():
            raise ModelLoadError(f"Model path not found: {model_path}")

        self.model = BertForTokenClassification.from_pretrained(
            str(model_path),
            num_labels=10,
        )
        self.model.eval()
        self.model.to(device)

        self._error_count = 0

    def can_resolve(self, word: WordInfo) -> bool:
        return self.dictionary.has_multiple_stresses(word.text)

    def resolve(self, word: WordInfo, context: SentenceResult) -> Optional[StressPosition]:
        """Single word resolution (fallback)."""
        entry = self.dictionary.lookup(word.text)
        if entry and entry.stress_positions:
            # Fallback to first dictionary variant
            pos = entry.stress_positions[0]
            vowels = vowel_positions(word.text.casefold())
            try:
                vowel_index = vowels.index(pos)
            except ValueError:
                return None
            return StressPosition(vowel_index=vowel_index, char_index=pos)
        return None

    def resolve_batch(
        self,
        batch: 'BertBatch',  # type: ignore
        doc: 'DocumentResult',  # type: ignore
    ) -> dict[int, list[tuple[int, int, StressPosition, StressMethod]]]:
        """
        Resolve stress for a batch of sentences using BERT.

        Returns:
            dict mapping doc_pos to list of (first, last, StressPosition, method)
        """
        from .batcher import BertBatch

        if not isinstance(batch, BertBatch):
            raise ResolutionError("Invalid batch type")

        results: dict[int, list[tuple[int, int, StressPosition, StressMethod]]] = {}

        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
                       
            if LOCAL_BERT:
                logits = outputs # outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                #if logits.dim() == 2:
                #    logits = logits.unsqueeze(0)  # Добавляем batch-измерение: [1, 
                            
            logits = logits.detach().cpu()
            
        for batch_idx, (doc_pos, spans) in enumerate(batch.sentence_spans):
            sentence = doc.sentences[doc_pos] if doc_pos < len(doc.sentences) else None
            if not sentence:
                continue

            word_results = []
            t_logits = logits[batch_idx]
            t_input_ids = input_ids[batch_idx].cpu().tolist()

            for (first_word, last_word), (first_token, last_token) in spans:
                if first_word == last_word:  # BOS/SEP
                    continue

                tword = sentence.original[first_word:last_word]

                # Check ё
                yo_pos = tword.casefold().find('ё')
                if yo_pos >= 0:
                    vowels = vowel_positions(tword.casefold())
                    try:
                        vidx = vowels.index(yo_pos)
                    except ValueError:
                        vidx = 0
                    word_results.append((first_word, last_word, 
                                        StressPosition(vidx, yo_pos), StressMethod.YO))
                    continue

                # Check vowels
                if count_vowels(tword) < 1:
                    word_results.append((first_word, last_word, None, StressMethod.NON_WORD))
                    continue

                # Dictionary lookup
                entry = self.dictionary.lookup(tword)
                if not entry:
                    # OOV — skip in BERT, will be handled by heuristic
                    continue

                if len(entry.stress_positions) == 1:
                    pos = entry.stress_positions[0]
                    vowels = vowel_positions(tword.casefold())
                    try:
                        vidx = vowels.index(pos)
                    except ValueError:
                        continue
                    word_results.append((first_word, last_word,
                                        StressPosition(vidx, pos), StressMethod.DICT_SINGLE))
                    continue

                # Multiple stresses — use BERT
                best_prob = -1.0
                best_pos = -1
                letter_pos = 0

                for token_idx in range(first_token, last_token):
                    token_id = t_input_ids[token_idx]
                    token_info = self.tokenizer.get_token_vowel_info(token_id)

                    if token_info is None:
                        self._error_count += 1
                        break

                    num_letters, vowel_pos = token_info
                    if not vowel_pos:
                        letter_pos += num_letters
                        continue

                    # Compute softmax probabilities
                    token_logits = t_logits[token_idx]
                    # print(token_idx, 'token_logits', token_logits)
                    probs = F.softmax(token_logits, dim=0).tolist()

                    for prob_idx, prob in enumerate(probs):
                        if prob_idx < 1:
                            continue
                        if prob_idx > len(vowel_pos):
                            break

                        tpos = letter_pos + vowel_pos[prob_idx - 1]

                        if tpos not in entry.stress_positions:
                            continue

                        if prob > best_prob:
                            best_prob = prob
                            best_pos = tpos

                    letter_pos += num_letters

                if best_pos >= 0:
                    vowels = vowel_positions(tword.casefold())
                    try:
                        vidx = vowels.index(best_pos)
                    except ValueError:
                        vidx = 0
                    word_results.append((first_word, last_word,
                                        StressPosition(vidx, best_pos), StressMethod.BERT))
                else:
                    # BERT failed — fallback to first variant
                    pos = entry.stress_positions[0]
                    vowels = vowel_positions(tword.casefold())
                    try:
                        vidx = vowels.index(pos)
                    except ValueError:
                        vidx = 0
                    word_results.append((first_word, last_word,
                                        StressPosition(vidx, pos), StressMethod.DICT_MULTI))

            results[doc_pos] = word_results

        return results
