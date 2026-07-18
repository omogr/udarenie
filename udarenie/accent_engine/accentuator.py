"""
Main accentuation engine with backward-compatible API.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch

from .core import (
    AccentConfig,
    DocumentResult,
    OutputFormat,
    StressMethod,
    StressPosition,
    WordInfo,
    ModelLoadError,
    RUSSIAN_ALPHABET,
)
from .parser import TextParser, SSMLParser
from .tokenizer import AccentTokenizer
from .batcher import BERTBatcher
from .resolvers import (
    MonosyllableResolver,
    YoResolver,
    DictionaryResolver,
    HeuristicResolver,
    BERTResolver,
    AccentDictionary,
)
from .resolvers import count_vowels
from .resolvers import vowel_positions
from .formatters import (
    AnnotatedFormatter,
    StressMarkFormatter,
    JSONFormatter,
    WordListFormatter,
    get_formatter,
)

logger = logging.getLogger(__name__)

# =============================================================================
# MAIN ENGINE
# =============================================================================

class AccentEngine:
    """
    Modern accentuation engine.

    Provides clean API for text accentuation with full metadata.
    """

    def __init__(self, config: AccentConfig):
        self.config = config
        self.device = self._get_device()

        # Components
        self.parser = TextParser(SSMLParser(
            preserve_tags=config.ssml_preserve_tags,
            void_tags=config.ssml_void_tags,
        ))
        self.dictionary = AccentDictionary(config.data_path)
        self.tokenizer = self._load_tokenizer()
        self.batcher = BERTBatcher(
            self.tokenizer,
            max_batch_tokens=config.max_batch_tokens,
            max_sentence_len=config.max_sentence_len,
        )

        # Resolvers (in order of priority)
        self.resolvers = self._build_resolvers()

        logger.info(f"AccentEngine initialized on {self.device}")

    def _get_device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _load_tokenizer(self) -> AccentTokenizer:
        vocab_path = self.config.data_path / 'model' / 'vocab.txt'
        return AccentTokenizer(vocab_path)

    def _build_resolvers(self) -> list:
        resolvers = []

        if self.config.stress_monosyllabic:
            resolvers.append(MonosyllableResolver())

        if self.config.stress_yo:
            resolvers.append(YoResolver())

        resolvers.append(DictionaryResolver(self.dictionary))

        if self.config.use_bert:
            model_path = self.config.data_path / 'model'
            resolvers.append(BERTResolver(
                model_path=model_path,
                tokenizer=self.tokenizer,
                dictionary=self.dictionary,
                device=self.device,
            ))

        if self.config.use_heuristic:
            resolvers.append(HeuristicResolver(
                data_path=self.config.data_path,
                device=self.device,
            ))

        return resolvers

    def accentuate(self, text: str) -> DocumentResult:
        """
        Accentuate text.

        Args:
            text: Input text (may contain SSML tags)

        Returns:
            DocumentResult with full metadata
        """
        # Parse text
        document = self.parser.parse(text)

        # Determine which sentences need BERT
        sentences_for_bert = []
        needs_bert = []

        for i, sentence in enumerate(document.sentences):
            has_ambiguous = False
            for word in sentence.words:
                if self.dictionary.has_multiple_stresses(word.text):
                    has_ambiguous = True
                    break

            sentences_for_bert.append((i, sentence.original))
            needs_bert.append(has_ambiguous)

        # Prepare batches
        easy_spans, bert_batches = self.batcher.prepare(sentences_for_bert, needs_bert)

        # Process easy sentences (no BERT needed)
        easy_results = self._process_easy(document, easy_spans)

        # Process BERT batches
        bert_results = self._process_bert(document, bert_batches)

        # Merge results
        self._merge_results(document, easy_results, bert_results)

        # Apply remaining resolvers (heuristic, etc.)
        self._apply_resolvers(document)

        return document

    def _resolve_single_word(
        self, tword: str
    ) -> tuple[Optional[StressPosition], Optional[StressMethod]]:
        """Resolve stress for a single word without hyphens."""
        # ё is always stressed
        yo_pos = tword.casefold().find('ё')
        if yo_pos >= 0:
            vowels = vowel_positions(tword.casefold())
            try:
                vidx = vowels.index(yo_pos)
            except ValueError:
                vidx = 0
            return StressPosition(vidx, yo_pos), StressMethod.YO

        # No vowels → non-word
        vcount = count_vowels(tword)
        if vcount < 1:
            return None, StressMethod.NON_WORD

        # Dictionary
        entry = self.dictionary.lookup(tword)
        if entry:
            pos = entry.stress_positions[0]
            vowels = vowel_positions(tword.casefold())
            try:
                vidx = vowels.index(pos)
            except ValueError:
                vidx = 0
            method = (
                StressMethod.DICT_SINGLE
                if len(entry.stress_positions) == 1
                else StressMethod.DICT_MULTI
            )
            return StressPosition(vidx, pos), method

        return None, None

    def _process_easy(
        self,
        document: DocumentResult,
        easy_spans: list[tuple[int, list]],
    ) -> dict[int, list[tuple[int, int, Optional[StressPosition], StressMethod]]]:
        """Process sentences that don't need BERT."""
        results = {}

        for doc_pos, spans in easy_spans:
            sentence = document.sentences[doc_pos]
            word_results = []

            for (first_word, last_word), _ in spans:
                if first_word == last_word:  # BOS/SEP
                    continue

                tword = sentence.original[first_word:last_word]

                # 1. Try whole word first
                stress, method = self._resolve_single_word(tword)
                if stress is not None:
                    word_results.append((first_word, last_word, stress, method))
                    continue

                # 2. If it contains hyphens and is not a non-word, try parts
                if '-' in tword and method != StressMethod.NON_WORD:
                    parts = tword.split('-')
                    offset = 0
                    found_any = False
                    for part in parts:
                        if not part:
                            offset += 1
                            continue
                        part_stress, part_method = self._resolve_single_word(part)
                        if part_stress is not None:
                            adjusted = StressPosition(
                                part_stress.vowel_index,
                                part_stress.char_index + offset,
                            )
                            word_results.append((first_word, last_word, adjusted, part_method))
                            found_any = True
                        offset += len(part) + 1
                    if found_any:
                        continue

                # 3. Truly unknown or non-word
                word_results.append((first_word, last_word, None, method or StressMethod.UNKNOWN))

            results[doc_pos] = word_results

        return results

    def _process_bert(
        self,
        document: DocumentResult,
        batches: list,
    ) -> dict[int, list[tuple[int, int, Optional[StressPosition], StressMethod]]]:
        """Process BERT batches."""
        results = {}

        bert_resolver = None
        for resolver in self.resolvers:
            if isinstance(resolver, BERTResolver):
                bert_resolver = resolver
                break

        if bert_resolver is None:
            return results

        for batch in batches:
            batch_results = bert_resolver.resolve_batch(batch, document)
            results.update(batch_results)

        return results

    def _merge_results(
        self,
        document: DocumentResult,
        easy_results: dict,
        bert_results: dict,
    ) -> None:
        """Merge easy and BERT results into document."""
        all_results = {**easy_results, **bert_results}

        for doc_pos, word_results in all_results.items():
            if doc_pos >= len(document.sentences):
                continue

            sentence = document.sentences[doc_pos]

            for first_word, last_word, stress, method in word_results:
                # Find corresponding WordInfo
                for word in sentence.words:
                    if word.start == sentence.spans[0].start + first_word:
                        if word.stress is None:
                            word.stress = stress
                            word.method = method
                        elif stress is not None:
                            word.sub_stresses.append(stress)
                        break

    def _apply_resolvers(self, document: DocumentResult) -> None:
        """Apply remaining resolvers for unresolved words."""
        for sentence in document.sentences:
            for word in sentence.words:
                if word.stress is not None:
                    continue

                # --- hyphenated words -------------------------------------------------
                if '-' in word.text:
                    parts = word.text.split('-')
                    offset = 0
                    found_any = False
                    for part in parts:
                        if not part:
                            offset += 1
                            continue

                        temp_word = WordInfo(
                            text=part,
                            start=word.start + offset,
                            end=word.start + offset + len(part),
                            is_russian_word=all(ch in RUSSIAN_ALPHABET for ch in part),
                        )

                        for resolver in self.resolvers:
                            if isinstance(resolver, BERTResolver):
                                continue
                            if resolver.can_resolve(temp_word):
                                try:
                                    stress = resolver.resolve(temp_word, sentence)
                                    if stress is not None:
                                        adjusted = StressPosition(
                                            stress.vowel_index,
                                            stress.char_index + offset,
                                        )
                                        if not found_any:
                                            word.stress = adjusted
                                            word.method = resolver.method
                                        else:
                                            word.sub_stresses.append(adjusted)
                                        found_any = True
                                        break
                                except Exception as e:
                                    logger.warning(
                                        f"Resolver {resolver.method.name} failed for part "
                                        f"'{part}' of '{word.text}': {e}"
                                    )

                        offset += len(part) + 1

                    if not found_any:
                        word.method = StressMethod.UNKNOWN
                    continue
                # ----------------------------------------------------------------------

                for resolver in self.resolvers:
                    if not isinstance(resolver, BERTResolver) and resolver.can_resolve(word):
                        try:
                            stress = resolver.resolve(word, sentence)
                            if stress is not None:
                                word.stress = stress
                                word.method = resolver.method
                                break
                        except Exception as e:
                            logger.warning(
                                f"Resolver {resolver.method.name} failed for '{word.text}': {e}"
                            )

                if word.stress is None:
                    word.method = StressMethod.UNKNOWN

    # Convenience methods
    def to_text(self, text: str) -> str:
        """Accentuate and return annotated text."""
        return self.accentuate(text).to_annotated_text()

    def to_stress_marks(self, text: str) -> str:
        """Accentuate and return text with stress marks."""
        return self.accentuate(text).to_stress_marks()

    def to_json(self, text: str) -> dict:
        """Accentuate and return JSON."""
        return self.accentuate(text).to_json()

    def to_word_list(self, text: str) -> list[list[tuple[str, str]]]:
        """Accentuate and return legacy word list format."""
        formatter = WordListFormatter()
        return formatter.format(self.accentuate(text))


# =============================================================================
# BACKWARD COMPATIBLE API
# =============================================================================

class Accentuator:
    """
    Backward-compatible wrapper matching original API.

    Provides:
    - accentuate(input_text) -> str or list[str]
    - accentuate_by_words(input_sentence_list) -> list[list[tuple[str, str]]]
    - accentuate_sentence_list(input_sentence_list) -> list[str]
    """

    def __init__(self, data_path: str, device_name: Optional[str] = None):
        config = AccentConfig(
            data_path=Path(data_path),
            device=device_name or "auto",
        )
        self._engine = AccentEngine(config)

    def accentuate(self, input_text: Union[str, list[str]]) -> Union[str, list[str]]:
        """
        Accentuate text(s).

        Args:
            input_text: Single string or list of strings

        Returns:
            Annotated text (str) or list of annotated texts (list[str])
        """
        if isinstance(input_text, str):
            return self._engine.to_text(input_text)

        if not isinstance(input_text, list):
            raise ValueError("a string or list of strings is required")

        if not all(isinstance(elem, str) for elem in input_text):
            raise ValueError("a string or list of strings is required")

        return [self._engine.to_text(t) for t in input_text]

    def accentuate_by_words(self, input_sentence_list: list[str]) -> list[list[tuple[str, str]]]:
        """
        Accentuate and return word list format.

        Returns list of sentences, each sentence is list of (punct, word) tuples.
        """
        if not input_sentence_list:
            raise ValueError("list of strings is required")

        if not isinstance(input_sentence_list, list):
            raise ValueError("list of strings is required")

        if not all(isinstance(elem, str) for elem in input_sentence_list):
            raise ValueError("list of strings is required")

        return self._engine.to_word_list("\n".join(input_sentence_list))

    def accentuate_sentence_list(self, input_sentence_list: list[str]) -> list[str]:
        """
        Accentuate list of sentences.

        Returns list of annotated strings.
        """
        if not input_sentence_list:
            raise ValueError("a list of strings is required")

        if not isinstance(input_sentence_list, list):
            raise ValueError("a list of strings is required")

        if not all(isinstance(elem, str) for elem in input_sentence_list):
            raise ValueError("a list of strings is required")

        return [self._engine.to_text(t) for t in input_sentence_list]

    def accentuate_all_easy(self, input_sentence_list: list[str]) -> list[str]:
        """
        Backward-compatible method for easy-only accentuation.

        In modern engine, this is equivalent to full accentuation
        (BERT is only used when needed).
        """
        return self.accentuate_sentence_list(input_sentence_list)