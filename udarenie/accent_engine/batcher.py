"""
BERT batching logic for accentuation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .core import (
    MAX_SENTENCE_LEN,
    MAX_POSITION_EMBEDDINGS,
    BERT_BOS_ID,
    BERT_SEP_ID,
    BERT_PAD_ID,
)
from .tokenizer import AccentTokenizer


@dataclass
class TokenizedSentence:
    """A sentence prepared for BERT processing."""
    doc_pos: int                # position in document
    input_ids: list[int]        # token IDs with BOS/SEP
    spans: list[tuple[tuple[int, int], tuple[int, int]]]  # (text_span, token_span)
    original: str               # original sentence text


@dataclass
class BertBatch:
    """A batch of sentences for BERT inference."""
    sentence_spans: list[tuple[int, list[tuple[tuple[int, int], tuple[int, int]]]]]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    max_length: int


class BERTBatcher:
    """
    Groups sentences into batches for efficient BERT inference.

    Sentences that are too long or don't need BERT are marked as easy.
    """

    def __init__(
        self,
        tokenizer: AccentTokenizer,
        max_batch_tokens: int = MAX_POSITION_EMBEDDINGS,
        max_sentence_len: int = MAX_SENTENCE_LEN,
    ):
        self.tokenizer = tokenizer
        self.max_batch_tokens = max_batch_tokens
        self.max_sentence_len = max_sentence_len

    def prepare(
        self,
        sentences: list[tuple[int, str]],
        needs_bert: list[bool],
    ) -> tuple[list[tuple[int, list[tuple[tuple[int, int], tuple[int, int]]]]], list[BertBatch]]:
        """
        Prepare sentences for processing.

        Args:
            sentences: list of (doc_pos, text)
            needs_bert: whether each sentence needs BERT (has ambiguous words)

        Returns:
            (easy_sentences, bert_batches)
        """
        easy_sentences: list[tuple[int, list[tuple[tuple[int, int], tuple[int, int]]]]] = []
        bert_candidates: list[TokenizedSentence] = []

        for (doc_pos, text), need_bert in zip(sentences, needs_bert):
            input_ids, spans = self._tokenize_sentence(text)

            if not need_bert:
                easy_sentences.append((doc_pos, spans))
                continue
                
            if len(input_ids) >= self.max_sentence_len:
                # Слишком длинное — BERT не осилит, отправляем в easy
                easy_sentences.append((doc_pos, spans))
                continue

            if len(input_ids) >= self.max_sentence_len:
                # Too long for BERT — treat as easy
                easy_sentences.append((doc_pos, spans))
                continue

            bert_candidates.append(TokenizedSentence(
                doc_pos=doc_pos,
                input_ids=input_ids,
                spans=spans,
                original=text,
            ))

        # Sort by length for efficient batching
        bert_candidates.sort(key=lambda x: len(x.input_ids))

        # Group into batches
        batches = self._create_batches(bert_candidates)

        return easy_sentences, batches

    def _tokenize_sentence(
        self,
        text: str,
    ) -> tuple[list[int], list[tuple[tuple[int, int], tuple[int, int]]]]:
        """
        Tokenize a sentence for BERT.

        Returns:
            (input_ids, spans)
        """
        # Remove existing stress marks
        text = text.replace('+', ' ')

        tokens: list[tuple[list[int], list[str], int, int]] = []
        # (word_tokens, punct_chars, first_char_pos, last_char_pos)

        # Add BOS
        tokens.append(([BERT_BOS_ID], [], 0, 0))

        word_chars: list[str] = []
        first_pos = -1

        for char_pos, char in enumerate(text):
            if char in '-абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ':
                if first_pos < 0:
                    first_pos = char_pos
                word_chars.append(char.lower())
                continue

            if word_chars:
                word_text = ''.join(word_chars)
                word_tokens = self.tokenizer.encode(word_text)
                tokens.append((word_tokens, [], first_pos, char_pos))
                word_chars = []
                first_pos = -1

            # Add punctuation to previous token's punct list
            if tokens:
                tokens[-1][1].append(char)

        # Handle last word
        if word_chars:
            word_text = ''.join(word_chars)
            word_tokens = self.tokenizer.encode(word_text)
            tokens.append((word_tokens, [], first_pos, len(text)))

        # Add SEP
        tokens.append(([BERT_SEP_ID], [], 0, 0))

        # Build input_ids and spans
        all_ids: list[int] = []
        spans: list[tuple[tuple[int, int], tuple[int, int]]] = []

        for word_tokens, punct_chars, first_char, last_char in tokens:
            first_token = len(all_ids)
            all_ids.extend(word_tokens)

            if last_char > 0 or first_char == 0:  # BOS/SEP have first_char==last_char==0
                last_token = len(all_ids)
                text_span = (first_char, last_char)
                token_span = (first_token, last_token)
                spans.append((text_span, token_span))

            # Add punctuation tokens
            if punct_chars:
                punct_text = ''.join(punct_chars).replace(' ', '')
                if punct_text:
                    punct_ids = self.tokenizer.encode(punct_text)
                    all_ids.extend(punct_ids)

        return all_ids, spans

    def _create_batches(self, candidates: list[TokenizedSentence]) -> list[BertBatch]:
        """Group sentences into batches."""
        batches: list[BertBatch] = []
        current_batch: list[TokenizedSentence] = []
        max_length = 1

        for candidate in candidates:
            new_max = max(max_length, len(candidate.input_ids))

            if new_max * (len(current_batch) + 1) > self.max_batch_tokens:
                # Start new batch
                if current_batch:
                    batches.append(self._build_batch(current_batch, max_length))
                current_batch = [candidate]
                max_length = len(candidate.input_ids)
            else:
                current_batch.append(candidate)
                max_length = new_max

        if current_batch:
            batches.append(self._build_batch(current_batch, max_length))

        return batches

    def _build_batch(
        self,
        candidates: list[TokenizedSentence],
        max_length: int,
    ) -> BertBatch:
        """Build a single BERT batch."""
        sentence_spans = []
        all_input_ids = []
        all_attention_mask = []

        for candidate in candidates:
            input_ids = candidate.input_ids[:max_length]
            attention_mask = [1] * len(input_ids)

            # Pad
            if len(input_ids) < max_length:
                padding = max_length - len(input_ids)
                input_ids += [BERT_PAD_ID] * padding
                attention_mask += [0] * padding

            sentence_spans.append((candidate.doc_pos, candidate.spans))
            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))

        return BertBatch(
            sentence_spans=sentence_spans,
            input_ids=torch.stack(all_input_ids),
            attention_mask=torch.stack(all_attention_mask),
            max_length=max_length,
        )
