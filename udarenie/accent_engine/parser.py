"""
Text parsing with SSML and technical tag support.
"""
from __future__ import annotations

import re
from typing import Optional

from .core import (
    RUSSIAN_ALPHABET,
    DocumentResult,
    SentenceResult,
    TextSpan,
    WordInfo,
    SSMLTag,
    DEFAULT_PRESERVE_TAGS,
    DEFAULT_VOID_TAGS,
    TextParseError,
)


class SSMLParser:
    """
    Parser for SSML and technical tags.

    Supports:
    - Void tags (<break/>, <phoneme/>) — removed from clean text
    - Preserve tags (<prosody>, <emphasis>) — content preserved but not processed
    - Custom tags can be configured as preserve or void
    """

    # Pattern for any XML-like tag
    TAG_RE = re.compile(r'<(/?)([a-zA-Z][a-zA-Z0-9]*)[^>]*?(/?)>')

    def __init__(
        self,
        preserve_tags: Optional[set[str]] = None,
        void_tags: Optional[set[str]] = None,
    ):
        self.preserve_tags = preserve_tags or set(DEFAULT_PRESERVE_TAGS)
        self.void_tags = void_tags or set(DEFAULT_VOID_TAGS)

    def parse(self, text: str) -> tuple[str, list[SSMLTag]]:
        """
        Extract tags from text.

        Returns:
            (clean_text, tags) where clean_text has void tags removed
            and preserve tags replaced with markers.
        """
        tags: list[SSMLTag] = []
        clean_parts: list[str] = []

        offset = 0          # position in original text
        clean_pos = 0       # position in clean text
        preserve_stack: list[tuple[str, int, int]] = []  # (tag_name, clean_start, original_start)

        for match in self.TAG_RE.finditer(text):
            is_close = bool(match.group(1))
            tag_name = match.group(2).lower()
            is_void = bool(match.group(3)) or tag_name in self.void_tags

            # Add text before tag
            before = text[offset:match.start()]
            clean_parts.append(before)
            clean_pos += len(before)

            if is_void or tag_name not in (self.preserve_tags | self.void_tags):
                # Unknown or void tag — skip entirely
                pass
            elif is_close:
                # Closing preserve tag
                if preserve_stack and preserve_stack[-1][0] == tag_name:
                    _, clean_start, orig_start = preserve_stack.pop()
                    tag = SSMLTag(
                        tag=match.group(0),
                        start=clean_start,
                        end=clean_pos,
                        original_start=orig_start,
                        original_end=match.end(),
                        is_preserve=True,
                    )
                    tags.append(tag)
            else:
                # Opening preserve tag
                preserve_stack.append((tag_name, clean_pos, match.start()))

            offset = match.end()

        # Add remaining text
        clean_parts.append(text[offset:])
        clean_text = ''.join(clean_parts)

        # Handle unclosed preserve tags — treat remaining content as preserved
        if preserve_stack:
            for tag_name, clean_start, orig_start in preserve_stack:
                tag = SSMLTag(
                    tag=f'</{tag_name}>',
                    start=clean_start,
                    end=clean_pos,
                    original_start=orig_start,
                    original_end=len(text),
                    is_preserve=True,
                )
                tags.append(tag)

        return clean_text, tags


class TextParser:
    """
    Parse text into sentences and words.

    Preserves exact positions in original text for every word and span.
    """

    # Sentence delimiters
    SENTENCE_END_RE = re.compile(r'[.!?…]+[\s\n]*|\n{2,}')

    # Word pattern: Russian letters and hyphens
    WORD_RE = re.compile(
        r'[\-абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ]+'
    )

    def __init__(self, ssml_parser: Optional[SSMLParser] = None):
        self.ssml_parser = ssml_parser or SSMLParser()

    def parse(self, text: str) -> DocumentResult:
        """
        Parse text into DocumentResult.

        Args:
            text: Input text, possibly with SSML tags

        Returns:
            DocumentResult with parsed sentences and spans
        """
        clean_text, ssml_tags = self.ssml_parser.parse(text)

        sentences: list[SentenceResult] = []
        offset = 0

        for sentence_text in self._split_sentences(clean_text):
            if not sentence_text.strip():
                offset += len(sentence_text)
                continue

            sent = self._parse_sentence(sentence_text, offset, ssml_tags)
            sentences.append(sent)
            offset += len(sentence_text)

        return DocumentResult(
            original_text=text,
            sentences=sentences,
            ssml_tags=ssml_tags,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences: list[str] = []
        start = 0

        for match in self.SENTENCE_END_RE.finditer(text):
            end = match.end()
            sentences.append(text[start:end])
            start = end

        if start < len(text):
            sentences.append(text[start:])

        return sentences

    def _parse_sentence(
        self,
        text: str,
        base_offset: int,
        ssml_tags: list[SSMLTag],
    ) -> SentenceResult:
        """Parse a sentence into words and non-word spans."""
        spans: list[TextSpan] = []
        words: list[WordInfo] = []

        pos = 0
        for match in self.WORD_RE.finditer(text):
            word_start = match.start()
            word_end = match.end()
            word_text = match.group(0)

            # Non-word before this word
            if word_start > pos:
                non_word_text = text[pos:word_start]
                is_preserved = self._is_preserved(base_offset + pos, base_offset + word_start, ssml_tags)
                spans.append(TextSpan(
                    text=non_word_text,
                    start=base_offset + pos,
                    end=base_offset + word_start,
                    is_word=False,
                    is_preserved=is_preserved,
                ))

            # Check if word is inside preserved region
            is_preserved = self._is_preserved(
                base_offset + word_start,
                base_offset + word_end,
                ssml_tags,
            )

            # Word
            word = WordInfo(
                text=word_text,
                start=base_offset + word_start,
                end=base_offset + word_end,
                is_russian_word=self._is_russian_word(word_text) and not is_preserved,
            )
            words.append(word)
            spans.append(TextSpan(
                text=word_text,
                start=base_offset + word_start,
                end=base_offset + word_end,
                is_word=True,
                word_info=word,
                is_preserved=is_preserved,
            ))

            pos = word_end

        # Non-word at end
        if pos < len(text):
            is_preserved = self._is_preserved(base_offset + pos, base_offset + len(text), ssml_tags)
            spans.append(TextSpan(
                text=text[pos:],
                start=base_offset + pos,
                end=base_offset + len(text),
                is_word=False,
                is_preserved=is_preserved,
            ))

        return SentenceResult(
            original=text,
            spans=spans,
            words=words,
        )

    def _is_russian_word(self, text: str) -> bool:
        """Check if text is a Russian word."""
        if not text:
            return False
        return all(ch in RUSSIAN_ALPHABET for ch in text)

    def _is_preserved(self, start: int, end: int, tags: list[SSMLTag]) -> bool:
        """Check if region is inside a preserved tag."""
        for tag in tags:
            if tag.is_preserve and tag.start <= start and end <= tag.end:
                return True
        return False
