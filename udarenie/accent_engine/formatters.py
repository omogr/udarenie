"""
Output formatters for accentuation results.
"""
from __future__ import annotations

from typing import Protocol

from .core import (
    DocumentResult,
    SentenceResult,
    TextSpan,
    WordInfo,
    StressMethod,
    OutputFormat,
)


class Formatter(Protocol):
    """Protocol for output formatters."""

    def format(self, result: DocumentResult) -> str:
        ...


class AnnotatedFormatter:
    """Legacy formatter: + before stressed vowel."""

    def format(self, result: DocumentResult) -> str:
        return result.to_annotated_text()


class StressMarkFormatter:
    """Formatter using combining acute accent U+0301."""

    def format(self, result: DocumentResult) -> str:
        return result.to_stress_marks()


class JSONFormatter:
    """JSON formatter with full metadata."""

    def format(self, result: DocumentResult) -> str:
        import json
        return json.dumps(result.to_json(), ensure_ascii=False, indent=2)


class WordListFormatter:
    """
    Legacy-compatible formatter returning list of (punct, word) tuples.

    Matches original accentuate_by_words() output format.
    """

    def format(self, result: DocumentResult) -> list[list[tuple[str, str]]]:
        output = []
        for sentence in result.sentences:
            word_list = []
            prev_end = sentence.spans[0].start if sentence.spans else 0

            for span in sentence.spans:
                if span.is_word and span.word_info:
                    # Punctuation before word
                    punct = result.original_text[prev_end:span.start]
                    word_text = span.text

                    if span.word_info.stress or span.word_info.sub_stresses:
                        all_stresses = []
                        if span.word_info.stress:
                            all_stresses.append(span.word_info.stress.char_index)
                        all_stresses.extend(s.char_index for s in span.word_info.sub_stresses)
                        all_stresses.sort(reverse=True)
                        for pos in all_stresses:
                            word_text = word_text[:pos] + '+' + word_text[pos:]

                    word_list.append((punct, word_text))
                    prev_end = span.end
                else:
                    # Non-word span — append to previous or create empty word
                    if word_list:
                        last_punct, last_word = word_list[-1]
                        word_list[-1] = (last_punct + span.text, last_word)
                    else:
                        word_list.append((span.text, ''))
                        prev_end = span.end

            # Trailing punctuation
            if sentence.spans:
                last_span = sentence.spans[-1]
                trailing = result.original_text[last_span.end:]
                if trailing:
                    if word_list:
                        last_punct, last_word = word_list[-1]
                        word_list[-1] = (last_punct + trailing, last_word)
                    else:
                        word_list.append((trailing, ''))

            output.append(word_list)

        return output


class TextListFormatter:
    """Formatter returning list of accentuated strings."""

    def format(self, result: DocumentResult) -> list[str]:
        return [result.to_annotated_text()]


def get_formatter(fmt: OutputFormat) -> Formatter:
    """Get formatter for given output format."""
    formatters = {
        OutputFormat.ANNOTATED: AnnotatedFormatter(),
        OutputFormat.STRESS_MARK: StressMarkFormatter(),
        OutputFormat.JSON: JSONFormatter(),
    }
    return formatters.get(fmt, AnnotatedFormatter())