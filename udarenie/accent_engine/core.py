"""
Core types, constants and exceptions for the accentuation engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# =============================================================================
# CONSTANTS
# =============================================================================

RUSSIAN_ALPHABET = frozenset(
    '-абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
)
RUSSIAN_VOWELS = frozenset('аеёиоуыэюяАЕЁИОУЫЭЮЯ')
RUSSIAN_VOWELS_LOWER = frozenset('аеёиоуыэюя')

MAX_SENTENCE_LEN = 510
MAX_POSITION_EMBEDDINGS = 512
BERT_BOS_ID = 2
BERT_SEP_ID = 3
BERT_PAD_ID = 0

DEFAULT_PRESERVE_TAGS = frozenset({
    'speak', 'prosody', 'emphasis', 'voice', 'audio', 'p', 's',
})
DEFAULT_VOID_TAGS = frozenset({
    'break', 'phoneme', 'mark',
})


# =============================================================================
# EXCEPTIONS
# =============================================================================

class AccentError(Exception):
    """Base exception for accentuation engine."""
    pass


class TextParseError(AccentError):
    """Error parsing input text."""
    pass


class ModelLoadError(AccentError):
    """Error loading model or dictionary."""
    pass


class ResolutionError(AccentError):
    """Error resolving stress position."""
    pass


# =============================================================================
# ENUMS
# =============================================================================

class StressMethod(Enum):
    """Method used to determine stress position."""
    MONO = auto()           # monosyllabic word (no ambiguity)
    YO = auto()             # letter ё is always stressed
    DICT_SINGLE = auto()    # unambiguous dictionary entry
    DICT_MULTI = auto()     # ambiguous dictionary entry (needs context)
    BERT = auto()           # BERT model for homographs
    HEURISTIC = auto()      # heuristic model for OOV words
    UNKNOWN = auto()        # could not determine
    NON_WORD = auto()       # not a Russian word (punctuation, tags, numbers)
    LLM = auto()
    MORPH = auto()


class OutputFormat(Enum):
    """Output format for accentuated text."""
    ANNOTATED = auto()      # + before stressed vowel (legacy)
    STRESS_MARK = auto()    # combining acute accent U+0301
    JSON = auto()           # full JSON structure
    RAW = auto()            # original text unchanged


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True, slots=True)
class StressPosition:
    """
    Stress position within a word.

    Attributes:
        vowel_index: which vowel is stressed (0 = first vowel)
        char_index: absolute character position within the word (0-based)
    """
    vowel_index: int
    char_index: int


@dataclass(slots=True)
class WordInfo:
    """Information about a word and its stress."""
    text: str                           # original word as in text
    start: int                          # start position in original text
    end: int                            # end position (exclusive)
    stress: Optional[StressPosition] = None
    method: StressMethod = StressMethod.UNKNOWN
    is_russian_word: bool = True
    sub_stresses: list[StressPosition] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class TextSpan:
    """A span of text — either a word or non-word (punctuation, spaces, tags)."""
    text: str
    start: int
    end: int
    is_word: bool = False
    word_info: Optional[WordInfo] = None
    is_preserved: bool = False          # text inside preserve tags


@dataclass(slots=True)
class SentenceResult:
    """Result for a single sentence."""
    original: str
    spans: list[TextSpan] = field(default_factory=list)
    words: list[WordInfo] = field(default_factory=list)

    def get_word_at(self, pos: int) -> Optional[WordInfo]:
        """Find word at given position in original text."""
        for word in self.words:
            if word.start <= pos < word.end:
                return word
        return None


@dataclass(frozen=True, slots=True)
class SSMLTag:
    """An SSML or technical tag in the text."""
    tag: str            # full tag text, e.g. '<break time="200ms"/>'
    start: int          # position in clean text (text without tags)
    end: int            # position in clean text (exclusive)
    original_start: int  # position in original text
    original_end: int    # position in original text (exclusive)
    is_preserve: bool = False  # whether content inside should be preserved


@dataclass(slots=True)
class DocumentResult:
    """Complete document processing result."""
    original_text: str
    sentences: list[SentenceResult] = field(default_factory=list)
    ssml_tags: list[SSMLTag] = field(default_factory=list)

    def to_annotated_text(self) -> str:
        """Return text with + before stressed vowel (legacy format)."""
        result = list(self.original_text)
        insertions = []
        for sent in self.sentences:
            for word in sent.words:
                if word.stress:
                    insertions.append((word.start + word.stress.char_index, '+'))
                for sub in word.sub_stresses:
                    insertions.append((word.start + sub.char_index, '+'))
        # Process right-to-left to avoid offset shifts
        insertions.sort(key=lambda x: x[0], reverse=True)
        for pos, char in insertions:
            result.insert(pos, char)
        return ''.join(result)

    def to_stress_marks(self) -> str:
        """Return text with combining acute accent U+0301."""
        result = list(self.original_text)
        insertions = []
        for sent in self.sentences:
            for word in sent.words:
                if word.stress:
                    # Insert after the stressed vowel
                    insertions.append((word.start + word.stress.char_index + 1, '\u0301'))
                for sub in word.sub_stresses:
                    insertions.append((word.start + sub.char_index + 1, '\u0301'))
        insertions.sort(key=lambda x: x[0], reverse=True)
        for pos, char in insertions:
            result.insert(pos, char)
        return ''.join(result)

    def to_json(self) -> dict:
        """Return full JSON structure with metadata."""
        return {
            "text": self.original_text,
            "ssml_tags": [
                {
                    "tag": tag.tag,
                    "start": tag.start,
                    "end": tag.end,
                    "original_start": tag.original_start,
                    "original_end": tag.original_end,
                    "is_preserve": tag.is_preserve,
                }
                for tag in self.ssml_tags
            ],
            "sentences": [
                {
                    "text": sent.original,
                    "words": [
                        {
                            "text": w.text,
                            "start": w.start,
                            "end": w.end,
                            "length": w.length,
                            "stress_vowel_index": w.stress.vowel_index if w.stress else None,
                            "stress_char_index": w.stress.char_index if w.stress else None,
                            "sub_stresses": [
                                {"vowel_index": s.vowel_index, "char_index": s.char_index}
                                for s in w.sub_stresses
                            ],
                            "method": w.method.name,
                            "is_russian_word": w.is_russian_word,
                        }
                        for w in sent.words
                    ],
                    "spans": [
                        {
                            "text": s.text,
                            "start": s.start,
                            "end": s.end,
                            "is_word": s.is_word,
                            "is_preserved": s.is_preserved,
                        }
                        for s in sent.spans
                    ],
                }
                for sent in self.sentences
            ],
        }

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AccentConfig:
    """Configuration for accentuation engine."""
    data_path: Path
    device: str = "auto"
    max_batch_tokens: int = MAX_POSITION_EMBEDDINGS
    max_sentence_len: int = MAX_SENTENCE_LEN # 510
    ssml_preserve_tags: set[str] = field(default_factory=lambda: {
        'speak', 'prosody', 'emphasis', 'voice', 'audio', 'p', 's',
    })
    ssml_void_tags: set[str] = field(default_factory=lambda: {
        'break', 'phoneme', 'mark',
    })
    output_format: OutputFormat = OutputFormat.ANNOTATED
    stress_yo: bool = True
    stress_monosyllabic: bool = False
    use_bert: bool = True
    use_heuristic: bool = True