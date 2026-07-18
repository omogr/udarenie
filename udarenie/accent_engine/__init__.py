"""
Russian text accentuation engine.

Modern API:
    from accent_engine import AccentEngine, AccentConfig

    config = AccentConfig(data_path=Path("./data"))
    engine = AccentEngine(config)

    result = engine.accentuate("Привет, мир!")
    print(result.to_annotated_text())
    print(result.to_json())

Legacy API (backward compatible):
    from accent_engine import Accentuator

    acc = Accentuator("./data")
    text = acc.accentuate("Привет, мир!")
"""
from .core import (
    AccentConfig,
    DocumentResult,
    SentenceResult,
    WordInfo,
    StressPosition,
    StressMethod,
    OutputFormat,
    SSMLTag,
    TextSpan,
    AccentError,
    TextParseError,
    ModelLoadError,
    ResolutionError,
)
from .accentuator import AccentEngine, Accentuator
from .parser import TextParser

__all__ = [
    'AccentEngine',
    'Accentuator',
    'AccentConfig',
    'DocumentResult',
    'SentenceResult',
    'WordInfo',
    'StressPosition',
    'StressMethod',
    'OutputFormat',
    'SSMLTag',
    'TextSpan',
    'TextParser',
    'AccentError',
    'TextParseError',
    'ModelLoadError',
    'ResolutionError',
]
