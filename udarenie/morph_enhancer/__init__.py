"""
Wiktionary Accent Enhancer — пост-обработка ударений через Wiktionary (kaikki.org)
и Natasha NewsMorphTagger.

Пакет работает поверх accent_engine и не требует изменений в её ядре
(достаточно добавить одну строку `MORPH = auto()` в StressMethod для красоты).
"""

from .morph_enhancer import MorphAccentEnhancer

# Re-export для удобства: пользователь может делать
#   from morph_enhancer import MorphStressFinder
# вместо отдельного импорта из morph_stress_finder.
try:
    from .morph_stress_finder import MorphStressFinder
except ImportError:
    # fallback: модуль может лежать рядом, но не внутри пакета
    try:
        from morph_stress_finder import MorphStressFinder
    except ImportError:
        MorphStressFinder = None  # type: ignore[misc]

__all__ = [
    "MorphAccentEnhancer",
    "MorphStressFinder",
]