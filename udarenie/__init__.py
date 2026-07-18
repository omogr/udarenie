"""
Russian Accentor — Unified Russian text accentuation library.

Combines accent_engine (base accentuation) with morph_enhancer
(post-processing via Morph) for high-quality stress placement.

Quick start:
    from russian_accentor import load_accentor
    accentor = load_accentor()
    print(accentor("Меня зовут Лёва."))

Or with explicit configuration:
    accentor = load_accentor(
        use_morph=True,
        data_dir="/path/to/data",
        auto_download=True
    )
"""

from pathlib import Path
from typing import Optional, Union
import logging

from .data_manager import ensure_data, get_default_data_dir, update_data

__version__ = "1.0.0"
__all__ = [
    "load_accentor",
    "Accentor",
    "ensure_data",
    "update_data",
    "get_default_data_dir",
]

logger = logging.getLogger(__name__)


class Accentor:
    """
    Unified accentor callable. Wraps either AccentEngine alone or
    AccentEngine + MorphAccentEnhancer.
    """

    def __init__(self, engine, enhancer=None):
        self._engine = engine
        self._enhancer = enhancer

    def __call__(self, text: str) -> str:
        """Accentuate text and return annotated string (+ before stressed vowel)."""
        return self.accentuate(text).to_annotated_text()

    def accentuate(self, text: str):
        """Return full DocumentResult object with metadata."""
        if self._enhancer is not None:
            return self._enhancer.accentuate(text)
        return self._engine.accentuate(text)

    def to_text(self, text: str) -> str:
        """Return text with + before stressed vowel (legacy format)."""
        return self.accentuate(text).to_annotated_text()

    def to_stress_marks(self, text: str) -> str:
        """Return text with combining acute accent U+0301."""
        return self.accentuate(text).to_stress_marks()

    def to_json(self, text: str) -> dict:
        """Return full JSON structure with metadata."""
        return self.accentuate(text).to_json()

    def to_word_list(self, text: str) -> list:
        """Return list of (punctuation, word) tuples (legacy format)."""
        from .accent_engine.formatters import WordListFormatter
        formatter = WordListFormatter()
        return formatter.format(self.accentuate(text))


def load_accentor(
    use_morph: bool = True,
    data_dir: Optional[Union[str, Path]] = None,
    auto_download: bool = True,
    force_download: bool = False,
    **engine_kwargs
) -> Accentor:
    """
    Load and return a unified accentor.

    Parameters
    ----------
    use_morph : bool, default True
        If True, use morph enhancer for better quality.
        If False, use only accent_engine (faster, lower quality).
    data_dir : str or Path, optional
        Path to data directory. If None, uses default cache location
        (~/.cache/russian_accentor/data).
    auto_download : bool, default True
        If True, download data automatically if missing.
    force_download : bool, default False
        If True, re-download data even if already present (useful for updates).
    **engine_kwargs
        Additional arguments passed to AccentConfig (e.g. device, use_bert).

    Returns
    -------
    Accentor
        Callable accentor instance.

    Examples
    --------
    >>> accentor = load_accentor()
    >>> accentor("Меня зовут Лёва.")
    'Мен+я зов+ут Л+ёва.'

    >>> accentor = load_accentor(use_morph=False)
    >>> accentor("Меня зовут Лёва.")
    'Мен+я зов+ут Л+ёва.'
    """
    
    if data_dir is None:
        data_dir = get_default_data_dir()
    else:
        data_dir = Path(data_dir)
    print('data_dir', data_dir)

    # Ensure data is available
    if auto_download or force_download:
        ensure_data(data_dir, force=force_download)

    # Paths to specific data
    accent_engine_data = data_dir / "accent_engine"
    morph_data = data_dir / "morph_enhancer" / "morph.pq"

    # Validate data exists
    if not accent_engine_data.exists():
        raise FileNotFoundError(
            f"AccentEngine data not found at {accent_engine_data}. "
            f"Call russian_accentor.ensure_data() to download."
        )

    # Import lazily to avoid heavy imports at package level
    from .accent_engine import AccentEngine, AccentConfig
    config = AccentConfig(data_path=accent_engine_data, **engine_kwargs)
    engine = AccentEngine(config)

    if use_morph:
        if not morph_data.exists():
            raise FileNotFoundError(
                f"Morph data not found at {morph_data}. "
                f"Call russian_accentor.ensure_data() to download."
            )
        from .morph_enhancer import MorphAccentEnhancer, MorphStressFinder
        finder = MorphStressFinder(str(morph_data))
        enhancer = MorphAccentEnhancer(engine, finder)
        return Accentor(engine, enhancer)

    return Accentor(engine)
