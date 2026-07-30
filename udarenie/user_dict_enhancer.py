"""
User Dictionary Accent Enhancer — пост-обработка ударений через пользовательский словарь.

Использование:
    from user_dict_enhancer import UserDictAccentEnhancer

    enhancer = UserDictAccentEnhancer(
        engine_or_enhancer=accent_engine,  # или morph_enhancer
        user_dict={"замок": "з+амок", "замок": "зам+ок"}
    )
    result = enhancer.accentuate("Замок был крепким.")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

from ..accent_engine import (
    DocumentResult,
    WordInfo,
    StressPosition,
    StressMethod,
)

logger = logging.getLogger(__name__)

RUSSIAN_VOWELS_LOWER = frozenset("аеёиоуыэюя")
STRESS_MARK = "\u0301"
LEGACY_STRESS_MARK = "+"  # знак + перед ударной гласной


def _vowel_positions(text: str) -> list[int]:
    """Возвращает индексы всех гласных в тексте (в нижнем регистре)."""
    return [i for i, ch in enumerate(text.lower()) if ch in RUSSIAN_VOWELS_LOWER]


class UserDictAccentEnhancer:
    """
    Обёртка над AccentEngine или MorphAccentEnhancer, которая переопределяет
    ударения на основе пользовательского словаря.

    Пользовательский словарь — это mapping: слово (в нижнем регистре) →
    форма с ударением. Форма с ударением может быть в двух форматах:
      - Legacy: "+" перед ударной гласной, например "з+амок"
      - Unicode: комбинируемый акут U+0301 после ударной, например "замо́к"

    Приоритет: пользовательский словарь выше всех остальных методов.
    """

    def __init__(
        self,
        engine_or_enhancer: Union["AccentEngine", "MorphAccentEnhancer"],
        user_dict: Optional[dict[str, str]] = None,
        user_dict_path: Optional[Union[str, Path]] = None,
    ):
        """
        Parameters
        ----------
        engine_or_enhancer : AccentEngine или MorphAccentEnhancer
            Предыдущий этап обработки.
        user_dict : dict[str, str], optional
            Словарь напрямую в виде dict.
        user_dict_path : str или Path, optional
            Путь к JSON-файлу со словарём. Формат: {"слово": "сло́во", ...}
            Можно использовать вместо или вместе с user_dict.
        """
        self._upstream = engine_or_enhancer

        self._user_dict: dict[str, str] = {}

        if user_dict is not None:
            self._user_dict.update(
                {k.lower(): v for k, v in user_dict.items()}
            )

        if user_dict_path is not None:
            self._load_from_file(Path(user_dict_path))

        logger.info(f"UserDictAccentEnhancer loaded with {len(self._user_dict)} entries")

    # ------------------------------------------------------------------
    def _load_from_file(self, path: Path) -> None:
        """Загрузить словарь из JSON-файла."""
        if not path.exists():
            raise FileNotFoundError(f"User dictionary file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"User dictionary must be a JSON object, got {type(data).__name__}")

        self._user_dict.update(
            {k.lower(): v for k, v in data.items()}
        )
        logger.info(f"Loaded {len(data)} entries from {path}")

    # ------------------------------------------------------------------
    def accentuate(self, text: str) -> DocumentResult:
        """
        Акцентуирует текст через upstream, затем переопределяет ударения
        по пользовательскому словарю.
        """
        doc = self._upstream.accentuate(text)
        self._enhance(doc)
        return doc

    # ------------------------------------------------------------------
    def _enhance(self, doc: DocumentResult) -> None:
        """Пост-обработка: замена ударений по пользовательскому словарю."""
        for sentence in doc.sentences:
            for word in sentence.words:
                if not word.is_russian_word:
                    continue

                word_lower = word.text.lower()
                stressed_form = self._user_dict.get(word_lower)
                if stressed_form is None:
                    continue

                stress = self._parse_stressed_form(word, stressed_form)
                if stress is not None:
                    word.stress = stress
                    word.method = StressMethod.DICT_SINGLE  # или можно добавить USER_DICT
                    word.sub_stresses.clear()  # очищаем под-ударения — пользователь диктует
                    logger.debug(
                        f"UserDict refined '{word.text}' → {stressed_form} "
                        f"(vowel_index={stress.vowel_index}, char_index={stress.char_index})"
                    )

    # ------------------------------------------------------------------
    def _parse_stressed_form(
        self, word: WordInfo, stressed_form: str
    ) -> Optional[StressPosition]:
        """
        Разбирает форму с ударением и возвращает StressPosition.

        Поддерживает два формата:
          - Legacy: "+" перед ударной гласной → "з+амок"
          - Unicode: U+0301 после ударной гласной → "замо́к"
        """
        # --- Формат с "+" (legacy) ---
        plus_idx = stressed_form.find(LEGACY_STRESS_MARK)
        if plus_idx >= 0:
            return self._parse_legacy_format(word, stressed_form, plus_idx)

        # --- Формат с U+0301 (unicode) ---
        stress_idx = stressed_form.find(STRESS_MARK)
        if stress_idx >= 0:
            return self._parse_unicode_format(word, stressed_form, stress_idx)

        # Ударение не найдено
        logger.warning(
            f"No stress mark found in user dict entry for '{word.text}': '{stressed_form}'"
        )
        return None

    # ------------------------------------------------------------------
    def _parse_legacy_format(
        self, word: WordInfo, stressed_form: str, plus_idx: int
    ) -> Optional[StressPosition]:
        """
        Разбирает формат "з+амок": '+' стоит ПЕРЕД ударной гласной.
        """
        # Убираем '+' для чистого сопоставления
        clean_form = stressed_form.replace(LEGACY_STRESS_MARK, "")
        word_lower = word.text.lower()

        if clean_form.lower() != word_lower:
            # Непрямое совпадение — маппим по гласным
            return self._map_by_vowels(word, clean_form, plus_idx - 1)

        # Прямое совпадение
        char_index = plus_idx  # '+' был перед гласной, значит гласная на позиции plus_idx
        # Но нужно учесть, что в оригинальном слове позиция та же
        if char_index >= len(word.text):
            return None

        word_vowels = _vowel_positions(word_lower)
        try:
            vowel_index = word_vowels.index(char_index)
        except ValueError:
            return None

        return StressPosition(vowel_index=vowel_index, char_index=char_index)

    # ------------------------------------------------------------------
    def _parse_unicode_format(
        self, word: WordInfo, stressed_form: str, stress_idx: int
    ) -> Optional[StressPosition]:
        """
        Разбирает формат "замо́к": U+0301 стоит ПОСЛЕ ударной гласной.
        """
        # Убираем все знаки ударения
        clean_form = stressed_form.replace(STRESS_MARK, "")
        word_lower = word.text.lower()

        # Сколько знаков ударения было до найденного
        num_marks_before = stressed_form[:stress_idx].count(STRESS_MARK)

        # Позиция ударной гласной в чистой строке
        stressed_char_in_clean = stress_idx - 1 - num_marks_before

        if stressed_char_in_clean < 0 or stressed_char_in_clean >= len(clean_form):
            return None

        if clean_form.lower() == word_lower:
            # Прямое совпадение
            char_index = stressed_char_in_clean
        else:
            # Непрямое совпадение — маппим по гласным
            form_vowels = _vowel_positions(clean_form.lower())
            word_vowels = _vowel_positions(word_lower)

            if len(form_vowels) != len(word_vowels):
                logger.warning(
                    f"Vowel count mismatch for '{word.text}' ({len(word_vowels)}) "
                    f"vs user dict entry '{stressed_form}' ({len(form_vowels)})"
                )
                return None

            try:
                stressed_vowel_idx = form_vowels.index(stressed_char_in_clean)
            except ValueError:
                return None

            if stressed_vowel_idx >= len(word_vowels):
                return None

            char_index = word_vowels[stressed_vowel_idx]

        # Проверяем валидность индекса
        if char_index >= len(word.text):
            return None

        word_vowels = _vowel_positions(word_lower)
        try:
            vowel_index = word_vowels.index(char_index)
        except ValueError:
            return None

        return StressPosition(vowel_index=vowel_index, char_index=char_index)

    # ------------------------------------------------------------------
    def _map_by_vowels(
        self, word: WordInfo, clean_form: str, stressed_vowel_pos_in_form: int
    ) -> Optional[StressPosition]:
        """
        Маппит ударение по порядковому номеру гласной, когда формы
        не совпадают посимвольно (например, "ё" vs "е").
        """
        form_vowels = _vowel_positions(clean_form.lower())
        word_vowels = _vowel_positions(word.text.lower())

        if len(form_vowels) != len(word_vowels):
            logger.warning(
                f"Vowel count mismatch for '{word.text}' ({len(word_vowels)}) "
                f"vs '{clean_form}' ({len(form_vowels)})"
            )
            return None

        if stressed_vowel_pos_in_form < 0 or stressed_vowel_pos_in_form >= len(form_vowels):
            return None

        stressed_vowel_idx = stressed_vowel_pos_in_form
        if stressed_vowel_idx >= len(word_vowels):
            return None

        char_index = word_vowels[stressed_vowel_idx]
        return StressPosition(vowel_index=stressed_vowel_idx, char_index=char_index)

    # ------------------------------------------------------------------
    # Удобные методы для управления словарём
    # ------------------------------------------------------------------

    def add_entry(self, word: str, stressed_form: str) -> None:
        """Добавить или обновить запись в словаре."""
        self._user_dict[word.lower()] = stressed_form

    def remove_entry(self, word: str) -> bool:
        """Удалить запись из словаря. Возвращает True, если запись была удалена."""
        return self._user_dict.pop(word.lower(), None) is not None

    def has_entry(self, word: str) -> bool:
        """Проверить наличие слова в словаре."""
        return word.lower() in self._user_dict

    def get_entry(self, word: str) -> Optional[str]:
        """Получить запись для слова."""
        return self._user_dict.get(word.lower())

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Сохранить текущий словарь в JSON-файл."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._user_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved user dictionary ({len(self._user_dict)} entries) to {path}")