"""
Morph Accent Enhancer — пост-обработка ударений через морфологический словарь
и Natasha NewsMorphTagger.

Использование:
    from accent_engine import AccentEngine, AccentConfig
    from morph_enhancer import MorphAccentEnhancer
    from morph_stress_finder import MorphStressFinder

    engine = AccentEngine(AccentConfig(data_path=Path("...")))
    finder = MorphStressFinder("kaikki-forms.jsonl")
    enhancer = MorphAccentEnhancer(engine, finder)

    result = enhancer.accentuate("Руки как ноги.")
    print(result.to_stress_marks())
"""

from __future__ import annotations

import logging
from typing import Optional

from ..accent_engine import (
    AccentEngine,
    DocumentResult,
    WordInfo,
    StressPosition,
    StressMethod,
)

from .morph_stress_finder import MorphStressFinder

logger = logging.getLogger(__name__)

RUSSIAN_VOWELS_LOWER = frozenset("аеёиоуыэюя")

STRESS_MARK = "\u0301"


def _vowel_positions(text: str) -> list[int]:
    """Возвращает индексы всех гласных в тексте (в нижнем регистре)."""
    return [i for i, ch in enumerate(text.lower()) if ch in RUSSIAN_VOWELS_LOWER]


class MorphAccentEnhancer:
    """
    Обёртка над AccentEngine, которая уточняет ударения через MorphStressFinder.

    Работает в два этапа:
      1. Быстрая акцентуация через AccentEngine.
      2. Пост-обработка: для слов, по которым MorphStressFinder даёт
         ровно один вариант ударения, заменяем результат на этот вариант.
         Во всех остальных случаях (0 или >1 вариантов) оставляем результат
         AccentEngine.
    """

    def __init__(
        self,
        accent_engine: AccentEngine,
        stress_finder: MorphStressFinder,
    ):
        self.accent_engine = accent_engine
        self.stress_finder = stress_finder

        # Пробуем использовать StressMethod.MORPH, если доступен в core.py
        try:
            self._morph_method = StressMethod.MORPH
        except AttributeError:
            self._morph_method = StressMethod.DICT_SINGLE
            logger.debug(
                "StressMethod.MORPH не найден — используем DICT_SINGLE как fallback"
            )

    # ------------------------------------------------------------------
    def accentuate(self, text: str) -> DocumentResult:
        """
        Акцентуирует текст с помощью AccentEngine, затем уточняет
        однозначные ударения через MorphStressFinder.
        """
        doc = self.accent_engine.accentuate(text)
        self._enhance(doc, text)
        return doc

    # ------------------------------------------------------------------
    def _enhance(self, doc: DocumentResult, text: str) -> None:
        """Пост-обработка: замена ударений там, где MorphAccentEnhancer даёт 1 вариант."""
        try:
            finder_results = self.stress_finder.find_stress(text)
        except Exception as exc:
            logger.warning(f"MorphStressFinder failed: {exc}")
            return

        # Индексируем результаты Natasha по абсолютной позиции в тексте
        morph_by_pos: dict[tuple[int, int], list[str]] = {}
        for token in finder_results:
            start = token.get("start")
            end = token.get("end")
            if start is None or end is None:
                continue
            morph_by_pos[(start, end)] = token.get("stress_options", [])

        # Проходим по словам из accent_engine и сопоставляем по (start, end)
        for sentence in doc.sentences:
            for word in sentence.words:
                if not word.is_russian_word:
                    continue

                options = morph_by_pos.get((word.start, word.end))
                if not options:
                    continue
                if len(options) != 1:
                    # 0 или >1 вариантов — доверяем accent_engine
                    continue

                stressed_form = options[0]
                stress = self._parse_stressed_form(word, stressed_form)
                if stress is not None:
                    word.stress = stress
                    word.method = self._morph_method
                    logger.debug(
                        f"Morph refined '{word.text}' → {stressed_form} "
                        f"(vowel_index={stress.vowel_index}, char_index={stress.char_index})"
                    )

    # ------------------------------------------------------------------
    def _parse_stressed_form(
        self, word: WordInfo, stressed_form: str
    ) -> Optional[StressPosition]:
        """
        Преобразует словарную форму вида 'дабы́' в StressPosition
        для оригинального слова (с учётом регистра и ё/е).

        Ударный знак U+0301 в Morph стоит ПОСЛЕ ударной гласной.
        """
        stress_idx = stressed_form.find(STRESS_MARK)
        if stress_idx < 0:
            return None

        # Убираем все знаки ударения для чистого сопоставления
        clean_form = stressed_form.replace(STRESS_MARK, "")
        word_lower = word.text.lower()

        # Сколько знаков ударения было до найденного (влияет на смещение)
        num_marks_before = stressed_form[:stress_idx].count(STRESS_MARK)

        # Базовый символ (ударная гласная) в оригинале находится перед знаком ударения.
        # После удаления всех combining marks до этого знака позиция в clean строке:
        stressed_char_in_clean = stress_idx - 1 - num_marks_before

        if stressed_char_in_clean < 0 or stressed_char_in_clean >= len(clean_form):
            return None

        if clean_form.lower() == word_lower:
            # Прямое совпадение — позиция гласной в word.text совпадает
            char_index = stressed_char_in_clean
        else:
            # Непрямое совпадение: маппим по порядку гласных
            form_vowels = _vowel_positions(clean_form.lower())
            word_vowels = _vowel_positions(word_lower)

            if len(form_vowels) != len(word_vowels):
                logger.warning(
                    f"Vowel count mismatch for '{word.text}' ({len(word_vowels)}) "
                    f"vs '{stressed_form}' ({len(form_vowels)})"
                )
                return None

            try:
                stressed_vowel_idx = form_vowels.index(stressed_char_in_clean)
            except ValueError:
                return None

            if stressed_vowel_idx >= len(word_vowels):
                return None

            char_index = word_vowels[stressed_vowel_idx]

        # Проверяем, что индекс валиден для оригинального слова
        if char_index >= len(word.text):
            return None

        # Считаем vowel_index (порядковый номер гласной в word.text)
        word_vowels = _vowel_positions(word_lower)
        try:
            vowel_index = word_vowels.index(char_index)
        except ValueError:
            return None

        return StressPosition(vowel_index=vowel_index, char_index=char_index)
  
  
