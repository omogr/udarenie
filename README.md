# udarenie

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Библиотека для автоматической расстановки ударений в текстах на русском языке.

`udarenie` сочетает BERT-модель, морфологический анализ (Natasha) и пользовательский словарь для высококачественной акцентуации, включая разрешение омографов по контексту.

## Возможности

- **BERT + эвристика** — базовая разметка ударений с контекстным разрешением омографов
- **Морфологический enhancer** — уточнение через словарь Natasha/Morph
- **Пользовательский словарь** — переопределение ударений для конкретных слов
- **SSML** — корректная обработка SSML-разметки
- **Несколько форматов вывода** — `+` перед ударной, Unicode combining acute, JSON, список слов
- **Оптимизация** — BERT запускается только для предложений с омографами

## Установка

```bash
pip install udarenie
```

При первом запуске данные (модель BERT, словари) автоматически скачаются в `~/.cache/udarenie/data`. Параметр data_path у load_accentor позволяет изменить путь, по которому скачиваются данные.

## Быстрый старт

```python
from udarenie import load_accentor

accentor = load_accentor()
result = accentor("Меня зовут Лёва.")
print(result)  # Мен+я зов+ут Л+ёва.
```

### Разные форматы вывода

```python
doc = accentor.accentuate("Меня зовут Лёва.")

# Текст с + перед ударной гласной
print(doc.to_annotated_text())   # Мен+я зов+ут Л+ёва.

# Текст с комбинируемым акутом U+0301
print(doc.to_stress_marks())     # Меня́ зову́т Лёва.

# Полная JSON-структура с метаданными
print(doc.to_json())
```

## Архитектура

Библиотека работает по цепочке из трёх этапов:

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ AccentEngine │  →  │ MorphAccent     │  →  │ UserDictAccent   │
│ (BERT-based) │     │ Enhancer        │     │ Enhancer         │
└──────────────┘     └─────────────────┘     └──────────────────┘
   Базовая разметка    Морфологическая         Пользовательские
                       коррекция               переопределения
```

Приоритет: **UserDict > Morph > BERT**. Каждый следующий этап может переопределить ударение, установленное предыдущим.

## Конфигурация

### Параметры `load_accentor()`

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `use_morph` | `bool` | `True` | Использовать морфологический enhancer |
| `user_dict` | `dict \| str \| Path` | `None` | Пользовательский словарь |
| `data_dir` | `str \| Path` | `~/.cache/...` | Путь к данным |
| `auto_download` | `bool` | `True` | Автоматически скачивать данные |
| `force_download` | `bool` | `False` | Принудительно перекачать данные |

### Параметры движка (`**engine_kwargs`)

```python
accentor = load_accentor(
    use_morph=True,
    device="cuda",              # "auto", "cpu", "cuda"
    max_batch_tokens=512,       # Макс. токенов в батче
    max_sentence_len=510,       # Макс. длина предложения
    stress_monosyllabic=False,  # Ударять односложные слова
    use_bert=True,              # Использовать BERT для омографов
    use_heuristic=True,         # Использовать эвристику для OOV
)
```

## Пользовательский словарь

Позволяет переопределить ударение для любого слова. Имеет наивысший приоритет.

### Загрузка из dict

```python
accentor = load_accentor(
    user_dict={
        "замок": "з+амок",      # формат с +
        "коса": "кос+а",        # формат с +
        "мука": "му́ка",         # формат с U+0301
    }
)
```

### Загрузка из файла

```python
# user_dict.json
# {"замок": "з+амок", "коса": "кос+а"}

accentor = load_accentor(user_dict="path/to/user_dict.json")
```

### Два формата ударения

| Формат | Пример | Описание |
|--------|--------|----------|
| Legacy | `"з+амок"` | `+` перед ударной гласной |
| Unicode | `"замо́к"` | Combining acute `U+0301` после ударной |

### Управление словарём программно

```python
from udarenie.user_dict_enhancer import UserDictAccentEnhancer

# Получить enhancer (после load_accentor)
enhancer = accentor._user_dict_enhancer

# Добавить запись
enhancer.add_entry("вёдра", "в+ёдра")

# Удалить запись
enhancer.remove_entry("замок")

# Сохранить в файл
enhancer.save_to_file("my_dict.json")

# Проверить наличие
print(enhancer.has_entry("замок"))
```

## SSML-поддержка

Библиотека корректно обрабатывает SSML-разметку:

```python
text = '<speak>Привет, <break time="200ms"/> мир!</speak>'
result = accentor.accentuate(text)
```

Сохраняемые теги (содержимое обрабатывается): `speak`, `prosody`, `emphasis`, `voice`, `audio`, `p`, `s`

Пустые теги (игнорируются при обработке): `break`, `phoneme`, `mark`

## Формат выходных данных

Краткий обзор структуры `DocumentResult`:

```python
doc = accentor.accentuate("Меня зовут Лёва.")

for sent in doc.sentences:
    for word in sent.words:
        print(f"{word.text}: {word.stress}, метод={word.method.name}")
# Меня: StressPosition(vowel_index=1, char_index=2), метод=BERT
# зовут: StressPosition(vowel_index=1, char_index=2), метод=DICT_SINGLE
# Лёва: StressPosition(vowel_index=0, char_index=1), метод=YO
```

### Основные типы

```python
@dataclass
class DocumentResult:
    original_text: str                  # Исходный текст
    sentences: list[SentenceResult]     # Список предложений
    ssml_tags: list[SSMLTag]            # SSML-теги (если есть)

@dataclass
class SentenceResult:
    original: str                       # Исходное предложение
    spans: list[TextSpan]               # Все спаны (слова + пунктуация)
    words: list[WordInfo]               # Только слова

@dataclass
class WordInfo:
    text: str                           # Слово как в тексте
    start: int                          # Начальная позиция в исходном тексте
    end: int                            # Конечная позиция (исключая)
    stress: Optional[StressPosition]    # Основное ударение
    method: StressMethod                # Метод определения ударения
    is_russian_word: bool               # Является ли русским словом
    sub_stresses: list[StressPosition]  # Дополнительные ударения

@dataclass(frozen=True)
class StressPosition:
    vowel_index: int    # Какая по счёту гласная ударная (0 = первая)
    char_index: int     # Абсолютная позиция символа в слове (0-based)
```

### Методы определения ударения (`StressMethod`)

| Значение | Описание |
|----------|----------|
| `MONO` | Односложное слово |
| `YO` | Буква `ё` всегда ударная |
| `DICT_SINGLE` | Однозначная запись в словаре |
| `DICT_MULTI` | Многозначная запись (требуется контекст) |
| `BERT` | BERT-модель для омографов |
| `HEURISTIC` | Эвристика для OOV-слов |
| `MORPH` | Морфологический анализатор (Natasha) |
| `USER_DICT` | Пользовательский словарь |
| `UNKNOWN` | Не удалось определить |
| `NON_WORD` | Не русское слово |

## Продвинутое использование

### Использование компонентов по отдельности

```python
from udarenie.accent_engine import AccentEngine, AccentConfig
from udarenie.morph_enhancer import MorphAccentEnhancer, MorphStressFinder
from udarenie.user_dict_enhancer import UserDictAccentEnhancer

config = AccentConfig(data_path=Path("..."))
engine = AccentEngine(config)

finder = MorphStressFinder("morph.pq")
morph = MorphAccentEnhancer(engine, finder)

user = UserDictAccentEnhancer(morph, user_dict={"замок": "з+амок"})

result = user.accentuate("Замок был крепким.")
```

### Создание собственного enhancer'а

```python
from udarenie import DocumentResult, StressMethod

class LoggingEnhancer:
    def __init__(self, upstream):
        self._upstream = upstream

    def accentuate(self, text: str) -> DocumentResult:
        doc = self._upstream.accentuate(text)
        for sent in doc.sentences:
            for word in sent.words:
                if word.stress:
                    print(f"{word.text}: {word.stress.vowel_index}")
        return doc
```

## Обработка ошибок

```python
from udarenie import AccentError, ModelLoadError, TextParseError

try:
    accentor = load_accentor()
except ModelLoadError as e:
    print(f"Не удалось загрузить модель: {e}")
```

## Лицензия

MIT License. См. [LICENSE](LICENSE).
