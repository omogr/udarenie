# Рефакторинг модулей расстановки ударений

## Структура нового пакета

```
accent_engine/
├── __init__.py          # Публичное API
├── core.py              # Типы, константы, исключения
├── parser.py            # Разбор текста + SSML-теги
├── tokenizer.py         # Обертка над BERT-токенизатором
├── batcher.py           # Группировка предложений в батчи
├── resolvers.py         # Resolver'ы ударений (Strategy pattern)
├── formatters.py        # Форматтеры вывода
└── accentuator.py       # Фасад + backward-compatible API
```

## Ключевые улучшения

### 1. Модульная архитектура
- Каждый модуль отвечает за одну задачу (SRP)
- Чёткие интерфейсы между компонентами
- Возможность тестирования каждого компонента отдельно

### 2. Типизация
- Полные type hints
- dataclasses вместо namedtuple
- Enum для методов и форматов

### 3. Новая модель данных
```python
@dataclass
class WordInfo:
    text: str                    # исходное слово
    start: int                   # позиция в тексте
    end: int                     # конец (exclusive)
    stress: Optional[StressPosition]
    method: StressMethod         # как определено ударение
    is_russian_word: bool
```

### 4. Поддержка SSML/технических тегов
```python
# Теги prosody — содержимое обрабатывается
<prosody rate="slow">Привет</prosody>

# Теги break — удаляются из текста
Привет <break time="200ms"/> мир!

# Кастомные preserve-теги — содержимое сохраняется как есть
<custom>не обрабатывать</custom>
```

### 5. Множественные форматы вывода
- `OutputFormat.ANNOTATED` — `+перед_ударной` (legacy)
- `OutputFormat.STRESS_MARK` — знак ударения U+0301
- `OutputFormat.JSON` — полная структура с метаданными

### 6. Strategy Pattern для resolver'ов
```python
resolvers = [
    MonosyllableResolver(),      # односложные слова
    YoResolver(),                 # буква ё
    DictionaryResolver(dict),   # словарь
    BERTResolver(model),          # BERT для омонимов
    HeuristicResolver(model),     # эвристики для OOV
]
```
Каждый resolver:
- Независимый
- Тестируемый
- Легко заменяемый

## Новое API

### Modern API (рекомендуется)
```python
from accent_engine import AccentEngine, AccentConfig
from pathlib import Path

config = AccentConfig(
    data_path=Path("./data"),
    device="cuda",
    ssml_preserve_tags={"prosody", "emphasis"},
    ssml_void_tags={"break", "phoneme"},
)

engine = AccentEngine(config)

# Обработка текста
result = engine.accentuate("Привет, мир!")

# Разные форматы вывода
print(result.to_annotated_text())   # Приве+т, ми+р!
print(result.to_stress_marks())      # Привéт, мир!
print(result.to_json())              # полная структура

# С SSML-тегами
ssml = '<speak>Привет, <break time="200ms"/> мир!</speak>'
result = engine.accentuate(ssml)
```

### Legacy API (backward compatible)
```python
from accent_engine import Accentuator

acc = Accentuator("./data")

# Все старые методы работают
text = acc.accentuate("Привет, мир!")
texts = acc.accentuate(["Привет!", "Мир!"])
words = acc.accentuate_by_words(["Привет, мир!"])
```

## Миграция словаря

Новый формат поддерживает исходный `.vcb` файл напрямую:

```python
# Старый формат (pickle)
(vocab: dict[str, int], vocab_index: dict[int, list[int]])

# Новый формат — загружается из wav2vec_words2.vcb
class DictionaryEntry:
    word: str
    stress_positions: tuple[int, ...]
    stress_vowels: tuple[int, ...]
    variants: dict[str, float]   # оригинальные варианты с весами
```

## Исправленные баги

1. **Name mismatch** в namedtuple — исправлено (dataclasses)
2. **Опечатка** `easy_sentences.append` → `self.easy_sentences` — исправлено
3. **Dead code** `assert False` — удалено
4. **check_ee_comu('кому')** возвращал 3 вместо 1 — исправлено в YoResolver
5. **Ручной softmax** → `torch.nn.functional.softmax`
6. **Ручной argmax** → `torch.argmax`

## Тестирование

```python
# Unit-тесты для каждого resolver'а
def test_yo_resolver():
    resolver = YoResolver()
    word = WordInfo(text="ёлка", start=0, end=4)
    assert resolver.can_resolve(word)
    stress = resolver.resolve(word, SentenceResult("ёлка"))
    assert stress.char_index == 0  # ё на позиции 0

def test_monosyllable_resolver():
    resolver = MonosyllableResolver()
    word = WordInfo(text="кот", start=0, end=3)
    assert resolver.can_resolve(word)
    stress = resolver.resolve(word, SentenceResult("кот"))
    assert stress.vowel_index == 0
```
