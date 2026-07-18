import json
from collections import defaultdict
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
import pyarrow.parquet as pq

class MorphStressFinder:
    """
    Находит ударения в русском тексте, сопоставляя морфологические атрибуты
    Natasha NewsMorphTagger с данными словаря.

    Алгоритм:
    1. При загрузке словаря: для каждой словоформы с ударением строим индекс
       нормализованная_форма → список (pos, morph_tags, stressed_form).
    2. При обработке текста: для каждого токена ищем его в индексе,
       фильтруем наборы по совместимости с Natasha-атрибутами.
    3. Возвращаем все допустимые варианты ударений или пустой список.
    """

    # --- Сопоставление POS Natasha → морфологический словарь ---
    POS_MAP = {
        'ADJ':   'adjective',
        'ADV':   'adverb',
        'NOUN':  'noun',
        'VERB':  'verb',
        'PRON':  'pronoun',
        'ADP':   'preposition',
        'SCONJ': 'conjunction',
        'CCONJ': 'conjunction',
        'PART':  'particle',
        'INTJ':  'intj',
        'NUM':   'numeral',
        'DET':   'det',
        'PROPN': 'name',
    }

    # --- Сопоставление feats Natasha → морфологический словарь ---
    FEAT_MAP = {
        'Number': {
            'Sing': 'singular',
            'Plur': 'plural',
        },
        'Gender': {
            'Masc': 'masculine',
            'Fem':  'feminine',
            'Neut': 'neuter',
        },
        'Case': {
            'Nom': 'nominative',
            'Acc': 'accusative',
            'Gen': 'genitive',
            'Dat': 'dative',
            'Ins': 'instrumental',
            'Loc': {'prepositional', 'locative'},
            'Voc': 'vocative',
            'Par': 'partitive',
        },
        'Tense': {
            'Past': 'past',
            'Pres': 'present',
            'Fut':  'future',
        },
        'Aspect': {
            'Perf': 'perfective',
            'Imp':  'imperfective',
        },
        'Mood': {
            'Imp': 'imperative',
        },
        'VerbForm': {
            'Inf':  'infinitive',
            'Part': 'participle',
            'Conv': 'adverbial',
        },
        'Voice': {
            'Act': 'active',
            'Pass': 'passive',
            'Mid':  'reflexive',
        },
        'Person': {
            '1': 'first-person',
            '2': 'second-person',
            '3': 'third-person',
        },
        'Animacy': {
            'Anim': 'animate',
            'Inan': 'inanimate',
        },
        'Degree': {
            'Cmp': 'comparative',
            'Sup': 'superlative',
        },
        'Variant': {
            'Short': 'short-form',
        },
    }

    # --- Теги морфологического словаря, которые НЕ относятся к морфологическим атрибутам ---
    NON_MORPH_TAGS = {
        'canonical', 'romanization', 'inflection-template', 'table-tags', 'class',
        'alternative', 'error-unrecognized-form', 'error-unknown-tag',
        'dated', 'rare', 'colloquial', 'poetic', 'obsolete', 'uncommon',
        'proscribed', 'archaic', 'dialectal', 'nonstandard', 'endearing',
        'clipping', 'literary', 'sometimes', 'demonym', 'common-gender',
        'standard', 'stressed', 'unstressed', 'Internet', 'pronunciation-spelling',
        'usually', 'also', 'Middle', 'Russian', 'no-perfect', 'no-short-form',
        'no-table-tags', 'abstract-noun', 'diminutive', 'augmentative', 'pejorative',
        'relational', 'adjective', 'adverb', 'noun-from-verb', 'emphatic', 'collective',
        'count-form', 'semelfactive', 'paucal', 'abbreviation', 'regional', 'Pskov',
        'by-personal-gender', 'common',
    }

    # --- Категории для проверки конфликтов ---
    _CATEGORIES = {
        'number':   {'singular', 'plural'},
        'gender':   {'masculine', 'feminine', 'neuter'},
        'case':     {'nominative', 'accusative', 'genitive', 'dative',
                     'instrumental', 'prepositional', 'locative', 'vocative', 'partitive'},
        'tense':    {'past', 'present', 'future'},
        'aspect':   {'perfective', 'imperfective'},
        'person':   {'first-person', 'second-person', 'third-person'},
        'animacy':  {'animate', 'inanimate'},
        'voice':    {'active', 'passive', 'reflexive'},
        'verbform': {'infinitive', 'participle', 'adverbial', 'imperative', 'short-form'},
        'degree':   {'comparative', 'superlative'},
    }

    def __init__(self, morph_path: str):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

        # Индекс: нормализованная_форма → [Record(pos, morph_tags, stressed_form)]
        self.index = defaultdict(list)
        self._load_morph_data(morph_path)

    # --------------------------------------------------------------------- #
    #  Внутренние утилиты
    # --------------------------------------------------------------------- #
    @staticmethod
    def _normalize(text: str) -> str:
        """Lower-case, без ударений, ё→е."""
        if not text:
            return ''
        return (text.lower()
                .replace('\u0301', '')
                .replace('\u0300', '')
                .replace('ё', 'е')
                .replace('Ё', 'Е'))

    @classmethod
    def _is_morph_tag(cls, tag: str) -> bool:
        return tag not in cls.NON_MORPH_TAGS

    @staticmethod
    def _has_stress(form: str) -> bool:
        """Проверяет наличие знака ударения U+0301."""
        return '\u0301' in form

    def _natasha_feats_to_morph(self, feats: dict) -> set:
        """Преобразует Natasha feats → множество тегов морфологического словаря."""
        morph_tags = set()
        for feat, value in (feats or {}).items():
            mapping = self.FEAT_MAP.get(feat)
            if not mapping:
                continue
            mapped = mapping.get(value)
            if not mapped:
                continue
            if isinstance(mapped, set):
                morph_tags.update(mapped)
            else:
                morph_tags.add(mapped)
        return morph_tags

    def _is_compatible(self, token_tags: set, form_tags: set) -> bool:
        """
        Проверяет совместимость двух наборов тегов.
        Возвращает False, если есть конфликт внутри одной категории.
        """
        for cat_vals in self._CATEGORIES.values():
            tok = token_tags & cat_vals
            frm = form_tags & cat_vals
            if tok and frm and not (tok & frm):
                return False
        return True

    # --------------------------------------------------------------------- #
    #  Загрузка morph
    # --------------------------------------------------------------------- #
    def _load_morph_data(self, path: str):
        """
        Загружает данные морфологического словаря и строит индекс по нормализованным словоформам.
        Для каждой формы с ударением сохраняем: (pos, morph_tags, stressed_form).
        """
        table = pq.read_table(path)
    
        keys = table.column(0).to_pylist()
        values = table.column(1).to_pylist()
        self.index = dict(zip(keys, values))


    def _load_morph_data_from_jsonl(self, path: str):
        """
        Загружает morph jsonl и строит индекс по нормализованным словоформам.
        Для каждой формы с ударением сохраняем: (pos, morph_tags, stressed_form).
        """
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                norm, value = entry
                self.index[norm] = [{
                            'pos': tv['pos'],
                            'morph_tags': set(tv['morph_tags']),
                            'stressed_form': tv['stressed_form'],
                        } for tv in value]

    # --------------------------------------------------------------------- #
    #  Публичный API
    # --------------------------------------------------------------------- #
    def find_stress(self, text: str) -> list[dict]:
        """
        Анализирует текст и возвращает список токенов с вариантами ударений.

        Каждый элемент — словарь:
            text          : исходное слово
            pos           : POS из Natasha
            feats         : feats из Natasha (dict)
            stress_options: список строк (форм с ударением из морфологического словаря)
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        results = []

        for token in doc.tokens:
            result = result = {
                'text': token.text,
                'start': token.start,
                'end': token.stop,
                'pos': token.pos,
                'feats': dict(token.feats) if token.feats else {},
                'stress_options': [],
            }


            # Пропускаем не-слова
            if token.pos in ('PUNCT', 'SYM', 'X'):
                results.append(result)
                continue

            # Преобразуем POS Natasha → POS из морфологического словаря
            morph_pos = self.POS_MAP.get(token.pos, token.pos.lower())

            # Преобразуем feats
            token_tags = self._natasha_feats_to_morph(
                dict(token.feats) if token.feats else {}
            )

            # Ищем словоформу в индексе
            norm_token = self._normalize(token.text)
            
            if norm_token in self.index:
                records = json.loads(self.index[norm_token])

                # Фильтруем по совместимости
                seen = set()
                for rec in records:
                    # Проверяем POS
                    if rec['pos'] != morph_pos:
                        continue

                    # Проверяем совместимость морфологических атрибутов
                    if not self._is_compatible(token_tags, set(rec['morph_tags'])):
                        continue
                        
                    stressed = rec['stressed_form']
                    if stressed not in seen:
                        result['stress_options'].append(stressed)
                        seen.add(stressed)

                results.append(result)

        return results

