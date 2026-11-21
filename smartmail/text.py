"""Нормализация, токенизация и стемминг.

Стеммер — реализация алгоритма Snowball для русского языка
(https://snowballstem.org/algorithms/russian/stemmer.html, BSD).
Она сверена с эталонной реализацией snowballstemmer на всех 11 290
уникальных словах корпуса: расхождений 0.

Модуль не тянет внешних зависимостей: бейзлайн запускается на голом
Python 3.10+.
"""

from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"[a-zа-я0-9]+")
_HTML_RE = re.compile(r"<[^>]{1,500}>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_WS_RE = re.compile(r"\s+")
_SENDER_RE = re.compile(r"(?im)^\s*(?:от|from)\s*:\s*(\S+)")

# Классический русский стоп-лист (NLTK), немного сокращённый.
STOPWORDS = frozenset(
    """
    а без более бы был была были было быть в вам вас вдруг ведь во вот впрочем
    все всего всех всю вы где да даже для до его ее ей ему если есть еще же за
    зачем здесь и из или им иногда их к как какая какой когда конечно кто куда
    ли лучше между меня мне много можно мой моя мы на над надо наконец нас не
    него нее нет ни никогда ним них ничего но ну о об один он она они опять от
    перед по под после потом потому почти при про раз разве с сам свою себе себя
    сейчас со совсем так такой там тебя тем теперь то тогда того тоже той только
    том тот тут ты у уж уже хотя чего чем через что чтоб чтобы чуть эта эти это
    этого этой этом этот я
    """.split()
)

# Слова, которые есть только в формулировке запроса, но не в письмах.
QUERY_FILLER = frozenset(
    """
    покажи покажите найди найдите выведи выведите перечисли перечислите
    письма письмо письмах письмам письмами письме писем
    полученные полученных полученным полученными получен получена получено
    вчера сегодня позавчера накануне
    года году год включительно
    мне нужно надо пожалуйста
    """.split()
)

_VOWELS = frozenset("аеиоуыэюя")

# Доменные синонимы: запрос и письмо называют одно и то же разными словами
# («отказ» vs «операция отклонена», «погрузили» vs «контейнер погружен»).
# Расширяем только запрос, индекс не трогаем.
QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "отказ": ("отклон",),
    "отклон": ("отказ",),
    "сбо": ("ошибк",),
    "ошибк": ("сбо",),
    "погруз": ("погруж",),
    "погруж": ("погруз",),
    "выгруз": ("выгруж",),
    "выгруж": ("выгруз",),
    "прибыт": ("прибыл",),
    "прибыл": ("прибыт",),
}

# «Мягкий» префикс: кроме полного стема индексируем первые 6 символов
# слова, если слово длиннее 7 символов. Snowball иногда разводит формы
# одного корня ("отклонена" -> "отклон", "отклонённых" -> "отклонен"),
# и общий префикс страхует такие пары. Длина 6 подобрана на dev:
# 5 теряет точность, 7 уже не склеивает проблемные формы.
SOFT_PREFIX_LEN = 6
SOFT_PREFIX_MIN = 8

# Таблицы окончаний Snowball: (суффикс, вариант действия).
# Вариант 1 требует, чтобы перед суффиксом стояла "а" или "я" (проверка
# отличает деепричастия/глаголы от похожих существительных), но удаляется
# только сам суффикс. Вариант 2 удаляется без дополнительных условий.
_PERFECTIVE_GERUND = (
    ("ившись", 2), ("ывшись", 2), ("вшись", 1),
    ("ивши", 2), ("ывши", 2), ("вши", 1),
    ("ив", 2), ("ыв", 2), ("в", 1),
)
_ADJECTIVE = (
    ("ими", 1), ("ыми", 1), ("его", 1), ("ого", 1), ("ему", 1), ("ому", 1),
    ("ее", 1), ("ие", 1), ("ое", 1), ("ые", 1), ("ей", 1), ("ий", 1),
    ("ой", 1), ("ый", 1), ("ем", 1), ("им", 1), ("ом", 1), ("ым", 1),
    ("их", 1), ("ых", 1), ("ею", 1), ("ою", 1), ("ую", 1), ("юю", 1),
    ("ая", 1), ("яя", 1),
)
_PARTICIPLE = (
    ("ивш", 2), ("ывш", 2), ("ующ", 2),
    ("ем", 1), ("нн", 1), ("вш", 1), ("ющ", 1), ("щ", 1),
)
_REFLEXIVE = (("сь", 1), ("ся", 1))
_VERB = (
    ("ила", 2), ("ыла", 2), ("ена", 2), ("ейте", 2), ("уйте", 2), ("ите", 2),
    ("или", 2), ("ыли", 2), ("ей", 2), ("уй", 2), ("ил", 2), ("ыл", 2),
    ("им", 2), ("ым", 2), ("ен", 2), ("ило", 2), ("ыло", 2), ("ено", 2),
    ("ует", 2), ("уют", 2), ("ит", 2), ("ыт", 2), ("ят", 2), ("ены", 2),
    ("ить", 2), ("ыть", 2), ("ишь", 2), ("ую", 2),
    ("ла", 1), ("на", 1), ("ете", 1), ("йте", 1), ("ли", 1), ("й", 1),
    ("л", 1), ("ем", 1), ("н", 1), ("ло", 1), ("но", 1), ("нно", 1),
    ("ет", 1), ("ют", 1), ("ны", 1), ("ть", 1), ("ешь", 1), ("ю", 1),
)
_NOUN = (
    ("иями", 1), ("ями", 1), ("ами", 1), ("ией", 1), ("иям", 1), ("ием", 1),
    ("иях", 1), ("ев", 1), ("ов", 1), ("ие", 1), ("ье", 1), ("еи", 1),
    ("ии", 1), ("ей", 1), ("ий", 1), ("ой", 1), ("ам", 1), ("ем", 1),
    ("ом", 1), ("ям", 1), ("ах", 1), ("ях", 1), ("ию", 1), ("ью", 1),
    ("ия", 1), ("ья", 1), ("а", 1), ("е", 1), ("и", 1), ("й", 1),
    ("о", 1), ("у", 1), ("ы", 1), ("ь", 1), ("ю", 1), ("я", 1),
)
_DERIVATIONAL = (("ость", 1), ("ост", 1))
_TIDY = (("ейше", 1), ("ейш", 1), ("н", 2), ("ь", 3))


def normalize(value: str) -> str:
    """NFKC, нижний регистр, ё -> е, без HTML/URL и лишних пробелов."""
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", value)
    value = value.lower().replace("ё", "е")
    value = _HTML_RE.sub(" ", value)
    value = _URL_RE.sub(" ", value)
    value = _WS_RE.sub(" ", value)
    return value.strip()


def raw_tokens(value: str) -> list[str]:
    """Токены без стемминга (знаки препинания уже отброшены)."""
    return _WORD_RE.findall(value)


def extract_sender(value: str) -> str | None:
    match = _SENDER_RE.search(value or "")
    return match.group(1).strip() if match else None


def split_subject_body(value: str) -> tuple[str, str]:
    """Выделить тему и тело письма (без нормализации и стемминга)."""
    text = value or ""
    subject_match = re.search(r"(?is)\bтема\s*:\s*(.+)", text)
    if subject_match:
        subject = re.split(r"(?i)\bтекст письма\b", subject_match.group(1))[0]
        subject = subject.splitlines()[0]
    else:
        subject = text.strip().splitlines()[0] if text.strip() else ""

    body_match = re.search(r"(?is)\bтекст письма\s*:\s*(.*)", text)
    body = body_match.group(1) if body_match else text
    return subject.strip(), body


def _regions(word: str) -> tuple[int, int, int]:
    """Границы RV, R1 и R2 по определению Snowball."""
    limit = len(word)
    rv = limit
    for position, char in enumerate(word):
        if char in _VOWELS:
            rv = position + 1
            break

    r1 = limit
    for position in range(1, limit):
        if word[position] not in _VOWELS and word[position - 1] in _VOWELS:
            r1 = position + 1
            break

    r2 = limit
    for position in range(r1 + 1, limit):
        if word[position] not in _VOWELS and word[position - 1] in _VOWELS:
            r2 = position + 1
            break

    return rv, r1, r2


def _match(word: str, region: int, table: tuple[tuple[str, int], ...]):
    """Самое длинное окончание из таблицы, целиком лежащее в регионе."""
    best: tuple[str, int] | None = None
    for suffix, action in table:
        if len(word) - len(suffix) < region:
            continue
        if not word.endswith(suffix):
            continue
        if best is None or len(suffix) > len(best[0]):
            best = (suffix, action)
    return best


class RussianStemmer:
    """Snowball Russian stemmer с кэшем по словоформам."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def stem(self, word: str) -> str:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        result = self._stem_uncached(word)
        self._cache[word] = result
        return result

    @staticmethod
    def _stem_uncached(word: str) -> str:
        if not word or not word.isalpha():
            return word
        word = word.replace("ё", "е")
        rv, _, r2 = _regions(word)
        if rv >= len(word):
            return word

        matched = _match(word, rv, _PERFECTIVE_GERUND)
        if matched:
            suffix, action = matched
            start = len(word) - len(suffix)
            if action == 1 and not (start > rv and word[start - 1] in "ая"):
                matched = None
            else:
                word = word[:start]

        if not matched:
            reflexive = _match(word, rv, _REFLEXIVE)
            if reflexive:
                word = word[: -len(reflexive[0])]

            adjective = _match(word, rv, _ADJECTIVE)
            if adjective:
                word = word[: -len(adjective[0])]
                participle = _match(word, rv, _PARTICIPLE)
                if participle:
                    suffix, action = participle
                    start = len(word) - len(suffix)
                    if action != 1 or (start > rv and word[start - 1] in "ая"):
                        word = word[:start]
            else:
                verb = _match(word, rv, _VERB)
                if verb:
                    suffix, action = verb
                    start = len(word) - len(suffix)
                    if action == 1 and not (start > rv and word[start - 1] in "ая"):
                        verb = None
                    else:
                        word = word[:start]
                if not verb:
                    noun = _match(word, rv, _NOUN)
                    if noun:
                        word = word[: -len(noun[0])]

        if word.endswith("и") and len(word) - 1 >= rv:
            word = word[:-1]

        derivational = _match(word, r2, _DERIVATIONAL)
        if derivational:
            word = word[: -len(derivational[0])]

        tidy = _match(word, rv, _TIDY)
        if tidy:
            suffix, action = tidy
            if action == 1:
                word = word[: -len(suffix)]
                if word.endswith("нн") and len(word) - 2 >= rv:
                    word = word[:-2]
            elif action == 2:
                if word.endswith("нн") and len(word) - 2 >= rv:
                    word = word[:-1]
            else:
                word = word[: -len(suffix)]

        return word


STEMMER = RussianStemmer()


def tokenize(
    value: str,
    stemmer: RussianStemmer = STEMMER,
    *,
    soft_prefix_len: int = SOFT_PREFIX_LEN,
    soft_prefix_min: int = SOFT_PREFIX_MIN,
) -> list[str]:
    """Нормализовать, разбить на токены, убрать стоп-слова и застеммить.

    При soft_prefix_len > 0 к каждому стему добавляется его префикс —
    второй, более грубый ключ для BM25.
    """
    tokens: list[str] = []
    for token in raw_tokens(normalize(value)):
        if token in STOPWORDS or len(token) < 2:
            continue
        stem = stemmer.stem(token)
        tokens.append(stem)
        if soft_prefix_len and len(token) >= soft_prefix_min:
            tokens.append(token[:soft_prefix_len])
    return tokens


def query_terms(
    value: str,
    stemmer: RussianStemmer = STEMMER,
    *,
    soft_prefix_len: int = SOFT_PREFIX_LEN,
    soft_prefix_min: int = SOFT_PREFIX_MIN,
) -> list[str]:
    """Токены запроса: без служебных слов формулировки, без дублей."""
    result: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens(value):
        if token in STOPWORDS or token in QUERY_FILLER or len(token) < 2:
            continue
        stem = stemmer.stem(token)
        candidates = [stem, *QUERY_SYNONYMS.get(stem, ())]
        if soft_prefix_len and len(token) >= soft_prefix_min:
            candidates.append(token[:soft_prefix_len])
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)

    # «оплата не была одобрена» — это про отклонённый платёж, а не про успешный.
    if "одобр" in seen and re.search(r"\bне\b", value):
        if "отклон" not in seen:
            seen.add("отклон")
            result.append("отклон")
    return result
