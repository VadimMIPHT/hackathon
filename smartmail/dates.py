"""Разбор даты из текста запроса.

Публичный dev-набор целиком состоит из запросов вида
"<что найти>, полученные <вчера | 2 ноября 2025 года | 02.11.2025>".
Дата в запросе — самый сильный сигнал, поэтому парсим её явно и
вырезаем из текста, чтобы она не засоряла BM25.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}
_MONTH_RE = "|".join(MONTHS)

_RE_DMY = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_RE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RE_RU_FULL = re.compile(
    r"\b(\d{1,2})\s+(" + _MONTH_RE + r")\s+(\d{4})\s*(?:года|г\.?)?"
)
_RE_RU_SHORT = re.compile(r"\b(\d{1,2})\s+(" + _MONTH_RE + r")\b(?!\s*\d{4})")
_RE_DAYS_AGO = re.compile(r"\b(\d{1,2})\s+дн(?:я|ей)\s+назад\b")
_RE_WEEKS_AGO = re.compile(r"\b(\d{1,2})\s+недел(?:ю|и|ь)\s+назад\b")
_RE_MONTHS_AGO = re.compile(r"\b(\d{1,2})\s+месяц(?:а|ев)?\s+назад\b")


@dataclass(frozen=True, slots=True)
class ParsedDate:
    """Найденная дата и её позиция в нормализованном тексте запроса."""

    date: date | None
    span: tuple[int, int] | None
    source: str


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _shift_months(value: date, months: int) -> date:
    total = value.year * 12 + (value.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def extract_date(query: str, query_date: date) -> ParsedDate:
    """Вытащить целевую дату из запроса.

    query должен быть уже нормализован (нижний регистр, ё -> е).
    Все явные даты проверяются на "не позже даты запроса": письма из
    будущего по условию задачи недоступны, а опечатка в годе не должна
    обнулять выдачу.
    """

    match = _RE_ISO.search(query)
    if match:
        value = _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if value and value <= query_date:
            return ParsedDate(value, match.span(), "iso")

    match = _RE_DMY.search(query)
    if match:
        value = _safe_date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        if value and value <= query_date:
            return ParsedDate(value, match.span(), "dmy")

    match = _RE_RU_FULL.search(query)
    if match:
        value = _safe_date(int(match.group(3)), MONTHS[match.group(2)], int(match.group(1)))
        if value and value <= query_date:
            return ParsedDate(value, match.span(), "ru_full")

    if "позавчера" in query:
        return ParsedDate(query_date - timedelta(days=2), None, "relative")
    if "вчера" in query:
        return ParsedDate(query_date - timedelta(days=1), None, "relative")
    if "сегодня" in query:
        return ParsedDate(query_date, None, "relative")

    match = _RE_DAYS_AGO.search(query)
    if match:
        return ParsedDate(query_date - timedelta(days=int(match.group(1))), match.span(), "relative")

    match = _RE_WEEKS_AGO.search(query)
    if match:
        return ParsedDate(query_date - timedelta(weeks=int(match.group(1))), match.span(), "relative")

    match = _RE_MONTHS_AGO.search(query)
    if match:
        return ParsedDate(_shift_months(query_date, int(match.group(1))), match.span(), "relative")

    match = _RE_RU_SHORT.search(query)
    if match:
        value = _safe_date(query_date.year, MONTHS[match.group(2)], int(match.group(1)))
        if value and value > query_date:
            # "2 января" при дате запроса 3 января — это прошлый год.
            value = _safe_date(query_date.year - 1, MONTHS[match.group(2)], int(match.group(1)))
        if value:
            return ParsedDate(value, match.span(), "ru_short")

    return ParsedDate(None, None, "none")


def strip_span(query: str, span: tuple[int, int] | None) -> str:
    """Убрать найденный фрагмент (дату) из текста запроса."""
    if span is None:
        return query
    start, end = span
    return f"{query[:start]} {query[end:]}"
