"""Разбор даты из текста запроса.

Запросы бывают не только про один день: «вчера», «2 ноября 2025 года»,
«9 или 10 октября», «с 30 января по 2 февраля», «25–27 сентября».
Поэтому парсер возвращает начало и конец диапазона (для одного дня они
совпадают), а найденный фрагмент вырезается из запроса, чтобы дата не
засоряла BM25.
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
    r"\b(\d{1,2})\s+(" + _MONTH_RE + r")\s+(\d{4})\s*(?:года|год|г\.?|-\s*го)?"
)
_RE_RU_SHORT = re.compile(r"\b(\d{1,2})\s+(" + _MONTH_RE + r")\b(?!\s*\d{4})")
_RE_DAYS_AGO = re.compile(r"\b(\d{1,2})\s+дн(?:я|ей)\s+назад\b")
_RE_WEEKS_AGO = re.compile(r"\b(\d{1,2})\s+недел(?:ю|и|ь)\s+назад\b")
_RE_MONTHS_AGO = re.compile(r"\b(\d{1,2})\s+месяц(?:а|ев)?\s+назад\b")

# «с 30 января по 2 февраля 2028»
_RE_RANGE_MONTHS = re.compile(
    r"\bс\s+(\d{1,2})\s+(" + _MONTH_RE + r")(?:\s+(\d{4}))?"
    r"\s+по\s+(\d{1,2})\s+(" + _MONTH_RE + r")(?:\s+(\d{4}))?"
)
# «с 26 по 30 ноября 2027», «с 7 по 10 октября 2028»
_RE_RANGE_SHARED = re.compile(
    r"\bс\s+(\d{1,2})\s+по\s+(\d{1,2})\s+(" + _MONTH_RE + r")(?:\s+(\d{4}))?"
)
# «9 или 10 октября 2028», «28 и 29 января 2028»
_RE_RANGE_OR = re.compile(
    r"\b(\d{1,2})\s+(?:или|и)\s+(\d{1,2})\s+(" + _MONTH_RE + r")(?:\s+(\d{4}))?"
)
# «25–27 сентября 2028»
_RE_RANGE_DASH = re.compile(
    r"\b(\d{1,2})\s*[-–—]\s*(\d{1,2})\s+(" + _MONTH_RE + r")(?:\s+(\d{4}))?"
)
# «в последний день января 2028»
_RE_LAST_DAY = re.compile(r"\bпоследн\w*\s+день\s+(" + _MONTH_RE + r")(?:\s+(\d{4}))?")


@dataclass(frozen=True, slots=True)
class ParsedDate:
    """Найденная дата (начало диапазона) и её позиция в тексте запроса."""

    date: date | None
    span: tuple[int, int] | None
    source: str
    end_date: date | None = None

    @property
    def is_range(self) -> bool:
        return self.date is not None and self.end_date is not None and self.end_date != self.date

    def bounds(self) -> tuple[date, date] | None:
        if self.date is None:
            return None
        return self.date, self.end_date or self.date


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


def _range_bounds(
    day1: int,
    month1: int,
    year1: int | None,
    day2: int,
    month2: int,
    year2: int | None,
    query_date: date,
) -> tuple[date, date] | None:
    """Достроить годы и вернуть корректный [start, end] не позже query_date."""
    if year1 is None and year2 is None:
        year1 = year2 = query_date.year
        if (month1, day1) > (month2, day2):
            year1 -= 1
    elif year1 is None:
        year1 = year2
        if (month1, day1) > (month2, day2):
            year1 -= 1
    elif year2 is None:
        year2 = year1
        if (month2, day2) < (month1, day1):
            year2 += 1

    start = _safe_date(year1, month1, day1)
    end = _safe_date(year2, month2, day2)
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start

    if start > query_date:
        # Год не указан, а дата уже в будущем — значит, имелся в виду прошлый.
        shifted_start = _safe_date(start.year - 1, start.month, start.day)
        shifted_end = _safe_date(end.year - 1, end.month, end.day)
        if shifted_start and shifted_end and shifted_end <= query_date:
            start, end = shifted_start, shifted_end

    if end > query_date:
        end = query_date
    if start > end:
        return None
    return start, end


def _range_from_match(
    match: re.Match[str],
    *,
    day1: int,
    month1: int,
    year1: int | None,
    day2: int,
    month2: int,
    year2: int | None,
    query_date: date,
    source: str,
) -> ParsedDate | None:
    bounds = _range_bounds(day1, month1, year1, day2, month2, year2, query_date)
    if bounds is None:
        return None
    return ParsedDate(bounds[0], match.span(), source, bounds[1])


def extract_date(query: str, query_date: date) -> ParsedDate:
    """Вытащить дату или диапазон дат из нормализованного запроса.

    Все даты проверяются на «не позже даты запроса»: письма из будущего
    по условию задачи недоступны.
    """

    match = _RE_RANGE_MONTHS.search(query)
    if match:
        result = _range_from_match(
            match,
            day1=int(match.group(1)),
            month1=MONTHS[match.group(2)],
            year1=int(match.group(3)) if match.group(3) else None,
            day2=int(match.group(4)),
            month2=MONTHS[match.group(5)],
            year2=int(match.group(6)) if match.group(6) else None,
            query_date=query_date,
            source="range",
        )
        if result:
            return result

    match = _RE_RANGE_SHARED.search(query)
    if match:
        result = _range_from_match(
            match,
            day1=int(match.group(1)),
            month1=MONTHS[match.group(3)],
            year1=int(match.group(4)) if match.group(4) else None,
            day2=int(match.group(2)),
            month2=MONTHS[match.group(3)],
            year2=int(match.group(4)) if match.group(4) else None,
            query_date=query_date,
            source="range",
        )
        if result:
            return result

    match = _RE_RANGE_OR.search(query)
    if match:
        result = _range_from_match(
            match,
            day1=int(match.group(1)),
            month1=MONTHS[match.group(3)],
            year1=int(match.group(4)) if match.group(4) else None,
            day2=int(match.group(2)),
            month2=MONTHS[match.group(3)],
            year2=int(match.group(4)) if match.group(4) else None,
            query_date=query_date,
            source="range",
        )
        if result:
            return result

    match = _RE_RANGE_DASH.search(query)
    if match:
        result = _range_from_match(
            match,
            day1=int(match.group(1)),
            month1=MONTHS[match.group(3)],
            year1=int(match.group(4)) if match.group(4) else None,
            day2=int(match.group(2)),
            month2=MONTHS[match.group(3)],
            year2=int(match.group(4)) if match.group(4) else None,
            query_date=query_date,
            source="range",
        )
        if result:
            return result

    match = _RE_LAST_DAY.search(query)
    if match:
        month = MONTHS[match.group(1)]
        year = int(match.group(2)) if match.group(2) else query_date.year
        day = calendar.monthrange(year, month)[1]
        value = _safe_date(year, month, day)
        if value and value <= query_date:
            return ParsedDate(value, match.span(), "last_day")

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

    if "накануне" in query:
        return ParsedDate(query_date - timedelta(days=1), None, "relative")
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
