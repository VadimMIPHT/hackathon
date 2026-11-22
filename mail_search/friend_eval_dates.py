import calendar
import re
from datetime import date, timedelta

from mail_search.dataset import Query
from mail_search.retrieval import MONTHS

DAY = r"(?<!\d)\d{1,2}"
MONTH = rf"(?:{'|'.join(MONTHS)})"
YEAR = r"\d{4}(?:-го)?"
DAY_SEQUENCE = re.compile(
    rf"(?P<first>{DAY})(?:\s+(?P<first_month>{MONTH}))?\s*"
    rf"(?P<connector>(?<=\s)(?:и|или|по|до)(?=\s)|[-–—])\s*"
    rf"(?P<last>{DAY})\s+(?P<last_month>{MONTH})\s+(?P<year>{YEAR})\b",
    re.IGNORECASE,
)
DAY_NAME = re.compile(rf"({DAY})\s+({MONTH})\s+({YEAR})\b", re.IGNORECASE)
NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)")
YESTERDAY = re.compile(r"\b(?:вчера|вчерашн[а-яё]+|накануне)\b", re.IGNORECASE)
LAST_OF_MONTH = re.compile(rf"\bпоследний\s+день\s+({MONTH})\s+({YEAR})\b", re.IGNORECASE)


def year_number(value: str) -> int:
    return int(value.split("-", 1)[0])


def as_date(day: str, month: str, year: str) -> date:
    return date(year_number(year), MONTHS[month.casefold()], int(day))


def requested_dates(query: Query) -> set[date] | None:
    dates: set[date] = set()
    try:
        text = query.text
        for match in DAY_SEQUENCE.finditer(text):
            first_month = match["first_month"] or match["last_month"]
            first = as_date(match["first"], first_month, match["year"])
            last = as_date(match["last"], match["last_month"], match["year"])
            if match["connector"].casefold() in {"и", "или"}:
                dates.update((first, last))
            else:
                width = (last - first).days
                if not 0 <= width <= 366:
                    return None
                dates.update(first + timedelta(days=offset) for offset in range(width + 1))
        text = DAY_SEQUENCE.sub(" ", text)
        dates.update(as_date(*match.groups()) for match in DAY_NAME.finditer(text))
        dates.update(date(int(year), int(month), int(day)) for day, month, year in NUMERIC_DATE.findall(text))
        for month_name, year_name in LAST_OF_MONTH.findall(text):
            year, month = year_number(year_name), MONTHS[month_name.casefold()]
            dates.add(date(year, month, calendar.monthrange(year, month)[1]))
        if YESTERDAY.search(text):
            dates.add(query.as_of - timedelta(days=1))
    except (OverflowError, ValueError):
        return None
    return dates or None
