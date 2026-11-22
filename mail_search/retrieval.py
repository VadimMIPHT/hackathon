import math
import re
from collections import Counter, defaultdict
from datetime import date, timedelta

from mail_search.dataset import Document, Query

TOKEN = re.compile(r"[^\W_]+(?:[-@.][^\W_]+)*")
SUBJECT = re.compile(r"\A\s*тема[ \t]*:[ \t]*([^\r\n]*)", re.IGNORECASE)
BODY_LABEL = re.compile(r"\A\s*(?:текст письма|тело письма)\s*:\s*", re.IGNORECASE)
MONTHS = {
    name: month for month, name in enumerate(
        "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(), 1
    )
}
CALENDAR_DATE = re.compile(
    rf"\b(\d{{1,2}})\s+({'|'.join(MONTHS)})\s+(\d{{4}})\b", re.IGNORECASE
)
YESTERDAY = re.compile(r"\bвчера\b", re.IGNORECASE)
BM25_K1 = 1.2
BM25_B = 0.75
SUBJECT_WEIGHT = 2
RRF_K = 60
RRF_DEPTH = 50


def fields(text: str) -> tuple[str, str]:
    match = SUBJECT.match(text)
    subject = match[1].strip() if match else ""
    body = text[match.end():] if match else text
    return subject, BODY_LABEL.sub("", body).strip()


def searchable_text(document: Document) -> str:
    return "\n".join(part for part in fields(document.text) if part)


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.casefold().replace("ё", "е"))


def day_from_query(query: Query) -> date | None:
    matches = CALENDAR_DATE.findall(query.text)
    yesterday = bool(YESTERDAY.search(query.text))
    if len(matches) + yesterday != 1:
        return None
    if yesterday:
        return query.as_of - timedelta(days=1)
    day, month, year = matches[0]
    try:
        return date(int(year), MONTHS[month.casefold()], int(day))
    except ValueError:
        return None


def eligible_indices(
    documents: list[Document], query: Query, use_query_day: bool,
) -> list[int]:
    target = day_from_query(query) if use_query_day else None
    return [
        i for i, document in enumerate(documents)
        if document.mailbox == query.mailbox
        and (document.received is None or document.received <= query.as_of)
        and (target is None or document.received == target)
    ]


class BM25:
    def __init__(self, documents: list[Document]) -> None:
        self.document_ids = [document.doc_id for document in documents]
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, document in enumerate(documents):
            subject, body = fields(document.text)
            frequencies = Counter(tokens(body))
            frequencies.update(tokens(subject) * SUBJECT_WEIGHT)
            self.lengths.append(sum(frequencies.values()))
            for term, frequency in frequencies.items():
                self.postings[term].append((index, frequency))

    def rank(self, query: str, eligible: list[int], limit: int = RRF_DEPTH) -> list[str]:
        size = len(eligible)
        if not size:
            return []
        available = set(eligible)
        mean_length = sum(self.lengths[i] for i in eligible) / size or 1
        scores: dict[int, float] = defaultdict(float)
        for term in dict.fromkeys(tokens(query)):
            matches = [(i, tf) for i, tf in self.postings.get(term, ()) if i in available]
            frequency = len(matches)
            idf = math.log1p((size - frequency + 0.5) / (frequency + 0.5))
            for i, tf in matches:
                norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[i] / mean_length)
                scores[i] += idf * tf * (BM25_K1 + 1) / (tf + norm)
        ranked = sorted(scores, key=lambda i: (-scores[i], self.document_ids[i]))[:limit]
        return [self.document_ids[i] for i in ranked]


def fuse(lexical: list[str], dense: list[str], limit: int = RRF_DEPTH) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in (lexical, dense):
        for rank, document_id in enumerate(ranking[:RRF_DEPTH], 1):
            scores[document_id] += 1 / (RRF_K + rank)
    return sorted(scores, key=lambda document_id: (-scores[document_id], document_id))[:limit]
