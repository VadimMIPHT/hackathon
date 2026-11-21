"""Аугментация обучающих данных: запросы «как я вспоминаю письмо».

Важно: аугментация не заменяет dev-набор организаторов. Она нужна, чтобы
(а) получить тысячи пар «запрос → письмо» для файнтюна би-энкодера,
(б) проверить устойчивость поиска к перефразировкам,
(в) собрать hard negatives.

Документы не генерируем: 16k писем достаточно для retrieval, синтетические
письма только добавляют шум в индекс и не помогают отвечать на реальные
запросы. Генерируем именно запросы, у которых положительный документ
известен по построению.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta

from .io import Document

STYLES = ("direct", "paraphrase", "vague", "keywords", "followup")

_SYSTEM_PROMPT = """Ты помогаешь собрать данные для поиска по почте.
По одному письму придумай запросы, которыми пользователь вспоминал бы это
письмо через неделю — по смыслу, а не по точным словам темы.

Требования:
- разговорный русский, разная длина (от 3 до 25 слов);
- не копируй тему письма дословно, перефразируй;
- не упоминай точную дату и не вставляй внутренние идентификаторы;
- используй разные стили: прямой вопрос, пересказ смысла, обрывки памяти,
  набор ключевых слов, уточняющий вопрос про суть или итог;
- каждый запрос должен быть самодостаточным: по нему реально найти письмо;
- если в письме есть номер заказа, сумма, имя или другой уникальный признак,
  используй его в 1–2 запросах, чтобы отличить письмо от похожих.

Верни строго JSON без пояснений:
{"queries": [{"text": "...", "style": "direct|paraphrase|vague|keywords|followup"}]}"""

_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Письмо:\nТема: Изменение времени вылета\n"
            "Текст: Рейс SU-1234 перенесён на 18:40, регистрация закрывается за 40 минут.\n"
            "Сгенерируй 3 запроса. JSON."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "queries": [
                    {"text": "Где писали, что самолёт улетит позже?", "style": "direct"},
                    {"text": "Письмо от авиакомпании про перенос рейса", "style": "keywords"},
                    {"text": "Что-то приходило про задержку вылета, не помню деталей", "style": "vague"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            "Письмо:\nТема: Возврат средств по заказу\n"
            "Текст: Возмещение инициировано, деньги вернутся на карту в течение 5 дней.\n"
            "Сгенерируй 3 запроса. JSON."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "queries": [
                    {"text": "Когда вернут деньги за заказ?", "style": "direct"},
                    {"text": "Мне писали, что возврат уже начали?", "style": "followup"},
                    {"text": "Письмо про возврат оплаты", "style": "keywords"},
                ]
            },
            ensure_ascii=False,
        ),
    },
]

_DATE_RE = re.compile(
    r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"
    r"|\b\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
    re.IGNORECASE,
)
_TOPIC_JUNK_RE = re.compile(r"(?i)\b(email|тема|текст письма|подписка на контейнер)\b")
_LEAD_WORD_RE = re.compile(r"(?i)^(есть|создан[аоы]?|выставлен[аоы]?|начислен[аоы]?|подтверждает)\s+")
_MONEY_RE = re.compile(r"\b\d+[.,]\d{2}\b")
_ORDER_RE = re.compile(r"\b\d{5,}\b")


@dataclass(frozen=True, slots=True)
class EmailView:
    doc_id: str
    mailbox: str
    date: date
    subject: str
    body: str
    sender: str | None


def split_email(document: Document) -> EmailView | None:
    """Выделить тему, тело и отправителя из текста письма."""
    text = document.text or ""
    if not text.strip() or document.date is None:
        return None

    subject_match = re.search(r"(?is)\bтема\s*:\s*(.+)", text)
    if subject_match:
        subject = re.split(r"(?i)\bтекст письма\b", subject_match.group(1))[0]
        subject = subject.splitlines()[0]
    else:
        subject = text.strip().splitlines()[0] if text.strip() else ""

    body_match = re.search(r"(?is)\bтекст письма\s*:\s*(.*)", text)
    body = body_match.group(1) if body_match else text

    sender_match = re.search(r"(?im)^\s*от\s*:\s*(\S+)", text)

    return EmailView(
        doc_id=document.doc_id,
        mailbox=document.mailbox,
        date=document.date,
        subject=_clean(subject),
        body=_clean(body),
        sender=sender_match.group(1) if sender_match else None,
    )


def _clean(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" «»\"'.,;:-—\n\t")


def _topic(email: EmailView, limit: int = 70) -> str:
    topic = _TOPIC_JUNK_RE.sub(" ", email.subject)
    topic = _LEAD_WORD_RE.sub("", topic)
    topic = _clean(topic)
    if len(topic) < 4:
        topic = _clean(" ".join(email.body.split()[:10]))
    if len(topic) > limit:
        topic = topic[:limit].rsplit(" ", 1)[0]
    return topic.rstrip(" .,:;—")


def _entity(email: EmailView) -> str | None:
    match = _ORDER_RE.search(email.body) or _MONEY_RE.search(email.body)
    return match.group(0) if match else None


def template_queries(email: EmailView, n: int) -> list[tuple[str, str]]:
    """Детерминированный fallback без сети.

    Это не замена LLM: запросы получаются близкими к теме письма и проверяют
    лексическую устойчивость, а не настоящий пересказ смысла.
    """
    topic = _topic(email)
    if not topic:
        return []

    variants: list[tuple[str, str]] = [
        ("direct", f"Где письмо про {topic}?"),
        ("paraphrase", f"Найди письмо про {topic}"),
        ("vague", f"Мне приходило что-то про {topic}, не помню деталей"),
        ("keywords", topic),
        ("followup", f"Что известно про {topic}?"),
    ]
    entity = _entity(email)
    if entity:
        variants.append(("keywords", f"Письмо про {topic}, там ещё номер {entity}"))

    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for style, text in variants:
        if len(result) >= n:
            break
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append((style, text))
    return result


def build_messages(email: EmailView, n: int) -> list[dict]:
    sender = email.sender or "не указан"
    body = email.body[:1200]
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *_FEW_SHOT,
        {
            "role": "user",
            "content": (
                f"Письмо:\nТема: {email.subject}\nОт: {sender}\n"
                f"Дата: {email.date.isoformat()}\nТекст: {body}\n"
                f"Сгенерируй {n} запросов. JSON."
            ),
        },
    ]


def parse_queries_response(raw: str) -> list[tuple[str, str]]:
    """Достать список (текст, стиль) из ответа модели, терпимо к обёрткам."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    items = payload.get("queries", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []

    result: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, str):
            result.append((item.strip(), "direct"))
        elif isinstance(item, dict) and item.get("text"):
            style = str(item.get("style") or "direct")
            result.append((str(item["text"]).strip(), style))
    return result


def is_valid_query(text: str) -> bool:
    if not 10 <= len(text) <= 220:
        return False
    lower = text.lower()
    if _DATE_RE.search(lower) or "syn-" in lower or "тема:" in lower:
        return False
    if re.search(r"\b(вчера|сегодня|позавчера)\b", lower):
        return False
    letters = re.findall(r"[а-яёa-z]", lower)
    if not letters:
        return False
    cyrillic = len(re.findall(r"[а-яё]", lower))
    return cyrillic / len(letters) >= 0.6


def make_query_id(query: str, doc_id: str) -> str:
    digest = hashlib.sha1(f"{doc_id}|{query.lower()}".encode("utf-8")).hexdigest()
    return f"AUG-{digest[:20]}"


class QueryGenerator:
    """Генератор запросов: template, openai (любой совместимый API) или ollama."""

    def __init__(
        self,
        backend: str = "template",
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.8,
        timeout: int = 60,
        retries: int = 3,
    ) -> None:
        if backend not in {"template", "openai", "ollama"}:
            raise ValueError(f"неизвестный backend: {backend}")
        self.backend = backend
        self.model = model or ("gpt-4o-mini" if backend == "openai" else "qwen2.5:7b")
        self.base_url = (base_url or self._default_base_url()).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "ollama"
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

    def _default_base_url(self) -> str:
        if self.backend == "ollama":
            return "http://localhost:11434/v1"
        return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def generate(self, email: EmailView, n: int) -> list[tuple[str, str]]:
        if self.backend == "template":
            return template_queries(email, n)
        return parse_queries_response(self._chat(build_messages(email, n)))

    def _chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        json_mode = True
        for attempt in range(1, self.retries + 1):
            try:
                return self._post(payload, json_mode=json_mode)
            except urllib.error.HTTPError as error:
                # Часть провайдеров не поддерживает response_format.
                if error.code == 400 and json_mode:
                    json_mode = False
                    continue
                if error.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt >= self.retries:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("не удалось получить ответ модели")

    def _post(self, payload: dict, *, json_mode: bool) -> str:
        body = dict(payload)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def generate_dataset(
    documents: list[Document],
    *,
    generator: QueryGenerator,
    per_email: int = 4,
    limit: int | None = None,
    seed: int = 42,
    negatives: int = 0,
    engine=None,
    progress=None,
) -> list[dict]:
    """Сгенерировать пары «запрос → положительное письмо».

    query_date = дата письма + 1 день, чтобы письмо было доступно на момент
    запроса и проходило фильтр доступности.
    """
    pool = [document for document in documents if document.date is not None]
    if limit is not None and limit < len(pool):
        pool = random.Random(seed).sample(pool, limit)

    records: list[dict] = []
    seen: set[str] = set()

    for index, document in enumerate(pool, start=1):
        view = split_email(document)
        if view is None:
            continue
        query_date = document.date + timedelta(days=1)
        for style, text in generator.generate(view, per_email):
            if not is_valid_query(text):
                continue
            key = re.sub(r"\s+", " ", text.lower()).strip()
            if key in seen:
                continue
            seen.add(key)

            record = {
                "query_id": make_query_id(text, document.doc_id),
                "doc_id": document.doc_id,
                "synthetic_user": document.mailbox,
                "query": text,
                "query_date": query_date.isoformat(),
                "style": style,
                "generator": generator.backend,
                "negatives": [],
            }
            if negatives and engine is not None:
                result = engine.search(
                    text, document.mailbox, query_date, top_k=negatives + 1
                )
                record["negatives"] = [
                    hit.doc_id for hit in result.hits if hit.doc_id != document.doc_id
                ][:negatives]
            records.append(record)

        if progress and index % 25 == 0:
            progress(index, len(pool), len(records))

    return records
