"""Проверки генератора аугментированных запросов."""

from __future__ import annotations

import json
import unittest
from datetime import date

from smartmail.augment import (
    EmailView,
    QueryGenerator,
    build_messages,
    generate_dataset,
    is_valid_query,
    make_query_id,
    parse_queries_response,
    split_email,
    template_queries,
)
from smartmail.io import Document


def document(doc_id: str = "d1", text: str = "", dt: str | None = "2028-09-15") -> Document:
    return Document(
        doc_id=doc_id,
        mailbox="box-a",
        text=text,
        date=date.fromisoformat(dt) if dt else None,
    )


class TestSplitEmail(unittest.TestCase):
    def test_subject_and_body(self) -> None:
        view = split_email(document(text="Тема: Операция отклонена\n\nПлатёж не прошёл."))
        self.assertIsNotNone(view)
        self.assertEqual(view.subject, "Операция отклонена")
        self.assertIn("Платёж не прошёл", view.body)

    def test_inline_subject_and_sender(self) -> None:
        view = split_email(
            document(text="ОТ: mail@example.invalid\nТЕМА: Погрузка контейнера ТЕКСТ ПИСЬМА: Контейнер погружен.")
        )
        self.assertIsNotNone(view)
        self.assertEqual(view.subject, "Погрузка контейнера")
        self.assertIn("Контейнер погружен", view.body)
        self.assertEqual(view.sender, "mail@example.invalid")

    def test_missing_date_is_skipped(self) -> None:
        self.assertIsNone(split_email(document(text="Тема: X", dt=None)))


class TestTemplateQueries(unittest.TestCase):
    def setUp(self) -> None:
        self.view = EmailView(
            doc_id="d1",
            mailbox="box-a",
            date=date(2028, 9, 15),
            subject="Операция отклонена",
            body="Операция по заказу 6989723515 отклонена. Сумма: 278.88.",
            sender="mail@example.invalid",
        )

    def test_returns_requested_number(self) -> None:
        queries = template_queries(self.view, 3)
        self.assertEqual(len(queries), 3)

    def test_all_queries_valid_and_unique(self) -> None:
        queries = template_queries(self.view, 6)
        texts = [text for _, text in queries]
        self.assertEqual(len(texts), len(set(texts)))
        for text in texts:
            self.assertTrue(is_valid_query(text), text)

    def test_queries_do_not_leak_dates_or_ids(self) -> None:
        for _, text in template_queries(self.view, 6):
            self.assertNotIn("2028", text)
            self.assertNotIn("SYN-", text)

    def test_entity_style_uses_order_number(self) -> None:
        texts = [text for _, text in template_queries(self.view, 6)]
        self.assertTrue(any("6989723515" in text for text in texts))


class TestValidation(unittest.TestCase):
    def test_rejects_dates(self) -> None:
        self.assertFalse(is_valid_query("Найди письмо от 15.09.2028"))
        self.assertFalse(is_valid_query("Письмо за 15 сентября"))
        self.assertFalse(is_valid_query("Что приходило вчера"))

    def test_rejects_internal_ids(self) -> None:
        self.assertFalse(is_valid_query("Письмо с контейнером SYN-abc12345"))

    def test_accepts_normal_query(self) -> None:
        self.assertTrue(is_valid_query("Где писали про возврат денег за билет?"))

    def test_rejects_english_only(self) -> None:
        self.assertFalse(is_valid_query("find the email about refund please"))


class TestParsing(unittest.TestCase):
    def test_plain_json(self) -> None:
        raw = '{"queries": [{"text": "Где письмо про оплату?", "style": "direct"}]}'
        self.assertEqual(parse_queries_response(raw), [("Где письмо про оплату?", "direct")])

    def test_fenced_json(self) -> None:
        raw = '```json\n{"queries": [{"text": "Письмо про возврат"}]}\n```'
        self.assertEqual(parse_queries_response(raw), [("Письмо про возврат", "direct")])

    def test_garbage(self) -> None:
        self.assertEqual(parse_queries_response("не json"), [])

    def test_build_messages_contains_subject_and_body(self) -> None:
        view = EmailView(
            doc_id="d1",
            mailbox="box-a",
            date=date(2028, 9, 15),
            subject="Тема письма",
            body="Текст письма",
            sender=None,
        )
        messages = build_messages(view, 4)
        joined = json.dumps(messages, ensure_ascii=False)
        self.assertIn("Тема письма", joined)
        self.assertIn("Текст письма", joined)


class TestLLMBackend(unittest.TestCase):
    def test_parses_chat_response(self) -> None:
        view = EmailView(
            doc_id="d1",
            mailbox="box-a",
            date=date(2028, 9, 15),
            subject="Тема",
            body="Текст письма",
            sender=None,
        )
        generator = QueryGenerator("openai", api_key="test")
        generator._chat = lambda messages: (
            '{"queries": [{"text": "Где письмо про оплату?", "style": "direct"}]}'
        )
        self.assertEqual(
            generator.generate(view, 1),
            [("Где письмо про оплату?", "direct")],
        )


class TestGenerateDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            document("d1", "Тема: Операция отклонена\n\nПлатёж не прошёл.", "2028-09-15"),
            document("d2", "Тема: Возврат средств\n\nВозмещение инициировано.", "2028-09-16"),
            document("d3", "Тема: Погрузка контейнера\n\nКонтейнер погружен.", "2028-09-17"),
            document("d4", "Тема: Без даты", None),
        ]
        self.generator = QueryGenerator("template")

    def test_query_date_is_next_day_and_doc_dated(self) -> None:
        records = generate_dataset(self.documents, generator=self.generator, per_email=2)
        self.assertTrue(records)
        dated = {doc.doc_id: doc.date for doc in self.documents if doc.date}
        for record in records:
            self.assertEqual(record["doc_id"] in dated, True)
            self.assertGreater(record["query_date"], dated[record["doc_id"]].isoformat())

    def test_every_query_has_positive_and_unique_id(self) -> None:
        records = generate_dataset(self.documents, generator=self.generator, per_email=3)
        ids = [record["query_id"] for record in records]
        self.assertEqual(len(ids), len(set(ids)))
        for record in records:
            self.assertTrue(record["query"].strip())

    def test_limit_is_respected(self) -> None:
        records = generate_dataset(
            self.documents, generator=self.generator, per_email=3, limit=1
        )
        self.assertEqual(len({record["doc_id"] for record in records}), 1)

    def test_query_id_is_stable(self) -> None:
        self.assertEqual(make_query_id("Где письмо?", "d1"), make_query_id("где письмо?", "d1"))


if __name__ == "__main__":
    unittest.main()
