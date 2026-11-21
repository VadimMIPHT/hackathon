"""Проверки бейзлайна: разбор дат, стеммер, фильтры, метрики, e2e."""

from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from smartmail import metrics
from smartmail.dates import extract_date
from smartmail.engine import SearchEngine
from smartmail.io import Document, load_documents, load_qrels, load_queries
from smartmail.text import RussianStemmer, normalize, query_terms, split_subject_body

DATA = Path(__file__).resolve().parent.parent / "data"
QUERY_DATE = date(2028, 9, 16)


def doc(doc_id: str, mailbox: str = "box-a", text: str = "", doc_date: str | None = None) -> Document:
    return Document(
        doc_id=doc_id,
        mailbox=mailbox,
        text=text,
        date=date.fromisoformat(doc_date) if doc_date else None,
    )


class TestDates(unittest.TestCase):
    def test_yesterday(self) -> None:
        parsed = extract_date("покажи письма, полученные вчера", QUERY_DATE)
        self.assertEqual(parsed.date, date(2028, 9, 15))

    def test_day_before_yesterday_is_not_yesterday(self) -> None:
        self.assertEqual(extract_date("позавчера", QUERY_DATE).date, date(2028, 9, 14))

    def test_explicit_russian_date_with_year(self) -> None:
        parsed = extract_date("письма, полученные 2 ноября 2025 года", QUERY_DATE)
        self.assertEqual(parsed.date, date(2025, 11, 2))
        # фрагмент даты вырезается из запроса
        self.assertIsNotNone(parsed.span)

    def test_explicit_short_russian_date_uses_query_year(self) -> None:
        parsed = extract_date("письма за 5 сентября", QUERY_DATE)
        self.assertEqual(parsed.date, date(2028, 9, 5))

    def test_short_date_in_future_rolls_to_previous_year(self) -> None:
        parsed = extract_date("письма за 20 декабря", QUERY_DATE)
        self.assertEqual(parsed.date, date(2027, 12, 20))

    def test_dotted_date(self) -> None:
        self.assertEqual(extract_date("письма от 03.09.2028", QUERY_DATE).date, date(2028, 9, 3))

    def test_no_date(self) -> None:
        parsed = extract_date("письма про возврат билета", QUERY_DATE)
        self.assertIsNone(parsed.date)
        self.assertIsNone(parsed.span)

    def test_future_explicit_date_is_ignored(self) -> None:
        parsed = extract_date("письма от 1 января 2030 года", QUERY_DATE)
        self.assertIsNone(parsed.date)


class TestText(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(normalize("  Ёлка   <b>и</b>  КОТ  "), "елка и кот")

    def test_stemmer_collapses_inflections(self) -> None:
        stemmer = RussianStemmer()
        forms = ["заказов", "заказа", "заказы", "заказ"]
        self.assertEqual(len({stemmer.stem(form) for form in forms}), 1)

    def test_stemmer_follows_snowball(self) -> None:
        stemmer = RussianStemmer()
        self.assertEqual(stemmer.stem("машины"), stemmer.stem("машина"))
        self.assertEqual(stemmer.stem("машины"), "машин")

    def test_soft_prefix_bridges_stemmer_gap(self) -> None:
        from smartmail.text import tokenize

        query = set(tokenize("отклонённых платежах"))
        document = set(tokenize("операция отклонена"))
        # стемы разные ("отклонен" / "отклон"), общий 6-символьный префикс — да
        self.assertIn("отклон", query & document)

    def test_soft_prefix_can_be_disabled(self) -> None:
        from smartmail.text import tokenize

        self.assertEqual(tokenize("отклонённых", soft_prefix_len=0), ["отклонен"])

    def test_stemmer_keeps_short_words(self) -> None:
        stemmer = RussianStemmer()
        self.assertEqual(stemmer.stem("чек"), "чек")

    def test_split_subject_body(self) -> None:
        subject, body = split_subject_body("Тема: Операция отклонена\n\nПлатёж не прошёл.")
        self.assertEqual(subject, "Операция отклонена")
        self.assertIn("Платёж не прошёл", body)

    def test_query_terms_drop_filler_and_date_words(self) -> None:
        terms = query_terms("покажи уведомления о сбоях модуля, полученные вчера")
        self.assertIn("уведомлен", terms)
        self.assertIn("сбо", terms)
        self.assertNotIn("вчер", terms)
        self.assertNotIn("покаж", terms)


class TestMetrics(unittest.TestCase):
    def test_recall(self) -> None:
        self.assertAlmostEqual(metrics.recall_at_k(["a", "b", "c"], {"b", "d"}, 3), 0.5)

    def test_mrr(self) -> None:
        self.assertAlmostEqual(metrics.mrr_at_k(["x", "y", "b"], {"b"}, 10), 1 / 3)

    def test_ndcg_perfect_order(self) -> None:
        self.assertAlmostEqual(metrics.ndcg_at_k(["a", "b"], {"a", "b"}, 10), 1.0)

    def test_ndcg_reversed_order_is_lower(self) -> None:
        good = metrics.ndcg_at_k(["a", "x", "b"], {"a", "b"}, 10)
        bad = metrics.ndcg_at_k(["x", "b", "a"], {"a", "b"}, 10)
        self.assertGreater(good, bad)

    def test_ndcg_matches_hand_calculation(self) -> None:
        # релевантный на позиции 2: DCG = 1/log2(3), IDCG = 1
        self.assertAlmostEqual(metrics.ndcg_at_k(["x", "a"], {"a"}, 10), 1 / 1.5849625, places=5)


class TestEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SearchEngine(
            [
                doc("d1", text="Создан автовизит для вывоза контейнеров 15.09.2028", doc_date="2028-09-15"),
                doc("d2", text="Уведомление о погрузке контейнера 15.09.2028", doc_date="2028-09-15"),
                doc("d3", text="Создан автовизит для вывоза контейнеров 16.09.2028", doc_date="2028-09-16"),
                doc("d4", mailbox="box-b", text="Создан автовизит 15.09.2028", doc_date="2028-09-15"),
            ]
        )

    def test_date_filter_exact_day(self) -> None:
        result = self.engine.search("автовизит контейнер вчера", "box-a", "2028-09-16", top_k=10)
        self.assertEqual(result.target_date, date(2028, 9, 15))
        self.assertEqual({hit.doc_id for hit in result.hits}, {"d1", "d2"})

    def test_mailbox_isolation(self) -> None:
        result = self.engine.search("автовизит", "box-b", "2028-09-16", top_k=10)
        self.assertEqual([hit.doc_id for hit in result.hits], ["d4"])

    def test_future_documents_excluded_when_no_date_in_query(self) -> None:
        result = self.engine.search("автовизит контейнер", "box-a", "2028-09-15", top_k=10)
        self.assertNotIn("d3", {hit.doc_id for hit in result.hits})

    def test_relevance_order_inside_day(self) -> None:
        result = self.engine.search(
            "уведомления о погрузке контейнеров", "box-a", "2028-09-16", top_k=2
        )
        self.assertEqual(result.hits[0].doc_id, "d2")

    def test_subject_boost_smoke(self) -> None:
        engine = SearchEngine(
            [doc("s1", text="Тема: Возврат средств\n\nПодробности.", doc_date="2028-09-15")],
            subject_boost=2,
        )
        self.assertTrue(engine.search("возврат", "box-a", "2028-09-16").hits)

    def test_unknown_dates_are_excluded_from_exact_day(self) -> None:
        engine = SearchEngine([doc("n1", text="письмо без даты", doc_date=None)])
        result = engine.search("письмо вчера", "box-a", "2028-09-16", top_k=10)
        self.assertEqual(result.hits, ())


class TestDevSet(unittest.TestCase):
    """Смоук на реальном наборе: бейзлайн обязан быть выше случайного."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_documents(DATA / "documents.jsonl")
        cls.queries = load_queries(DATA / "queries_dev.jsonl")
        cls.qrels = load_qrels(DATA / "qrels_dev.jsonl")
        cls.engine = SearchEngine(cls.documents)

    def test_qrels_are_inside_mailbox_and_day(self) -> None:
        by_id = {document.doc_id: document for document in self.documents}
        for query in self.queries:
            parsed = extract_date(normalize(query.text), query.date)
            self.assertIsNotNone(parsed.date, query.text)
            for doc_id in self.qrels[query.query_id]:
                document = by_id[doc_id]
                self.assertEqual(document.mailbox, query.mailbox)
                self.assertEqual(document.date, parsed.date)

    def test_recall_at_10_above_floor(self) -> None:
        recalls = []
        for query in self.queries:
            result = self.engine.search(query.text, query.mailbox, query.date, top_k=10)
            ranked = [hit.doc_id for hit in result.hits]
            recalls.append(metrics.recall_at_k(ranked, self.qrels[query.query_id], 10))
        mean_recall = sum(recalls) / len(recalls)
        self.assertGreater(mean_recall, 0.5, f"Recall@10 = {mean_recall:.3f}")


if __name__ == "__main__":
    unittest.main()
