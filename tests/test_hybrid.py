"""Тесты гибридного слоя: RRF, дедуп, стратификация, sender-фильтр."""

from __future__ import annotations

import unittest
from datetime import date

from smartmail.engine import SearchEngine
from smartmail.hybrid import HybridConfig, HybridSearcher
from smartmail.io import Document


def document(doc_id: str, text: str, sender: str = "a@example.invalid") -> Document:
    return Document(
        doc_id=doc_id,
        mailbox="box-a",
        text=f"ОТ: {sender}\n{text}",
        date=date(2028, 9, 15),
    )


class FakeDense:
    """Управляемая dense-выдача для проверки логики слияния."""

    def __init__(self, ranking: list[int]) -> None:
        self.ranking = ranking

    def search(self, query: str, candidates, top_k: int):
        allowed = set(candidates)
        return [
            (index, 1.0 / (rank + 1))
            for rank, index in enumerate(self.ranking)
            if index in allowed
        ][:top_k]


class TestHybrid(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            document("d1", "Тема: Отклонённые платежи\n\nОперация отклонена, платёж не прошёл."),
            document("d2", "Тема: Отклонённые платежи\n\nПлатёж отклонён банком."),
            document(
                "d3",
                "Тема: Успешная оплата\n\nПлатёж прошёл успешно.",
                sender="b@example.invalid",
            ),
        ]
        self.engine = SearchEngine(self.documents)

    def searcher(self, dense=None, **config) -> HybridSearcher:
        return HybridSearcher(
            self.documents,
            dense,
            engine=self.engine,
            config=HybridConfig(**config),
        )

    def test_rrf_keeps_both_sources(self) -> None:
        result = self.searcher(FakeDense([2, 0, 1])).search(
            "отклонённые платежи", "box-a", "2028-09-16", top_k=3
        )
        ids = [hit.doc_id for hit in result.hits]
        self.assertIn("d1", ids)
        self.assertIn("d3", ids)

    def test_dedup_collapses_same_sender_and_subject(self) -> None:
        result = self.searcher(dedup=True).search(
            "отклонённые платежи", "box-a", "2028-09-16", top_k=10
        )
        ids = [hit.doc_id for hit in result.hits]
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids.count("d1") + ids.count("d2"), 1)

    def test_stratify_round_robin_by_sender(self) -> None:
        plain = self.searcher().search(
            "платёж отклонён", "box-a", "2028-09-16", top_k=3
        )
        plain_ids = [hit.doc_id for hit in plain.hits]
        self.assertEqual(set(plain_ids[:2]), {"d1", "d2"})

        stratified = self.searcher(stratify=True, group_by="sender").search(
            "платёж отклонён", "box-a", "2028-09-16", top_k=3
        )
        self.assertEqual(
            [hit.doc_id for hit in stratified.hits],
            [plain_ids[0], "d3", plain_ids[1]],
        )

    def test_sender_in_query_filters_candidates(self) -> None:
        result = self.searcher().search(
            "письма от b@example.invalid про оплату", "box-a", "2028-09-16", top_k=5
        )
        self.assertEqual([hit.doc_id for hit in result.hits], ["d3"])

    def test_rm3_adds_terms_from_top_documents(self) -> None:
        searcher = self.searcher(expansion="rm3")
        result = searcher.search("платёж", "box-a", "2028-09-16", top_k=3)
        self.assertTrue(any(term not in {"платеж"} for term in result.terms))

    def test_empty_query_returns_nothing(self) -> None:
        result = self.searcher().search("", "box-a", "2028-09-16", top_k=3)
        self.assertEqual(result.hits, ())


if __name__ == "__main__":
    unittest.main()
