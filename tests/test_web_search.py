from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mail_search.dataset import Dataset, Document
from mail_search.web import ArchiveSearch


class WebSearchTest(TestCase):
    def test_default_idf_uses_candidates_after_mailbox_and_date_filters(self):
        query_day = date(2028, 9, 14)
        first_day = date(2028, 9, 12)
        docs = [
            Document("quartz", "box", "Кварц", first_day),
            Document("granite-first", "box", "Гранит", first_day),
            Document("granite-next", "box", "Гранит", date(2028, 9, 13)),
            Document("too-old", "box", "Кварц", date(2028, 9, 11)),
            Document("future", "box", "Кварц", date(2028, 9, 15)),
            Document("unknown-date", "box", "Кварц", None),
            *(Document(f"other-{i}", "other", "Кварц", first_day) for i in range(4)),
        ]
        with TemporaryDirectory() as folder:
            app = ArchiveSearch(Dataset("test", {doc.doc_id: doc for doc in docs}, {}, {}),
                                None, Path(folder) / "reminders.db")
            result = app.search({"mailbox": "box", "asOf": str(query_day),
                                 "query": "Покажи кварц и гранит с 12 по 13 сентября 2028 года."})
        self.assertEqual(result["ids"], ["quartz", "granite-next", "granite-first"])
        self.assertEqual((result["date"], result["dateEnd"]), ("2028-09-12", "2028-09-13"))

    def test_preserves_original_date_and_matches_joined_archive_identifier(self):
        docs = [
            Document("right", "box", "От: логистика@example.org\nконтейнерSYN-112233aabb прибыл", date(2028, 9, 12)),
            Document("wrong-day", "box", "От: логистика@example.org\nконтейнерSYN-112233aabb прибыл", date(2028, 9, 13)),
            Document("wrong-sender", "box", "От: другой@example.org\nконтейнерSYN-112233aabb прибыл", date(2028, 9, 12)),
        ]
        with TemporaryDirectory() as folder:
            app = ArchiveSearch(Dataset("test", {doc.doc_id: doc for doc in docs}, {}, {}),
                                None, Path(folder) / "reminders.db")
            result = app.search({
                "mailbox": "box", "asOf": "2028-09-15", "query": "контейнер прибыл 13 сентября 2028",
                "filters": {"sender": "логистика@example.org"}, "orderIds": ["SYN-112233aabb"],
            }, day_text="письма за 12 сентября 2028 года")
        self.assertEqual(result["ids"], ["right"])
        self.assertEqual(result["date"], "2028-09-12")
