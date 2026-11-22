from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from mail_search.conditional import confirm, default_anchors, reminder_draft
from mail_search.dataset import Dataset, Document
from mail_search.intent import Decision
from mail_search.web import ArchiveSearch


class ConditionalReminderTest(TestCase):
    def test_chat_proposes_condition_and_deadline_for_archive_snapshot(self):
        as_of = date(2028, 9, 12)
        self.assertEqual(reminder_draft(
            "Напомни 15 сентября в 9:30, если машина так и не приехала.", as_of,
        ), {"condition": "Машина приехала", "dueAt": "2028-09-15T09:30"})
        self.assertEqual(reminder_draft(
            "Напомни завтра, если не будет подтверждения, что договор подписан обеими сторонами.", as_of,
        ), {"condition": "Договор подписан обеими сторонами", "dueAt": "2028-09-13T09:00"})
        self.assertIsNone(reminder_draft("Напомни завтра, если договор подписан.", as_of))

    def test_chat_keeps_full_source_anchors_and_requires_user_to_save(self):
        docs = [
            Document("first", "box", "Тема: Автовизит SYN-112233aabb\n\n" + "А" * 800 +
                     "\nКлиентSYN-112233ddee", date(2028, 9, 12)),
            Document("other", "box", "Клиент SYN-aabb112233", date(2028, 9, 12)),
            Document("foreign", "other", "Клиент SYN-123456abcd", date(2028, 9, 12)),
        ]
        data = Dataset("test", {doc.doc_id: doc for doc in docs}, {}, {})
        question = "Напомни 15 сентября в 9:00, если машина ещё не приехала."
        request = {"question": question, "query": question, "sourceIds": ["first", "other"],
                   "mailbox": "box", "asOf": "2028-09-12", "autoRoute": "agent"}
        with TemporaryDirectory() as tmp:
            app = ArchiveSearch(data, None, Path(tmp) / "app.db", condition_checks=3)
            with patch.object(app.assistant, "answer") as model:
                choices = app.ask(request)["answer"]["conditionalDraft"]["sources"]
                self.assertEqual([item["id"] for item in choices], ["first", "other"])
                only = app.ask({**request, "focusedId": "first"})["answer"]["conditionalDraft"]["sources"]
                self.assertEqual(only, [{"id": "first", "anchors": ["SYN-112233AABB", "SYN-112233DDEE"]}])
                with self.assertRaises(ValueError):
                    app.ask({**request, "focusedId": "foreign"})
                model.assert_not_called()
            self.assertEqual(app.conditional_reminders("box", date(2028, 9, 12))["reminders"], [])

    def test_matches_only_new_mail_for_the_same_case_and_keeps_the_evidence(self):
        source = Document("start", "box", "Заказ SYN-112233aabb, клиентSYN-112233ddee", date(2028, 9, 12))
        wrong = Document("wrong", "box", "Контейнер SYN-112233aabb прибыл. Клиент SYN-aabb112233.", date(2028, 9, 13))
        promised = Document("promise", "box", "SYN-112233aabb. SYN-112233ddee. Машина скоро выедет.", date(2028, 9, 13))
        right = Document("right", "box", "SYN-112233aabb SYN-112233ddee\nАвтомашина прибыла.", date(2028, 9, 14))
        late = Document("late", "box", "SYN-112233aabb SYN-112233ddee\nАвтомашина прибыла.", date(2028, 9, 15))
        other_box = Document("other", "other", right.text, date(2028, 9, 13))
        data = Dataset("test", {doc.doc_id: doc for doc in (source, wrong, promised, right, late, other_box)}, {}, {})
        payload = {"messageId": "start", "mailbox": "box", "asOf": "2028-09-12",
                   "dueAt": "2028-09-15T09:00", "condition": "Грузовик приехал",
                   "anchors": default_anchors(source.text)}
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENROUTER_API_KEY": "mock"}):
            db = Path(tmp) / "app.db"
            app = ArchiveSearch(data, None, db, use_openrouter_env=True, condition_checks=8)
            with patch("mail_search.web.confirm", side_effect=[None, "Автомашина прибыла."]) as judge:
                app.set_conditional_reminder(payload, remove=False)
                self.assertEqual(app.conditional_reminders("box", date(2028, 9, 11))["reminders"], [])
                self.assertEqual(app.conditional_reminders("box", date(2028, 9, 12))["reminders"][0]["status"], "pending")
                self.assertEqual(app.conditional_reminders("box", date(2028, 9, 13))["reminders"][0]["status"], "pending")
                app = ArchiveSearch(data, None, db, use_openrouter_env=True, condition_checks=8)
                result = app.conditional_reminders("box", date(2028, 9, 15))["reminders"][0]
                self.assertEqual((result["status"], result["evidence"]["id"], result["quote"]),
                                 ("fulfilled", "right", "Автомашина прибыла."))
                self.assertEqual([call.args[3]["id"] for call in judge.call_args_list], ["promise", "right"])
                offline = ArchiveSearch(data, None, db, condition_checks=8)
                self.assertEqual(offline.conditional_reminders("box", date(2028, 9, 15))["reminders"][0], result)

    def test_only_accepts_a_high_confidence_quote(self):
        message = {"subject": "Заявка", "body": "Документ подписан."}
        with patch("mail_search.conditional.choose", side_effect=[
            Decision("p2", .88, 0, 0, 0), Decision("p2", .94, 0, 0, 0),
        ]):
            self.assertIsNone(confirm("mock", "Файл подписали", [], message))
            self.assertEqual(confirm("mock", "Файл подписали", [], message), "Документ подписан.")

    def test_past_deadline_without_confirmation_stays_due(self):
        source = Document("start", "box", "Счёт 4172", date(2028, 9, 12))
        paid = Document("paid", "box", "Счёт 4172 оплачен.", date(2028, 9, 16))
        data = Dataset("test", {"start": source, "paid": paid}, {}, {})
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENROUTER_API_KEY": "mock"}):
            app = ArchiveSearch(data, None, Path(tmp) / "app.db", use_openrouter_env=True, condition_checks=1)
            app.set_conditional_reminder({"messageId": "start", "mailbox": "box", "asOf": "2028-09-12",
                                          "dueAt": "2028-09-15T09:00", "condition": "Счёт оплачен",
                                          "anchors": ["4172"]}, remove=False)
            with patch("mail_search.web.confirm") as judge:
                self.assertEqual(app.conditional_reminders("box", date(2028, 9, 16))["reminders"][0]["status"], "due")
                judge.assert_not_called()
