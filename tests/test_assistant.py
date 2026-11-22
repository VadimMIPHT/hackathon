import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mail_search.assistant import MailAssistant, evidence_from, reminder_request, timeline_from
from mail_search.intent import Decision


QUESTION = "Вернули ли деньги за заказ № 12345? Если нет, напомни завтра в 10:30."
PROMISE = {"id": "a", "subject": "Возврат по заказу № 12345",
           "body": "Клиенту обещали сообщить срок сегодня до 16:00."}
CONFIRMED = {"id": "b", "subject": "Возврат по заказу № 12345",
             "body": "Деньги уже зачислены на карту клиента."}
NEIGHBOR = {"id": "c", "subject": "Возврат по заказу № 67890",
            "body": "Деньги уже зачислены на карту клиента. Соседний заказ № 12345 пока ждёт."}


class AssistantTest(unittest.TestCase):
    def test_auto_search_skips_deepseek_and_routes_on_text_only(self):
        query = "Покажи письма от customer@example.org."
        request = {"query": query, "autoRoute": "agent", "context": ["Письма о доставке"], "sourceIds": ["a"]}
        with TemporaryDirectory() as folder:
            assistant = MailAssistant(Path(folder) / "usage.sqlite3", None, 1, use_env=True)
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-only"}), patch(
                "mail_search.assistant.classify", return_value=Decision("search", .99, 0, 0, 0),
            ) as classify, patch("mail_search.assistant.complete") as complete:
                result = assistant.answer(query, request, lambda *_: {"ids": ["a"]}, lambda _: self.fail("Read a mail"))
            classify.assert_called_once_with("test-only", "Покажи письма от [контакт].", "Письма о доставке")
            complete.assert_not_called()
            self.assertEqual(result, {"ids": ["a"], "answer": None, "routing": {"route": "search", "router": "jev"}})

    def test_auto_uses_agent_when_jev_is_uncertain_about_search(self):
        query = "Найди письма. А что изменилось?"
        request = {"query": query, "autoRoute": "search", "context": ["Возврат"], "sourceIds": ["a"]}
        with TemporaryDirectory() as folder:
            assistant = MailAssistant(Path(folder) / "usage.sqlite3", None, 1, use_env=True)
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-only"}), patch(
                "mail_search.assistant.classify", return_value=Decision("search", .55, 0, 0, 0),
            ), patch("mail_search.assistant.complete", return_value={}) as complete:
                result = assistant.answer(query, request, lambda *_: {"ids": []}, lambda _: None)
            complete.assert_called_once()
            self.assertEqual(result["routing"], {"route": "agent", "router": "jev"})
            self.assertIsNotNone(result["answer"])

    def test_auto_uses_local_route_when_jev_fails(self):
        query = "Найди письма с вложениями."
        request = {"query": query, "autoRoute": "search"}
        with TemporaryDirectory() as folder:
            assistant = MailAssistant(Path(folder) / "usage.sqlite3", None, 1, use_env=True)
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-only"}), patch(
                "mail_search.assistant.classify", side_effect=TimeoutError,
            ), patch("mail_search.assistant.complete") as complete:
                result = assistant.answer(query, request, lambda *_: {"ids": []}, lambda _: None)
            complete.assert_not_called()
            self.assertEqual(result["routing"], {"route": "search", "router": "rules"})

    def test_promise_has_source_and_proposes_reminder(self):
        rows = evidence_from(QUESTION, [PROMISE], True, ["12345"])
        self.assertEqual(rows, [{"id": "a", "text": PROMISE["body"], "kind": "promised"}])
        self.assertEqual(reminder_request(QUESTION, rows, True, False), {"id": "a", "days": 1, "time": "10:30"})

    def test_confirmation_suppresses_conditional_reminder(self):
        rows = evidence_from(QUESTION, [PROMISE, CONFIRMED], True, ["12345"])
        self.assertEqual({row["kind"] for row in rows}, {"confirmed", "promised"})
        self.assertIsNone(reminder_request(QUESTION, rows, True, True))

    def test_negative_and_neighbor_are_not_confirmations(self):
        negated = {**CONFIRMED, "body": "Деньги ещё не зачислены на карту клиента."}
        rows = evidence_from(QUESTION, [PROMISE, NEIGHBOR, negated], True, ["12345"])
        self.assertEqual([row["id"] for row in rows], ["a"])

    def test_timeline_tracks_matching_reference_and_approval_anchor(self):
        reference = "SYN-abcdef1234"
        docs = [
            {"id": "later", "date": "2027-11-18", "body": "",
             "subject": f"Возврат {reference} от 18.11.2027 09:11:03: статус изменился с В работе на Заявка выполнена"},
            {"id": "approval", "date": "2027-11-17", "body": "",
             "subject": f"Возврат {reference}: статус изменился с «Новая» на «Заявка одобрена»."},
            {"id": "other", "date": "2027-11-20", "body": "",
             "subject": "Возврат SYN-7777777777: статус изменился с «Новая» на «Заявка одобрена»."},
        ]
        timeline = timeline_from("А что изменилось после одобрения?", docs, [reference])
        self.assertEqual([event["id"] for event in timeline], ["approval", "later"])
        self.assertEqual(timeline[-1]["time"], "09:11:03")
        self.assertEqual(timeline[-1]["status"], "Заявка выполнена")

    def test_action_uses_dialogue_reference_without_model_call(self):
        with TemporaryDirectory() as folder:
            assistant = MailAssistant(Path(folder) / "usage.sqlite3", None, 1, use_env=True)
            question = "Тогда напомни завтра в 10:30 и подготовь короткий ответ поддержке."
            request = {
                "query": question, "mailbox": "box", "sourceIds": ["a"], "orderIds": ["12345"],
                "context": [QUESTION],
            }
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-only"}), patch(
                "mail_search.assistant.complete",
            ) as provider:
                answer = assistant.answer(
                    question, request, lambda *_: {"ids": ["a"]}, lambda _: PROMISE,
                )["answer"]
            provider.assert_not_called()
            self.assertTrue(answer["actionOnly"])
            self.assertEqual(answer["reminder"], {"id": "a", "days": 1, "time": "10:30"})
            self.assertIn("12345", answer["draft"]["text"])

    def test_model_only_rewrites_query_and_selects_known_snippets(self):
        with TemporaryDirectory() as folder:
            db = Path(folder) / "usage.sqlite3"
            assistant = MailAssistant(db, None, 1, use_env=True)
            seen = []

            def search(request, day_text):
                seen.append((request.copy(), day_text))
                return {"ids": ["a"], "messages": [PROMISE], "date": None}

            request = {"query": QUESTION, "mailbox": "box", "orderIds": ["12345"]}
            response = {"tool_calls": [{"id": "tool", "type": "function", "function": {
                "name": "search_mail", "arguments": json.dumps({"query": "возврат", "mailbox": "other"}),
            }}]}
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-only"}), patch(
                "mail_search.assistant.complete", side_effect=[response, {"content": '{"snippets":[1]}' }],
            ) as provider:
                answer = assistant.answer(QUESTION, request, search, lambda _: PROMISE)["answer"]
                self.assertEqual(answer["provider"], "deepseek")
                self.assertEqual(answer["evidence"][0]["text"], PROMISE["body"])
                self.assertEqual(seen[0][0]["mailbox"], "box")
                self.assertEqual(seen[0][0]["orderIds"], ["12345"])
                self.assertEqual(seen[0][0]["query"], "возврат")
                self.assertIn("Вернули ли деньги", seen[0][1])
                self.assertEqual(provider.call_count, 2)

                restarted = MailAssistant(db, None, 1, use_env=True)
                exhausted = restarted.answer(QUESTION, request, search, lambda _: PROMISE)["answer"]
                self.assertEqual(exhausted["provider"], "rules")
                self.assertEqual(provider.call_count, 2)


if __name__ == "__main__":
    unittest.main()
