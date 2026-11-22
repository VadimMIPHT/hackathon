"""Serve the Sites frontend and search the organizer archive on loopback."""
import argparse
import json
import logging
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from email.utils import parseaddr
from functools import partial
from http.client import HTTPConnection
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn, UnixStreamServer
from threading import Lock
from urllib.parse import parse_qs, urlencode, urlsplit

from mail_search.assistant import MailAssistant
from mail_search.conditional import (
    ConditionalReminders, MAX_CANDIDATES, confirm, contains_anchors, default_anchors, reminder_draft,
)
from mail_search.dataset import Dataset, Document, Query, load_archive
from mail_search.reranker_data import documents_by_mailbox
from mail_search.retrieval import BM25, CALENDAR_DATE, YESTERDAY, day_from_query, eligible_indices, fields, searchable_text

LOGGER = logging.getLogger(__name__)
SENDER_HEADER = re.compile(r"\A\s*(?:от|отправитель)[ \t]*:[ \t]*([^\r\n]*)\r?\n", re.IGNORECASE)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "mipt_mail_search_participants.zip"
PROTOCOL = PROJECT_ROOT / "configs" / "final-test-protocol.json"
PREVIEW_CHARS = 180
TOP_K = 50


class ThreadingUnixHTTPServer(ThreadingMixIn, UnixStreamServer):
    daemon_threads = True


class RemoteError(Exception):
    pass


def present(doc: Document, *, full: bool = False) -> dict:
    header = SENDER_HEADER.match(doc.text)
    sender = address = ""
    text = doc.text
    if header:
        raw_name = header[1].strip()
        parsed_name, parsed_address = parseaddr(raw_name)
        sender = parsed_name or raw_name
        if "@" in parsed_address:
            sender = parsed_name or parsed_address
            address = parsed_address
        text = text[header.end():]
    subject, body = fields(text)
    return {
        "id": doc.doc_id, "mailbox": doc.mailbox, "sender": sender,
        "address": address, "subject": subject,
        "body": body if full else body[:PREVIEW_CHARS],
        "date": doc.received.isoformat() if doc.received else None,
    }


class NeuralRanker:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        from mail_search.e5 import E5, rank
        from mail_search.reranker import Reranker

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Select exactly one available CUDA device with CUDA_VISIBLE_DEVICES")
        torch.set_num_threads(2)
        memory_limit = 6 * (1 << 30) / torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(memory_limit, 1.0), device=0)
        protocol = json.loads(PROTOCOL.read_text())
        self.e5 = E5(args.e5_model)
        self.rank = rank
        self.bge = Reranker(
            args.bge_model, protocol["bge_revision"], protocol["bge_max_length"],
            protocol["bge_train_layers"],
        )
        self.bge.load_weights(args.bge_weights)
        self.bge.model.eval()
        self.batch_size = protocol["bge_batch_size"]
        self.matrices = {}
        self.lock = Lock()

    def search(self, query: Query, docs: list[Document], eligible: list[int]) -> list[str]:
        if not eligible:
            return []
        ids = [doc.doc_id for doc in docs]
        texts = [searchable_text(doc) for doc in docs]
        with self.lock:
            if query.mailbox not in self.matrices:
                LOGGER.info("Indexing a mailbox with %d documents", len(docs))
                self.matrices[query.mailbox] = self.e5.encode(texts, "passage: ")
            vector = self.e5.encode([query.text], "query: ")[0]
            candidates = self.rank(self.matrices[query.mailbox], vector, eligible, ids, TOP_K)
            by_id = dict(zip(ids, texts, strict=True))
            scoring_ids = sorted(candidates)
            scores = []
            for start in range(0, len(scoring_ids), self.batch_size):
                batch = scoring_ids[start:start + self.batch_size]
                scores.extend(self.bge.score(query.text, [by_id[doc_id] for doc_id in batch]))
            logits = dict(zip(scoring_ids, scores, strict=True))
        return sorted(candidates, key=lambda doc_id: (-logits[doc_id], doc_id))


class ArchiveSearch:
    def __init__(
        self, dataset: Dataset, neural: NeuralRanker | None, reminders_db: Path,
        openrouter_key_file: Path | None = None, assistant_limit: int = 20, *,
        use_openrouter_env: bool = False, upstream_port: int | None = None, condition_checks: int = 0,
    ) -> None:
        self.dataset = dataset
        boxes = documents_by_mailbox(dataset)
        self.boxes = {box: sorted(docs, key=lambda doc: (doc.received or date.min, doc.doc_id), reverse=True)
                      for box, docs in sorted(boxes.items())}
        self.lexical = {box: BM25(docs) for box, docs in self.boxes.items()}
        self.neural = neural
        self.upstream_port = upstream_port
        self.reminders_db = reminders_db
        reminders_db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(reminders_db)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reminders ("
                "dataset TEXT NOT NULL, mailbox TEXT NOT NULL, message_id TEXT NOT NULL, "
                "due_at TEXT NOT NULL, PRIMARY KEY (dataset, mailbox, message_id))"
            )
        self.assistant = MailAssistant(reminders_db, openrouter_key_file, assistant_limit, use_env=use_openrouter_env)
        self.conditional = ConditionalReminders(reminders_db, dataset.archive_sha256, condition_checks) if condition_checks else None
        self.profiles = {}
        for position, (box, docs) in enumerate(self.boxes.items(), start=1):
            sample = [query for query in dataset.queries.values() if query.mailbox == box]
            snapshot = max((q.as_of for q in sample), default=docs[0].received or date.today())
            examples = list(dict.fromkeys(q.text for q in sample if q.as_of == snapshot))[:3]
            self.profiles[box] = {
                "id": box, "name": f"Ящик {position:02d}", "initials": f"{position:02d}",
                "asOf": snapshot.isoformat(), "examples": examples,
            }

    def messages(self, mailbox: str, as_of: date) -> dict:
        query = Query("web", mailbox, "", as_of)
        docs = self.boxes[mailbox]
        return {"messages": [present(docs[i]) for i in eligible_indices(docs, query, use_query_day=False)]}

    def upstream(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        try:
            with closing(HTTPConnection("127.0.0.1", self.upstream_port, timeout=60)) as connection:
                connection.request(method, path, body, {"Content-Type": "application/json"})
                response = connection.getresponse()
                if response.status != 200:
                    raise RemoteError("H200 HTTP request failed")
                return json.loads(response.read(2_000_000))
        except (OSError, ValueError) as error:
            raise RemoteError("Could not connect to H200") from error

    def search(self, request: dict, *, day_text: str | None = None) -> dict:
        mailbox = request["mailbox"]
        text = request["query"]
        mode = request["mode"]
        if not isinstance(text, str) or len(text.strip()) > 500:
            raise ValueError("Invalid query length")
        if mode not in ("semantic", "lexical"):
            raise ValueError("Unknown search mode")
        as_of = date.fromisoformat(request.get("asOf", self.profiles[mailbox]["asOf"]))
        query = Query("web", mailbox, text.strip(), as_of)
        date_query = query if day_text is None else Query("web", mailbox, day_text, as_of)
        docs = self.boxes[mailbox]
        eligible = eligible_indices(docs, date_query, use_query_day=True)
        filters = request.get("filters", {})
        ids = request.get("orderIds", [])
        if not isinstance(filters, dict) or not isinstance(ids, list) or len(ids) > 16:
            raise ValueError("Invalid filters")
        sender = filters.get("sender", "")
        if not isinstance(sender, str) or len(sender) > 150:
            raise ValueError("Invalid sender")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9]{2,16}|SYN-[0-9a-f]{6,24}", value, re.I) for value in ids):
            raise ValueError("Invalid order IDs")
        start = date.fromisoformat(filters["from"]) if filters.get("from") else None
        end = date.fromisoformat(filters["to"]) if filters.get("to") else None
        if start and end and end < start:
            raise ValueError("Invalid search period")
        if not query.text and not (sender or start or end or ids):
            raise ValueError("Empty search")
        if start or end:
            eligible = [i for i in eligible if docs[i].received is not None
                        and (not start or docs[i].received >= start)
                        and (not end or docs[i].received <= end)]
        if sender:
            initials = sender.casefold().replace("ё", "е")
            eligible = [i for i in eligible if (header := SENDER_HEADER.match(docs[i].text))
                        and initials in header[1].casefold().replace("ё", "е")]
        if ids:
            alternatives = "|".join(ids)
            exact = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.I)
            eligible = [i for i in eligible if exact.search(docs[i].text)]
        archive_reference = any(value.upper().startswith("SYN-") for value in ids)
        if self.upstream_port and query.text and mode == "semantic" and not archive_reference:
            retrieval_text = query.text
            if day_text is not None:
                # Preserve a date in the user's query. A tool rewrite cannot introduce a date filter.
                retrieval_text = day_text if day_from_query(date_query) else YESTERDAY.sub(
                    "", CALENDAR_DATE.sub("", retrieval_text),
                ).strip() or day_text
            return self.upstream("POST", "/api/search", {
                "mailbox": mailbox, "query": retrieval_text, "mode": mode,
                "asOf": as_of.isoformat(), "filters": filters, "orderIds": ids,
            })
        if not query.text or archive_reference and self.upstream_port:
            ranked_ids = [docs[i].doc_id for i in eligible[:TOP_K]]
        elif mode == "semantic":
            if self.neural is None:
                raise ValueError("Semantic search is disabled")
            ranked_ids = self.neural.search(query, docs, eligible)
        else:
            ranked_ids = self.lexical[mailbox].rank(query.text, eligible, TOP_K)
        day = day_from_query(date_query)
        return {
            "ids": ranked_ids, "date": day.isoformat() if day else None,
            "messages": [present(self.dataset.documents[doc_id]) for doc_id in ranked_ids],
        }

    def ask(self, request: dict) -> dict:
        question, mailbox, query = request["question"], request["mailbox"], request["query"]
        if not isinstance(question, str) or not 0 < len(question.strip()) <= 500:
            raise ValueError("Invalid question")
        if mailbox not in self.boxes or not isinstance(query, str) or len(query) > 500:
            raise ValueError("Invalid search scope")
        if request.get("autoRoute", "") not in ("", "search", "agent"):
            raise ValueError("Invalid automatic route")
        context = request.get("context", [])
        source_ids = request.get("sourceIds", [])
        order_ids = request.get("orderIds", [])
        if not isinstance(context, list) or len(context) > 2 or any(
            not isinstance(text, str) or len(text) > 500 for text in context
        ):
            raise ValueError("Invalid conversation context")
        if not isinstance(source_ids, list) or len(source_ids) > TOP_K or not all(isinstance(id_, str) for id_ in source_ids):
            raise ValueError("Invalid conversation scope")
        if not isinstance(order_ids, list) or len(order_ids) > 16 or not all(
            isinstance(id_, str) and re.fullmatch(r"[0-9]{2,16}|SYN-[0-9a-f]{6,24}", id_, re.I) for id_ in order_ids
        ):
            raise ValueError("Invalid reference")
        as_of = date.fromisoformat(request.get("asOf", self.profiles[mailbox]["asOf"]))
        docs = [self.dataset.documents[id_] for id_ in dict.fromkeys(source_ids)]
        if any(doc.mailbox != mailbox or doc.received and doc.received > as_of for doc in docs):
            raise ValueError("Message is outside conversation scope")
        scoped_result = {"ids": [doc.doc_id for doc in docs], "date": None, "messages": [present(doc) for doc in docs]}
        draft = reminder_draft(question, as_of) if self.conditional else None
        if draft is not None:
            focused = request.get("focusedId")
            if focused:
                doc = self.dataset.documents[focused]
                if doc.mailbox != mailbox or doc.received and doc.received > as_of:
                    raise ValueError("Message is outside conversation scope")
                docs = [doc]
                scoped_result = {"ids": [doc.doc_id], "date": None, "messages": [present(doc)]}
            elif not docs and order_ids:
                search = self.search({**request, "query": " ".join(order_ids)}, day_text="")
                scoped_result = search
                docs = [self.dataset.documents[doc_id] for doc_id in search["ids"]]
            draft["sources"] = []
            for doc in docs[:6]:
                matched_ids = [item.upper() for item in order_ids if contains_anchors(doc.text, [item])]
                anchors = list(dict.fromkeys([*matched_ids, *default_anchors(doc.text)]))[:6]
                draft["sources"].append({"id": doc.doc_id, "anchors": anchors})
            return {**scoped_result, "answer": {"evidence": [], "conditionalDraft": draft}}

        return self.assistant.answer(
            question.strip(), request,
            lambda params, day_text: scoped_result if source_ids else self.search(params, day_text=day_text),
            lambda doc_id: present(self.dataset.documents[doc_id], full=True),
        )

    def reminders(self, mailbox: str) -> dict:
        if mailbox not in self.boxes:
            raise KeyError("Unknown mailbox")
        if self.upstream_port:
            return self.upstream("GET", "/api/reminders?" + urlencode({"mailbox": mailbox}))
        with closing(sqlite3.connect(self.reminders_db)) as connection:
            rows = connection.execute(
                "SELECT message_id, due_at FROM reminders WHERE dataset = ? AND mailbox = ? "
                "ORDER BY due_at, message_id", (self.dataset.archive_sha256, mailbox),
            ).fetchall()
        return {"reminders": [
            {"message": present(self.dataset.documents[doc_id]), "dueAt": due_at}
            for doc_id, due_at in rows
        ]}

    def set_reminder(self, request: dict, *, remove: bool) -> dict:
        mailbox, doc_id = request["mailbox"], request["messageId"]
        if self.dataset.documents[doc_id].mailbox != mailbox:
            raise KeyError("Unknown message")
        if self.upstream_port:
            return self.upstream("DELETE" if remove else "PUT", "/api/reminders", request)
        key = (self.dataset.archive_sha256, mailbox, doc_id)
        with closing(sqlite3.connect(self.reminders_db)) as connection, connection:
            if remove:
                connection.execute(
                    "DELETE FROM reminders WHERE dataset = ? AND mailbox = ? AND message_id = ?", key,
                )
            else:
                due_at = datetime.fromisoformat(request["dueAt"].replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if due_at.tzinfo is None or not now < due_at <= now + timedelta(days=365):
                    raise ValueError("Invalid reminder time")
                connection.execute(
                    "INSERT INTO reminders VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (dataset, mailbox, message_id) DO UPDATE SET due_at = excluded.due_at",
                    (*key, due_at.astimezone(timezone.utc).isoformat()),
                )
        return self.reminders(mailbox)

    def confirmation_candidates(
        self, source: Document, start: date, end: date, condition: str, anchors: list[str],
    ) -> list[Document]:
        docs = self.boxes[source.mailbox]
        eligible = [i for i, doc in enumerate(docs) if doc.doc_id != source.doc_id and doc.received
                    and start <= doc.received <= end and contains_anchors(doc.text, anchors)]
        if len(eligible) <= MAX_CANDIDATES:
            return [docs[i] for i in reversed(eligible)]
        ranked = self.lexical[source.mailbox].rank(condition, eligible, MAX_CANDIDATES)
        if not anchors and (self.neural or self.upstream_port):
            try:
                result = self.search({"mailbox": source.mailbox, "query": condition, "asOf": end.isoformat(),
                                      "filters": {"from": start.isoformat()}, "mode": "semantic"}, day_text="")
            except RemoteError:
                result = {"ids": []}
            eligible_ids = {docs[i].doc_id for i in eligible}
            ranked = [doc_id for doc_id in result["ids"] if doc_id in eligible_ids][:MAX_CANDIDATES] + ranked
        shortlist = [*ranked, *(docs[i].doc_id for i in eligible)]
        return [self.dataset.documents[doc_id] for doc_id in list(dict.fromkeys(shortlist))[:MAX_CANDIDATES]]

    def conditional_reminders(self, mailbox: str, as_of: date) -> dict:
        if not self.conditional or mailbox not in self.boxes:
            raise ValueError("Conditional reminders are unavailable")
        key = self.assistant.key()
        def judge(source: Document, condition: str, anchors: list[str], doc: Document) -> str | None:
            return confirm(key, condition, anchors, present(doc, full=True), present(source, full=True))

        return {"reminders": self.conditional.list(
            mailbox, as_of, self.dataset.documents, self.confirmation_candidates, present, judge if key else None,
        )}

    def set_conditional_reminder(self, request: dict, *, remove: bool) -> dict:
        doc = self.dataset.documents[request["messageId"]]
        if not self.conditional or doc.mailbox != request["mailbox"]:
            raise ValueError("Unknown reminder")
        as_of = date.fromisoformat(request["asOf"])
        if remove:
            self.conditional.remove(doc)
        else:
            condition, anchors = request["condition"], request["anchors"]
            if not isinstance(condition, str) or not isinstance(anchors, list) or not all(isinstance(a, str) for a in anchors):
                raise ValueError("Invalid condition")
            self.conditional.save(doc, as_of, datetime.fromisoformat(request["dueAt"]), condition.strip(),
                                  list(dict.fromkeys(a.strip() for a in anchors)))
        return {"saved": True}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, app: ArchiveSearch, port: int, **kwargs) -> None:
        self.app = app
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        if len(args) > 1 and str(args[1]).isdigit() and int(args[1]) >= 500:
            LOGGER.warning("Local HTTP request failed: %s", args[1])

    def reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def local_host(self) -> bool:
        if self.headers.get("Host") in self.hosts:
            return True
        self.send_error(403)
        return False

    def do_HEAD(self) -> None:
        if self.local_host():
            super().do_HEAD()

    def do_GET(self) -> None:
        if not self.local_host():
            return
        url = urlsplit(self.path)
        if not url.path.startswith("/api/"):
            super().do_GET()
            return
        params = parse_qs(url.query)
        try:
            match url.path:
                case "/api/mailboxes":
                    payload = {
                        "mailboxes": list(self.app.profiles.values()),
                        "engine": "e5-bge" if self.app.neural or self.app.upstream_port else "bm25",
                        "assistant": True,
                        "conditionalReminders": self.app.conditional is not None,
                    }
                case "/api/messages":
                    mailbox = params["mailbox"][0]
                    as_of = date.fromisoformat(params.get("asOf", [self.app.profiles[mailbox]["asOf"]])[0])
                    payload = self.app.messages(mailbox, as_of)
                case "/api/message":
                    doc = self.app.dataset.documents[params["id"][0]]
                    if doc.mailbox != params["mailbox"][0]:
                        raise KeyError("Unknown message")
                    payload = {"message": present(doc, full=True)}
                    if self.app.conditional:
                        payload["message"]["conditionAnchors"] = default_anchors(doc.text)
                case "/api/reminders":
                    payload = self.app.reminders(params["mailbox"][0])
                case "/api/conditional-reminders":
                    payload = self.app.conditional_reminders(params["mailbox"][0], date.fromisoformat(params["asOf"][0]))
                case _:
                    self.reply(404, {"error": "Unknown endpoint"})
                    return
        except (KeyError, ValueError):
            self.reply(400, {"error": "Invalid mailbox, date or message"})
            return
        except RemoteError:
            self.reply(503, {"error": "H200 service is unavailable"})
            return
        self.reply(200, payload)

    def do_POST(self) -> None:
        if self.path == "/api/ask":
            self.mutate("/api/ask", self.app.ask)
        else:
            self.mutate("/api/search", self.app.search)

    def do_PUT(self) -> None:
        if self.path == "/api/conditional-reminders":
            return self.mutate(self.path, partial(self.app.set_conditional_reminder, remove=False))
        self.mutate("/api/reminders", partial(self.app.set_reminder, remove=False))

    def do_DELETE(self) -> None:
        if self.path == "/api/conditional-reminders":
            return self.mutate(self.path, partial(self.app.set_conditional_reminder, remove=True))
        self.mutate("/api/reminders", partial(self.app.set_reminder, remove=True))

    def mutate(self, endpoint: str, action: Callable[[dict], dict]) -> None:
        if not self.local_host():
            return
        if self.path != endpoint:
            self.reply(404, {"error": "Unknown endpoint"})
            return
        host = self.headers["Host"]
        if self.headers.get("Origin", f"http://{host}") != f"http://{host}":
            self.reply(403, {"error": "Same-origin requests only"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit = 12288 if endpoint == "/api/ask" else 4096
            if self.headers.get_content_type() != "application/json" or not 0 < length <= limit:
                raise ValueError("Invalid request")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("Invalid JSON payload")
            result = action(request)
        except (AttributeError, KeyError, TypeError, ValueError):
            self.reply(400, {"error": "Invalid request"})
            return
        except RemoteError:
            self.reply(503, {"error": "H200 service is unavailable"})
            return
        except Exception:
            LOGGER.exception("Local API request failed")
            self.reply(500, {"error": "Request failed"})
            return
        self.reply(200, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help=f"Organizer ZIP. Default: data/{ARCHIVE_NAME}")
    parser.add_argument("--dist", type=Path, default=PROJECT_ROOT / "sites" / "pismo" / "dist")
    parser.add_argument("--port", type=int, default=8940)
    parser.add_argument("--unix-socket", type=Path, help="Bind a private Unix socket for SSH port forwarding")
    parser.add_argument("--reminders-db", type=Path, default=Path("runs/pismo/reminders.sqlite3"))
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--openrouter-key-file", type=Path, help="Owner-only file with the OpenRouter API key")
    parser.add_argument("--openrouter-env", action="store_true", help="Use the OPENROUTER_API_KEY environment variable")
    parser.add_argument("--assistant-limit", type=int, default=20, help="Total number of questions sent to OpenRouter")
    parser.add_argument("--condition-checks", type=int, default=0,
                        help="Enable conditional reminders and set the separate Jev request budget")
    parser.add_argument("--upstream-port", type=int, help="Use an SSH-forwarded H200 search/reminder API on localhost")
    for name in ("e5-model", "bge-model", "bge-weights"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    archive = args.archive or PROJECT_ROOT / "data" / ARCHIVE_NAME
    if not archive.is_file():
        parser.error(f"Archive not found at {archive}. Pass --archive /path/to/{ARCHIVE_NAME}.")
    if not (args.dist / "index.html").is_file():
        parser.error("--dist must point to the Sites dist folder")
    if args.assistant_limit < 0 or args.condition_checks < 0:
        parser.error("OpenRouter request limits must be nonnegative")
    if args.openrouter_env and args.openrouter_key_file:
        parser.error("Select one OpenRouter key source")
    if args.upstream_port and not (0 < args.upstream_port < 65536 and args.upstream_port != args.port):
        parser.error("--upstream-port must be a different local TCP port")
    if not (args.lexical_only or args.upstream_port or all((args.e5_model, args.bge_model, args.bge_weights))):
        parser.error("Supply --e5-model, --bge-model and --bge-weights, or select --lexical-only")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset = load_archive(archive)
    neural = None if args.lexical_only or args.upstream_port else NeuralRanker(args)
    app = ArchiveSearch(
        dataset, neural, args.reminders_db, args.openrouter_key_file, args.assistant_limit,
        use_openrouter_env=args.openrouter_env, upstream_port=args.upstream_port,
        condition_checks=args.condition_checks,
    )
    handler = partial(Handler, app=app, port=args.port, directory=str(args.dist.resolve()))
    if args.unix_socket:
        socket_path = args.unix_socket.resolve()
        if socket_path.parent.stat().st_mode & 0o077:
            parser.error("The Unix socket directory must be private (chmod 700)")
        server = ThreadingUnixHTTPServer(str(socket_path), handler)
        socket_path.chmod(0o600)
        LOGGER.info("Письмо is listening on a private Unix socket")
    else:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        LOGGER.info("Письмо is available at http://127.0.0.1:%d/", args.port)
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
