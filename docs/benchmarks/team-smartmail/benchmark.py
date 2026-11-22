import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.request import Request, urlopen
from zipfile import ZipFile

from smartmail.engine import SearchEngine
from smartmail.io import load_documents, load_qrels, load_queries

from mail_search.benchmark import METRICS, measure


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(rows: list[dict], metric_set: str) -> dict:
    metrics = {key: [row[metric_set][key] for row in rows if row[metric_set][key] is not None]
               for key in METRICS}
    return {"queries": len(rows), **{key: mean(values) if values else None for key, values in metrics.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jev", type=Path, required=True, help="Extracted vk-jev-dataset directory")
    parser.add_argument("--public", type=Path, required=True, help="Original VK archive")
    parser.add_argument("--mailbox-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--team-commit", required=True)
    parser.add_argument("--metrics-commit", required=True)
    parser.add_argument("--idf-mode", choices=("global", "local"), default="global")
    parser.add_argument("--search-url", help="Verify rankings against the running website's /api/search endpoint")
    args = parser.parse_args()
    paths = {name: args.jev / name for name in ("documents.jsonl", "queries_dev.jsonl", "qrels_dev.jsonl")}
    manifest = json.loads((args.jev / "manifest.json").read_text())
    file_hashes = {name: digest(path) for name, path in paths.items()}
    if any(value != manifest["sha256"][name] for name, value in file_hashes.items()):
        raise ValueError("Jev files differ from manifest")
    documents = load_documents(paths["documents.jsonl"])
    queries = load_queries(paths["queries_dev.jsonl"])
    jev = load_qrels(paths["qrels_dev.jsonl"])
    with ZipFile(args.public) as archive:
        with archive.open("documents.jsonl") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != file_hashes["documents.jsonl"]:
                raise ValueError("VK and Jev document corpora differ")
        public = {}
        for line in archive.open("qrels_dev.jsonl"):
            row = json.loads(line)
            if row["relevance"] == 1:
                public.setdefault(row["query_id"], set()).add(row["doc_id"])
    if len(queries) != 139 or len(public) != 99 or len(jev) != 139 or len(documents) != 16376:
        raise ValueError("Unexpected corpus or evaluation cohorts")
    start = perf_counter()
    engine = SearchEngine(documents, idf_mode=args.idf_mode)
    indexing_seconds = perf_counter() - start
    rows = []
    for query in queries:
        result = engine.search(query.text, query.mailbox, query.date, top_k=50)
        ids = [hit.doc_id for hit in result.hits]
        row = {"query_id": query.query_id, "mailbox": query.mailbox,
               "eligible_documents": result.candidates, "ranked_doc_ids": ids,
               "parsed_start": result.target_date.isoformat() if result.target_date else None,
               "parsed_end": result.target_end.isoformat() if result.target_end else None,
               "jev": measure(ids, dict.fromkeys(jev[query.query_id], 1))}
        if query.query_id in public:
            row["public_vk"] = measure(ids, dict.fromkeys(public[query.query_id], 1))
        if args.search_url:
            request = Request(args.search_url, json.dumps({
                "mailbox": query.mailbox, "query": query.text, "asOf": query.date.isoformat(),
            }).encode(), {"Content-Type": "application/json"})
            started = perf_counter()
            with urlopen(request, timeout=15) as response:
                remote_ids = json.load(response)["ids"]
            row["http_ms"] = (perf_counter() - started) * 1000
            if remote_ids != ids:
                raise ValueError(f"Search endpoint ranks {query.query_id} differently from {args.idf_mode} BM25")
        rows.append(row)
    original = [r for r in rows if "public_vk" in r]
    new = [r for r in rows if "public_vk" not in r]
    split = json.loads(args.mailbox_split.read_text())
    test_boxes = set(split["splits"]["test"]["mailboxes"])
    test_jev = [r for r in rows if r["mailbox"] in test_boxes]
    test_public = [r for r in original if r["mailbox"] in test_boxes]
    output = {
        "team_commit": args.team_commit, "metrics_commit": args.metrics_commit, "script_sha256": digest(Path(__file__)),
        "python": platform.python_version(), "platform": platform.system(), "cpu_architecture": platform.machine(),
        "config": {"engine": "smartmail.engine.SearchEngine", "k1": 1.5, "b": 0.75, "idf": args.idf_mode,
                   "subject_boost": 1, "top_k": 50, "seed": None, "dates": "native team-smartmail parser"},
        "data_sha256": {**file_hashes, "public_vk_archive": digest(args.public),
                        "mailbox_split": digest(args.mailbox_split)},
        "mailbox_split_seed": split["seed"],
        "indexing_seconds": indexing_seconds,
        "summary": {"original99_public_vk": summarize(original, "public_vk"),
                    "original99_jev": summarize(original, "jev"), "new40_jev": summarize(new, "jev"),
                    "all139_jev": summarize(rows, "jev"),
                    "test23_public_vk": summarize(test_public, "public_vk"),
                    "test30_jev": summarize(test_jev, "jev")},
        "per_query": rows,
    }
    if args.search_url:
        elapsed = sorted(row["http_ms"] for row in rows)
        output["http"] = {"exact_rankings": len(elapsed), "p99_ms": elapsed[math.ceil(0.99 * len(elapsed)) - 1],
                          "scope": "sequential requests, prebuilt index, HTTP round-trip including JSON parsing"}
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
