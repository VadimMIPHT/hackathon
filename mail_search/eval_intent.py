"""Replay labeled synthetic mail queries through the Jev decision endpoint."""
import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError, URLError

from mail_search.intent import MIN_SEARCH_CONFIDENCE, MODEL, classify


ROOT = Path(__file__).resolve().parents[1]
ROUTING_SOURCES = ("mail_search/intent.py", "sites/pismo/dist/assistant.js")


def provenance() -> dict:
    version = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {
        "base_commit": version.stdout.strip() or None,
        "routing_sources_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ROUTING_SOURCES
        },
        "python": platform.python_version(),
        "node": subprocess.check_output(["node", "--version"], text=True).strip(),
    }


def browser_routes(cases: list[dict]) -> list[str]:
    script = """
import { readFileSync } from 'node:fs';
import { routeQuestion } from './sites/pismo/dist/assistant.js';
const queries = JSON.parse(readFileSync(0, 'utf8'));
console.log(JSON.stringify(queries.map(query =>
  ['ask', 'auto-ask'].includes(routeQuestion(query)) ? 'agent' : 'search'
)));
"""
    result = subprocess.run(
        ["node", "--experimental-default-type=module", "--input-type=module", "--eval", script],
        input=json.dumps([case["query"] for case in cases], ensure_ascii=False), cwd=ROOT,
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def metrics(rows: list[dict], prediction: str) -> dict:
    return {
        "correct": sum(row[prediction] == row["expected"] for row in rows),
        "avoided_agent": sum(row["expected"] == row[prediction] == "search" for row in rows),
        "extra_agent": sum(row[prediction] == "agent" and row["expected"] == "search" for row in rows),
        "missed_agent": sum(row[prediction] == "search" and row["expected"] == "agent" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "configs/jev-routing-eval.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (key := os.getenv("OPENROUTER_API_KEY")):
        parser.error("Set OPENROUTER_API_KEY to run the live replay")
    content = args.cases.read_bytes()
    cases = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    fallback_routes = browser_routes(cases)

    def evaluate(item: tuple[dict, str]) -> dict:
        case, fallback = item
        try:
            decision = classify(key, case["query"], case.get("previous", ""))
        except (URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            error = f"HTTP {exc.code}" if isinstance(exc, HTTPError) else type(exc).__name__
            return {**case, "rules": fallback, "decision": None, "error": error, "effective": fallback}
        return {**case, "rules": fallback, "decision": asdict(decision), "error": None,
                "effective": decision.effective_route}

    with ThreadPoolExecutor(max_workers=3) as workers:
        rows = list(workers.map(evaluate, zip(cases, fallback_routes, strict=True)))
    successful = [{**row, "jev": row["decision"]["route"]} for row in rows if row["decision"]]
    elapsed = sorted(row["decision"]["seconds"] for row in successful)
    report = {
        "model": MODEL, "cases_sha256": hashlib.sha256(content).hexdigest(), "provenance": provenance(),
        "min_search_confidence": MIN_SEARCH_CONFIDENCE, "total": len(rows), "completed": len(successful),
        "expected_agent": sum(row["expected"] == "agent" for row in rows),
        "routing": {**{name: metrics(rows, name) for name in ("rules", "effective")},
                    "raw_jev": metrics(successful, "jev")},
        "seconds": {"median": statistics.median(elapsed), "p95": elapsed[math.ceil(len(elapsed) * 0.95) - 1]}
        if elapsed else None,
        "cost_usd": sum(row["decision"]["cost"] for row in successful),
        "input_tokens": sum(row["decision"]["input_tokens"] for row in successful),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**report, "cases": rows}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
