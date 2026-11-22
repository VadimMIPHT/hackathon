import hashlib
import json
import math
import statistics
from pathlib import Path

from mail_search.benchmark import measure
from mail_search.friend_eval_data import load_friend_data, sha256
from mail_search.reranker_data import documents_by_mailbox

folder = Path("runs/friend-eval-r1")
download = Path("/home/maxim/Downloads/Telegram Desktop")
data = load_friend_data(
    download.parent / "mipt_mail_search_participants.zip",
    download / "vk_mail_search_augmented.zip",
    Path("splits/mailboxes.json"),
)
corpus = data.dataset
qwen_training = Path("runs/qwen-train-r1/metrics.json")
bge_reference = Path("runs/depth/bge-returns.json")

def load(filename: str) -> dict:
    return json.loads((folder / filename).read_text())

candidates = load("candidates.json")
model_runs = {name: load(filename) for name,filename in (
    ("BGE после обучения", "bge.json"), ("Qwen готовый", "qwen.json"),
    ("Qwen после обучения", "qwen-ft.json"),
)}
assert candidates["protocol"] == data.protocol
boxes = documents_by_mailbox(corpus)
for run in model_runs.values():
    assert run["protocol"] == data.protocol
    assert run["candidates_sha256"] == sha256(folder / "candidates.json")
assert model_runs["BGE после обучения"]["model"]["checkpoint_sha256"] == json.loads(
    bge_reference.read_text())["model"]["checkpoint_sha256"]
assert model_runs["Qwen после обучения"]["model"]["checkpoint_sha256"] == json.loads(
    qwen_training.read_text())["checkpoint_sha256"]
assert model_runs["Qwen после обучения"]["model"]["base_weights_sha256"] == model_runs[
    "Qwen готовый"]["model"]["base_weights_sha256"]

ranking_modes = {}
for mode in ("frozen", "date-aware"):
    expected = {row["query_id"]: row for row in candidates["modes"][mode]["per_query"]}
    assert len(expected) == 40 and expected.keys() == corpus.queries.keys()
    for qid, row in expected.items():
        q = corpus.queries[qid]
        _, allowed = data.date_pool(q, boxes[q.mailbox], mode)
        rank = row["ranked_doc_ids"]
        assert len(set(rank)) == min(50, len(allowed)) == len(rank) and set(rank) <= set(allowed)
        assert row["retrieved_relevant"] == len(set(rank) & corpus.qrels[qid].keys())
        assert row["relevant_documents"] == row["eligible_relevant"] + row["excluded_by_query_day"]
        assert row["eligible_relevant"] == (row["retrieved_relevant"] +
            row["excluded_by_top50_capacity"] + row["excluded_by_e5_ordering"])

    ranking_modes[mode] = {"E5": expected}
    for name, result in model_runs.items():
        rows = {row["query_id"]: row for row in result["modes"][mode]["per_query"]}
        assert rows.keys() == expected.keys()
        for qid, row in rows.items():
            orig = expected[qid]["ranked_doc_ids"]
            actual = row["reranked_doc_ids"]
            assert row["ranked_doc_ids"] == orig and len(actual) == len(orig)
            assert set(orig) == set(actual) == set(row["reranker_logits"])
            assert actual == sorted(orig, key=lambda d: (-row["reranker_logits"][d], d))
            for metric, target in measure(actual, corpus.qrels[qid]).items():
                got = row["reranking_metrics"][metric]
                assert got is None if target is None else math.isclose(got, target, abs_tol=1e-12)
            assert row["reranking_metrics"]["recall@50"] == expected[qid]["retrieval_metrics"]["recall@50"]
        ranking_modes[mode][name] = rows

def results_for(mode: str, name: str, subset) -> dict:
    source = ranking_modes[mode][name]
    key = "retrieval_metrics" if name == "E5" else "reranking_metrics"
    return {metric: statistics.mean(row[key][metric] for row in source.values() if subset(row))
            for metric in ("ndcg@10", "recall@10", "recall@50")}

def fmt(value: float) -> str:
    return f"{value:.4f}".replace(".", ",")

table = [
    "| Модель | nDCG@10 текущая дата | nDCG@10 расширенная дата | Recall@10 расширенная дата |",
    "| --- | ---: | ---: | ---: |",
]
for name in ranking_modes["frozen"]:
    frozen = results_for("frozen", name, lambda row: True)
    extended = results_for("date-aware", name, lambda row: True)
    table.append(f"| {name} | {fmt(frozen['ndcg@10'])} | {fmt(extended['ndcg@10'])} | {fmt(extended['recall@10'])} |")

groups = {
    "Все запросы": lambda row: True,
    "Перефразы": lambda row: row["query_type"] == "paraphrase",
    "Интервалы": lambda row: row["query_type"] == "date_range",
    "Ящики train": lambda row: row["mailbox_split"] == "train",
    "Ящики val": lambda row: row["mailbox_split"] == "val",
    "Ящики test": lambda row: row["mailbox_split"] == "test",
    "Ящики вне train": lambda row: row["mailbox_split"] != "train",
}
slices = {}
for mode in ranking_modes:
    slices[mode] = {}
    for group, predicate in groups.items():
        slices[mode][group] = {
            "queries": sum(predicate(row) for row in ranking_modes[mode]["E5"].values()),
            "models": {name: results_for(mode, name, predicate) for name in ranking_modes[mode]},
        }
slices_table = [
    "| Режим | Группа | N | E5 | BGE дообученный | Qwen готовый | Qwen дообученный |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for mode, mode_data in slices.items():
    for group, details in mode_data.items():
        values = " | ".join(fmt(metrics["ndcg@10"]) for metrics in details["models"].values())
        slices_table.append(f"| {mode} | {group} | {details['queries']} | {values} |")

coverage_table = [
    "| Режим | Потери фильтра | Потери из-за лимита 50 | Потери при отборе E5 сверх лимита | Попало в top-50 | Recall@50 | Потолок nDCG@10 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for mode, run in candidates["modes"].items():
    r = run["summary"]
    coverage_table.append(f"| {mode} | {r['excluded_by_query_day']} | {r['excluded_by_top50_capacity']} | "
        f"{r['excluded_by_e5_ordering']} | {r['retrieved_relevant']} из {r['relevant_documents']} | "
        f"{fmt(r['retrieval_metrics']['recall@50'])} | {fmt(r['candidate_oracle_metrics']['ndcg@10'])} |")
audit = load("audit.json")
assert audit["manual_letter_classification"]["rows"] == 1000
report = f"""# Поиск на дополнительных запросах VK

Расширенный ZIP добавил 40 запросов с 1 143 размеченными релевантными попаданиями. 32 запроса перефразируют старые, восемь объединяют несколько дней. Корпус писем побайтно совпадает. Положительные пары унаследованы из исходной разметки, а 2 078 дополнительных ручных пар в ZIP относятся к прежним запросам. Файл `vk-mail-manual-1000.jsonl` содержит 1 000 писем с 13 метками категорий. Письма из этого файла совпадают по тексту, ящику и дате. У отдельного JSONL нет запросов и связей «запрос — письмо», поэтому отдельная retrieval-метрика по нему не определена.

Модели скорят один и тот же top-50 E5 для каждого режима. Дообученный Qwen использует те же 1 355 пар, что и BGE. Ни одна дополнительная пара из новых файлов не использовалась для обучения. Базовый Qwen с готовыми весами и Qwen после второй эпохи сравниваются на одинаковых кандидатах.

{chr(10).join(table)}

### Покрытие кандидатов

{chr(10).join(coverage_table)}

Потери приведены в парах «запрос — релевантное письмо». Recall@50 — среднее по запросам, а не отношение суммы восстановленных пар к 1 143. Дополнительный парсер использует только текст запроса и `query_date`. Он подобран для встреченных формулировок, поэтому расширенный режим следует считать отдельным диагностическим экспериментом. MRR на запросах с одним положительным письмом не применялся: в новых запросах по 10 и более размеченных писем.

### nDCG@10 по группам

{chr(10).join(slices_table)}

Часть добавленных запросов наследует релевантные письма из train или val. Из test-ящиков здесь только один ящик и семь запросов. Итог по 40 запросам показывает устойчивость к новым словам и датам на данном корпусе. Для оценки на независимых письмах потребуется отдельная разметка.

[Аудит файлов](audit.json) · [Сводные метрики по разрезам](comparison.json) · [Кандидаты E5](candidates.json) · [Выдачи BGE](bge.json) · [Выдачи готового Qwen](qwen.json) · [Выдачи дообученного Qwen](qwen-ft.json) · [Дообучение и старый val](../qwen-train-r1/analysis-report.md)
"""
(folder / "analysis-report.md").write_text(report, encoding="utf-8")
(folder / "comparison.json").write_text(json.dumps({
    "protocol": data.protocol,
    "sources_sha256": {p.name: sha256(p) for p in
        [*(folder / n for n in ("candidates.json", "bge.json", "qwen.json", "qwen-ft.json", "audit.json")),
         qwen_training, bge_reference]},
    "script_sha256": sha256(Path(__file__)),
    "coverage": {mode: v["summary"] for mode, v in candidates["modes"].items()},
    "methods": {name: {"config": run["model"], "environment": run["environment"]} for name, run in model_runs.items()},
    "slices": slices,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("\n".join(table + [""] + coverage_table + [""] + slices_table))
