import hashlib
import json
from pathlib import Path

folder = Path(__file__).resolve().parent
coverage = json.loads((folder / "coverage.json").read_text())
expected = {
    row["query_id"]: row["available"] for row in coverage["depths"]["all"]["per_query"]
}
names = {
    "bge-returns": "BGE после дообучения на возвратах",
    "rubert": "RuBERT",
    "qwen": "Qwen3-Reranker-0.6B",
    "bge-frozen": "Готовый BGE",
    "catboost": "CatBoost с E5, BM25 и готовым BGE",
}
def summary_on_pool(result: dict, expected_counts: dict) -> float:
    assert {row["query_id"] for row in result["per_query"]} == set(expected_counts)
    return result["summary"]["ndcg@10"]


results = {}
for key, label in names.items():
    path = folder / f"{key}.json"
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    for signature in ("archive_sha256", "split_manifest_sha256", "e5_candidates_sha256"):
        assert result["config"][signature] == coverage["config"][signature], (key, signature)
    assert result["config"]["split"] == "val"
    assert {r["query_id"]: len(r["doc_ids"]) for r in result["scores"]} == expected
    for depth, metrics in result["depths"].items():
        assert summary_on_pool(metrics, expected) <= coverage["depths"][depth]["oracle_ndcg@10"] + 1e-12
    results[label] = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "depths": {depth: run["summary"] for depth, run in result["depths"].items()},
    }

if not results:
    raise SystemExit("No completed model run is available")
output = {"config": coverage["config"], "model_results": results}
(folder / "comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")

table = [
    "| Вариант | nDCG@10 top-20 | nDCG@10 top-50 | nDCG@10 полный пул | Recall@10 полный пул | Офлайн p99 полного пула, мс |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for name, run in results.items():
    depths = run["depths"]
    values = [f"{depths[depth]['ndcg@10']:.4f}" for depth in ("20", "50", "all")]
    values.extend((f"{depths['all']['recall@10']:.4f}", f"{depths['all']['p99_ms_offline']:.0f}"))
    table.append("| " + " | ".join([name, *(value.replace(".", ",") for value in values)]) + " |")

text = """# Ранжирование при разных глубинах E5

Для каждой модели прямой скоринг всех писем за подходящий день выполняется одним проходом. Затем одни и те же скоры сортируются отдельно внутри top-20 E5, top-50 E5 и всего пула. В ничьих используется идентификатор письма. Ограниченные варианты сохраняют E5-хвост до позиции 50. Выборка состоит из 21 val-запроса. Test зарезервирован.

""" + "\n".join(table) + """

В этом прогоне GPU была занята другими задачами, для скоринга предусмотрена пауза 50 мс между батчами. Офлайн p99 полного пула включает эти паузы, подготовку текста, скоринг и сортировку. Для сравнения качества моделей используется одна разметка, для проверки серверного SLA потребуется отдельный замер. CatBoost продолжает использовать признаки E5 и BM25 и замеряется без переобучения. Его офлайн p99 включает готовый BGE, парные признаки и predict после расчёта E5/BM25.

[Потолки и покрытие](coverage-report.md) · [Сводные и исходные метрики](comparison.json) · [Команда H200](run_h200.sh)
"""
(folder / "analysis-report.md").write_text(text)
print("\n".join(table))
