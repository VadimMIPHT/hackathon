import hashlib
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mail_search.benchmark import measure
from mail_search.dataset import load_archive, read_split
from mail_search.reranker_data import load_training_groups

root = Path.cwd()
folder = root / "runs" / "compare-r1"
dataset = load_archive(Path("/home/maxim/Downloads/mipt_mail_search_participants.zip"))
split_path = root / "splits" / "mailboxes.json"
train, _ = read_split(split_path, dataset, "train")
val, _ = read_split(split_path, dataset, "val")
test, _ = read_split(split_path, dataset, "test")
assert not (train & val or train & test or val & test)
extras_path = root / "runs" / "comparison-inputs" / "refund-judgments.json"
extras = load_training_groups(extras_path, dataset, train)
assert len(extras) == 8 and sum(len(docs) for _, docs, _ in extras) == 31

def read(path: str) -> dict:
    return json.loads((folder / path).read_text())

baseline = read("../bge-top20-r1/metrics.json")
known_rrf = read("../rank-fusion-val.json")["runs"]["finetuned_epoch2_rrf60"]
aug = read("bge-returns/metrics.json")
replay = read("bge-verification.json")
features = read("boost-features.json")
assert aug["config"]["extra_judgements_sha256"] == hashlib.sha256(extras_path.read_bytes()).hexdigest()
assert aug["config"]["eval_split"] == replay["config"]["eval_split"] == "val"
assert hashlib.sha256((folder / "boost-features.json").read_bytes()).hexdigest() == read("catboost.json")["config"]["input_sha256"]
best_aug = aug["epochs"][aug["best_finetuned_epoch"] - 1]
assert len(aug["epochs"]) == 2
assert [r["ranked_doc_ids"] for r in best_aug["per_query"]] == [
    r["ranked_doc_ids"] for r in replay["weights_evaluation"]["per_query"]
]

runs = {
    "E5": baseline["e5"],
    "BGE исходный": baseline["epochs"][1],
    "E5 + BGE через RRF": known_rrf,
    "Qwen3 0.6B": read("qwen.json"),
    "RuBERT": read("rubert.json"),
    "CatBoostRanker": read("catboost.json"),
    "BGE + 8 запросов": best_aug,
}
expected = {q.query_id for q in dataset.queries.values() if q.mailbox in val}
original = {r["query_id"]:r["ranked_doc_ids"] for r in baseline["e5"]["per_query"]}
depth = baseline["config"]["top_k"]
assert set(original) == expected and len(expected) == 21
assert all(not any(dataset.documents[doc_id].mailbox in val | test for doc_id in rel)
           for _, _, rel in extras)
for name, run in runs.items():
    rows = run["per_query"]
    assert len(rows) == len(expected) and {r["query_id"] for r in rows} == expected, name
    for row in rows:
        qid = row["query_id"]
        ids = row["ranked_doc_ids"]
        ref = original[qid]
        assert set(ids[:depth]) == set(ref[:depth]) and ids[depth:] == ref[depth:], name
        metrics = measure(ids, dataset.qrels[qid])
        assert all(row["metrics"][key] is None if value is None
                   else abs(row["metrics"][key] - value) < 1e-10 for key, value in metrics.items()), name

def mean_metrics(rows: list[dict]) -> dict:
    return {key: statistics.mean(values) if (values := [x["metrics"][key] for x in rows
            if x["metrics"][key] is not None]) else None for key in rows[0]["metrics"]}

def add_fusion(name: str, run: dict) -> dict:
    rows = []
    for row in run["per_query"]:
        qid = row["query_id"]
        e5 = original[qid]
        rrank = {doc_id: pos for pos,doc_id in enumerate(row["ranked_doc_ids"][:depth], 1)}
        fused = sorted(enumerate(e5[:depth], 1), key=lambda item: (
            -(1 / (60 + item[0]) + 1 / (60 + rrank[item[1]])), item[0],
        ))
        ranked = [doc_id for _, doc_id in fused] + e5[depth:]
        rows.append({"query_id": qid, "mailbox": row["mailbox"], "ranked_doc_ids": ranked,
                     "metrics": measure(ranked, dataset.qrels[qid])})
    return {"summary": mean_metrics(rows), "per_query": rows}

aug_extra_rows = []
for qid, ranking in original.items():
    qrels = dataset.qrels[qid]
    ideal = sorted(ranking[:depth], key=lambda doc_id: -qrels.get(doc_id, 0)) + ranking[depth:]
    measured = measure(ideal, qrels)
    aug_extra_rows.append({"query_id": qid, "mailbox": dataset.queries[qid].mailbox,
                           "ranked_doc_ids": ideal, "metrics": measured})
oracle = {"summary": mean_metrics(aug_extra_rows), "per_query": aug_extra_rows}
assert abs(best_aug["summary"]["ndcg@10"] - oracle["summary"]["ndcg@10"]) < 1e-10

for name, run in list(runs.items())[3:]:
    runs[f"E5 + {name}, RRF"] = add_fusion(name, run)

fmt = lambda value: f"{value:.4f}".replace(".", ",")
scorers = list(runs)[:7]
comparison = json.dumps({
    "config": {"archive_sha256": dataset.archive_sha256, "split": "val", "top_k": depth,
               "rrf_constant": 60, "split_seed": aug["config"]["split_seed"]},
    "sources": {filename: hashlib.sha256((folder / filename).read_bytes()).hexdigest()
                for filename in ("qwen.json", "rubert.json", "catboost.json", "boost-features.json",
                                 "bge-returns/metrics.json", "bge-verification.json",
                                 "../bge-top20-r1/metrics.json", "../rank-fusion-val.json",
                                 "../comparison-inputs/refund-judgments.json")},
    "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "summaries": {name: run["summary"] for name, run in runs.items()},
    "rrf_per_query": {name: run["per_query"] for name, run in runs.items() if name.startswith("E5 +")},
    "oracle_top20": oracle,
}, ensure_ascii=False, indent=2) + "\n"
(folder / "comparison.json").write_text(comparison, encoding="utf-8")

summary = [
    "| Вариант | nDCG@10 | Recall@10 | MRR@10¹ | Офлайн p99 rerank, мс |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for name, run in runs.items():
    s = run["summary"]
    latency = f"{s['p99_ms_offline']:.0f}" if "p99_ms_offline" in s and name != "E5" else "—"
    summary.append(f"| {name} | {fmt(s['ndcg@10'])} | {fmt(s['recall@10'])} | {fmt(s['mrr@10_singleton'])} | {latency} |")
table = "\n".join(summary)

topic_labels = {"возврат": "Возвраты", "отклон": "Отклонённые платежи",
                "успеш": "Успешная оплата", "заказ": "Новые заказы"}
topics = {qid: next(label for word,label in topic_labels.items()
                   if word in dataset.queries[qid].text.casefold()) for qid in original}
groups = {name: {qid for qid,label in topics.items() if name == label} for name in topic_labels.values()}
groups.update({f"Ящик …{box[-4:]}": {qid for qid in original if dataset.queries[qid].mailbox == box}
               for box in sorted({dataset.queries[qid].mailbox for qid in original})})
sliced = {name: {group: mean_metrics([row for row in run["per_query"] if row["query_id"] in ids])
                 for group,ids in groups.items()} for name,run in runs.items()}

report = f"""# Сравнение reranker на top-20 E5

Все варианты переупорядочивают одни и те же кандидаты для 21 запроса из двух val-ящиков. Лучший локальный результат получил BGE с восемью дополнительными train-запросами о возвратах. Его nDCG@10 совпадает с oracle при идеальной перестановке этих top-20. CatBoostRanker дал близкий результат с исходной train-разметкой и готовым BGE в признаках.

{table}

¹ MRR считается для пяти запросов с одним релевантным письмом. Recall@50 одинаковый у всех и равен {fmt(oracle['summary']['recall@50'])}. Для CatBoost задержка включает готовый BGE, подготовку парных признаков и predict, E5 и BM25 были рассчитаны заранее. Для нейросетей время включает токенизацию, скоринг и перестановку. Это последовательные офлайн-замеры на H200. Проверка серверного p99 требует отдельного замера.

Дополнительные запросы составлены по восьми письмам train-ящика за восемь разных дней. Проверены 8 положительных и 23 отрицательных пары. BGE обучался с прежней конфигурацией на 1 355 парах по 63 запросам. После загрузки чекпоинт второй эпохи воспроизвёл выдачу. Тема возвратов выбрана по ошибкам на val, поэтому результат на этой же val считается разведочным. Test зарезервирован. Результат около oracle относится только к этим 21 запросу.

[Сырые метрики и oracle](comparison.json) · [Разрезы и параметры](stats-appendix.md) · [Графики](figure-catalog.md) · [Дополнительные метки](../comparison-inputs/refund-judgments.json)
"""
(folder / "analysis-report.md").write_text(report, encoding="utf-8")

cols = [f"{group} ({len(ids)})" for group,ids in groups.items()]
lines = [
    "# Разрезы и параметры", "", "Среднее nDCG@10 по каждому разрезу. Объём указан в запросах.", "",
    "| Вариант | " + " | ".join(cols) + " |",
    "| --- | " + " | ".join(["---:"] * len(cols)) + " |",
]
for name in scorers:
    lines.append("| " + " | ".join([name, *(fmt(sliced[name][key]["ndcg@10"]) for key in groups)]) + " |")
lines.extend([
    "",
    "Модели сравниваются попарно на одних кандидатах. При двух размеченных ящиках независимого доверительного интервала по новым ящикам нет. Контроль seed не заменяет независимую выборку.",
    "",
    f"CatBoost 1.2.10, PairLogit, depth 2, 100 деревьев, seed 42. Обучение использовало {read('catboost.json')['train']['mixed_queries']} train-запроса с парами разных классов. BGE в признаках CatBoost взят до дообучения. Признаки включают позиции E5 и BM25, E5-косинус, логит BGE и совпадения по словам, теме, триграммам и идентификаторам.",
    "",
    f"Qwen и RuBERT используют зафиксированные ревизии `{read('qwen.json')['config']['revision']}` и `{read('rubert.json')['config']['revision']}`. Бюджет до 256 токенов, включая инструкцию Qwen. Qwen ранжирует по разности логитов yes и no.",
    "",
    f"Чекпоинт BGE хранится на H200 в `~/vk-mail-search-hackathon/runs/compare-r1/bge-returns/best-finetuned.safetensors`. После загрузки его SHA-256 `{replay['config']['checkpoint_sha256']}`. Семя BGE 42, две эпохи, AdamW с lr 2e-5, последние два слоя и классификационная голова.",
    "",
    f"Архив SHA-256 `{dataset.archive_sha256}`, seed разбиения 42, базовый коммит `{aug['environment']['code_base_commit']}`. Среда, SHA-256 кода и использованных входов записаны в JSON прогонов. Метки расширения привязаны к архиву и проверены по дате и train-ящику.",
    "",
    "[Qwen](qwen.json) · [RuBERT](rubert.json) · [CatBoost](catboost.json) · [Признаки](boost-features.json) · [Среда CatBoost](catboost-environment.txt) · [BGE с возвратами](bge-returns/metrics.json) · [Перезагрузка BGE](bge-verification.json) · [Команды на H200](h200-commands.sh)",
])
(folder / "stats-appendix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

figures = folder / "figures"
figures.mkdir(exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                    "axes.axisbelow": True, "font.size": 11})
fig, ax = plt.subplots(figsize=(8.1, 5.2), layout="constrained")
numbers = [runs[name]["summary"]["ndcg@10"] for name in scorers]
bars = ax.barh(range(len(scorers)), numbers, color=["#5c6978"]*3+["#255a91"]*4)
for index, score in enumerate(numbers):
    ax.text(1.005, index, fmt(score), va="center")
ax.set_yticks(range(len(scorers)), scorers)
ax.set_xlim(0,1.08)
ax.invert_yaxis()
ax.axvline(oracle["summary"]["ndcg@10"], color="#b45c21", linestyle="--", linewidth=1,
           label="Потолок top-20")
ax.grid(axis="x", alpha=.22)
ax.set_xlabel("Средний nDCG@10")
ax.set_title("Сравнение reranker · 21 val-запрос", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.savefig(figures / "ndcg-methods.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 4.9), layout="constrained")
x=np.arange(len(scorers))
for i,(topic,color) in enumerate((("Возвраты","#205783"),("Отклонённые платежи","#b45c21"))):
    data=[sliced[name][topic]["ndcg@10"] for name in scorers]
    ax.barh(x + (i - .5)*.32, data, height=.31, color=color, label=topic)
ax.set_yticks(x, scorers)
ax.set_xlim(0,1.02)
ax.invert_yaxis()
ax.grid(axis="x", alpha=.22)
ax.set_xlabel("Средний nDCG@10 на девяти запросах каждого типа")
ax.set_title("Запросы о возвратах и отклонённых платежах", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
fig.savefig(figures / "ndcg-topics.png", dpi=160)
plt.close(fig)
(folder / "figure-catalog.md").write_text("""# Графики сравнения reranker

## Среднее качество на val

![Среднее качество](figures/ndcg-methods.png)

По одному столбцу на вариант. Среднее по 21 val-запросу из двух ящиков. Пунктир показывает oracle для уже отобранного top-20. График показывает, сколько запаса для перестановки оставляет каждый метод на этих кандидатах.

## Возвраты и отклонённые платежи

![Качество по типам запросов](figures/ndcg-topics.png)

В каждом разрезе девять запросов. Проверяем, сохраняет ли метод сильные результаты E5 по возвратам, пока улучшает результаты на отклонённых платежах. Точечные значения без интервалов.

[Полные разрезы](stats-appendix.md)
""",encoding="utf-8")

print(table)
print("\n".join(lines[4:14]))
