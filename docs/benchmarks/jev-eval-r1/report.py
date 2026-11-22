import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from evaluate import read_jev
from mail_search.benchmark import METRICS, measure
from mail_search.depth_benchmark import write_json
from mail_search.friend_eval import MODES
from mail_search.friend_eval_data import sha256


ROOT = Path(__file__).resolve().parent
FILES = {
    "E5": "candidates-original99.json",
    "BGE": "bge-original99.json",
    "Qwen": "qwen-original99.json",
    "Qwen FT": "qwen-ft-original99.json",
}
REPLAY_FILES = {name: file.replace("-original99", "") for name, file in FILES.items()}


def read(path):
    return json.loads(path.read_text())


def summarize(rows, key):
    return {
        metric: {"n": len(values), "mean": statistics.mean(values) if values else None,
                 "std": statistics.stdev(values) if len(values) > 1 else None}
        for metric in METRICS
        for values in [[row[key][metric] for row in rows if row[key][metric] is not None]]
    }


def contrast(rows, baseline, key):
    assert rows.keys() == baseline.keys()
    deltas = {qid: row[key]["ndcg@10"] - baseline[qid][key]["ndcg@10"] for qid, row in rows.items()}
    boxes = {row["mailbox"] for row in rows.values()}
    leave_one_out = [statistics.mean(d for qid, d in deltas.items() if rows[qid]["mailbox"] != box)
                     for box in sorted(boxes)] if len(boxes) > 1 else []
    return {
        "mean_delta": statistics.mean(deltas.values()),
        "per_query_delta_std": statistics.stdev(deltas.values()),
        "wins": sum(d > 1e-12 for d in deltas.values()),
        "ties": sum(abs(d) <= 1e-12 for d in deltas.values()),
        "losses": sum(d < -1e-12 for d in deltas.values()),
        "leave_one_mailbox_out_range": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None,
    }


def check_metrics(actual, expected):
    for metric, v in expected.items():
        assert actual[metric] is None if v is None else math.isclose(actual[metric], v, abs_tol=1e-12)


def combined(args):
    previous, _, _, meta = read_jev(args)
    qrels = meta["positives"]
    prior99 = {qid: {} for qid in qrels if not qid.startswith("N-")}
    for row in meta["original_positive"]:
        prior99[row["query_id"]][row["doc_id"]] = row["relevance"]
    replay = read(ROOT / "replay-new40.json")
    artifacts = {name: read(ROOT / filename) for name, filename in FILES.items()}
    candidates_sha256 = sha256(ROOT / FILES["E5"])
    for name, filename in REPLAY_FILES.items():
        assert sha256(args.cache / filename) == replay["sources"][name]
    for name, data in artifacts.items():
        assert data["protocol"] == {**replay["protocol"], "cohort": "99 original queries"}
        earlier = replay["models"][name]["config"]
        keys = (("revision", "weights_sha256", "max_length") if name == "E5" else
                ("revision", "base_weights_sha256", "checkpoint_sha256", "max_length", "train_layers"))
        for key in keys:
            assert data.get("model", data.get("e5"))[key] == earlier[key]
        if name != "E5":
            assert data["candidates_sha256"] == candidates_sha256

    output = {"sources": {filename: sha256(ROOT / filename) for filename in
                          ("replay-new40.json", "evaluate.py", "evaluate-h200.py", "run_original99.sh",
                           "run_gpu.sh", "report.py", *FILES.values())},
              "replay_protocol": replay["protocol"], "original99_protocol": artifacts["E5"]["protocol"],
              "groups": {}, "modes": {}}
    training = {"BGE": ROOT.parent / "compare-r1/bge-returns/metrics.json",
                "Qwen FT": ROOT.parent / "qwen-train-r1/metrics.json"}
    output["training_sources"] = {name: {"path": str(path.relative_to(ROOT.parent)), "sha256": sha256(path)}
                                  for name, path in training.items()}
    for mode in MODES:
        rows_by_model = {}
        for name, data in artifacts.items():
            rows = {}
            for row in data["modes"][mode]["per_query"]:
                qid = row["query_id"]
                ranking = row.get("reranked_doc_ids", row["ranked_doc_ids"])
                metric = row.get("reranking_metrics", row["retrieval_metrics"])
                assert len(set(ranking)) == len(ranking) <= replay["protocol"]["top_k"]
                assert row["relevant_documents"] == len(qrels[qid])
                if name != "E5":
                    assert row["ranked_doc_ids"] == rows_by_model["E5"][qid]["ranked_doc_ids"]
                    assert set(ranking) == set(row["ranked_doc_ids"])
                    scores = row["reranker_logits"]
                    assert len(scores) == len(ranking) and all(map(math.isfinite, scores.values()))
                    assert ranking == sorted(scores, key=lambda doc: (-scores[doc], doc))
                check_metrics(metric, measure(ranking, qrels[qid]))
                rows[qid] = {**row, "metrics": metric, "previous": measure(ranking, prior99[qid])}
            assert rows.keys() == prior99.keys() and len(data["modes"][mode]["per_query"]) == len(rows)
            for row in replay["models"][name]["modes"][mode]["per_query"]:
                qid = row["query_id"]
                assert qid not in rows
                ranking = row["ranked_doc_ids"]
                if name != "E5":
                    assert set(ranking) == set(rows_by_model["E5"][qid]["ranked_doc_ids"])
                check_metrics(row["reranking_metrics"], measure(ranking, qrels[qid]))
                check_metrics(row["previous_metrics"], measure(ranking, previous.dataset.qrels[row["prior_query_id"]]))
                rows[qid] = {**row, "metrics": row["reranking_metrics"], "previous": row["previous_metrics"]}
            assert rows.keys() == qrels.keys()
            rows_by_model[name] = rows

        groups = {"all139": set(qrels), "original99": set(prior99), "new40": qrels.keys() - prior99.keys()}
        for split in ("train", "val", "test"):
            groups[split] = {qid for qid, row in rows.items() if row["mailbox_split"] == split}
            for cohort in ("original99", "new40"):
                groups[f"{cohort}-{split}"] = groups[cohort] & groups[split]
        output["modes"][mode] = {}
        for group, ids in groups.items():
            cohort_rows = {name: {qid: rows[qid] for qid in sorted(ids)}
                           for name, rows in rows_by_model.items()}
            output["groups"][group] = {"queries": len(ids), "mailboxes": len({rows[q]["mailbox"] for q in ids}),
                                       "positive_pairs": sum(len(qrels[q]) for q in ids)}
            output["modes"][mode][group] = {
                name: {"current": summarize(list(subset.values()), "metrics"),
                       "previous": summarize(list(subset.values()), "previous"),
                       "change_qrels": contrast(subset, {q: {**r, "metrics": r["previous"]} for q, r in subset.items()}, "metrics"),
                       "versus_e5": contrast(subset, cohort_rows["E5"], "metrics"),
                       "versus_qwen_base": contrast(subset, cohort_rows["Qwen"], "metrics")}
                for name, subset in cohort_rows.items()}
        output["modes"][mode]["candidate_oracle_original99"] = summarize(
            [rows_by_model["E5"][q] for q in prior99], "candidate_oracle_metrics")
    return output


def render(data):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    sets = data["modes"]["date-aware"]
    names = list(FILES)
    colors = ["#466988", "#008273", "#a86530", "#7e568c"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        ax.plot([row for row in range(3)], [sets[g][name]["current"]["ndcg@10"]["mean"]
                for g in ("original99", "new40", "test")], "o-", lw=1.5, color=color, label=name)
    ax.set(xticks=range(3), xticklabels=["Исходные (99)", "Дополнительные (40)", "Test-ящики (30)"],
           ylabel="Средний nDCG@10", ylim=(0.78, 1.012))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.savefig(destination / "ndcg-by-cohort.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    for i, name in enumerate(names):
        for offset, group, color, label in [(-0.17, "original99", "#466988", "Исходные (99)"),
                                            (0.17, "new40", "#008273", "Дополнительные (40)")]:
            delta = sets[group][name]["change_qrels"]["mean_delta"]
            ax.barh(i + offset, delta, 0.29, color=color, label=label if i == 0 else None)
    ax.axvline(0, color="#202028", lw=0.8)
    ax.set(yticks=range(len(names)), yticklabels=names, xlabel="Изменение nDCG@10 после замены разметки")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.savefig(destination / "qrel-effect.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("original-archive", "augmented-archive", "jev-archive", "splits", "cache"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    metrics = combined(parser.parse_args())
    write_json(ROOT / "metrics.json", metrics)
    render(metrics)
