# Замеры поиска и маршрутизации

В каждом отчёте указаны выборка и версия разметки. JSON рядом хранит результаты по запросам, параметры запуска, ревизии моделей, контрольные суммы и окружение, если они были записаны при замере. `runs/` используется для локальных запусков, исходные архивы и чекпойнты хранятся отдельно.

| Эксперимент | Результаты и сырые метрики |
| --- | --- |
| BM25, E5 и RRF на исходных train и val | [Отчёт](analysis-output/analysis-report.md) · [Train](train-e5-baselines-r2.json) · [Val](val-e5-baselines-r2.json) · [Ранний val](val-e5-baselines-r1.json) · [BM25 smoke](smoke-train-bm25.json) |
| BGE на top‑20 и слияние рангов | [Отчёт](bge-top20-r1/analysis-report.md) · [Метрики по эпохам](bge-top20-r1/metrics.json) · [RRF](rank-fusion-val.json) |
| BGE, Qwen, RuBERT и CatBoost на общем пуле E5 | [Отчёт](compare-r1/analysis-report.md) · [Сравнение](compare-r1/comparison.json) |
| Top‑20, top‑50 и ранжирование полного пула | [Отчёт](depth/analysis-report.md) · [Метрики](depth/comparison.json) |
| Зафиксированный тест на 23 запросах из test-ящиков | [Отчёт](final-test/analysis-report.md) · [Метрики](final-test/metrics.json) |
| Дообучение Qwen | [Отчёт](qwen-train-r1/analysis-report.md) · [Метрики по эпохам](qwen-train-r1/metrics.json) |
| 40 дополнительных запросов, прежняя разметка | [Отчёт](friend-eval-r1/analysis-report.md) · [Срезы](friend-eval-r1/comparison.json) |
| Все 139 запросов, полная разметка Jev | [Отчёт](jev-eval-r1/analysis-report.md) · [Срезы](jev-eval-r1/metrics.json) · [Потолки Recall@10](jev-eval-r1/recall10-ceilings.json) |
| BGE обучен на 40 новых, тест на 99 исходных | [Отчёт](jev-transfer-r1/analysis-report.md) · [Срезы](jev-transfer-r1/comparison.json) |
| BM25 из `team-smartmail` на тех же запросах и метках | [Разбор ветки](team-smartmail/README.md) · [Метрики](team-smartmail/metrics.json) |
| Маршрутизация Pismo через Jev | [Разработка](pismo/jev-routing-20261121.json) · [Holdout](pismo/jev-routing-holdout-20261121.json) · [Конфигурации](../../configs/) |

[Разбиение по почтовым ящикам, seed 42](mailbox-split.json). В результатах маршрутизации сохранены метрики и решения по кейсам. Тексты запросов доступны в соответствующих конфигурациях. В первом замере маршрутизации исходный JSON не записал ревизию кода и окружение.
