# Умный поиск в Почте

Офлайн-пайплайн для корпуса VK. Поиск учитывает почтовый ящик и дату, выбирает top-50 через `multilingual-e5-small` и переставляет кандидатов по скору `bge-reranker-v2-m3`. У E5 есть самостоятельная выдача. Эмбеддинги писем вычисляются заранее, поиск по ним использует точное косинусное сходство.

## Веб-демо «Письмо»

Для локального запуска нужен Python 3.10+ и ZIP, выданный организаторами. Из корня репозитория выполните команду и откройте http://127.0.0.1:8940/.

```bash
python3 -m mail_search.web --archive /path/to/mipt_mail_search_participants.zip --lexical-only
```

Если архив лежит в `data/mipt_mail_search_participants.zip`, флаг `--archive` можно опустить. Интерфейс читает письма из всех 12 ящиков архива. Дата в шапке задаёт срез выбранного ящика и точку отсчёта для относительных дат. Поиск BM25 доступен без ML-зависимостей. Чтобы включить E5 + BGE, используйте [команду запуска с весами](sites/pismo/README.md#подключение-архива-хакатона).

[Исходники интерфейса, Worker и схема напоминаний](sites/pismo/). Приватная страница Sites работает на отдельном вымышленном наборе, архив организаторов используется локально.

## Данные и окружение

Положите архив организаторов в `data/mipt_mail_search_participants.zip`. Он читается напрямую. Для нейросетевого поиска нужны Python 3.11, CUDA-версия PyTorch, NumPy, Transformers, tokenizers, safetensors и одна доступная GPU. Замеренные версии окружения включали PyTorch 2.10.0+cu128 и Transformers 4.57.6.

Подготовьте локальные копии весов и токенизаторов указанных ревизий.

| Модель | Ревизия | Локальный путь |
| --- | --- | --- |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | `models/e5-small-614241f` |
| [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | `models/bge-reranker-v2-m3-953dc6f` |

Исходный архив содержит 16 376 писем и 99 публичных запросов. Генератор проверяет контрольные суммы и назначает каждый ящик целиком в один сплит. При seed 42 в train попадает 55 запросов, в val 21, в test 23.

```bash
python -m mail_search.dataset \
  --archive data/mipt_mail_search_participants.zip \
  --output splits/mailboxes.json --seed 42
```

Фильтр оставляет письма нужного ящика, доступные на `query_date`. Если запрос содержит конкретную дату или «вчера», отбор дополнительно ограничивается этим днём. E5 кодирует до 256 токенов на письмо.

## Получить кандидатов и дообучить BGE

Укажите доступную GPU. На общей машине можно использовать её UUID.

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
BASE_COMMIT=$(git rev-parse HEAD)

python -m mail_search.benchmark \
  --archive data/mipt_mail_search_participants.zip \
  --splits splits/mailboxes.json --split val \
  --e5-model models/e5-small-614241f \
  --e5-revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --base-commit "$BASE_COMMIT" --output runs/val-e5.json

python -m mail_search.reranker \
  --archive data/mipt_mail_search_participants.zip \
  --splits splits/mailboxes.json --e5-candidates runs/val-e5.json \
  --model models/bge-reranker-v2-m3-953dc6f \
  --bge-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --extra-judgements configs/refund-judgments.json \
  --base-commit "$BASE_COMMIT" --output-dir runs/bge-returns
```

Обучение использует исходный train и [31 вручную проверенную пару](configs/refund-judgments.json) для восьми дополнительных train-запросов о возвратах. Обновляются два последних слоя и классификационная голова. BCE получает одинаковый суммарный вес от каждого запроса. После двух эпох скрипт сохраняет лучший по `nDCG@10` checkpoint в `runs/bge-returns/best-finetuned.safetensors`.

## Воспроизвести зафиксированный замер test

[Протокол](configs/final-test-protocol.json) фиксирует top-50, обработку дат, версии моделей и контрольную сумму checkpoint. Для точного воспроизведения нужен сохранённый checkpoint команды с указанной суммой. Архив и веса хранятся отдельно. [Отчёты и результаты прогонов](docs/benchmarks/).

```bash
python -m mail_search.final_test \
  --archive data/mipt_mail_search_participants.zip \
  --splits splits/mailboxes.json --protocol configs/final-test-protocol.json \
  --e5-model models/e5-small-614241f --bge-model models/bge-reranker-v2-m3-953dc6f \
  --bge-weights runs/bge-returns/best-finetuned.safetensors \
  --base-commit "$BASE_COMMIT" --output runs/final-test/metrics.json
```

На локальном test из 23 запросов `nDCG@10` составил 0,9986 у E5 и 0,9940 у каскада. Обе выдачи получили `Recall@50 = 1,0000`. JSON сохраняет выдачу по запросам и параметры окружения. Для серверного p99 нужен отдельный нагрузочный замер.
