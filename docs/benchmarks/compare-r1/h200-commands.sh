set -euo pipefail
cd "$HOME/vk-mail-search-hackathon"
umask 077
mkdir -p runs/compare-r1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2 < 512 && $3 < 5 {print $1; exit}')
test -n "$gpu"
printf 'Assigned GPU %s\n' "$gpu"
nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
py="$HOME/project_venv/bin/python"
data=(--archive data/mipt_mail_search_participants.zip --splits splits/mailboxes.json)
commit=(--base-commit 0f47058c0da551eedf9270b3c7eb513d624dac3d)
candidates=(--e5-candidates runs/val-e5-baselines-r2.json)
bge_revision=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
bge_directory=models/bge-reranker-v2-m3-953dc6f
failed=0
stage() {
    local name="$1"
    shift
    printf 'Starting %s\n' "$name"
    if timeout 180s nice -n 10 "$@"; then
        printf 'Completed %s\n' "$name"
    else
        printf 'Failed %s\n' "$name" >&2
        failed=1
    fi
}
stage qwen "$py" -m mail_search.zero_shot "${data[@]}" "${candidates[@]}" "${commit[@]}" --name qwen \
    --model models/qwen3-reranker-0.6b-e61197e --revision e61197ed45024b0ed8a2d74b80b4d909f1255473 \
    --output runs/compare-r1/qwen.json
stage rubert "$py" -m mail_search.zero_shot "${data[@]}" "${candidates[@]}" "${commit[@]}" --name rubert \
    --model models/dity-russian-msmarco-9029bab --revision 9029bab08103ad171724b510d312befa5b476293 \
    --output runs/compare-r1/rubert.json
stage bge-returns "$py" -m mail_search.reranker "${data[@]}" "${candidates[@]}" "${commit[@]}" \
    --model "$bge_directory" --bge-revision "$bge_revision" --extra-judgements runs/comparison-inputs/refund-judgments.json \
    --output-dir runs/compare-r1/bge-returns
stage feature-table "$py" -m mail_search.boost_features "${data[@]}" "${commit[@]}" \
    --train-candidates runs/train-e5-baselines-r2.json --val-candidates runs/val-e5-baselines-r2.json \
    --e5-model models/e5-small-614241f --bge-model "$bge_directory" --bge-revision "$bge_revision" \
    --output runs/compare-r1/boost-features.json
nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader
if test -s runs/compare-r1/boost-features.json; then
    export CUDA_VISIBLE_DEVICES=
    stage catboost .venv-catboost/bin/python -m mail_search.boost "${data[@]}" \
        --input runs/compare-r1/boost-features.json --output runs/compare-r1/catboost.json
fi
exit "$failed"
