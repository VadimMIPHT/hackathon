set -euo pipefail
cd "$HOME/vk-mail-search-hackathon"
umask 077
directory=runs/jev-eval-r1
common=(--original-archive data/mipt_mail_search_participants.zip
        --augmented-archive data/friend/vk_mail_search_augmented.zip
        --jev-archive data/friend/vk-jev-dataset.zip --splits splits/mailboxes.json
        --base-commit 2c4de80d904e430f642c658ff11fc10ef4b7beff --gpu-memory-limit-gib 5)
snapshot=$(nvidia-smi --query-gpu=index,memory.free,compute_mode --format=csv,noheader,nounits)
printf '%s\n' "$snapshot"
gpu=$(printf '%s\n' "$snapshot" | awk -F, '
    $2 >= 24576 && $2 > best && $3 ~ /^[[:space:]]*Default[[:space:]]*$/ {best=$2; gpu=$1}
    END {if (best) print gpu}')
test -n "$gpu"
candidates=$directory/candidates-original99.json
qwen_revision=e61197ed45024b0ed8a2d74b80b4d909f1255473
qwen_path=models/qwen3-reranker-0.6b-e61197e
stage() {
    local name=$1
    shift
    if [[ ! -s "$directory/$name.json" ]]; then
        bash "$directory/run_gpu.sh" "$gpu" "$@" "${common[@]}" --output "$directory/$name.json"
        printf 'Completed %s\n' "$name"
    fi
}
stage candidates-original99 retrieve --e5-model models/e5-small-614241f \
    --e5-revision 614241f622f53c4eeff9890bdc4f31cfecc418b3
stage bge-original99 score --candidates "$candidates" --model bge \
    --model-path models/bge-reranker-v2-m3-953dc6f --model-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
    --weights runs/compare-r1/bge-returns/best-finetuned.safetensors --batch-pause-ms 30
stage qwen-original99 score --candidates "$candidates" --model qwen \
    --model-path "$qwen_path" --model-revision "$qwen_revision" --batch-pause-ms 30
stage qwen-ft-original99 score --candidates "$candidates" --model qwen-ft \
    --model-path "$qwen_path" --model-revision "$qwen_revision" \
    --weights runs/qwen-train-r1/best-finetuned.safetensors --batch-pause-ms 30
