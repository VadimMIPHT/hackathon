set -euo pipefail
cd "$HOME/vk-mail-search-hackathon"
umask 077
mkdir -p runs/depth-r1
shared=${VK_ALLOW_SHARED_GPU:-0}
snapshot=$(nvidia-smi --query-gpu=index,uuid,memory.free,memory.used,utilization.gpu,compute_mode --format=csv,noheader,nounits)
printf '%s\n' "$snapshot"
gpu=$(printf '%s\n' "$snapshot" | awk -F, -v shared="$shared" '
    $4 < 512 && $5 < 5 && idle == "" {idle=$1}
    $6 ~ /^[[:space:]]*Default[[:space:]]*$/ && $3 >= 24576 && $3 > best {best=$3; headroom=$1}
    END {if (idle != "") print idle; else if (shared == 1 && best > 0) print headroom}')
if test -z "$gpu"; then
    printf 'No GPU with the required headroom.\n' >&2
    exit 75
fi
free_gib_floor_mib=16000
pause_ms=$(printf '%s\n' "$snapshot" | awk -F, -v gpu="$gpu" '$1 == gpu {print $4 < 512 && $5 < 5 ? 0 : 50}')
printf 'Assigned GPU %s; batch pause %s ms; PyTorch allocator cap 6 GiB.\n' "$gpu" "$pause_ms"
nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader
gpu_uuid=$(printf '%s\n' "$snapshot" | awk -F, -v gpu="$gpu" '$1 == gpu {gsub(/[[:space:]]/, "", $2); print $2}')
test -n "$gpu_uuid"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_uuid" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
py="$HOME/project_venv/bin/python"
"$py" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
common=(--archive data/mipt_mail_search_participants.zip --splits splits/mailboxes.json
        --e5-candidates runs/val-e5-baselines-r2.json --base-commit 0f47058c0da551eedf9270b3c7eb513d624dac3d)
limits=(--gpu-memory-limit-gib 6 --batch-pause-ms "$pause_ms")
bge=(--name bge --model models/bge-reranker-v2-m3-953dc6f
     --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e)
gpu_headroom() {
    nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits |
        awk -v floor="$1" '$1 >= floor {ok=1} END {exit !ok}'
}
failed=0
stage() {
    local name="$1"
    shift
    printf 'Starting %s\n' "$name"
    if timeout 240s nice -n 10 "$@"; then
        printf 'Completed %s\n' "$name"
    else
        printf 'Failed %s\n' "$name" >&2
        failed=1
    fi
}
gpu_stage() {
    local name="$1"
    shift
    if ! gpu_headroom 24576; then
        printf 'GPU headroom below 24 GiB before %s.\n' "$name" >&2
        failed=1
        return
    fi
    printf 'Starting %s\n' "$name"
    timeout --kill-after=5s 240s nice -n 15 "$@" &
    local job=$!
    while kill -0 "$job" 2>/dev/null; do
        sleep 3
        if ! gpu_headroom "$free_gib_floor_mib" && kill -0 "$job" 2>/dev/null; then
            printf 'Low GPU headroom; stopping owned stage %s.\n' "$name" >&2
            kill -TERM "$job"
            break
        fi
    done
    if wait "$job"; then
        printf 'Completed %s\n' "$name"
    else
        printf 'Failed %s\n' "$name" >&2
        failed=1
    fi
}
gpu_stage bge-return "$py" -m mail_search.depth_benchmark "${common[@]}" "${limits[@]}" "${bge[@]}" \
    --weights runs/compare-r1/bge-returns/best-finetuned.safetensors --output runs/depth-r1/bge-returns.json
gpu_stage rubert "$py" -m mail_search.depth_benchmark "${common[@]}" "${limits[@]}" --name rubert \
    --model models/dity-russian-msmarco-9029bab --revision 9029bab08103ad171724b510d312befa5b476293 \
    --output runs/depth-r1/rubert.json
gpu_stage qwen "$py" -m mail_search.depth_benchmark "${common[@]}" "${limits[@]}" --name qwen \
    --model models/qwen3-reranker-0.6b-e61197e --revision e61197ed45024b0ed8a2d74b80b4d909f1255473 \
    --output runs/depth-r1/qwen.json
gpu_stage bge-frozen "$py" -m mail_search.depth_benchmark "${common[@]}" "${limits[@]}" "${bge[@]}" \
    --output runs/depth-r1/bge-frozen.json
if test -s runs/depth-r1/bge-frozen.json; then
    gpu_stage catboost-features "$py" -m mail_search.depth_boost prepare "${common[@]}" --gpu-memory-limit-gib 6 \
        --e5-model models/e5-small-614241f --frozen-bge-scores runs/depth-r1/bge-frozen.json \
        --output runs/depth-r1/catboost-features.json
fi
nvidia-smi -i "$gpu" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader
if test -s runs/depth-r1/catboost-features.json; then
    export CUDA_VISIBLE_DEVICES=
    stage catboost .venv-catboost/bin/python -m mail_search.depth_boost predict "${common[@]}" \
        --features runs/depth-r1/catboost-features.json --model runs/compare-r1/catboost.cbm \
        --output runs/depth-r1/catboost.json
fi
exit "$failed"
