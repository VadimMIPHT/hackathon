set -euo pipefail
cd "$HOME/vk-mail-search-hackathon"
umask 077
snapshot=$(nvidia-smi --query-gpu=index,uuid,memory.free,memory.used,utilization.gpu,compute_mode --format=csv,noheader,nounits)
printf '%s\n' "$snapshot"
gpu=$(printf '%s\n' "$snapshot" | awk -F, '
    $4 < 512 && $5 < 5 && idle == "" {idle=$1}
    $6 ~ /^[[:space:]]*Default[[:space:]]*$/ && $3 >= 24576 && $3 > best {best=$3; headroom=$1}
    END {if (idle != "") print idle; else if (best > 0) print headroom}')
if test -z "$gpu"; then
    printf 'No GPU with the required headroom.\n' >&2
    exit 75
fi
gpu_uuid=$(printf '%s\n' "$snapshot" | awk -F, -v gpu="$gpu" '$1 == gpu {gsub(/[[:space:]]/, "", $2); print $2}')
test -n "$gpu_uuid"
pause_ms=$(printf '%s\n' "$snapshot" | awk -F, -v gpu="$gpu" '$1 == gpu {print $4 < 512 && $5 < 5 ? 0 : 50}')
printf 'Assigned GPU %s. Batch pause %s ms, PyTorch allocator cap 6 GiB.\n' "$gpu" "$pause_ms"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_uuid"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
py="$HOME/project_venv/bin/python"
"$py" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
timeout --kill-after=5s 180s nice -n 15 "$py" -m mail_search.final_test \
    --archive data/mipt_mail_search_participants.zip --splits splits/mailboxes.json \
    --protocol runs/final-test/protocol.json --e5-model models/e5-small-614241f \
    --bge-model models/bge-reranker-v2-m3-953dc6f \
    --bge-weights runs/compare-r1/bge-returns/best-finetuned.safetensors \
    --base-commit 0f47058c0da551eedf9270b3c7eb513d624dac3d \
    --gpu-memory-limit-gib 6 --batch-pause-ms "$pause_ms" --output runs/final-test/metrics.json &
job=$!
while kill -0 "$job" 2>/dev/null; do
    sleep 3
    if ! nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1>=16000 {ok=1} END {exit !ok}'; then
        if kill -0 "$job" 2>/dev/null; then
            printf 'Low GPU headroom, stopping the test process.\n' >&2
            kill -TERM "$job"
        fi
        break
    fi
done
status=0
wait "$job" || status=$?
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader
exit "$status"
