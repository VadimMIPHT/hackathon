set -euo pipefail
cd "$HOME/vk-mail-search-hackathon"
umask 077
gpu=$1
if ! nvidia-smi -i "$gpu" --query-gpu=memory.free,compute_mode --format=csv,noheader,nounits |
    awk -F, '$1 >= 24576 && $2 ~ /^[[:space:]]*Default[[:space:]]*$/ {ok=1} END {exit !ok}'; then
    printf 'Insufficient available GPU headroom.\n' >&2
    exit 75
fi
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
py="$HOME/project_venv/bin/python"
"$py" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
timeout --kill-after=5s 400s nice -n 15 "$py" -m mail_search.qwen_train \
    --archive data/mipt_mail_search_participants.zip --splits splits/mailboxes.json \
    --extra-judgements runs/comparison-inputs/refund-judgments.json \
    --bge-reference runs/compare-r1/bge-returns/metrics.json \
    --e5-candidates runs/val-e5-baselines-r2.json --model models/qwen3-reranker-0.6b-e61197e \
    --revision e61197ed45024b0ed8a2d74b80b4d909f1255473 --base-commit 2c4de80d904e430f642c658ff11fc10ef4b7beff \
    --gpu-memory-limit-gib 8 --batch-pause-ms 50 --output-dir runs/qwen-train-r1 &
job=$!
while kill -0 "$job" 2>/dev/null; do
    sleep 3
    if ! nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits |
        awk '$1>=16000 {ok=1} END {exit !ok}'; then
        if kill -0 "$job" 2>/dev/null; then
            printf 'Low GPU headroom; stopping owned Qwen job.\n' >&2
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
