set -euo pipefail
cd "$HOME/vk-mail-search-hackathon"
umask 077
gpu=$1
shift
if ! nvidia-smi -i "$gpu" --query-gpu=memory.free,compute_mode --format=csv,noheader,nounits |
    awk -F, '$1 >= 24576 && $2 ~ /^[[:space:]]*Default[[:space:]]*$/ {ok=1} END {exit !ok}'; then
    printf 'GPU headroom below 24 GiB.\n' >&2
    exit 75
fi
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
py="$HOME/project_venv/bin/python"
"$py" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
mkdir -p runs/friend-eval-r1
timeout --kill-after=5s 420s nice -n 15 "$py" -m mail_search.friend_eval "$@" &
job=$!
while kill -0 "$job" 2>/dev/null; do
    sleep 3
    if ! nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits |
        awk '$1>=16000 {ok=1} END {exit !ok}'; then
        if kill -0 "$job" 2>/dev/null; then
            printf 'Low GPU headroom; stopping owned evaluation job.\n' >&2
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
