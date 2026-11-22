set -euo pipefail
cd "$HOME/vk-mail-search-hackathon"
umask 077
run=runs/jev-transfer-r1
python="$HOME/project_venv/bin/python"
export PYTHONPATH="$PWD:$PWD/runs/jev-eval-r1" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
snapshot=$(nvidia-smi --query-gpu=index,memory.free,compute_mode --format=csv,noheader,nounits)
printf '%s\n' "$snapshot"
gpu=$(awk -F, '$2 >= 24576 && $2 > best && $3 ~ /^[[:space:]]*Default[[:space:]]*$/ {best=$2; gpu=$1}
    END {if (best) print gpu}' <<< "$snapshot")
test -n "$gpu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES
common=(--model models/bge-reranker-v2-m3-953dc6f --base-commit "$1"
        --output-dir "$run" --jev-archive data/friend/vk-jev-dataset.zip)
test_inputs=(--original-archive data/mipt_mail_search_participants.zip
             --augmented-archive data/friend/vk_mail_search_augmented.zip --splits splits/mailboxes.json
             --candidates runs/jev-eval-r1/candidates-original99.json --split "$run/split.json")
run_stage() {
    local job status
    timeout --kill-after=5s 360s nice -n 15 "$python" "$run/transfer.py" "$@" &
    job=$!
    trap 'kill -TERM "$job" 2>/dev/null || true' HUP TERM INT
    while kill -0 "$job" 2>/dev/null; do
        sleep 3
        if ! nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits |
            awk '$1 >= 16000 {ok=1} END {exit !ok}'; then
            printf 'Stopping owned job due to low GPU headroom.\n' >&2
            kill -TERM "$job" 2>/dev/null || true
            break
        fi
    done
    status=0
    wait "$job" || status=$?
    trap - HUP TERM INT
    nvidia-smi -i "$gpu" --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader
    return "$status"
}
run_stage train "${common[@]}" --train-manifest "$run/train_manifest.json"
run_stage test "${common[@]}" "${test_inputs[@]}"
