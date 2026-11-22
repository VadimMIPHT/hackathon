set -euo pipefail
common=(--original-archive data/mipt_mail_search_participants.zip
        --augmented-archive data/friend/vk_mail_search_augmented.zip
        --splits splits/mailboxes.json --base-commit 2c4de80d904e430f642c658ff11fc10ef4b7beff
        --gpu-memory-limit-gib 5)
directory=runs/friend-eval-r1
candidates=$directory/candidates.json
qwen_revision=e61197ed45024b0ed8a2d74b80b4d909f1255473
qwen_directory=models/qwen3-reranker-0.6b-e61197e
bge_checkpoint=runs/compare-r1/bge-returns/best-finetuned.safetensors
qwen_checkpoint=runs/qwen-train-r1/best-finetuned.safetensors
if [[ ! -s "$candidates" ]]; then
    bash "$1" "$2" retrieve "${common[@]}" \
        --e5-model models/e5-small-614241f --e5-revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
        --output "$candidates"
fi
if [[ ! -s "$directory/bge.json" ]]; then
    bash "$1" "$2" score "${common[@]}" --candidates "$candidates" \
        --model bge --model-path models/bge-reranker-v2-m3-953dc6f \
        --model-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
        --weights "$bge_checkpoint" --batch-size 20 --output "$directory/bge.json"
fi
if [[ ! -s "$directory/qwen.json" ]]; then
    bash "$1" "$2" score "${common[@]}" --candidates "$candidates" \
        --model qwen --model-path "$qwen_directory" --model-revision "$qwen_revision" \
        --batch-size 20 --output "$directory/qwen.json"
fi
if [[ ! -s "$directory/qwen-ft.json" ]]; then
    bash "$1" "$2" score "${common[@]}" --candidates "$candidates" \
        --model qwen-ft --model-path "$qwen_directory" --model-revision "$qwen_revision" \
        --weights "$qwen_checkpoint" --batch-size 20 --output "$directory/qwen-ft.json"
fi
