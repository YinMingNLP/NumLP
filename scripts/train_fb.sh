#!/usr/bin/env bash
TASK="FB15K237"

KGE_MODEL="CoLE"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    KGE_MODEL=$1
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/output/${TASK}/$(date +%Y%m%d-%H%M%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/dataset/${TASK}"
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
python train_fb.py \
    --model_class KGEQwen \
    --kge_model "${KGE_MODEL}" \
    --embedding_dim 768 \
    --train_path "${DATA_DIR}/${KGE_MODEL}/data_KGEQwen/train.json" \
    --eval_path "${DATA_DIR}/${KGE_MODEL}/data_KGEQwen/valid.json" \
    --test_path "${DATA_DIR}/${KGE_MODEL}/data_KGEQwen/test.json" \
    --model_name_or_path "/home/ym/models/Llama-3-8B-Instruct" \
    --dataset "${DATA_DIR}" \
    --source_max_len 3072 \
    --target_max_len 64 \
    --full_finetune False \
    --use_quant True \
    --bf16 \
    --do_train True \
    --do_eval True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 4.0 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --save_strategy steps \
    --save_steps 2 \
    --save_total_limit 20 \
    --optim paged_adamw_32bit \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant \
    --warmup_ratio 0.03 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --dataloader_num_workers 32 \
