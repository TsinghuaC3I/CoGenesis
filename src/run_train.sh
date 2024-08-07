#!/bin/sh
# export NCCL_P2P_LEVEL=NVL
data_dir="../data/ablation"
task="ctx"
lr=1e-5
# set data_type to empty for with context, wo for without context, outline for sketch-based method
data_type=$1

echo "Start training ${task} with lr=${lr}, data type = ${data_type}"

torchrun --nproc_per_node=4 --master_port=20001 run_train.py \
    --model_name_or_path Qwen/Qwen1.5-1.8B-Chat  \
    --data_path ${data_dir}/${task}_train_"${data_type}".json \
    --eval_data_path ${data_dir}/${task}_dev_"${data_type}".json \
    --bf16 True \
    --output_dir ./outputs/${task}_"${data_type}"/qwen1.5_1_8b-Chat-${lr} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --trust_remote_code True \
    --load_best_model_at_end
