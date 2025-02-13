#!/bin/bash

wandb online

deepspeed  --include localhost:1,4,6,7 --master_port 61001  train/train_mem_nograph_relation_preds.py \
--deepspeed ./scripts/zero3.json \
--model_name_or_path /disk/NarGINA/weights/vicuna-7b-v1.5 \
--data_path /disk/NarGINA/relation_extraction/dataset/train_data.json \
--eval_data_path /disk/NarGINA/relation_extraction/dataset/validate_data.json \
--version conv_edge_pred \
--cache_dir ../../checkpoint \
--bf16 True \
--output_dir ./checkpoints/relation_preds/vicuna_7b \
--num_train_epochs 1 \
--per_device_train_batch_size 10 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy epoch \
--save_strategy "steps"  \
--save_steps 300 \
--learning_rate 3e-4 \
--weight_decay 0.01 \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--lazy_preprocess False \
--report_to wandb \
--run_name relation_preds_vicuna_7b \
--save_total_limit 3 \
--lora_enable True \
--is_trait_comment False \
--is_trait False 