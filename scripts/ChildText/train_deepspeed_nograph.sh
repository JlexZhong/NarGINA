#!/bin/bash

wandb online

deepspeed  --include localhost:1,3,4,7 --master_port 61000  train/train_mem_nograph.py \
--deepspeed ./scripts/zero3.json \
--model_name_or_path /disk/NarGINA/weights/Llama-2-7b-chat-hf \
--data_path /disk/NarGINA/dataset/ChildText_onlytext/score_trait/trait_train_data.json \
--eval_data_path /disk/NarGINA/dataset/ChildText_onlytext/score_trait/trait_validate_data.json \
--version conv_childtext_llama2 \
--cache_dir ../../checkpoint \
--bf16 True \
--output_dir ./checkpoints/ChildText/score_trait/llama2_7b-nograph-trait-lora \
--num_train_epochs 10 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy epoch \
--save_strategy epoch \
--learning_rate 5e-4 \
--weight_decay 0.01 \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--lazy_preprocess False \
--report_to wandb \
--run_name ChildText_llama2_7b_nograph_trait_lora \
--eval_accumulation_steps 20 \
--save_total_limit 3 \
--lora_enable True \
--is_trait_comment False \
--is_trait True