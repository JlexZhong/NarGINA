#!/bin/bash

wandb online

deepspeed  --include localhost:2,3  --master_port 61000  train/train_mem.py \
--deepspeed ./scripts/zero3.json \
--model_name_or_path /disk/NarGINA/weights/vicuna-7b-v1.5 \
--version conv_childtext \
--cache_dir ../../checkpoint \
--pretrained_embedding_type GRACE_512 \
--tune_mm_mlp_adapter True \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--bf16 True \
--output_dir ./checkpoints/ChildText/score/llaga-vicuna7b-GRACE_512_STv2-mlpv2-no_edgestr-pre_metric-stage2-lora-v3 \
--num_train_epochs 20 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 2 \
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
--lazy_preprocess True \
--report_to wandb \
--sample_neighbor_size 10 \
--mm_projector_type attn_mlp_v2 \
--use_task graph_score \
--use_dataset ChildText \
--run_name ChildText_vicun7b_proj_GRACE_512_STv2_mlpv2-no_edgestr-pre_metric-stage2-lora-v3 \
--eval_accumulation_steps 20 \
--save_total_limit 2 \
--lora_enable True \
--data_path /disk/NarGINA/dataset/ChildText/embedings/GRACE_ST_v2/pretrained_GAT_hidden=512_train.pkl \
--eval_data_path /disk/NarGINA/dataset/ChildText/embedings/GRACE_ST_v2/pretrained_GAT_hidden=512_val.pkl \
--graph_encoder_type None \
--tune_graph_encoder False \
--graph_encoder_num_hidden 2048 \
--is_only_graph False \
--is_trait_comment False \
--is_trait False \
--is_precompute_micro_metric True \
--is_edge_str False \
--pretrain_mm_mlp_adapter /disk/NarGINA/checkpoints/ChildText/graph_match/llaga_vicuna7b_GRACE_512_STv2_mlpv2_graph_match/mm_projector.bin



# deepspeed  --include localhost:3,4  --master_port 61001  train/train_mem.py \
# --deepspeed ./scripts/zero3.json \
# --model_name_or_path /disk/NarGINA/weights/vicuna-7b-v1.5 \
# --version conv_childtext \
# --cache_dir ../../checkpoint \
# --pretrained_embedding_type GRACE_512 \
# --tune_mm_mlp_adapter True \
# --mm_use_graph_start_end False \
# --mm_use_graph_patch_token False \
# --bf16 True \
# --output_dir ./checkpoints/ChildText/llaga-vicuna7b-GRACE_512_STv1-mlpv2-no_edgestr-pre_metric-stage2-lora \
# --num_train_epochs 20 \
# --per_device_train_batch_size 12 \
# --per_device_eval_batch_size 1 \
# --gradient_accumulation_steps 1 \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --learning_rate 5e-4 \
# --weight_decay 0.01 \
# --warmup_ratio 0.03 \
# --lr_scheduler_type cosine \
# --logging_steps 1 \
# --tf32 True \
# --model_max_length 4096 \
# --gradient_checkpointing True \
# --lazy_preprocess True \
# --report_to wandb \
# --use_hop 2 \
# --sample_neighbor_size 10 \
# --mm_projector_type attn_mlp_v2 \
# --use_task graph_score \
# --use_dataset ChildText \
# --template none \
# --run_name ChildText_vicun7b_proj_GRACE_512_STv1_mlpv2-no_edgestr-pre_metric-stage2-lora \
# --eval_accumulation_steps 20 \
# --save_total_limit 2 \
# --lora_enable True \
# --data_path /disk/NarGINA/dataset/ChildText/embedings/GRACE_ST_v1/pretrained_GAT_hidden=512_train.pkl \
# --eval_data_path /disk/NarGINA/dataset/ChildText/embedings/GRACE_ST_v1/pretrained_GAT_hidden=512_val.pkl \
# --graph_encoder_type None \
# --tune_graph_encoder False \
# --graph_encoder_num_hidden 2048 \
# --is_only_graph False \
# --is_trait_comment False \
# --is_trait False \
# --is_precompute_micro_metric True \
# --is_edge_str False \
# --pretrain_mm_mlp_adapter /disk/NarGINA/checkpoints/ChildText/graph_match/llaga_vicuna7b_GRACE_512_STv1_mlpv2_graph_match/checkpoint-644/mm_projector.bin