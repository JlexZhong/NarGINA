#!/bin/bash

wandb online

deepspeed  --include localhost:2,3,4,6 --master_port 61000  train/train_mem.py \
--deepspeed ./scripts/zero3.json \
--cache_dir ../../checkpoint \
--tune_mm_mlp_adapter True \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--bf16 True \
--weight_decay 0.01 \
--warmup_ratio 0.05 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--lazy_preprocess True \
--report_to wandb \
--mm_projector_type attn_mlp_v2 \
--use_task graph_score \
--use_dataset ChildText \
--save_total_limit 2 \
--run_name ChildText_llama2_7b_GRACE_512_STv1_mlpv2_trait_lora_v2 \
--model_name_or_path /disk/NarGINA/weights/Llama-2-7b-chat-hf \
--output_dir ./checkpoints/ChildText/llaga_llama2_7b_GRACE_512_STv1_mlpv2_trait_lora_v2 \
--lora_enable True \
--pretrained_embedding_type GRACE_512 \
--version conv_childtext_llama2 \
--data_path /disk/NarGINA/dataset/ChildText/embedings/GRACE_ST_v1/pretrained_GAT_hidden=512_train.pkl \
--eval_data_path /disk/NarGINA/dataset/ChildText/embedings/GRACE_ST_v1/pretrained_GAT_hidden=512_val.pkl \
--graph_encoder_type None \
--tune_graph_encoder False \
--graph_encoder_num_hidden 2048 \
--is_only_graph False \
--num_train_epochs 20 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy epoch \
--save_strategy epoch \
--learning_rate 1e-3 \
--is_trait_comment False \
--is_trait True \
--is_precompute_micro_metric False \
--is_edge_str False \
#--pretrain_mm_mlp_adapter /disk/NarGINA/checkpoints/ChildText/graph_match/llaga_vicuna7b_GRACE_512_STv1_mlpv2_graph_match_v2/mm_projector.bin

# "--include","localhost:7",
# "--master_port","61000",
# "train/train_mem.py",
# "--deepspeed", "./scripts/zero2.json",
# "--cache_dir", "../../checkpoint",
# "--tune_mm_mlp_adapter", "True",
# "--mm_use_graph_start_end", "False",
# "--mm_use_graph_patch_token", "False",
# "--bf16", "True",
# "--weight_decay", "0.01",
# "--warmup_ratio", "0.03",
# "--lr_scheduler_type", "cosine",
# "--logging_steps", "1",
# "--tf32", "True",
# "--model_max_length", "4096",
# "--gradient_checkpointing", "True",
# "--lazy_preprocess", "True",
# "--report_to", "wandb",
# "--use_hop", "2",
# "--sample_neighbor_size", "10",
# "--mm_projector_type", "attn_mlp",
# "--use_task", "graph_score",
# "--use_dataset", "ChildText",
# "--template", "none",
# "--save_total_limit", "2",
# "--run_name", "ChildText_vicun7b_ncla_projector_trait_comment",
# "--model_name_or_path", "/disk/NarGINA/weights/vicuna-7b-v1.5",
# "--output_dir", "./checkpoints/ChildText/llaga-vicun7b_ncla_projector_trait_comment",
# "--lora_enable", "False",
# "--pretrained_embedding_type", "ncla_7b",
# "--version", "conv_childtext",
# "--data_path", "/disk/NarGINA/dataset/ChildText/embedings/NCLA_vicuna_7b/pretrained_GraphAttModel_train.pkl",
# "--eval_data_path", "/disk/NarGINA/dataset/ChildText/embedings/NCLA_vicuna_7b/pretrained_GraphAttModel_val.pkl",
# "--graph_encoder_type", "None",
# "--tune_graph_encoder", "False",
# "--graph_encoder_num_hidden", "2048",
# "--is_only_graph", "False",
# "--num_train_epochs", "5",
# "--per_device_train_batch_size", "8",
# "--per_device_eval_batch_size", "4",
# "--gradient_accumulation_steps", "1",
# "--evaluation_strategy", "steps",
# "--eval_steps", "5",
# "--save_strategy", "epoch",
# "--learning_rate", "2e-3",
# "--eval_accumulation_steps", "20",
# "--is_trait_comment", "True"