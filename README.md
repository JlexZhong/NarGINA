# <center> NarGINA: Narrative Graph-based Interpretable Children's Narrative Ability Assessment</center>

#### Step 1: Environment  

```shell
# create a new environment
conda create -n nargina python=3.10
conda activate nargina

# install pytorch. Modify the command to align with your own CUDA version.
pip3 install torch  --index-url https://download.pytorch.org/whl/cu118

# install related libraries
pip install -r requirements.txt

# install flash-attn
pip install flash-attn --no-build-isolation

# install pyg
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

```
#### Step 2:  Data Preparation
Download our datasets from https://huggingface.co/datasets/JlexZzz/NarGINA-corpus.

#### Step 3: Training
To execute the training process, you can run either `./scripts/ChildText/train_deepspeed_trait.sh`. The usage instructions are as follows:
```shell
#!/bin/bash

wandb online

deepspeed  --include localhost:1,2,3,4 --master_port 61000  train/train_mem.py \
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
--sample_neighbor_size 10 \
--mm_projector_type attn_mlp_v2 \
--use_task graph_score \
--use_dataset ChildText \
--save_total_limit 2 \
--run_name ChildText_vicuna7b_GRACE_512_t5uie-vicuna_mlpv2_trait_lora \
--model_name_or_path /disk/NarGINA/weights/vicuna-7b-v1.5 \
--output_dir ./checkpoints/ChildText/llaga_vicuna7b_GRACE_512_t5uie-vicuna_mlpv2_trait_lora \
--lora_enable True \
--pretrained_embedding_type GRACE_512 \
--version conv_childtext \
--data_path /disk/NarGINA/dataset/ChildText/embedings/GRACE_ST_v1/pretrained_GAT_hidden=512_train.pkl \
--eval_data_path /disk/NarGINA/dataset/ChildText/embedings/GRACE_ST_v1/pretrained_GAT_hidden=512_val.pkl \
--graph_encoder_type None \
--tune_graph_encoder False \
--graph_encoder_num_hidden 2048 \
--is_only_graph True \
--num_train_epochs 20 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy epoch \
--save_strategy epoch \
--learning_rate 3e-4 \
--is_trait_comment False \
--is_trait True \
--is_precompute_micro_metric False \
--is_edge_str False \
```

#### Step 4: Evaluation
You can evaluate NarGINA with running `eval/ChildText/eval_pretrain_ChildText_trait.py`:

```shell
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/disk/NarGINA/checkpoints/ChildText/score_trait/llaga_vicuna7b_GRACE_512_STv1_mlpv2_no_edgestr_trait_lora_v2")
    parser.add_argument("--model_base", type=str, default="/disk/NarGINA/weights/vicuna-7b-v1.5")
    parser.add_argument("--data_path", type=str, default="/disk/NarGINA/dataset/ChildText_test/teacher_2/embedings/GRACE_ST_t5uie-vicuna/pretrained_GAT_hidden=512_test.pkl")
    #parser.add_argument("--pretrained_embedding_type", type=str, default="GRACE_512")#ST_encoder,GRACE_512
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="/disk/NarGINA/output/ChildText/score_trait_v2/answer_childtext_vicuna_7b_mlpv2_GRACE_512_t5uie-vicuna_trait_lora")
    parser.add_argument("--conv_mode", type=str, default="conv_childtext") #conv_childtext_llama2
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="graph_score")
    parser.add_argument("--dataset", type=str, default="ChildText")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--is_only_graph", type=bool, default=False)#TODO
    parser.add_argument("--is_trait_comment", type=bool, default=False)#TODO
    parser.add_argument("--is_trait", type=bool, default=True)#TODO
    parser.add_argument("--is_precompute_micro_metric", type=bool, default=False)#TODO
    parser.add_argument("--is_edge_str", type=bool, default=False)#TODO
    args = parser.parse_args()
```

#### Step 5: interpretability
Please run `serve/my_app.py`.
```shell
demo.launch(server_name="0.0.0.0", server_port=7860)
```


