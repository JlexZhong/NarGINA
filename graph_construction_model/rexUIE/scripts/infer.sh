echo "experiment for test, ${data_name}"

torchrun --nproc_per_node=1 event_extraction.py \
        --bert_model_dir=log/debug-trigger \
        --data_path=data/narrative \
        --run_name=debug \
        --task_metrics=trigger \
        --do_predict=True \
        --per_device_train_batch_size=16 \
        --gradient_accumulation_steps=1 \
        --per_device_eval_batch_size=16 \
        --evaluation_strategy=no \
        --num_train_epochs=10 \
        --learning_rate=3e-5 \
        --lr_scheduler_type=linear \
        --log_level=info \
        --logging_strategy=epoch \
        --logging_steps=100 \
        --seed=42 \
        --fp16 \ #  --no_cuda=True when only cpu available \
        --report_to=none \
        --save_strategy=epoch \
        --save_total_limit=3 \
        --greater_is_better=True \
        --metric_for_best_model=f1 \
        --verbose_debug \
        --remove_unused_columns=False \
        --in_low_res=true \
        --output_dir=./log \
        --load_checkpoint=./log/debug-trigger