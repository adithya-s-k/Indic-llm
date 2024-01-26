```
accelerate launch examples/research_projects/stack_llama_2/scripts/sft_llama2.py \
    --output_dir="./sft" \
    --max_steps=500 \
    --logging_steps=10 \
    --save_steps=10 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --bf16=True \
    --remove_unused_columns=False \
    --run_name="sft_llama2" \
    --report_to="wandb"
```


accelerate launch dpo.py \
    --model_name_or_path="Cognitive-Lab/Ambari-7B-Instruct-v0.1" \
    --tokenizer_name_or_path="Cognitive-Lab/Ambari-7B-Instruct-v0.1" \
    --num_train_epochs=1 \
    --logging_steps=10 \
    --save_steps=10 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=True \
    --group_by_length=False \
    --lora_alpha=128
    --lora_dropout=0.05
    --lora_r=256
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --bf16=True \
    --remove_unused_columns=False \
    --run_name="dpo_finetuning" \
    --report_to="wandb"
    --output_dir="dpo"