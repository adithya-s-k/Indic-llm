## Continual Pretraining Overview

### Introduction
Continual pretraining serves as a pivotal stage in developing large language models (LLMs) like [Amabri](https://www.cognitivelab.in/blog/introducing-ambari). This process involves enriching the model's vocabulary, improving its language understanding, and enhancing its generation capabilities. The objective is to familiarize the model with diverse linguistic nuances and optimize its performance on specific language tasks.


### Model Arguments
These arguments pertain to the model, configuration, and tokenizer settings.

- `model_name_or_path`: Model checkpoint for weights initialization (default: `None`).
- `tokenizer_name_or_path`: Tokenizer for weights initialization (default: `None`).
- `model_type`: Type of the model when training from scratch (choices: "bpe", "unigram", "char", "word") if not loading from a checkpoint (default: `None`).
- `config_overrides`: Override existing default config settings when training from scratch (default: `None`).
- `config_name`: Pretrained config name or path if not the same as `model_name`.
- `tokenizer_name`: Pretrained tokenizer name or path if not the same as `model_name`.
- `cache_dir`: Directory to store the pretrained models downloaded from huggingface.co (default: `None`).
- `use_fast_tokenizer`: Whether to use one of the fast tokenizers (backed by the tokenizers library) or not (default: `True`).
- `model_revision`: Specific model version to use (default: "main").
- `use_auth_token`: Use the token generated when running `huggingface-cli login` (default: `False`).
- `torch_dtype`: Override the default `torch.dtype` and load the model under this dtype (choices: "auto", "bfloat16", "float16", "float32", default: `None`).

### Data Training Arguments
Arguments pertaining to the data used for training and evaluation.

- `dataset_dir`: Name of the dataset to use (via the datasets library, default: `None`).
- `dataset_config_name`: Configuration name of the dataset to use (via the datasets library, default: `None`).
- `train_file`: Input training data file (a text file, default: `None`).
- `validation_file`: Optional input evaluation data file to evaluate the perplexity on (a text file, default: `None`).
- `max_train_samples`: Truncate the number of training examples to this value for debugging purposes (default: `None`).
- `max_eval_samples`: Truncate the number of evaluation examples to this value for debugging purposes (default: `None`).
- `streaming`: Enable streaming mode (default: `False`).
- `block_size`: Optional input sequence length after tokenization (default: `None`).
- `overwrite_cache`: Overwrite the cached training and evaluation sets (default: `False`).
- `validation_split_percentage`: Percentage of the train set used as the validation set (default: 0.05).
- `preprocessing_num_workers`: Number of processes to use for preprocessing (default: `None`).
- `keep_linebreaks`: Whether to keep line breaks when using TXT files or not (default: `True`).
- `data_cache_dir`: Directory to store the datasets processed (default: "./").

### Custom Training Arguments
Extends `TrainingArguments` with custom parameters.

- `trainable`: Comma-separated list of trainable components in the model (default: "q_proj,v_proj").
- `lora_rank`: LoRA rank (default: 8).
- `lora_dropout`: LoRA dropout rate (default: 0.05).
- `lora_alpha`: LoRA alpha value (default: 32.0).
- `modules_to_save`: Comma-separated list of modules to save (default: "embed_tokens,lm_head").
- `debug_mode`: Enable debug mode (default: `False`).
- `peft_path`: Path to PEFT file (default: `None`).

### Example Command
An example command to run the script:

```shell
torchrun --nnodes 1 --nproc_per_node 1 pretrain.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path path/to/hf/llama/dir \
    --tokenizer_name_or_path path/to/chinese/llama/tokenizer/dir \
    --dataset_dir path/to/pt/data/dir \
    --data_cache temp_data_cache_dir \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 200 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir output_dir \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --trainable q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
    --modules_to_save embed_tokens,lm_head \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
```

### Tokenization Efficiency
Efficient tokenization is a foundational step in the pretraining pipeline. A robust tokenizer helps the model understand and process text effectively. In our approach, we utilized an efficient tokenizer, facilitating the subsequent pretraining phases.

### Pretraining Approaches
During the pretraining phase, we explored two main approaches:

1. **Lora-based Pretraining:**
   - **Methodology:** Utilizing LoRA (Learned Random Access) for efficient and targeted pretraining.
   - **Observations:** Initial observations revealed nuanced language understanding; however, further comparison with full-weight fine-tuning was necessary to determine the optimal approach.

2. **Full-weight Fine-tuning:**
   - **Methodology:** Training the model comprehensively on the enriched vocabulary.
   - **Observations:** Contrary to expectations, full-weight fine-tuning did not significantly degrade performance. Notably, the fully fine-tuned model demonstrated improved token prediction, especially in scenarios involving specific languages like Kannada.

### Initial Observations
The preliminary results indicated several noteworthy observations:

1. **Performance Comparison:**
   - Full-weight fine-tuning did not lead to a significant decrease in performance compared to Lora-based pretraining.
   - The fully fine-tuned model showcased enhanced language understanding, particularly in predicting tokens of specific languages.

2. **Robustness in Generation:**
   - The fully fine-tuned model demonstrated heightened robustness in generating text, indicating improved language capabilities.

### Computational Resources
The pretraining phase was executed on a cluster equipped with 2xA100 GPUs. The process took approximately 25 hours for full-weight pre-training on a substantial corpus comprising 500 million tokens.

### Conclusion
Continual pretraining serves as a critical phase in LLM development, optimizing models for language understanding and generation tasks. The exploration of different pretraining approaches provides insights into model behavior, guiding further refinement and development. Ongoing testing will likely refine these observations, contributing to the continual evolution of the language model.


