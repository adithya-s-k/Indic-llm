python ./merge_adapter.py \
    --base_model_name="Cognitive-Lab/Ambari-7B-Instruct-v0.1" \
    --base_tokenizer_name="Cognitive-Lab/Ambari-7B-Instruct-v0.1" \
    --adapter_model_name="./dpo/checkpoint-2000/" \
    --output_name="dpo-ambari"