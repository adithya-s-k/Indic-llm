## Data Sources and Preparation

The following example are with respect to downloading and converting kannada data 

### Download dataset (Optional)
To download [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)
kannada subset 

   ```shell
   python ./scripts/download_dataset.py \
       --hf-dataset "uonlp/CulturaX" \
       --hf-subset "kn" \
       --dataset-split "train" 
   ```


### Convert Dataset to corpus

1. [wikimedia/wikipedia (Kannada rows - 31.4k)](https://huggingface.co/datasets/wikimedia/wikipedia)

   ```shell

   python3 ./scripts/download_dataset.py \
       --hf-dataset "wikimedia/wikipedia" \
       --hf-subset "20231101.kn" \
       --generate-corpus "True" \
       --text-col "text" \
       --output-file-name "kannada_sentence_corpus_wikipedia-31.4k.txt"
   ```

2. [mc4 (Kannada rows -1.06M)](https://huggingface.co/datasets/mc4)
   ```shell
   python3 ./scripts/download_dataset.py \
       --hf-dataset "mc4" \
       --hf-subset "kn" \
       --generate-corpus "True" \
       --text-col "text" \
       --output-file-name "kannada_sentence_corpus_mc4-1.06M.txt"
   ```

3. [uonlp/CulturaX-1.3M](https://huggingface.co/datasets/uonlp/CulturaX/tree/main/kn)
   ```shell
   python3 ./scripts/download_dataset.py \
       --hf-dataset "uonlp/CulturaX" \
       --hf-subset "kn" \
       --generate-corpus "True" \
       --text-col "text" \
       --output-file-name "kannada_sentence_corpus_CulturaX.txt"
   ```
   
4. [oscar-corpus/OSCAR-2201-151k](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201)
5. [indic_glue-13k](https://huggingface.co/datasets/indic_glue)
6. [kannada_news-6k](https://huggingface.co/datasets/kannada_news)
7. [oscar 251k](https://huggingface.co/datasets/oscar/viewer/unshuffled_deduplicated_kn)

## Translation Datasets
1. [bigscience](https://huggingface.co/datasets/bigscience/xP3/tree/main/kn)
2. [opus100](https://huggingface.co/datasets/opus100/viewer/en-kn)


## Language Specific Datasets

### Aya Indic Collection 
1. [CognitiveLab](https://www.cognitivelab.in/) - [Aya Indic Suite](https://huggingface.co/collections/Cognitive-Lab/aya-indic-suite-65eaa0e34a2307f30bbd55e5)



### Hindi datasets

1. [sarvamai/samvaad-hi-v1 · Datasets at Hugging Face](https://huggingface.co/datasets/sarvamai/samvaad-hi-v1)
2. [ai4bharat/indic-instruct-data-v0.1 · Datasets at Hugging Face](https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1)
3. [Hindi Translated Datasets - a manishiitg Collection (huggingface.co)](https://huggingface.co/collections/manishiitg/hindi-translated-datasets-65d80ec92a17b13e8fdd3362)
4. [GenVRadmin/Samvaad-Mixed-Language-2 · Datasets at Hugging Face](https://huggingface.co/datasets/GenVRadmin/Samvaad-Mixed-Language-2)
5. [CohereForAI/aya_dataset · Datasets at Hugging Face](https://huggingface.co/datasets/CohereForAI/aya_dataset)
6. [CohereForAI/aya_collection · Datasets at Hugging Face](https://huggingface.co/datasets/CohereForAI/aya_collection)
7. [SherryT997/HelpSteer-hindi · Datasets at Hugging Face](https://huggingface.co/datasets/SherryT997/HelpSteer-hindi)
8. [FreedomIntelligence/sharegpt-hindi · Datasets at Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/sharegpt-hindi)
9. [sam2ai/hindi_alpaca_dolly_67k · Datasets at Hugging Face](https://huggingface.co/datasets/sam2ai/hindi_alpaca_dolly_67k)
10. [OdiaGenAI/health_hindi_200 · Datasets at Hugging Face](https://huggingface.co/datasets/OdiaGenAI/health_hindi_200)
11. [FreedomIntelligence/evol-instruct-hindi · Datasets at Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-hindi)
12. [FreedomIntelligence/alpaca-gpt4-hindi · Datasets at Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-hindi)
13. [rohansolo/BB-Ultrachat-IndicLingual6-12k · Datasets at Hugging Face](https://huggingface.co/datasets/rohansolo/BB-Ultrachat-IndicLingual6-12k)
14. [rohansolo/BB_HindiHinglishV2 · Datasets at Hugging Face](https://huggingface.co/datasets/rohansolo/BB_HindiHinglishV2)
15. [Tensoic/Bhandara · Datasets at Hugging Face](https://huggingface.co/datasets/Tensoic/Bhandara) (pretraining)
16. [ravithejads/samvaad-hi-filtered · Datasets at Hugging Face](https://huggingface.co/datasets/ravithejads/samvaad-hi-filtered)
17. [HydraIndicLM/hindi_alpaca_dolly_67k · Datasets at Hugging Face](https://huggingface.co/datasets/HydraIndicLM/hindi_alpaca_dolly_67k)


### Kannada datasets

1. [Cognitive-Lab/Kannada-Instruct-dataset · Datasets at Hugging Face](https://huggingface.co/datasets/Cognitive-Lab/Kannada-Instruct-dataset)
2. [Tensoic/airoboros-3.2_kn · Datasets at Hugging Face](https://huggingface.co/datasets/Tensoic/airoboros-3.2_kn)
3. [Tensoic/nvidia_helpsteer_kn · Datasets at Hugging Face](https://huggingface.co/datasets/Tensoic/nvidia_helpsteer_kn)
4. [Tensoic/no_robots_kn · Datasets at Hugging Face](https://huggingface.co/datasets/Tensoic/no_robots_kn)
5. [Tensoic/gpt-teacher_kn · Datasets at Hugging Face](https://huggingface.co/datasets/Tensoic/gpt-teacher_kn)


### Tamil datasets

1. [abhinand/tamil-alpaca-orca · Datasets at Hugging Face](https://huggingface.co/datasets/abhinand/tamil-alpaca-orca)
2. [abhinand/tamil-alpaca · Datasets at Hugging Face](https://huggingface.co/datasets/abhinand/tamil-alpaca)


### Telgu datasets

1. [Telugu-LLM-Labs/uonlp_culturaX_telugu_romanized_100k · Datasets at Hugging Face](https://huggingface.co/datasets/Telugu-LLM-Labs/uonlp_culturaX_telugu_romanized_100k)
2. [Telugu-LLM-Labs/yahma_alpaca_cleaned_telugu_filtered_and_romanized · Datasets at Hugging Face](https://huggingface.co/datasets/Telugu-LLM-Labs/yahma_alpaca_cleaned_telugu_filtered_and_romanized)

### Odia datasets

1. [OdiaGenAIdata/culturax-odia · Datasets at Hugging Face](https://huggingface.co/datasets/OdiaGenAIdata/culturax-odia) (pretraining)
2. [OdiaGenAI/odia_master_data_llama2 · Datasets at Hugging Face](https://huggingface.co/datasets/OdiaGenAI/odia_master_data_llama2)
3. [OdiaGenAI/Odia_Alpaca_instructions_52k · Datasets at Hugging Face](https://huggingface.co/datasets/OdiaGenAI/Odia_Alpaca_instructions_52k)
4. [OdiaGenAI/gpt-teacher-roleplay-odia-3k · Datasets at Hugging Face](https://huggingface.co/datasets/OdiaGenAI/gpt-teacher-roleplay-odia-3k)
3. [Telugu-LLM-Labs/teknium_GPTeacher_general_instruct_telugu_filtered_and_romanized · Datasets at Hugging Face](https://huggingface.co/datasets/Telugu-LLM-Labs/teknium_GPTeacher_general_instruct_telugu_filtered_and_romanized)


### Malayalam datasets

1. [VishnuPJ/Alpaca_Instruct_Malayalam · Datasets at Hugging Face](https://huggingface.co/datasets/VishnuPJ/Alpaca_Instruct_Malayalam)
