## Pretraining DataSources

1. [wikimedia/wikipedia-31.4k](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.kn) \
```python3 ./scripts/dataset/generate_text_corpus.py --hf-dataset "wikimedia/wikipedia" --hf-corpus-path "20231101.kn" --text-col "text" --output-file-name "kannada_sentence_corpus_wikipedia-31.4k.txt"```
2. [mc4-1.06M](https://huggingface.co/datasets/mc4/viewer/kn) \
```python3 ./scripts/dataset/generate_text_corpus.py --hf-dataset "mc4" --hf-corpus-path "kn" --text-col "text" --output-file-name "kannada_sentence_corpus_mc4-1.06M.txt"```
3. [oscar-corpus/OSCAR-2201-151k](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201/viewer/kn)
4. [indic_glue-13k](https://huggingface.co/datasets/indic_glue/viewer/csqa.kn)
5. [kannada_news-6k](https://huggingface.co/datasets/kannada_news)
6. [uonlp/CulturaX-1.3M](https://huggingface.co/datasets/uonlp/CulturaX/tree/main/kn) \
```python3 ./scripts/dataset/generate_text_corpus.py --hf-dataset "uonlp/CulturaX" --hf-corpus-path "kn" --text-col "text" --output-file-name "kannada_sentence_corpus_CulturaX_1.3M.txt"```
7. [oscar 251k](https://huggingface.co/datasets/oscar/viewer/unshuffled_deduplicated_kn)

## Translation Datasets
1. [bigscience](https://huggingface.co/datasets/bigscience/xP3/tree/main/kn)
2. [opus100](https://huggingface.co/datasets/opus100/viewer/en-kn)

## Instruction Datasets
