# Tokenizer Training and Merging

This section focuses on expanding the vocabulary size of the tokenizer for improved performance, especially in handling languages with specific tokenization challenges.

### Why Tokenizer Training and Merging?

Llama2 and Mistral share a common tokenizer trained using SentencePiece, a type of Byte Pair Encoding (BPE) tokenizer, with a vocabulary size of 32K. The vocabulary size refers to the number of unique subwords or tokens generated during the tokenization process. Each token is assigned a unique number for representation.

While Llama2 is proficient in tokenizing English, it falls short when dealing with indic languages. It generally employs character-level tokenization, which is suboptimal for inference and deployment. To enhance the tokenizer's efficiency, a new tokenizer is trained using SentencePiece exclusively on the text of the specific language. This new tokenizer is then merged with the existing one to significantly improve efficiency (by 80 to 90%). Continual pretraining is then performed to acclimate the tokenizer to the newly added tokens in the vocabulary.

### Important Points to Note

- **Not Mandatory:** This stage is optional and not mandatory for all scenarios.
- **Not Required for Gemma:** Gemma already possesses a substantial vocabulary size of 256k and excels in tokenizing indic languages.
- **Performance Impact:** It's essential to be aware that extending the vocabulary may lead to a decrease in model performance.

Certainly! Here's a brief documentation for each command parser:

### Train Tokenizer
Use this command to train a tokenizer based on the provided input corpus.

```shell
python ./scripts/tokenizer.py \
    --train \
    --input-file /path/to/input_corpus.txt \
    --output-dir ./models \
    --model-prefix SP_tokenizer \
    --vocab-size 20000 \
    --character-coverage 1.0 \
    --model-type bpe
```

- `--train`: Enable training the tokenizer.
- `--input-file`: Path to the input text corpus file (should be a .txt file).
- `--output-dir`: Directory to save the trained model and vocabulary.
- `--model-prefix`: Name to save the SentencePiece model as.
- `--vocab-size`: Total vocabulary size of the tokenizer.
- `--character-coverage`: Character coverage for the model (default: 1.0).
- `--model-type`: Type of SentencePiece model (choices: "bpe", "unigram", "char", "word").

### Merge Tokenizer
Use this command to merge two tokenizers.

```shell
python ./scripts/tokenizer.py \
    --merge \
    --base-tokenizer /path/to/base_tokenizer.model \
    --trained-tokenizer /path/to/trained_tokenizer.model
```

- `--merge`: Enable merging two tokenizers.
- `--base-tokenizer`: Base tokenizer name or path.
- `--trained-tokenizer`: Tokenizer name or path to merge with the base tokenizer.

### Test Tokenizer
Use this command to test the tokenizer on a sample text.

```shell
python ./scripts/tokenizer.py \
    --test \
    --tokenizer-model /path/to/tokenizer_model.model \
    --text "Your sample text for testing tokenization."
```

- `--test`: Enable testing the tokenizer.
- `--tokenizer-model`: Name or path of the tokenizer model.
- `--text`: Input text to tokenize.

### Counting Indic Tokens
Use this command to count the number of Indic tokens using UTF-8 ranges.

```shell
python ./scripts/tokenizer.py \
    --count-indic-tokens \
    --tokenizer-model /path/to/tokenizer_model.model
```

- `--count-indic-tokens`: Enable counting the number of Indic tokens.
- `--tokenizer-model`: Name or path to the tokenizer model.


To fine all the argument parsers for each operation -[tokenizer.py](https://github.com/adithya-s-k/LLama-K/blob/main/scripts/tokenizer.py)