# Tokenizer Training and Merging

This section focuses on expanding the vocabulary size of the tokenizer for improved performance, especially in handling languages with specific tokenization challenges.

### Why Tokenizer Training and Merging?

Llama2 and Mistral share a common tokenizer trained using SentencePiece, a type of Byte Pair Encoding (BPE) tokenizer, with a vocabulary size of 32K. The vocabulary size refers to the number of unique subwords or tokens generated during the tokenization process. Each token is assigned a unique number for representation.

While Llama2 is proficient in tokenizing English, it falls short when dealing with indic languages. It generally employs character-level tokenization, which is suboptimal for inference and deployment. To enhance the tokenizer's efficiency, a new tokenizer is trained using SentencePiece exclusively on the text of the specific language. This new tokenizer is then merged with the existing one to significantly improve efficiency (by 80 to 90%). Continual pretraining is then performed to acclimate the tokenizer to the newly added tokens in the vocabulary.

### Important Points to Note

- **Not Mandatory:** This stage is optional and not mandatory for all scenarios.
- **Not Required for Gemma:** Gemma already possesses a substantial vocabulary size of 256k and excels in tokenizing indic languages.
- **Performance Impact:** It's essential to be aware that extending the vocabulary may lead to a decrease in model performance.

## Train Tokenizer


## Merge Tokenizer


## Test Tokenizer

