## Ambari

In this blog, I am thrilled to share insights into the meticulous approach we undertook to train Amabri Base ([Cognitive-Lab/Ambari-7B-Instruct-v0.1](https://huggingface.co/Cognitive-Lab/Ambari-7B-Instruct-v0.1)) and Amabri Instruct ([Cognitive-Lab/Ambari-7B-Instruct-v0.1](https://huggingface.co/Cognitive-Lab/Ambari-7B-Instruct-v0.1)). Offering a high-level glimpse into our process, this narrative serves as a precursor to the forthcoming revelation of all technical details— the culmination of extensive testing and evaluation. Stay tuned as we unravel the intricacies that led to the creation of Amabri, an innovative open-source bilingual Kannada-English Large Language Model.

# Why We Built Amabri

**Purpose Behind Amabri**

In the dynamic landscape of Large Language Models (LLMs), the creation of Amabri stemmed from a multifaceted purpose:

- **Language Adaptation of LLMs:** Our primary objective was to pioneer language adaptability within LLMs, bridging the linguistic gap between Kannada and English.
- **Training/Finetuning on a Modest 1B-Token Dataset:** Recognizing the constraints posed by smaller datasets, we aimed to push the boundaries of efficiency by training and finetuning Amabri on a relatively compact dataset of 1 billion tokens.
- **Identifying the Most Efficient Process:** The quest for efficiency led us to meticulously explore and determine the most effective processes at each stage of Amabri's development.
- **Observing World Knowledge Acquisition:** Amabri was conceived as a lens through which we could observe the accrual of world knowledge throughout the training process, shedding light on its adaptability and expansive learning capabilities.
- **Optimizing Training Methods for Each Stage:** A crucial aspect of our endeavor was to discern and optimize the training methods suited for each developmental stage of Amabri.

As LLMs increasingly permeate mainstream usage, open-source models, while enriched in world knowledge, predominantly emerge from English-centric training. Amabri serves as a pioneering initiative to broaden this scope and adapt LLMs to diverse languages.

## Introduction

In the evolving landscape of LLMs, the demand for vast amounts of training data, ranging from 1 trillion to 10 trillion tokens, has become a norm. However, this poses a challenge for languages with limited documented resources. In our pursuit, we focused on the adaptation of a pre-trained LLM, such as Llama/Mistral, to comprehend the nuances of a new language—Kannada in the case of Amabri. Despite Kannada not being classified as a very low-resource language, it served as an ideal candidate to test our hypotheses and methodologies. Rigorously defining the stages of training and finetuning, we set a cap of 1 billion training tokens for the entire process.

Subsequently, we meticulously crafted datasets, distributed them accordingly, and delineated the stages of our process:

- **Pre-training:** 500 Million tokens
- **Bilingual Next Token Prediction/Translation:** ~300 Million tokens
- **Instruct Finetuning/DPO Finetuning:** ~200 Million tokens

This deliberate approach laid the foundation for Amabri's development, pushing the boundaries of language adaptability within the realm of LLMs.

## Tokenization

Tokenization, a critical component in the efficiency of language models, posed a unique challenge for Kannada text within the context of open-source LLMs. Many existing models inefficiently resort to character-level tokenization, especially during inference, impacting overall performance. To address this, we developed a specialized tokenization model for Kannada text using SentencePiece. This model was seamlessly integrated with the base Llama tokenizer, resulting in a comprehensive vocabulary of 49,600 , expanded by 17,600 .

Our approach involved training the tokenizer model on three different dataset sizes, revealing optimal results with a dataset comprising 100,000 tokens. As we evolve Amabri, the upcoming iteration will feature a refined tokenization strategy, employing a reduced vocabulary size of 48,000. This adjustment, validated by insights shared by Andrej Karpathy in his Twitter post ([Andrej Karpathy on Twitter](https://twitter.com/karpathy/status/1621578354024677377)), is geared towards enhancing overall efficiency.

Curious to explore the efficiency gains firsthand? You can test out the tokenizer in action [here](https://github.com/adithya-s-k/LLM-Alchemy-Chamber/blob/main/LLMs/ambari/tokeniser.ipynb).

## Continual Pre-Training

**Pre-Training**

With an efficient tokenizer in place, our next crucial step was the pre-training phase, aimed at familiarizing the model with the newly enriched vocabulary. To optimize this process, we curated a comprehensive dataset from diverse sources. Notably, we explored two distinct approaches during this phase—pre-training with Lora and fully training the model. This strategic decision stemmed from our desire to discern the optimal path for Amabri's development.

A detailed comparison between these methodologies will be unveiled shortly, but we've gleaned some initial observations:

- Contrary to our hypothesis, full-weight fine-tuning did not result in a significant performance decrease compared to Lora.
- The fully fine-tuned model exhibited a remarkable increase in predicting Kannada tokens when the preceding token was Kannada, indicating enhanced language understanding.
- Additionally, we noted a heightened robustness in the generation capabilities of the fully fine-tuned model.

While we acknowledge that our ongoing testing may refine these observations, this snapshot provides valuable insights into our progress. The pre-training phase employed a cluster of 2xA100 GPUs, taking approximately 25 hours for full-weight pre-training on a substantial corpus comprising 500 million tokens.

It's worth mentioning that the weights of the fully fine-tuned model are now available on [Hugging Face](https://huggingface.co/Cognitive-Lab/Ambari-7B-base-v0.1)🤗 - https://huggingface.co/Cognitive-Lab/Ambari-7B-base-v0.1, contributing to the open-source knowledge sharing within the community.

## Bilingual Next Token Prediction and Translation

**Bilingual Next Token Prediction**

This phase, inspired by the open Hathi series by [sarvam.ai](http://sarvam.ai/), was an unplanned yet pivotal addition to our training strategy. Creating a dataset of 200,000 tokens, we utilized Lora for fine-tuning, aiming to equip the model with enhanced language understanding. As we progressed, our focus shifted towards instilling 'world knowledge' in Kannada. Given the scarcity of Kannada content, especially compared to English, we turned to translation. Leveraging IndicTrans2, we translated English content, primarily sourced from Wikipedia, into Kannada. However, instead of conventional monolingual next token prediction, we introduced a groundbreaking approach — bilingual next token prediction. Alternating sentences between Kannada and English, this method compelled the model to cross-lingually attend to information during next-token prediction. This nuanced approach not only fostered increased alignment between Kannada and English but also naturally balanced exposure to Hindi and English tokens during training. This stage added an extra layer of sophistication to Amabri's training journey.

**Translation Finetuning**

The intention behind this phase was to establish a coherent relationship between English and corresponding Kannada tokens. Employing low-rank adaptation for fine-tuning, we encountered some challenges, notably with the decision to use a very low-rank value, which proved less effective. With a dataset size of 100,000 tokens, this stage presented limitations, and we acknowledge the need for improvements. As we refine this aspect of the training process, our commitment to enhancing the bilingual capabilities of Amabri remains unwavering.

## Bilingual Instruct Fine-tuning

**Bilingual Instruct Fine-tuning**

In this pivotal stage, we employed supervised fine-tuning with low-rank adaptation to mold the model's responsiveness. Embracing a chat template structure consisting of user prompts/instructions and corresponding responses, we ventured into the realm of Bilingual Instruct Fine-tuning. This approach involved training the model to adeptly respond in either English or Kannada based on the language specified in the user prompt or instruction.

Chat Template

```bash
<|user|>
{user prompt / instruction}
<|endoftext|>
<|assistant|>
{response}
<|endoftext|>
```

For instance, given a user prompt like 

"Give me 10 Study tips in Kannada," 

> Response
> 

the model seamlessly generates a response in Kannada, maintaining linguistic coherence. To enrich the training process, we amalgamated various instruction datasets, including [Alpaca Instruct](https://huggingface.co/datasets/tatsu-lab/alpaca), [Dolly Instruct](https://huggingface.co/datasets/c-s-ale/dolly-15k-instruction-alpaca-format), and more. Leveraging translation APIs such as Google, Azure, and a custom deployment of the IndicTrans2 model from [ai4bharat](https://ai4bharat.iitm.ac.in/), we crafted a comprehensive bilingual instruct dataset.

The dataset, now publicly available on Hugging Face [here](https://huggingface.co/datasets/Cognitive-Lab/Kannada-Instruct-dataset), encompasses diverse linguistic scenarios. During training, we implemented supervised fine-tuning with four distinct representations:

1. Kannada Instruction → Kannada Output
2. English Instruction → Kannada Output
3. Kannada Instruction → English Output
4. English Instruction → Kannada Output

This meticulous approach not only familiarized the model with responding in different languages but also laid the groundwork for mastering various cross-lingual tasks.

 The weights of this finely-tuned model are accessible on Hugging Face, and for a hands-on experience, you can explore the 4-bit quantized version on [chat.cognitivelab.in](https://chat.cognitivelab.in/).

## DPO Fine-tuning

In the culminating phase of our model refinement, we delved into the world of Direct Preference Optimization (DPO). This strategic choice, inspired by the success observed in various open-source models, aimed not only to align our model but also to drive improvements in benchmarks. Embarking on this experimental journey, we leveraged the [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset. Translating it to Kannada, we subjected the model to DPO fine-tuning, currently undergoing a comprehensive evaluation to gauge its performance impact.

## Learnings and Conclusion: Navigating Challenges and Unveiling Potential

**Scope of Improvement**

- **Lack of World Knowledge:** The model, trained with a capped dataset of 1 billion Kannada tokens, exhibits occasional hallucinations in response to highly specific queries, signaling a scope for improvement in imparting broader world knowledge.
- **Translation Challenges:** Notably, translation nuances surface when handling nouns like names and places. Addressing this challenge involves allocating more training data specifically for the translation phase, a key focus for future enhancements.
- **Full Weight Fine-tuning Dilemma:** An observation surfaced regarding the model's slight overfitting to predict Kannada tokens due to full weight fine-tuning. Future iterations will strategically decide between full weight and Lora for continual pre-training based on extensive tests and evaluations.


## Usage Note

It's crucial to be aware that the models provided in this framework have not undergone detoxification. While they showcase impressive linguistic capabilities, there is a potential for generating content that may be considered harmful or offensive. Users are strongly advised to exercise discretion and closely monitor the model's outputs, especially in public or sensitive applications.

## Contributions

We welcome contributions to enhance and expand this project. If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the GNU GPL v3.0 license. For details, refer to the [LICENSE.md](LICENSE.md) file.

**IMPORTANT:** The GPL 3.0 License applies solely to the source code and datasets provided in this repository. As Indic-LLM is a derivative of Meta's LLama 2 model, it is subject to the original licensing of LLama 2, which cannot be altered. Therefore, for comprehensive details regarding the licensing of the model, please consult the LLAMA2-LICENSE file.