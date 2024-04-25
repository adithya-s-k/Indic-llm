# Indic-LLM

Indic-LLM is a framework that provides the foundation to adapt Language Models (LLMs) for Indic languages supporting LLMs such as LLama 2,Mistral,Gemma 

## Installation

```bash
git clone https://github.com/adithya-s-k/Indic-llm.git
cd Indic-llm

conda create -n indic-venv python=3.10
conda activate indic-venv

pip3 install -r requirements.txt
```

## Model Support 
| Model  | Tokeniser | Pretraining(Lora) | SFT | DPO | Evaluation |
|--------|-----------|---------------------|-----|-----|------------|
| LLama2 | ✅        | ✅                 | ✅  | ✅  | ✅         |
| Mistral| ✅        | ✅                 | ✅  | ✅  | ✅         |
| Gemma  | -        | ✅                 | ✅  | ✅  | ✅         |
| Qwen   | -         | -                 | -  | -  | -         |


## Quick Start
Please Refer to the [Docs](./docs)

## Usage Note

It's crucial to be aware that the models provided in this framework have not undergone detoxification. While they showcase impressive linguistic capabilities, there is a potential for generating content that may be considered harmful or offensive. Users are strongly advised to exercise discretion and closely monitor the model's outputs, especially in public or sensitive applications.

## Contributions

We welcome contributions to enhance and expand this project. If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the GNU GPL v3.0 license. For details, refer to the [LICENSE.md](LICENSE.md) file.

**IMPORTANT:** The GPL 3.0 License applies solely to the source code and datasets provided in this repository. As Indic-LLM is a derivative of Meta's LLama 2 model, it is subject to the original licensing of LLama 2, which cannot be altered. Therefore, for comprehensive details regarding the licensing of the model, please consult the LLAMA2-LICENSE file.

<!-- ## Citation

If you use ambari model or the kannada-instruct dataset in your research, please cite:

*Insert citation information here* -->

## Acknowledgment

This repository draws inspiration from the following repositories:
- [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [Tamil-LLaMA](https://github.com/abhinand5/tamil-llama)
