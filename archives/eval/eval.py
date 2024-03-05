import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
import evaluate
from datasets import load_dataset
from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    dynamic_import_function,
)
from bleurt import score

lang_map = {
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "brx_Deva": "Bodo",
    "doi_Deva": "Dogri",
    "eng_Latn": "English",
    "guj_Gujr": "Gujarati",
    "gom_Deva": "Konkani",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "kas_Arab": "Kashmiri",
    "mai_Deva": "Maithili",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "mni_Mtei": "Manipuri",
    "npi_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "san_Deva": "Sanskrit",
    "sat_Olck": "Santali",
    "snd_Deva": "Sindhi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
}


def format_example(src_text, src_lang, tgt_lang, tgt_text=None):
    prompt = f"{lang_map[src_lang]}: {src_text}"
    prompt += f"\n{lang_map[tgt_lang]}:"
    if tgt_text is not None:
        prompt += f" {tgt_text}\n\n"
    return prompt


def gen_prompt(dev_data, src_lang, tgt_lang, k=-1):
    prompt = f"Translate the following sentence(s) from {lang_map[src_lang]} into {lang_map[tgt_lang]}.\n\n"
    if k > 0:
        exemplars = dev_data.select(range(k))
        for example in exemplars:
            prompt += format_example(
                src_text=example[f"sentence_{src_lang}"],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                tgt_text=example[f"sentence_{tgt_lang}"],
            )
    return prompt


def main(args):
    random.seed(args.seed)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
        )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    dataset = load_dataset(args.dataset, f"{args.src_lang}-{args.tgt_lang}")
    dataset = dataset.map(
        lambda x: {
            f"sentence_{args.src_lang}": x[f"sentence_{args.src_lang}"].strip(),
            f"sentence_{args.tgt_lang}": x[f"sentence_{args.tgt_lang}"].strip(),
        }
    )
    test_data = dataset["gen"] if args.dataset == "ai4bharat/IN22-Gen" else dataset["conv"]
    # test_data = test_data.select(range(50))

    prompts = []
    for i, example in enumerate(test_data):
        dev_data = test_data.filter(
            lambda x: x[f"sentence_{args.src_lang}"] != example[f"sentence_{args.src_lang}"]
        ).shuffle(args.seed)
        k = args.ntrain
        prompt_end = format_example(
            src_text=example[f"sentence_{args.src_lang}"], src_lang=args.src_lang, tgt_lang=args.tgt_lang
        )
        train_prompt = gen_prompt(dev_data, args.src_lang, args.tgt_lang, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += f"The {lang_map[args.tgt_lang]} translation is: "
            else:
                prompt += f" The {lang_map[args.tgt_lang]} translation is: "

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_data, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += f"The {lang_map[args.tgt_lang]} translation is: "
                else:
                    prompt += f" The {lang_map[args.tgt_lang]} translation is: "

            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        prompts.append(prompt)

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=256,
        batch_size=args.eval_batch_size,
        stop_id_sequences=None,
    )
    # remove unnecessary space
    outputs = [output.strip().split("\n")[0] for output in outputs]

    with open(os.path.join(args.save_dir, f"in22_{args.src_lang}_{args.tgt_lang}_predictions.jsonl"), "w") as fout:
        for example, output in zip(test_data, outputs):
            example["prediction_text"] = output
            fout.write(json.dumps(example) + "\n")

    # flush all the GPU memory
    del model
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    print("Calculating bleu, chrf, chrf++, bleurt ...")
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    bleurt = score.BleurtScorer(args.bleurt_model_name_or_path)

    predictions = [output for output in outputs]
    references = [[example[f"sentence_{args.tgt_lang}"]] for example in test_data]

    metrics = {
        "bleu": sacrebleu.compute(predictions=predictions, references=references)["score"],
        "chrf": chrf.compute(predictions=predictions, references=references)["score"],
        "chrf2": chrf.compute(predictions=predictions, references=references, word_order=2)["score"],
        "bleurt": np.mean(
            bleurt.score(candidates=predictions, references=[ref for sublist in references for ref in sublist])
        ),
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5, help="number of examples to use for few-shot evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset", type=str, default="ai4bharat/IN22-Gen", choices=["ai4bharat/IN22-Gen", "ai4bharat/IN22-Conv"]
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        choices=list(lang_map.keys()),
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="hin_Deva",
        choices=list(lang_map.keys()),
    )
    parser.add_argument("--save_dir", type=str, default="results/in22-gen/llama-7B/")
    parser.add_argument(
        "--bleurt_model_name_or_path",
        type=str,
        default="/data/jaygala/bleurt/BLEURT-20",
        help="bleurt model to load for evaluation.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )
    args = parser.parse_args()
    main(args)