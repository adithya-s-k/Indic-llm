import sys
import os
import json
import shutil
import logging
import argparse
from tqdm import tqdm


import pandas as pd
from datasets import load_dataset
from dataclasses import asdict
from pathlib import Path

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from indic_llm.tokenization import SentencePieceTrainer
from indic_llm import print_indic_llm_text_art

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TokenizerCLI(SentencePieceTrainer):
    def __init__(self) -> None:
        super().__init__()
    
    def run(self):
        parser = argparse.ArgumentParser(
            description="Tokenizer Script"
        )
        
        # train arguments
        
        parser.add_argument("--train", action="store_true", help="Train Tokeniser")
        parser.add_argument(
            "--input-file",
            required=True,
            help="Path to the input text corpus file.",
        )
        parser.add_argument(
            "--output-dir",
            default="./models",
            help="Directory where the trained model and vocabulary will be saved.",
        )
        parser.add_argument(
            "--model-prefix",
            default="SP_tokenizer",
            help="Prefix for the model and vocabulary filenames.",
        )
        parser.add_argument(
            "--vocab-size",
            type=int,
            default=self.vocab_size,
            help="Size of the vocabulary.",
        )
        parser.add_argument(
            "--character-coverage",
            type=float,
            default=self.character_coverage,
            help="Character coverage for the model.",
        )
        parser.add_argument(
            "--model-type",
            default=self.model_type,
            choices=["bpe", "unigram", "char", "word"],
            help="Type of SentencePiece model.",
        )
        
        # merge arguments
        
        parser.add_argument("--merge", action="store_true", help="Merge Tokenizer")
        parser.add_argument("--base-tokenizer", type=str, help="Base Tokenizer for Merging")
        parser.add_argument("--trained-tokenizer", type=str, help="Base Tokenizer for Merging")

        
        # test arguments
        parser.add_argument("--test", action="store_true", help="Merge Tokenizer")
        parser.add_argument("--tokenizer-model", type=str, help="tokenizer to determine the number of indic tokens")
        parser.add_argument("--text", type=str, help="Pass in the text you want to tokenize")
        
        # test arguments
        parser.add_argument("--test-dataset", action="store_true", help="Merge Tokenizer")
        parser.add_argument("--base-tokenizer", type=str, help="Base Tokenizer for Merging")
        parser.add_argument("--merged-tokenizer", type=str, help="Merged Tokenizer for testing")
        parser.add_argument("--dataset", type=str, help="Dataset to test the tokenizer on")
        parser.add_argument("--text-cols", type=str, help="Column to test the tokenizer on")
        
        # count indic tokens in the tokenizer
        parser.add_argument("--count-tokens", action="store_true", help="Merge Tokenizer")
        parser.add_argument("--tokenizer-model", type=str, help="tokenizer to determine the number of indic tokens")


        args = parser.parse_args()
        
        # Initialize logger with appropriate log level
                # Log parsed arguments
        logger.info("Parsed arguments:")
        for arg, value in vars(args).items():
            logger.info(f"{arg}: {value}")

        print_indic_llm_text_art()

        logger.setLevel(logging.INFO)
        
        
        self.vocab_size = args.vocab_size # 20000 by default
        self.character_coverage = args.character_coverage # 1.0 by default
        self.model_type = args.model_type #[BPE, unigram, char, word]

        os.makedirs(self.output_dir, exist_ok=True)
        
        if args.train:
            self.train_tokenizer(args.text_corpus, args.model_prefix, args.output_dir)
        elif args.merge:
            self.merge_tokenizer(args.base_tokenizer,args.trained_tokenizer ,args.merged_output)
        elif args.test:
            self.test_tokenizer(args.tokenizer_model, args.text)
        elif args.count_tokens:
            args.count_language_tokens(args.tokenizer_model)
        elif args.test_dataset:
            pass # feature to be added
        else:
            logger.error("Please provide either --train or --merge or --test option.")

if __name__ == "__main__":
    tokenizer = TokenizerCLI()
    tokenizer.run()