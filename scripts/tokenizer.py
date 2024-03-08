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
            default=self.output_dir,
            help="Directory where the trained model and vocabulary will be saved.",
        )
        parser.add_argument(
            "--model-prefix",
            default=self.model_prefix,
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
        parser.add_argument("--merged-output", type=str, help="Output Path for Merged Tokenizer")
        
        # test arguments
        parser.add_argument("--test", action="store_true", help="Merge Tokenizer")
        parser.add_argument("--base-tokenizer", type=str, help="Base Tokenizer for Merging")
        parser.add_argument("--merged-tokenizer", type=str, help="Base Tokenizer for Merging")
        parser.add_argument("--dataset", type=str, help="Dataset to test the tokenizer on")
        parser.add_argument("--text-cols", type=str, help="Column to test the tokenizer on")

        args = parser.parse_args()
        
        # Initialize logger with appropriate log level
                # Log parsed arguments
        logger.info("Parsed arguments:")
        for arg, value in vars(args).items():
            logger.info(f"{arg}: {value}")

        print_indic_llm_text_art()

        logger.setLevel(logging.INFO)
        
        
        self.output_dir = args.output_dir
        self.model_prefix = args.model_prefix
        self.vocab_size = args.vocab_size
        self.character_coverage = args.character_coverage
        self.model_type = args.model_type

        os.makedirs(self.output_dir, exist_ok=True)
        
        if args.train:
            self.train_tokenizer(args.text_corpus)
        elif args.merge:
            self.merge_tokenizer(args.base_tokenizer,args.trained_tokenizer ,args.merged_output)
        elif args.test:
            self.test_tokeniser(args.text_corpus, args.output)
        else:
            logger.error("Please provide either --train or --merge or --test option.")

if __name__ == "__main__":
    tokenizer = TokenizerCLI()
    tokenizer.run()