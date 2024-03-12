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

        # Train arguments
        parser.add_argument("--train", action="store_true", help="Enable training the tokenizer.")
        parser.add_argument("--input-file", required=True, help="Path to the input text corpus file (should be a .txt file).")
        parser.add_argument("--output-dir", default="./models", help="Directory to save the trained model and vocabulary.")
        parser.add_argument("--model-prefix", default="SP_tokenizer", help="Name to save the SentencePiece model as.")
        parser.add_argument("--vocab-size", type=int, default=self.vocab_size, help="Total vocabulary size of the tokenizer.")
        parser.add_argument("--character-coverage", type=float, default=self.character_coverage, help="Character coverage for the model (default: 1.0).")
        parser.add_argument("--model-type", default=self.model_type, choices=["bpe", "unigram", "char", "word"], help="Type of SentencePiece model.")

        # Merge arguments
        parser.add_argument("--merge", action="store_true", help="Enable merging two tokenizers.")
        parser.add_argument("--base-tokenizer", type=str, help="Base tokenizer name or path.")
        parser.add_argument("--trained-tokenizer", type=str, help="Tokenizer name or path to merge with the base tokenizer.")

        # Test arguments
        parser.add_argument("--test", action="store_true", help="Enable testing the tokenizer.")
        parser.add_argument("--tokenizer-model", type=str, help="Name or path of the tokenizer model.")
        parser.add_argument("--text", type=str, help="Input text to tokenize.")

        # Count Indic tokens arguments
        parser.add_argument("--count-indic-tokens", action="store_true", help="Count the number of Indic tokens using UTF-8 ranges.")
        parser.add_argument("--tokenizer-model", type=str, help="Name or path to the tokenizer model.")


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
        elif args.count_indic_tokens:
            args.count_language_tokens(args.tokenizer_model)
        # elif args.test_dataset:
        #     pass # feature to be added
        else:
            logger.error("Please provide either --train or --merge or --test option.")

if __name__ == "__main__":
    tokenizer = TokenizerCLI()
    tokenizer.run()