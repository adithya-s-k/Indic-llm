import argparse
import os
import time

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import LlamaTokenizer



class SentencePieceTrainer:
    def __init__(self):
        self.corpus_dir = "./corpus"
        self.output_dir = "./models"
        self.model_prefix = "kannada_sp"
        self.vocab_size = 20000
        self.character_coverage = 1.0
        self.model_type = "unigram"

    def train_tokenizer(self, input_file):
        start_time = time.time()
        
        output_model_path = os.path.join(self.output_dir, f"{self.model_prefix}.model")

        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            model_type=self.model_type,
            train_extremely_large_corpus=True
        )

        os.rename(
            f"{self.model_prefix}.vocab",
            os.path.join(self.output_dir, f"{self.model_prefix}.vocab"),
        )
        os.rename(
            f"{self.model_prefix}.model",
            os.path.join(self.output_dir, f"{self.model_prefix}.model"),
        )
        
        end_time = time.time()
        total_time_seconds = end_time - start_time
        total_time_minutes = total_time_seconds / 60.0

        print(f"Total time taken to train the model: {total_time_minutes:.2f} minutes")


        return output_model_path
    
    def merge_tokenizer():
        pass
    
    def test_tokenizer():
        pass

