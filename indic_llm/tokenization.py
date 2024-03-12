import argparse
import os
import time

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import AutoTokenizer


class SentencePieceTrainer:
    def __init__(self):
        self.vocab_size = 20000
        self.character_coverage = 1.0
        self.model_type = "unigram"
        # This is NOT perfect but gives high level idea about the composition of the tokenizer
        self.language_unicode_ranges = {
            'European': ('\u0000', '\u007F'),
            'Chinese (Basic)': ('\u4E00', '\u9FFF'),
            'Tamil': ('\u0B80', '\u0BFF'),
            'Hindi': ('\u0900', '\u097F'),
            'Telugu': ('\u0C00', '\u0C7F'),
            'Malayalam': ('\u0D00', '\u0D7F'),
            'Kannada': ('\u0C80', '\u0CFF'),
            'Marathi': ('\u0900', '\u097F'),  # Marathi shares the range with Hindi
            'Bengali': ('\u0980', '\u09FF'),
        }
        
        self.indic_language_unicode_ranges = {
            'Devanagari': ('\u0900', '\u097F'),
            'Bengali': ('\u0980', '\u09FF'),
            'Gurmukhi': ('\u0A00', '\u0A7F'),
            'Gujarati': ('\u0A80', '\u0AFF'),
            'Oriya': ('\u0B00', '\u0B7F'),
            'Tamil': ('\u0B80', '\u0BFF'),
            'Telugu': ('\u0C00', '\u0C7F'),
            'Kannada': ('\u0C80', '\u0CFF'),
            'Malayalam': ('\u0D00', '\u0D7F'),
            'Marathi': ('\u0900', '\u097F'),
        }


    def train_tokenizer(self, input_file, model_prefix , output_dir):
        start_time = time.time()
        
        output_model_path = os.path.join(output_dir, f"{self.model_prefix}.model")

        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            model_type=self.model_type,
            train_extremely_large_corpus=True
        )

        os.rename(
            f"{model_prefix}.vocab",
            os.path.join(output_dir, f"{model_prefix}.vocab"),
        )
        os.rename(
            f"{model_prefix}.model",
            os.path.join(output_dir, f"{model_prefix}.model"),
        )
        
        end_time = time.time()
        total_time_seconds = end_time - start_time
        total_time_minutes = total_time_seconds / 60.0

        print(f"Total time taken to train the model: {total_time_minutes:.2f} minutes")


        return output_model_path
    
    def merge_tokenizer(self, base_tokenizer_dir, extended_tokenizer_dir):
        
        # load
        # logger.
        base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_dir)
        extended_tokenizer = spm.SentencePieceProcessor()
        extended_tokenizer.Load(extended_tokenizer_dir)
        
        base_spm = sp_pb2_model.ModelProto()
        base_spm.ParseFromString(base_tokenizer.sp_model.serialized_model_proto())
        extended_spm = sp_pb2_model.ModelProto()
        extended_spm.ParseFromString(extended_tokenizer.serialized_model_proto())
        
        # print number of tokens
        print(len(base_tokenizer), len(extended_tokenizer))
        print(base_tokenizer.all_special_tokens)
        print(base_tokenizer.all_special_ids)
        print(base_tokenizer.special_tokens_map)
        
        ## Add kannada tokens to LLaMA tokenizer
        llama_spm_tokens_set = set(p.piece for p in base_spm.pieces)
        print(len(llama_spm_tokens_set))
        print(f"Before:{len(llama_spm_tokens_set)}")
        for p in extended_spm.pieces:
            piece = p.piece
            if piece not in llama_spm_tokens_set:
                new_p = sp_pb2_model.ModelProto().SentencePiece()
                new_p.piece = piece
                new_p.score = 0
                base_spm.pieces.append(new_p)
        print(f"New model pieces: {len(base_spm.pieces)}")
        
        ## Save
        output_sp_dir = "merged_tokenizer_sp"
        output_hf_dir = "merged_tokenizer_hf"  # the path to save kannada-LLaMA tokenizer
        os.makedirs(output_sp_dir, exist_ok=True)
        with open(output_sp_dir + f"/merged_tokenizer.model", "wb") as f:
            f.write(base_spm.SerializeToString())
        tokenizer = AutoTokenizer(vocab_file=output_sp_dir + "/merged_tokenizer.model")
        
        tokenizer.save_pretrained(output_hf_dir)
        print(f"Extended tokenizer has been saved to {output_hf_dir}")
        
        
    
    # source code for "count language tokens" taken from https://github.com/abhinand5/tamil-llama/blob/main/scripts/utils/count_indic_tokens.py
    def count_language_tokens(self, tokenizer_model):
        
        def is_language(token, ranges):
            return any(ranges[0] <= char <= ranges[1] for char in token)

        def count_language_tokens(tokenizer, ranges):
            return sum(is_language(token, ranges) for token in tokenizer.get_vocab().keys())
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        total_vocab_size = len(tokenizer.get_vocab())

        print("\n---Note: These calculations are approximate!---\n")
        print(f"Total vocabulary size of '{tokenizer_model}': {total_vocab_size}\n")
        print(f"{'Language':<20} | {'Tokens':>10} | {'Percentage':>10}")
        print("-" * 50)

        for language, ranges in self.indic_language_unicode_ranges.items():
            count = count_language_tokens(tokenizer, ranges)
            percentage = (count / total_vocab_size) * 100
            print(f"{language:<20} | {count:>10} | {percentage:>9.2f}%")


    def test_tokenizer(self, tokenizer_model, text):
        sp = spm.SentencePieceProcessor(model_file=tokenizer_model)
        encoded_text = sp.encode(text, out_type=str)
        
        print(encoded_text)    

        

    
    
    