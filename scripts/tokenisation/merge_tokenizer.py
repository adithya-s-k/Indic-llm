# Based on https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizer/merge_tokenizers.py
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--llama_tokenizer_dir", default=None, type=str, required=True)
parser.add_argument("--kannada_sp_model_file", default="./kannada_sp.model", type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
kannada_sp_model_file = args.kannada_sp_model_file

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
kannada_sp_model = spm.SentencePieceProcessor()
kannada_sp_model.Load(kannada_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
kannada_spm = sp_pb2_model.ModelProto()
kannada_spm.ParseFromString(kannada_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer), len(kannada_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

## Add kannada tokens to LLaMA tokenizer
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in kannada_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = "merged_tokenizer_sp"
output_hf_dir = "merged_tokenizer_hf"  # the path to save kannada-LLaMA tokenizer
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + "/kannada_llama.model", "wb") as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + "/kannada_llama.model")

tokenizer.save_pretrained(output_hf_dir)
print(f"kannada-LLaMA tokenizer has been saved to {output_hf_dir}")


# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
kannada_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text = """
ರಸ್ತುತ ದಿನಗಳಲ್ಲಿ ಸ್ಮಾರ್ಟ್‌ಫೋನ್‌ಗಳು ಮಾತ್ರವಲ್ಲ ಸ್ಮಾರ್ಟ್‌ವಾಚ್‌ಗಳು ಕೂಡ ಸಿಕ್ಕಾಪಟ್ಟೆ ಸೌಂಡ್‌ ಮಾಡುತ್ತಿವೆ. ಫಿಟ್ನೆಸ್‌ ಆಧಾರಿತ ಸ್ಮಾರ್ಟ್‌ವಾಚ್‌ಗಳಿಗೆ ಟೆಕ್‌ ವಲಯದಲ್ಲಿ ಭಾರಿ ಬೇಡಿಕೆ ಇದೆ. ಇದೇ ಕಾರಣಕ್ಕೆ ಹಲವು ಕಂಪೆನಿಗಳು ವೈವಿಧ್ಯಮಯ ಸ್ಮಾರ್ಟ್‌ವಾಚ್‌ಗಳನ್ನು ಪರಿಚಯಿಸಿವೆ. ಇವುಗಳಲ್ಲಿ ಅಮಾಜ್‌ಫಿಟ್‌ ಕೂಡ ಬಳಕೆದಾರರ ನೆಚ್ಚಿನ ಬ್ರಾಂಡ್‌ ಎನಿಸಿಕೊಂಡಿದೆ. ಸದ್ಯ ಅಮಾಜ್‌ಫಿಟ್‌ ತನ್ನ ವಿಭಿನ್ನ ಸ್ಮಾರ್ಟ್‌ವಾಚ್‌ಗಳಿಂದ ಗುರುತಿಸಿಕೊಂಡಿದೆ. ಇದೀಗ ತನ್ನ ಸ್ಮಾರ್ಟ್‌ವಾಚ್‌ಗಳ ಮೇಲೆ ಭರ್ಜರಿ ಡಿಸ್ಕೌಂಟ್‌ ಅನ್ನು ನೀಡುತ್ತಿದೆ.
ಹೌದು, ಅಮಾಜ್‌ಫಿಟ್‌ ಕಂಪೆನಿ ತನ್ನ ಕೆಲವು ಸ್ಮಾರ್ಟ್‌ವಾಚ್‌ಗಳ ಮೇಲೆ ಬಿಗ್‌ ಡಿಸ್ಕೌಂಟ್‌ ನೀಡುತ್ತಿದೆ. ನೀವು ಕೂಡ ಅಮಾಜ್‌ಫಿಟ್‌ ಸ್ಮಾರ್ಟ್‌ವಾಚ್‌ಗಳನ್ನು ರಿಯಾಯಿತಿ ದರದಲ್ಲಿ ಖರೀದಿಸಲು ಬಯಸಿದರೆ ಇದು ಉತ್ತಮ ಸಮಯವಾಗಿದೆ. ಅದರಲ್ಲೂ ಜನಪ್ರಿಯ ಸ್ಮಾರ್ಟ್‌ವಾಚ್‌ಗಳಾದ ಅಮಾಜ್‌ಫಿಟ್ ಜಿಟಿಎಸ್ 2 ಮಿನಿ, ಬಿಪ್ ಯು ಪ್ರೊ ಮತ್ತು ಬಿಪ್‌ಯು ವಾಚ್‌ಗಳಿಗೆ ರಿಯಾಯಿತಿ ಘೋಷಿಸಿದೆ. ಹಾಗಾದ್ರೆ ಅಮಾಜ್‌ಫಿಟ್‌ ವಾಚ್‌ಗಳಿಗೆ ಯಾವೆಲ್ಲಾ ರಿಯಾಯಿತಿ ದೊರೆಯುತ್ತಿದೆ ಅನ್ನೊದನ್ನ ಈ ಲೇಖನದಲ್ಲಿ ತಿಳಿಸಿಕೊಡ್ತೀವಿ ಓದಿರಿ.
ಆನ್‌ಲೈನ್‌ ಶಾಪಿಂಗ್‌ ದೈತ್ಯ ಅಮೆಜಾನ್‌ ಸೈಟ್‌ನಲ್ಲಿ ಅಮಾಜ್‌ಫಿಟ್ ಬ್ರಾಂಡ್ ಡೇ ಸೇಲ್ ಲೈವ್‌ ಆಗಿದೆ. ಅಲ್ಲದೆ ಅಮಾಜ್ ಫಿಟ್ ನ ಅಧಿಕೃತ ವೆಬ್ ಸೈಟ್ ನಲ್ಲಿ ಕೂಡ ರಿಯಾಯಿತಿ ಸೇಲ್‌ ಲೈವ್ ಆಗಿದೆ. ಈ ಸೇಲ್‌ ಇದೇ ಸೆಪ್ಟೆಂಬರ್ 12 ರವರೆಗೆ ನಡೆಯಲಿದೆ. ಇದರಲ್ಲಿ ಅಮಾಜ್‌ಫಿಟ್ ಜಿಟಿಎಸ್ 2 ಮಿನಿ ಅಮೆಜಾನ್‌ನಲ್ಲಿ ಮಾತ್ರ ಲಭ್ಯವಿದೆ. ಆದರೆ ಬಿಪ್‌ ಯು ಮತ್ತು ಬಿಪ್‌ ಯು ಪ್ರೊ ಫ್ಲಿಪ್‌ಕಾರ್ಟ್‌ನಲ್ಲಿ ರಿಯಾಯಿತಿ ದರದಲ್ಲಿ ಲಭ್ಯವಾಗಲಿದೆ.
"""
print("Test text:\n", text)
llama_tokenized = llama_tokenizer.tokenize(text)
kannada_llama_tokenized = kannada_llama_tokenizer.tokenize(text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenized}")
print(f"LLaMA tokenizer n_tokens={len(llama_tokenized)}")
print(f"Tokenized by kannada-LLaMA tokenizer:{kannada_llama_tokenized}")
print(f"kannada LLaMA tokenizer n_tokens={len(kannada_llama_tokenized)}")
