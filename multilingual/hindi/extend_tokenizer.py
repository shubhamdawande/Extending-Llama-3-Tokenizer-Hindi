"""
Code borrowed from https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizer/merge_tokenizers.py
"""

import os
import fire
import re
from transformers import LlamaTokenizer, AutoTokenizer

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from tokenizers import Tokenizer

llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# llama_model_id = "meta-llama/Llama-2-7b-chat-hf"

def main(new_tokenizer_path, extended_tokenizer_save_path):
    original_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    new_tokenizer_spm = sp_pb2_model.ModelProto()
    new_tokenizer_spm.ParseFromString(open(os.path.join(new_tokenizer_path, "tokenizer.model"), "rb").read())
            
    original_tokenizer_tokenset = set(original_tokenizer.get_vocab().keys())
    print(f"Number of tokens before merge: {len(original_tokenizer_tokenset)}")
    print(f"Number of tokens in new tokenizer: {len(new_tokenizer_spm.pieces)}")

    new_pieces = []
    for p in new_tokenizer_spm.pieces:
        piece = p.piece
        if piece not in original_tokenizer_tokenset:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_pieces.append(piece)
    original_tokenizer.add_tokens(new_pieces)
    print(f"Number of tokens after merge: {len(set(original_tokenizer.get_vocab().keys()))}")

    os.makedirs(extended_tokenizer_save_path, exist_ok=True)
    original_tokenizer.save_pretrained(extended_tokenizer_save_path)
    print(f"Tokenizer saved to {extended_tokenizer_save_path}")

    # Verify that the extended tokenizer's English vocab matches with that of the original Llama tokenizer
    tok1 = AutoTokenizer.from_pretrained(llama_model_id)
    tok2 = Tokenizer.from_file(os.path.join(extended_tokenizer_save_path, "tokenizer.json"))
    for i in range(len(tok1)):
        assert tok1.convert_ids_to_tokens(i) == tok2.id_to_token(i), f"Token mismatch at index {i}."


if __name__ == "__main__":
    fire.Fire(main)