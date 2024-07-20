from tokenizers import Tokenizer, AddedToken
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
import sentencepiece as spm
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import PreTokenizer
import regex as re


def get_tokenizer(version=0):
    llama3_model_id = "meta-llama/Meta-Llama-3-8B"
    
    # tokenizer 2: Huggingface AutoTokenizer
    if version == 0:
        tokenizer = AutoTokenizer.from_pretrained(llama3_model_id)

    # tokenizer 1: Fast Tokenizer
    elif version == 1:
        tokenizer = Tokenizer.from_pretrained(llama3_model_id)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    # tokenizer 3: Raw BPE
    elif version == 2: 
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()

    return tokenizer
    
def tokenize_text(text, tokenizer):
    tokens_encoded = tokenizer.encode(text)
    tokens_decoded = []
    for token in tokens_encoded:
        tokens_decoded.append(tokenizer.decode(token))
        
    print('\n--------------\nNumber of tokens:', len(tokens_decoded))
    print(tokens_decoded)
    return tokens_decoded

def get_token_id(tokenizer, token):
    return tokenizer.get_vocab().get(token)

def contains_english(text):
    return bool(re.search('[a-zA-Z]', text))
        
texts = [
    "I will make him an offer he can't refuse.",
    "मैं उसे एक ऐसा प्रस्ताव दूँगा जिसे वह अस्वीकार नहीं कर सकेगा।",
    "कर्मणयेवाधिकारस्ते मा फलेषु कदाचन।",
    "समस्याएं हमारे जीवन मे बिना किसी वजह के नहीं आती। उनका आना इशारा है की हमें अपने जीवन मे कुछ बदलना है।"
]

# Load base tokenizer
tokenizer = get_tokenizer()

# Tokenize text
for text in texts:
    tokenize_text(text, tokenizer)

# Load 2nd tokenizer
hi_spm = spm.SentencePieceProcessor()
hi_spm.Load('./multilingual/hindi/hi_tokenizer/tokenizer.model')
# vocab = [str(hi_spm.decode(i)) for i in range(len(hi_spm))]

hindi_vocab = []
for i in range(hi_spm.GetPieceSize()):
    token = hi_spm.IdToPiece(i)

    if token.startswith('▁'):
        token = token.replace("▁", " ")
    
    if not contains_english(token):
        hindi_vocab.append(token)

# Add new token and Tokenize again
new_tokens = []
base_vocab = tokenizer.get_vocab()
for token in hindi_vocab:
    if token and token not in base_vocab and token.strip():
        new_tokens.append(token)
tokenizer.add_tokens(new_tokens)

for text in texts:
    tokenize_text(text, tokenizer)