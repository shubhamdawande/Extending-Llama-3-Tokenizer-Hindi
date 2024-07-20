from tokenizers import Tokenizer, AddedToken, pre_tokenizers
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_tokenizer(version=0):
    llama3_model_id = "meta-llama/Meta-Llama-3-8B"
    
    # tokenizer 2: AutoTokenizer
    if version == 0:
        tokenizer = AutoTokenizer.from_pretrained(llama3_model_id)

    # tokenizer 1: Tokenizer lib
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
        
    print('Number of tokens:', len(tokens_decoded))
    print(tokens_decoded)
    return tokens_decoded

text = "मैं उसे एक ऐसा प्रस्ताव दूँगा जिसे वह अस्वीकार नहीं कर सकेगा।"
text = "I will make him an offer he can't refuse"
tokenizer = get_tokenizer()

# Tokenize text
tokenize_text(text, tokenizer)

# Add new token and Tokenize again 
new_token = "स्वीकार"
tokenizer.add_tokens([AddedToken(new_token, normalization=False, special=False)])
tokenize_text(text, tokenizer)