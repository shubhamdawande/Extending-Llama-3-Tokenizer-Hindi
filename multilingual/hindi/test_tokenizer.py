from transformers import AutoTokenizer, LlamaTokenizer
from tokenizers import Tokenizer

texts = [
    # "I'm gonna make him an offer he can't refuse.",
    "मैं उसे एक ऐसा प्रस्ताव दूँगा जिसे वह अस्वीकार नहीं कर सकेगा।",
    # "Я сделаю ему предложение, от которого он не сможет отказаться.",
    # "હું તેને એવી ઓફર કરીશ કે તે ના પાડી શકે નહીં.",
]

# Llama2 tokenizer
# llama2_model_id = "meta-llama/Llama-2-7b-chat-hf"
# llama2_tokenizer = AutoTokenizer.from_pretrained(llama2_model_id, force_download=False)
# for text in texts:
#     llama2_res = llama2_tokenizer.tokenize(text)
#     print("llama2 response:", llama2_res)

llama3_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llama3_tokenizer = AutoTokenizer.from_pretrained(llama3_model_id, force_download=False)
# for text in texts:
#     llama3_res = llama3_tokenizer.tokenize(text)
#     for el in llama3_res:
#         print(el)
#         for char in el:
#             print(ord(char), hex(ord(char)))
#         print('-------')
#     print("llama3 response:", llama3_res)
    
# print(ord('क'), hex(ord('क')))

vocabs = llama3_tokenizer.get_vocab()
for i, vocab in enumerate(vocabs):
    if vocab == 'क' or vocab == '_क':
        print(i, vocabs[i-500:i+500])
        break

# our_tokenizer = Tokenizer.from_file('./extended_tokenizer/tokenizer.json')
# our_res = our_tokenizer.encode(text)
# print("our_res", our_res.tokens)