import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='./multilingual/hi_tokenizer/tokenizer.model')

vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]

for i, vocab in enumerate(vocabs):
    # if vocab == 'क' or vocab == '_क':
    print(i, vocabs[500:1000])
    break

# # print(len(vocabs))
# with open('tokenizer.vocab', 'w') as f:
#     for vocab in vocabs:
#         f.write(vocab + '\n')