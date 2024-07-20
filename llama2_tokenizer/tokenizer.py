import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='./llama2_tokenizer/tokenizer.model')

vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]

for i, vocab in enumerate(vocabs[0:32000]):
    # if vocab == 'क' or vocab == '_क':
    print(i, vocabs[0:0+500])
    break

# # print(len(vocabs))
# with open('tokenizer.vocab', 'w') as f:
#     for vocab in vocabs:
#         f.write(vocab + '\n')