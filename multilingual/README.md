# Training a SentencePiece Tokenizer for a New Language

This guide outlines the process of training a SentencePiece tokenizer for a new language, specifically Hindi. The code is adapted from the [Sarvam AI Multilingual Guide](https://github.com/meta-llama/llama-recipes/tree/main/recipes/use_cases/multilingual).

## Quick Start

You can build the tokenizer by executing the `train.sh` script.

## Step-by-Step Instructions

Alternatively, you can run the following commands manually:

1. Generate Hindi language data using the rahular/varta dataset:

```bash
python prepare_data.py --split=validation --lang=hi --docs_to_sample=10000 --save_path=./data
```

2. Train the SentencePiece tokenizer:

```bash
python train_tokenizer.py --data_file=./data/hi.txt --save_path=./hi_tokenizer --vocab_size=16000
```

Note

- The prepare_data.py script samples 10,000 documents from the validation split of the Hindi subset of the rahular/varta dataset.
- The resulting tokenizer will have a vocabulary size of 16,000 tokens.
