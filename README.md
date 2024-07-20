# Extending Llama 3 Tokenizer to a New Language

## About Llama 3 Tokenizer

- Uses [tiktoken](https://github.com/openai/tiktoken) instead of SentencePiece
- 128,000 token vocabulary
- Supports multiple languages, including Hindi

## Building a New Tokenizer

1. **Generate Target Language Corpus**
   - Collect diverse, representative text in your target language

2. **Train a SentencePiece Tokenizer**
   - Use SentencePiece library on your new language data. We use sentencepiece considering it works well for most languages. (`./multilingual/hindi/train_tokenizer.py`)

3. **Merge with Base Tokenizer**
   - Load Llama 3 tokenizer and integrate new language tokens (`extend_tokenizer_llama3.py`)

### Project Structure
```
.
├── extend_tokenizer_llama3.py
└── multilingual/
    └── hindi/
        ├── data/
        ├── tokenizer_model/
        └── train_tokenizer.py
```

## Llama 3 vs Extended Tokenizer

Let's consider the hindi sentence:
```
मैं उसे एक ऐसा प्रस्ताव दूँगा जिसे वह अस्वीकार नहीं कर सकेगा।

Number of words = 13
```


Llama 3 tokenizer:

```
Target Tokens: ['म', 'ैं', ' उस', 'े', ' एक', ' ऐस', 'ा', ' प', '्रस', '्त', 'ाव', ' द', 'ूँ', 'ग', 'ा', ' ज', 'िस', 'े', ' वह', ' अस', '्व', 'ीक', 'ार', ' नह', 'ीं', ' कर', ' सक', 'ेग', 'ा।']

Target tokens produced: 30 Tokens
Fertility Score ≈ 28/13 = 2.15 tokens per word
```


Extended tokenizer:
```
Target Tokens: ['मैं', ' उसे', ' एक', ' ऐसा', ' प्रस्ताव', ' दू', 'ँ', 'गा', ' जिसे', ' वह', ' अस', '्वी', 'कार', ' नहीं', ' कर', ' सकेगा', '।']

Target tokens produced: 18 Tokens
Fertility ≈ 16/13 = 1.23 tokens per word
```

### Key Observations

1. Efficiency and Fertility: The new tokenizer is 40% more efficient and has a lower fertility score, producing fewer subword units per word on average.

2. Improved Segmentation: The new tokenizer demonstrates better understanding of Hindi morphology:
- It keeps more words intact: "मैं", "उसे", "एक", "ऐसा", "जिसे", "वह"
- It makes more linguistically sensible splits: "प्रस्ताव" instead of "प", "्रस", "्त", "ाव"

3. Diacritic Handling: The new tokenizer handles diacritical marks more effectively, often keeping them attached to the base character (e.g., "दू" instead of "द", "ूँ").

Overall this helps in improving training and inference since with fewer tokens, more meaningful content can fit within the model's context window.