# Generate hindi langauage data using rahular/varta dataset
python prepare_data.py --split=validation --lang=hi --docs_to_sample=10000 --save_path=./data

# Train sentencepiece tokenizer
python train_tokenizer.py --data_file=./data/hi.txt --save_path=./hi_tokenizer --vocab_size=16000