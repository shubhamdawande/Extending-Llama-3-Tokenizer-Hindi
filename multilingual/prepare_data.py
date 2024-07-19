import fire
import os
from datasets import load_dataset

def main(split="validation", lang="hi", docs_to_sample=10_000, save_path="data"):

    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, f"{lang}.txt"), "w") as f:

        # Dataset 1
        DATASET = "rahular/varta"
        dataset = load_dataset(DATASET, split=split, streaming=True)
        count = 0
        for idx, d in enumerate(dataset):
            if idx % 10_000 == 0:
                print(f"Searched {idx} documents for {lang} documents. Found {count} documents.")
            if count >= docs_to_sample:
                break
            if d["langCode"] == lang:
                f.write(d["headline"] + "\n" + d["text"] + "\n")
                count += 1

        # Dataset 2
        # DATASET = "findnitai/english-to-hinglish"
        # dataset = load_dataset(DATASET, split="train", streaming=True)
        # for idx, d in enumerate(dataset):
        #     f.write(d["translation"]["hi_ng"] + "\n")
        #     if idx % 10_000 == 0:
        #         print(f"Searched {idx} documents for {lang} documents. Found {idx} documents.")
        #     if idx >= 100000:
        #         print(f"Done {idx} documents for {lang} documents. Exiting.")
        #         break

if __name__ == "__main__":
    fire.Fire(main)