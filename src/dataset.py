import sys
sys.path.append("src")

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from preprocess import encode_text, label_map, get_text, build_vocab


class LLMResponseDataset(Dataset):

    def __init__(self, dataframe, vocab, text_mode="output"):
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.text_mode = text_mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        text = get_text(row, self.text_mode)

        encoded_text = encode_text(
            text,
            self.vocab
        )

        label = label_map[row["LLM_name"]]

        return (
            torch.tensor(encoded_text, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


if __name__ == "__main__":

    train_df = pd.read_csv("data/train.csv")

    for mode in ["input", "output", "combined"]:

        vocab = build_vocab(
            train_df,
            mode
        )

        dataset = LLMResponseDataset(
            train_df,
            vocab,
            text_mode=mode
        )

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True
        )

        texts, labels = next(iter(loader))

        print(f"\nMode: {mode}")
        print("Vocabulary size:", len(vocab))
        print("Text shape:", texts.shape)
        print("Label shape:", labels.shape)
