import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from preprocess import encode_text, label_map


class LLMResponseDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Convert LLM response into 300 token IDs
        encoded_text = encode_text(row["LLM_output"])

        # Convert model name into class number
        label = label_map[row["LLM_name"]]

        return (
            torch.tensor(encoded_text, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


if __name__ == "__main__":

    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    train_dataset = LLMResponseDataset(train_df)
    val_dataset = LLMResponseDataset(val_df)
    test_dataset = LLMResponseDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    print("\nDataset sizes:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    # Look at one batch
    texts, labels = next(iter(train_loader))

    print("\nBatch text shape:", texts.shape)
    print("Batch label shape:", labels.shape)

    print("\nFirst 10 labels:")
    print(labels[:10])

    print("\nFirst example, first 20 token IDs:")
    print(texts[0][:20])
