import torch
import torch.nn as nn


class RNNClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_classes=3,
        dropout=0.5
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):
        x = self.embedding(x)

        _, (hidden, _) = self.lstm(x)

        x = hidden[-1]

        x = self.dropout(x)

        return self.fc(x)
