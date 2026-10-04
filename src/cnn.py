import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        num_filters=100,
        num_classes=3,
        dropout=0.5
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # Look for patterns of 3, 4, and 5 tokens
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=3),
            nn.Conv1d(embedding_dim, num_filters, kernel_size=4),
            nn.Conv1d(embedding_dim, num_filters, kernel_size=5)
        ])

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            num_filters * 3,
            num_classes
        )

    def forward(self, x):

        # [batch, sequence] -> [batch, sequence, embedding]
        x = self.embedding(x)

        # Conv1d expects embedding dimension before sequence
        x = x.permute(0, 2, 1)

        pooled_outputs = []

        for conv in self.convs:
            # Convolution + ReLU
            conv_out = torch.relu(conv(x))

            # Keep strongest feature from each filter
            pooled = torch.max(conv_out, dim=2).values

            pooled_outputs.append(pooled)

        # Combine results from 3, 4, and 5 token filters
        x = torch.cat(pooled_outputs, dim=1)

        x = self.dropout(x)

        return self.fc(x)
