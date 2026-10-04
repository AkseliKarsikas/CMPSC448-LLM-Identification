import sys
sys.path.append("src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from preprocess import build_vocab
from dataset import LLMResponseDataset
from cnn import CNNClassifier
from rnn import RNNClassifier


# Command format:
# python3 src/train.py cnn input
# python3 src/train.py cnn output
# python3 src/train.py cnn combined

if len(sys.argv) < 3:
    print("Usage: python3 src/train.py <cnn|rnn> <input|output|combined>")
    sys.exit()

model_type = sys.argv[1].lower()
text_mode = sys.argv[2].lower()

if model_type not in ["cnn", "rnn"]:
    print("Model must be 'cnn' or 'rnn'")
    sys.exit()

if text_mode not in ["input", "output", "combined"]:
    print("Text mode must be 'input', 'output', or 'combined'")
    sys.exit()


torch.manual_seed(42)

# Load data
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

# Build vocabulary ONLY from training data for this text mode
vocab = build_vocab(train_df, text_mode)

print(f"Model: {model_type.upper()}")
print(f"Text mode: {text_mode}")
print(f"Vocabulary size: {len(vocab)}")

# Datasets
train_dataset = LLMResponseDataset(
    train_df,
    vocab,
    text_mode=text_mode
)

val_dataset = LLMResponseDataset(
    val_df,
    vocab,
    text_mode=text_mode
)

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

# Model
if model_type == "cnn":
    model = CNNClassifier(vocab_size=len(vocab))
else:
    model = RNNClassifier(vocab_size=len(vocab))

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

save_path = f"results/{model_type}_{text_mode}_model.pt"


def evaluate(loader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for texts, labels in loader:

            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


best_val_accuracy = 0.0
best_epoch = 0


for epoch in range(epochs):

    model.train()
    total_loss = 0

    for texts, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    val_loss, val_accuracy = evaluate(val_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            save_path
        )


print(f"\n{model_type.upper()} ({text_mode}) training complete.")
print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Saved best model to {save_path}")
