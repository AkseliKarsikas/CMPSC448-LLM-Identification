import sys
sys.path.append("src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from preprocess import vocab
from dataset import LLMResponseDataset
from cnn import CNNClassifier
from rnn import RNNClassifier


# Choose model from command line
if len(sys.argv) < 2:
    print("Usage: python3 src/train.py cnn")
    print("   or: python3 src/train.py rnn")
    sys.exit()

model_type = sys.argv[1].lower()

# Reproducibility
torch.manual_seed(42)

# Load data
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

train_dataset = LLMResponseDataset(train_df)
val_dataset = LLMResponseDataset(val_df)

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

# Create selected model
if model_type == "cnn":
    model = CNNClassifier(vocab_size=len(vocab))
    save_path = "results/cnn_model.pt"

elif model_type == "rnn":
    model = RNNClassifier(vocab_size=len(vocab))
    save_path = "results/rnn_model.pt"

else:
    print("Model must be 'cnn' or 'rnn'")
    sys.exit()


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10


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

    accuracy = correct / total

    return total_loss / len(loader), accuracy


# Keep the model with the best validation accuracy
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

    # Save best model instead of automatically saving final epoch
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            save_path
        )


print(f"\n{model_type.upper()} training complete.")
print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Saved best model to {save_path}")
