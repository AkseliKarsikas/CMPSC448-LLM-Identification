import sys
sys.path.append("src")

import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import build_vocab
from dataset import LLMResponseDataset
from cnn import CNNClassifier
from rnn import RNNClassifier


if len(sys.argv) < 3:
    print("Usage: python3 src/evaluate.py <cnn|rnn> <input|output|combined>")
    sys.exit()

model_type = sys.argv[1].lower()
text_mode = sys.argv[2].lower()

if model_type not in ["cnn", "rnn"]:
    print("Model must be 'cnn' or 'rnn'")
    sys.exit()

if text_mode not in ["input", "output", "combined"]:
    print("Text mode must be 'input', 'output', or 'combined'")
    sys.exit()


# Vocabulary must be built from training data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

vocab = build_vocab(
    train_df,
    text_mode
)

test_dataset = LLMResponseDataset(
    test_df,
    vocab,
    text_mode=text_mode
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


if model_type == "cnn":
    model = CNNClassifier(vocab_size=len(vocab))
else:
    model = RNNClassifier(vocab_size=len(vocab))


model_path = f"results/{model_type}_{text_mode}_model.pt"

model.load_state_dict(
    torch.load(model_path, weights_only=True)
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for texts, labels in test_loader:

        outputs = model(texts)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

names = [
    "GPT-5.6 Luna",
    "Gemini 3.5 Flash-Lite",
    "Qwen 3.8 27B"
]

print(f"\n{model_type.upper()} - {text_mode.upper()} TEST RESULTS")
print("=" * 45)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=names,
        digits=4,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)
