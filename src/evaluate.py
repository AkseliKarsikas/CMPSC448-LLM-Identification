import sys
sys.path.append("src")

import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import vocab
from dataset import LLMResponseDataset
from cnn import CNNClassifier
from rnn import RNNClassifier


if len(sys.argv) < 2:
    print("Usage: python3 src/evaluate.py cnn")
    print("   or: python3 src/evaluate.py rnn")
    sys.exit()

model_type = sys.argv[1].lower()

# Load untouched test set
test_df = pd.read_csv("data/test.csv")
test_dataset = LLMResponseDataset(test_df)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Load selected model
if model_type == "cnn":
    model = CNNClassifier(vocab_size=len(vocab))
    model.load_state_dict(
        torch.load("results/cnn_model.pt", weights_only=True)
    )

elif model_type == "rnn":
    model = RNNClassifier(vocab_size=len(vocab))
    model.load_state_dict(
        torch.load("results/rnn_model.pt", weights_only=True)
    )

else:
    print("Model must be 'cnn' or 'rnn'")
    sys.exit()


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

print(f"\n{model_type.upper()} TEST RESULTS")
print("=" * 40)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "GPT-5.6 Luna",
            "Gemini 3.5 Flash-Lite",
            "Qwen 3.8 27B"
        ],
        digits=4
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)
