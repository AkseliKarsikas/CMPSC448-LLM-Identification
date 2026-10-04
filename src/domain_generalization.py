import sys
sys.path.append("src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score

from preprocess import build_vocab
from dataset import LLMResponseDataset
from cnn import CNNClassifier


torch.manual_seed(42)

df = pd.read_csv("data/llm_responses.csv")

domains = [
    "gymnastics",
    "random",
    "oulu",
    "finland",
    "academic"
]

results = []


for test_domain in domains:

    print("\n" + "=" * 50)
    print(f"HELD-OUT DOMAIN: {test_domain}")
    print("=" * 50)

    # Train on four domains
    train_df = df[df["domain"] != test_domain].copy()

    # Test only on unseen domain
    test_df = df[df["domain"] == test_domain].copy()

    vocab = build_vocab(
        train_df,
        "output"
    )

    train_dataset = LLMResponseDataset(
        train_df,
        vocab,
        text_mode="output"
    )

    test_dataset = LLMResponseDataset(
        test_df,
        vocab,
        text_mode="output"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    model = CNNClassifier(
        vocab_size=len(vocab)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Same number of epochs as main experiment
    for epoch in range(10):

        model.train()

        for texts, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(texts)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    # Evaluate on unseen domain
    model.eval()

    predictions = []
    labels_all = []

    with torch.no_grad():

        for texts, labels in test_loader:

            outputs = model(texts)

            predicted = outputs.argmax(dim=1)

            predictions.extend(predicted.tolist())
            labels_all.extend(labels.tolist())

    accuracy = accuracy_score(
        labels_all,
        predictions
    )

    print(f"Test responses: {len(test_df)}")
    print(f"Accuracy: {accuracy:.4f}")

    results.append({
        "held_out_domain": test_domain,
        "accuracy": accuracy
    })


results_df = pd.DataFrame(results)

print("\nFINAL CROSS-DOMAIN RESULTS")
print("=" * 50)

print(results_df)

print(
    "\nAverage accuracy:",
    round(results_df["accuracy"].mean(), 4)
)

results_df.to_csv(
    "results/domain_generalization.csv",
    index=False
)

print("\nSaved results/domain_generalization.csv")

