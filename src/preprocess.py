import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/llm_responses.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Get unique prompt IDs
prompt_ids = df["prompt_id"].unique()

# 70% train, 30% temporary
train_ids, temp_ids = train_test_split(
    prompt_ids,
    test_size=0.30,
    random_state=42
)

# Split remaining 30% equally into validation and test
val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.50,
    random_state=42
)

# Keep all three LLM responses for a prompt in the same split
train_df = df[df["prompt_id"].isin(train_ids)].copy()
val_df = df[df["prompt_id"].isin(val_ids)].copy()
test_df = df[df["prompt_id"].isin(test_ids)].copy()

# Save splits
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Dataset split complete.")

print("\nTrain:")
print("Responses:", len(train_df))
print("Prompts:", train_df["prompt_id"].nunique())
print(train_df["LLM_name"].value_counts())

print("\nValidation:")
print("Responses:", len(val_df))
print("Prompts:", val_df["prompt_id"].nunique())
print(val_df["LLM_name"].value_counts())

print("\nTest:")
print("Responses:", len(test_df))
print("Prompts:", test_df["prompt_id"].nunique())
print(test_df["LLM_name"].value_counts())

# Verify that no prompt appears in multiple splits
train_set = set(train_ids)
val_set = set(val_ids)
test_set = set(test_ids)

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)

assert len(train_set | val_set | test_set) == 150

print("\nNo prompt leakage detected.")
import re
from collections import Counter

# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def tokenize(text):
    """
    Convert text to lowercase tokens.
    Keeps words, numbers, and punctuation as tokens.
    """
    text = str(text).lower()
    return re.findall(r"\w+|[^\w\s]", text)


# Build vocabulary ONLY from training outputs
counter = Counter()

for text in train_df["LLM_output"]:
    counter.update(tokenize(text))

# Special tokens
vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

# Add words that appear at least twice in training data
for token, count in counter.items():
    if count >= 2:
        vocab[token] = len(vocab)

print("\nVocabulary size:", len(vocab))


def encode_text(text, max_length=300):
    """
    Convert text into token IDs and pad/truncate
    every sequence to max_length.
    """
    tokens = tokenize(text)

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    # Truncate long responses
    ids = ids[:max_length]

    # Pad short responses
    if len(ids) < max_length:
        ids += [vocab["<PAD>"]] * (max_length - len(ids))

    return ids


# Model labels
label_map = {
    "GPT-5.6 Luna": 0,
    "Gemini 3.5 Flash-Lite": 1,
    "Qwen 3.8 27B": 2
}

print("\nLabel mapping:")
print(label_map)

# Test preprocessing on one response
example_text = train_df.iloc[0]["LLM_output"]
example_encoded = encode_text(example_text)

print("\nExample original text:")
print(example_text[:200])

print("\nFirst 20 encoded tokens:")
print(example_encoded[:20])

print("\nEncoded sequence length:")
print(len(example_encoded))
