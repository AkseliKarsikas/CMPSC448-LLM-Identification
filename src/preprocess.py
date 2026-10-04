import re
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/llm_responses.csv"

label_map = {
    "GPT-5.6 Luna": 0,
    "Gemini 3.5 Flash-Lite": 1,
    "Qwen 3.8 27B": 2
}


def tokenize(text):
    text = str(text).lower()
    return re.findall(r"\w+|[^\w\s]", text)


def get_text(row, text_mode):

    if text_mode == "input":
        return str(row["LLM_Input"])

    elif text_mode == "output":
        return str(row["LLM_output"])

    elif text_mode == "combined":
        return str(row["LLM_Input"]) + " <SEP> " + str(row["LLM_output"])

    else:
        raise ValueError(
            "text_mode must be 'input', 'output', or 'combined'"
        )


def build_vocab(dataframe, text_mode):
    counter = Counter()

    for _, row in dataframe.iterrows():
        text = get_text(row, text_mode)
        counter.update(tokenize(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for token, count in counter.items():
        if count >= 2:
            vocab[token] = len(vocab)

    return vocab


def encode_text(text, vocab, max_length=300):
    tokens = tokenize(text)

    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]

    ids = ids[:max_length]

    if len(ids) < max_length:
        ids += [vocab["<PAD>"]] * (max_length - len(ids))

    return ids


def create_splits():

    df = pd.read_csv(INPUT_FILE)

    prompt_ids = df["prompt_id"].unique()

    train_ids, temp_ids = train_test_split(
        prompt_ids,
        test_size=0.30,
        random_state=42
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=42
    )

    train_df = df[df["prompt_id"].isin(train_ids)].copy()
    val_df = df[df["prompt_id"].isin(val_ids)].copy()
    test_df = df[df["prompt_id"].isin(test_ids)].copy()

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    # Verify no prompt leakage
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    print("Dataset split complete.")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))
    print("No prompt leakage detected.")


if __name__ == "__main__":

    create_splits()

    train_df = pd.read_csv("data/train.csv")

    print("\nVocabulary sizes:")

    for mode in ["input", "output", "combined"]:
        vocab = build_vocab(train_df, mode)
        print(f"{mode}: {len(vocab)}")
