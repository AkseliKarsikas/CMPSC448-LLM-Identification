import pandas as pd
import re


df = pd.read_csv("data/llm_responses.csv")


def get_features(text):
    text = str(text)

    words = re.findall(r"\b\w+\b", text.lower())
    unique_words = set(words)

    word_count = len(words)
    char_count = len(text)

    if word_count > 0:
        lexical_diversity = len(unique_words) / word_count
    else:
        lexical_diversity = 0

    return pd.Series({
        "word_count": word_count,
        "char_count": char_count,
        "lexical_diversity": lexical_diversity,
        "bullet_count": text.count("- "),
        "bold_count": text.count("**") // 2,
        "heading_count": len(re.findall(r"(?m)^#+\s", text))
    })


features = df["LLM_output"].apply(get_features)

analysis_df = pd.concat(
    [df[["prompt_id", "domain", "LLM_name"]], features],
    axis=1
)

print("\nAVERAGE CHARACTERISTICS BY MODEL")
print("=" * 60)

summary = analysis_df.groupby("LLM_name")[
    [
        "word_count",
        "char_count",
        "lexical_diversity",
        "bullet_count",
        "bold_count",
        "heading_count"
    ]
].mean()

print(summary.round(3))


print("\nMEDIAN WORD COUNT")
print(
    analysis_df.groupby("LLM_name")["word_count"]
    .median()
)


print("\nWORD COUNT STANDARD DEVIATION")
print(
    analysis_df.groupby("LLM_name")["word_count"]
    .std()
    .round(2)
)


# Save detailed features
analysis_df.to_csv(
    "results/response_characteristics.csv",
    index=False
)

# Save summary table
summary.to_csv(
    "results/characteristics_summary.csv"
)

print("\nSaved:")
print("results/response_characteristics.csv")
print("results/characteristics_summary.csv")
import matplotlib.pyplot as plt

# -----------------------------
# Plot 1: Average word count
# -----------------------------

word_means = analysis_df.groupby("LLM_name")["word_count"].mean()

plt.figure(figsize=(8, 5))
word_means.plot(kind="bar")

plt.title("Average Response Length by LLM")
plt.ylabel("Average Word Count")
plt.xlabel("LLM")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "results/average_word_count.png",
    dpi=300
)

plt.close()


# -----------------------------
# Plot 2: Formatting behavior
# -----------------------------

format_means = analysis_df.groupby("LLM_name")[
    ["bullet_count", "bold_count", "heading_count"]
].mean()

ax = format_means.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title("Average Formatting Features by LLM")
ax.set_ylabel("Average Count per Response")
ax.set_xlabel("LLM")

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "results/formatting_features.png",
    dpi=300
)

plt.close()

print("Saved plots:")
print("results/average_word_count.png")
print("results/formatting_features.png")
