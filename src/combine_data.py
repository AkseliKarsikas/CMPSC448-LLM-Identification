import pandas as pd

files = [
    "data/gemini_responses.csv",
    "data/qwen_responses.csv",
    "data/openai_responses.csv"
]

dfs = [pd.read_csv(file) for file in files]
df = pd.concat(dfs, ignore_index=True)

# Sort so responses to the same prompt are together
df = df.sort_values(["prompt_id", "LLM_name"]).reset_index(drop=True)

# Validation
print("Total responses:", len(df))
print("Unique prompts:", df["prompt_id"].nunique())

print("\nResponses per model:")
print(df["LLM_name"].value_counts())

print("\nResponses per domain:")
print(df["domain"].value_counts())

print("\nMissing values:")
print(df.isna().sum())

print(
    "\nDuplicate model/prompt pairs:",
    df.duplicated(subset=["prompt_id", "LLM_name"]).sum()
)

# Save final combined dataset
df.to_csv("data/llm_responses.csv", index=False)

print("\nSaved data/llm_responses.csv")
