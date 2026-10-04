import csv
import os
import time
from google import genai

INPUT_FILE = "data/prompts.csv"
OUTPUT_FILE = "data/gemini_responses.csv"

client = genai.Client()

# Find prompts that have already been completed
completed_ids = set()

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed_ids.add(row["prompt_id"])

# Read all prompts
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    prompts = list(csv.DictReader(f))

print(f"Found {len(prompts)} prompts.")
print(f"Already completed: {len(completed_ids)}")

# Create output file with header if it doesn't exist
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_id",
            "domain",
            "task_type",
            "LLM_name",
            "LLM_Input",
            "LLM_output"
        ])

# Generate responses
for row in prompts:

    if row["prompt_id"] in completed_ids:
        continue

    print(
        f'Processing prompt {row["prompt_id"]}/{len(prompts)} '
        f'({row["domain"]})...'
    )

    try:
        interaction = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=row["prompt"]
        )

        response_text = interaction.output_text

        # Save immediately
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                row["prompt_id"],
                row["domain"],
                row["task_type"],
                "Gemini 3.5 Flash-Lite",
                row["prompt"],
                response_text
            ])

        print("Saved.")

        # Small delay to be nicer to the API
        time.sleep(2)

    except Exception as e:
        print(f"Error on prompt {row['prompt_id']}:")
        print(e)
        print("Stopping so progress is not lost.")
        break

print("Finished.")
