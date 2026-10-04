from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6-luna",
    input="Explain why maintaining body tension is important during a handstand.",
    max_output_tokens=500
)

print(response.output_text)
