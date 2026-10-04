from google import genai

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input="Explain why maintaining body tension is important during a handstand."
)

print(interaction.output_text)
