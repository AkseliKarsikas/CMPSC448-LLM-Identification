from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
        {
            "role": "user",
            "content": "Explain why maintaining body tension is important during a handstand."
        }
    ]
)

print(response.choices[0].message.content)
