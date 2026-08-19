# Test Ollama

import ollama


response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Explain customer churn in one simple sentence."
        }
    ]
)

print("\nLLM RESPONSE:")
print(response.message.content)