from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

user_likes = ["PlayStation 5", "iPhone 15"]
suggest_item = "Sony WH-1000XM5 headphone"

# Build the prompt dynamically
user_likes_str = ", ".join(user_likes)
prompt = (
    f"The user likes {user_likes_str}. "
    f"Can you explain why a {suggest_item} would be a great addition for them? "
    "Please respond in English with a friendly and informative tone."
)

completion = client.chat.completions.create(
    model="deepseek/deepseek-r1:free",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(completion.choices[0].message.content)