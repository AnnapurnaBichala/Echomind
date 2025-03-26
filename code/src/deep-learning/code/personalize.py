from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
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