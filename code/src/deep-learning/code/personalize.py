from openai import OpenAI

client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = "",
)

def generate_personalized_reason(user_likes: list[str], suggest_item: str) -> str:
    user_likes_str = ", ".join(user_likes)
    prompt = (
        f"The user likes {user_likes_str}. "
        f"Can you explain why a {suggest_item} would be a great addition for them? "
        "Please respond in English with a friendly and informative tone."
    )

    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content.strip()