import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")


client = genai.Client(api_key=api_key)

resp = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
)

print(resp.text)
usage = resp.usage_metadata

print(f"Prompt Tokens: {usage.prompt_token_count}")
print(f"Response Tokens: {usage.candidates_token_count}")
