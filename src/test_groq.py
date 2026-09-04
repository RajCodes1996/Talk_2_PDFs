"""
test_groq.py  —  Quick Groq API key verification
Run: python test_groq.py
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads from .env if present

def test_groq():
    api_key = os.getenv("GROQ_API_KEY")

    # ── 1. Check key is set ──────────────────────────────────────────────────
    if not api_key:
        print("❌  GROQ_API_KEY not found.")
        print("    Either set it in your .env file or run:")
        print("    set GROQ_API_KEY=gsk_xxxxxxx   (Windows CMD)")
        return

    print(f"✅  Key found: {api_key[:8]}{'*' * (len(api_key) - 8)}")

    # ── 2. Try importing groq ────────────────────────────────────────────────
    try:
        from groq import Groq
        print("✅  groq package installed")
    except ImportError:
        print("❌  groq package not installed. Run: pip install groq")
        return

    # ── 3. Make a real API call ──────────────────────────────────────────────
    print("\n🔄  Sending test message to llama3-70b-8192 via Groq...")

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Reply in one short sentence."
                },
                {
                    "role": "user",
                    "content": "Say hello and confirm you are working!"
                }
            ],
            max_tokens=60,
            temperature=0.5,
        )

        reply = response.choices[0].message.content.strip()
        model_used = response.model
        usage = response.usage

        print(f"\n✅  Groq responded successfully!")
        print(f"   Model     : {model_used}")
        print(f"   Response  : {reply}")
        print(f"   Tokens    : {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total")
        print("\n🎉  Your Groq API key is working. You're good to go!")

    except Exception as e:
        err = str(e)
        print(f"\n❌  API call failed: {err}")

        if "401" in err or "invalid_api_key" in err.lower():
            print("    → Your API key is invalid or expired.")
            print("    → Get a fresh key at: https://console.groq.com")
        elif "429" in err:
            print("    → Rate limit hit. Wait a moment and try again.")
        elif "connection" in err.lower():
            print("    → Network error. Check your internet connection.")
        else:
            print("    → Check https://console.groq.com/docs for help.")


if __name__ == "__main__":
    test_groq()
