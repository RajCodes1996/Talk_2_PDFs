"""
llm.py
Wraps the Groq API for three core accessibility tasks:
  1. Plain-language summarisation
  2. RAG-based question answering
  3. Simplification (ELI5 mode)
"""

from groq import Groq
from typing import List, Tuple
import os

GROQ_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a helpful reading assistant for visually impaired people.
Your job is to explain documents clearly and simply so anyone can understand them.
Always use:
- Short sentences (under 20 words each)
- Plain, everyday words (no jargon unless explained)
- A warm, patient, and supportive tone
- Numbered or bulleted lists when listing things
- Clear structure so the listener can follow along easily
Never assume the reader can see images, charts, or tables — describe them in words."""


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def summarise_document(text: str, mode: str = "standard") -> str:
    client = get_groq_client()

    mode_instructions = {
        "standard": (
            "Write a clear summary of this document in plain English. "
            "Cover the main topic, key points, and any important conclusions. "
            "Use 3–5 short paragraphs."
        ),
        "simple": (
            "Explain this document as if talking to someone who finds reading difficult. "
            "Use very simple words. Imagine explaining to a 10-year-old. Keep it to 150 words."
        ),
        "bullet": (
            "List the 5–7 most important points from this document as clear, "
            "short bullet points. Each bullet must be one sentence. "
            "Start each bullet with a key word in CAPITALS."
        ),
    }

    instruction = mode_instructions.get(mode, mode_instructions["standard"])
    truncated_text = text[:6000] if len(text) > 6000 else text

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\nDocument:\n{truncated_text}"},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def answer_question(
    question: str,
    context_chunks: List[Tuple[str, float]],
    chat_history: List[dict] = None,
) -> str:
    client = get_groq_client()

    # Build context from retrieved chunks
    context_parts = []
    for i, (chunk, score) in enumerate(context_chunks, 1):
        context_parts.append(f"[Excerpt {i} | relevance: {score:.2f}]\n{chunk}")
    context = "\n\n".join(context_parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for turn in chat_history[-6:]:
            messages.append(turn)

    messages.append({
        "role": "user",
        "content": (
            f"A visually impaired user is asking about a document. "
            f"Use the excerpts below as your PRIMARY source. "
            f"Give a clear, detailed, plain-language answer. "
            f"If the excerpts don't fully answer the question, say what you found "
            f"and what is unclear — do NOT just say 'it is not in the document'.\n\n"
            f"Document excerpts:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer in simple, clear language. Be specific and helpful."
        ),
    })

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip()


def simplify_passage(passage: str) -> str:
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Rewrite the following passage in the simplest possible language. "
                    "Use short sentences. Replace every difficult word with a simple one. "
                    "Keep the same meaning.\n\n"
                    f"Passage:\n{passage}"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()
