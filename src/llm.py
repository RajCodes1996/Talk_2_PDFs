"""
llm.py
Wraps the Groq API for three core accessibility tasks:
  1. Plain-language summarisation
  2. RAG-based question answering
  3. Simplification (ELI5 mode)

This version intentionally avoids hard dependency on LangChain so the app
still works in minimal environments where only the Groq SDK is installed.
"""

from __future__ import annotations

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

META_QUESTIONS = {
    "what is in the pdf",
    "what is in this pdf",
    "what is this",
    "what is this about",
    "what does it contain",
    "what does this contain",
    "summarize",
    "summarise",
    "give me a summary",
    "overview",
    "tell me about this",
    "what is the document about",
    "what is this document",
}

LOW_SCORE_THRESHOLD = 0.35


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def summarise_document(text: str, mode: str = "standard") -> str:
    """Produce a plain-language summary of the full document text."""
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
    truncated_text = text[:12_000] if len(text) > 12_000 else text

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{instruction}\n\nDocument:\n{truncated_text}",
            },
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
    """Answer a user question using retrieved document chunks."""
    client = get_groq_client()

    if any(q in question.lower() for q in META_QUESTIONS):
        combined = "\n\n".join(chunk for chunk, _ in context_chunks)
        return summarise_document(combined, mode="standard")

    if context_chunks:
        avg_score = sum(score for _, score in context_chunks) / len(context_chunks)
        if avg_score < LOW_SCORE_THRESHOLD:
            combined = "\n\n".join(chunk for chunk, _ in context_chunks)
            return summarise_document(combined, mode="standard")

    context_parts = []
    for i, (chunk, score) in enumerate(context_chunks, 1):
        context_parts.append(f"[Excerpt {i} | relevance: {score:.2f}]\n{chunk[:1_500]}")
    context = "\n\n".join(context_parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for turn in chat_history[-6:]:
            messages.append(turn)

    messages.append(
        {
            "role": "user",
            "content": (
                f"A visually impaired user is asking about a document.\n"
                f"Use the excerpts below as your PRIMARY source of information.\n"
                f"Give a clear, detailed, plain-language answer.\n"
                f"Synthesize ALL excerpts — never say you 'only see a short part'.\n"
                f"For broad questions, give a full overview of everything you find.\n"
                f"If something is genuinely absent from the excerpts, say so briefly "
                f"then answer with what you do know.\n\n"
                f"Document excerpts:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer in simple, clear language. Be specific and helpful."
            ),
        }
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip()


def simplify_passage(passage: str) -> str:
    """Rewrite a passage in the simplest possible plain language."""
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


def ask(
    question: str,
    chunks: List[Tuple[str, float]],
    full_text: str,
    filename: str = "",
) -> str:
    """Compatibility wrapper used by older callers; falls back to RAG answer generation."""
    return answer_question(question, chunks)


memory = None
doc_memory = None


def reset_session() -> None:
    """Compatibility no-op for previous memory-aware callers."""
    return None


def extract_and_remember(text: str, filename: str) -> dict:
    """Compatibility helper returning a minimal metadata payload."""
    return {
        "summary": summarise_document(text, mode="standard")[:400],
        "key_facts": [],
        "topics": [],
    }
