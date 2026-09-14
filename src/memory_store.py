"""
memory_store.py

This file contains optional LangChain-based memory helpers. The app does not
require them to run, so missing LangChain packages no longer crash imports.
"""

import json
import os
from pathlib import Path
from typing import Optional

try:
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, AIMessage
except Exception:  # pragma: no cover - optional dependency path
    ConversationSummaryBufferMemory = None  # type: ignore
    ChatGroq = None  # type: ignore
    HumanMessage = AIMessage = None  # type: ignore

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODEL = "openai/gpt-oss-20b"
MEMORY_FILE = Path("memory_store.json")
MAX_TOKEN_LIMIT = 1000


def _get_llm():
    if ChatGroq is None:
        raise ModuleNotFoundError(
            "LangChain dependencies are missing. Install langchain, langchain-groq, and langgraph to use memory features."
        )
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")
    return ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0.3)


class ConversationMemoryManager:
    def __init__(self):
        if ConversationSummaryBufferMemory is None:
            self._memory = None
        else:
            self._memory = ConversationSummaryBufferMemory(
                llm=_get_llm(),
                max_token_limit=MAX_TOKEN_LIMIT,
                return_messages=True,
                memory_key="chat_history",
            )

    def add_turn(self, question: str, answer: str) -> None:
        if self._memory is None:
            return
        self._memory.chat_memory.add_user_message(question)
        self._memory.chat_memory.add_ai_message(answer)

    def get_history(self) -> list[dict]:
        if self._memory is None:
            return []
        messages = self._memory.chat_memory.messages
        history = []
        for msg in messages:
            if HumanMessage is not None and isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif AIMessage is not None and isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history

    def get_summary(self) -> str:
        if self._memory is None:
            return ""
        return self._memory.moving_summary_buffer or ""

    def clear(self) -> None:
        if self._memory is not None:
            self._memory.clear()


class DocumentMemory:
    def __init__(self, filepath: Path = MEMORY_FILE):
        self._path = filepath
        self._store: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._store, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def save_document(
        self, filename: str, summary: str, key_facts: list[str], topics: list[str]
    ) -> None:
        from datetime import datetime, timezone

        self._store[filename] = {
            "filename": filename,
            "summary": summary,
            "key_facts": key_facts,
            "topics": topics,
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def get_document(self, filename: str) -> Optional[dict]:
        return self._store.get(filename)

    def has_document(self, filename: str) -> bool:
        return filename in self._store

    def list_documents(self) -> list[str]:
        return list(self._store.keys())

    def get_all_summaries(self) -> str:
        if not self._store:
            return "No documents have been stored yet."
        parts = []
        for name, data in self._store.items():
            parts.append(
                f"Document: {name}\nSummary: {data['summary']}\nTopics: {', '.join(data['topics'])}"
            )
        return "\n\n".join(parts)

    def clear_document(self, filename: str) -> None:
        self._store.pop(filename, None)
        self._save()

    def clear_all(self) -> None:
        self._store = {}
        self._save()
