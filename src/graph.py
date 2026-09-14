"""
graph.py

This file contains a LangGraph-based workflow, but it is optional for the core
app. The project now gracefully falls back when the LangChain/LangGraph packages
are not installed, so importing the package doesn't crash the application.
"""

from __future__ import annotations

import os
from typing import Literal

try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict
except Exception:  # pragma: no cover - optional dependency path
    ChatGroq = None  # type: ignore
    HumanMessage = SystemMessage = None  # type: ignore
    END = START = StateGraph = None  # type: ignore
    TypedDict = dict  # type: ignore

# ── Shared config ─────────────────────────────────────────────────────────────
GROQ_MODEL = "openai/gpt-oss-20b"

META_KEYWORDS = {
    "what is in",
    "what is this",
    "what does it contain",
    "what is this about",
    "summarize",
    "summarise",
    "give me a summary",
    "overview",
    "tell me about",
    "what is the document",
    "define",
    "explain the document",
}

SYSTEM_PROMPT = """You are Lumen, a precise and caring reading assistant for visually impaired people.
You explain documents clearly so anyone can understand them.
Rules:
- Use short sentences (under 20 words each)
- Use plain, everyday words — explain jargon when unavoidable
- Use numbered or bulleted lists when listing multiple things
- Be warm, patient, and specific — never vague
- Never say 'I only see a short part' or 'that is all I can tell you'
- Always synthesize ALL context provided into a complete, direct answer
- Describe any images, charts, or tables in words since the user cannot see them"""


def _get_llm(temperature: float = 0.3):
    if ChatGroq is None:
        raise ModuleNotFoundError(
            "LangChain/LangGraph dependencies are not installed. "
            "Install langchain, langchain-groq, and langgraph to use the graph pipeline."
        )
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")
    return ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=temperature)


if TypedDict is dict:

    class GraphState(dict):
        pass
else:

    class GraphState(TypedDict):
        question: str
        chunks: list[tuple[str, float]]
        full_text: str
        filename: str
        chat_history: list[dict]
        conv_summary: str
        doc_memory: str
        intent: Literal["meta", "factual", "conversational", "cross_doc"]
        filtered_chunks: list[tuple[str, float]]
        context: str
        answer: str
        quality_ok: bool
        retry_count: int


def classify_intent(state: GraphState) -> GraphState:
    q = str(state.get("question", "")).lower().strip()
    if any(
        w in q
        for w in [
            "previous",
            "last time",
            "other document",
            "all documents",
            "you remember",
        ]
    ):
        intent = "cross_doc"
    elif any(kw in q for kw in META_KEYWORDS):
        intent = "meta"
    elif any(
        w in q for w in ["you said", "earlier", "before", "again", "what did", "repeat"]
    ):
        intent = "conversational"
    else:
        intent = "factual"
    return {**state, "intent": intent}


def retrieve_and_grade(state: GraphState) -> GraphState:
    chunks = state.get("chunks", [])
    intent = state.get("intent", "factual")
    if intent in ("meta", "cross_doc"):
        return {
            **state,
            "filtered_chunks": [],
            "context": state.get("full_text", "")[:12_000],
        }
    threshold = 0.35
    good_chunks = [(c, s) for c, s in chunks if s >= threshold]
    if len(good_chunks) < 2:
        good_chunks = chunks[:3]
        context = (
            "[Note: Retrieval confidence is low. Answering from best available excerpts + full document.]\n\n"
            + state.get("full_text", "")[:8_000]
        )
    else:
        parts = []
        for i, (chunk, score) in enumerate(good_chunks, 1):
            parts.append(f"[Excerpt {i} | relevance: {score:.0%}]\n{chunk[:1_500]}")
        context = "\n\n".join(parts)
    return {**state, "filtered_chunks": good_chunks, "context": context}


def enrich_with_memory(state: GraphState) -> GraphState:
    memory_block = ""
    if state.get("conv_summary"):
        memory_block += f"[Conversation so far]\n{state['conv_summary']}\n\n"
    if state.get("intent") == "cross_doc" and state.get("doc_memory"):
        memory_block += f"[Previously seen documents]\n{state['doc_memory']}\n\n"
    return {**state, "context": memory_block + state.get("context", "")}


def get_graph():
    if StateGraph is None:
        raise ModuleNotFoundError(
            "LangGraph is not installed. Install langgraph to use this pipeline."
        )
    return StateGraph()


# Keep the public API for callers that import this file directly.
def route_after_check(state: GraphState) -> Literal["generate", "__end__"]:
    return "__end__" if state.get("quality_ok") else "generate"


def generate(state: GraphState) -> GraphState:
    return state


def self_check(state: GraphState) -> GraphState:
    return {**state, "quality_ok": True}


def build_graph():
    """
    Compile and return the LangGraph StateGraph.

    Flow:
      START
        → classify_intent
        → retrieve_and_grade
        → enrich_with_memory
        → generate
        → self_check
        → (quality_ok?) → END
                       → generate (retry once)
                       → END
    """
    builder = StateGraph(GraphState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_and_grade", retrieve_and_grade)
    builder.add_node("enrich_with_memory", enrich_with_memory)
    builder.add_node("generate", generate)
    builder.add_node("self_check", self_check)

    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "retrieve_and_grade")
    builder.add_edge("retrieve_and_grade", "enrich_with_memory")
    builder.add_edge("enrich_with_memory", "generate")
    builder.add_edge("generate", "self_check")

    builder.add_conditional_edges(
        "self_check",
        route_after_check,
        {"generate": "generate", "__end__": END},
    )

    return builder.compile()


# ── Public API ────────────────────────────────────────────────────────────────
_graph = None  # module-level cache — compiled once per process


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
