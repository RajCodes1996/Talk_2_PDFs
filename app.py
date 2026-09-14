"""
app.py — RAG Accessibility Reader
A calm, high-contrast reading assistant for visually impaired users.
"""

import streamlit as st
import tempfile
import os
import time
from dotenv import load_dotenv

from src.pdf_loader import extract_text_from_pdf, get_pdf_metadata
from src.chunker import chunk_text
from src.vector_store import FAISSVectorStore
from src.llm import summarise_document, answer_question

try:
    from src.tts import text_to_speech_bytes, SUPPORTED_LANGUAGES
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    SUPPORTED_LANGUAGES = {"English": "en"}

    def text_to_speech_bytes(*args, **kwargs):
        raise ImportError("gTTS not installed")

load_dotenv()

st.set_page_config(
    page_title="Lumen Reader",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
C = {
    "paper": "#F5F2EC",
    "white": "#FFFFFF",
    "ink": "#1C1C1C",
    "ink_soft": "#4A4A4A",
    "ink_faint": "#7A7A7A",
    "forest": "#2D6A4F",
    "forest_dark": "#1B4332",
    "forest_tint": "#D8F3DC",
    "line": "#DDD8CE",
    "line_strong": "#C4BDB0",
    "warn_bg": "#FFF8E7",
    "warn_text": "#8B6914",
    "danger": "#9B2226",
}


def inject_css(font_px: int):
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

#MainMenu, footer, header {{ visibility: hidden; height: 0; }}

:root {{
    --paper: {C["paper"]};
    --white: {C["white"]};
    --ink: {C["ink"]};
    --ink-soft: {C["ink_soft"]};
    --forest: {C["forest"]};
    --forest-dark: {C["forest_dark"]};
    --forest-tint: {C["forest_tint"]};
    --line: {C["line"]};
}}

.stApp {{
    background: var(--paper);
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    color: var(--ink);
}}

html, body, [class*="css"] {{
    font-size: {font_px}px;
    line-height: 1.7;
}}

/* ── Top bar ── */
.lumen-topbar {{
    background: var(--white);
    border-bottom: 1px solid var(--line);
    padding: 0.85rem 2rem;
    margin: -1rem -1rem 1.75rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.lumen-logo {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.lumen-logo-mark {{
    width: 40px;
    height: 40px;
    background: var(--forest);
    border-radius: 10px;
    display: grid;
    place-items: center;
    color: white;
    font-family: 'IBM Plex Serif', Georgia, serif;
    font-weight: 600;
    font-size: 1.1rem;
}}
.lumen-logo-text {{
    font-family: 'IBM Plex Serif', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--ink);
    margin: 0;
    line-height: 1.2;
}}
.lumen-logo-sub {{
    font-size: 0.78rem;
    color: var(--ink-soft);
    margin: 0;
    font-weight: 400;
}}
.lumen-doc-pill {{
    background: var(--forest-tint);
    color: var(--forest-dark);
    border: 1px solid #B7E4C7;
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 500;
}}

/* ── Landing layout ── */
.lumen-hero {{
    max-width: 960px;
    margin: 0 auto;
    padding: 1rem 0 3rem;
}}
.lumen-hero h1 {{
    font-family: 'IBM Plex Serif', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.15;
    margin: 0 0 1rem 0;
    letter-spacing: -0.02em;
}}
.lumen-hero h1 em {{
    font-style: normal;
    color: var(--forest);
}}
.lumen-lead {{
    font-size: 1.12rem;
    color: var(--ink-soft);
    max-width: 540px;
    margin-bottom: 2.5rem;
}}
.lumen-steps {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}}
.lumen-step {{
    background: var(--white);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.25rem;
}}
.lumen-step-num {{
    width: 28px;
    height: 28px;
    background: var(--forest);
    color: white;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}}
.lumen-step h3 {{
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 0.35rem 0;
    color: var(--ink);
}}
.lumen-step p {{
    font-size: 0.85rem;
    color: var(--ink-soft);
    margin: 0;
    line-height: 1.5;
}}

/* ── Upload card ── */
.lumen-upload-card {{
    background: var(--white);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}}
.lumen-upload-card h2 {{
    font-family: 'IBM Plex Serif', Georgia, serif;
    font-size: 1.3rem;
    margin: 0 0 0.5rem 0;
    color: var(--ink);
}}

/* ── Chat area ── */
.lumen-chat-wrap {{
    max-width: 780px;
    margin: 0 auto;
    padding-bottom: 1rem;
}}

[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0.35rem 0 !important;
}}
[data-testid="stChatMessageContent"] {{
    background: var(--white) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    color: var(--ink) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{
    background: var(--forest) !important;
    border-color: var(--forest) !important;
    color: #FFFFFF !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p {{
    color: #FFFFFF !important;
}}

/* Confidence tag */
.lumen-tag {{
    display: inline-block;
    margin-top: 10px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 500;
}}
.lumen-tag-high {{ background: #D8F3DC; color: #1B4332; }}
.lumen-tag-med  {{ background: #FFF8E7; color: #8B6914; }}
.lumen-tag-low  {{ background: #FDE8E8; color: #9B2226; }}

/* Typewriter */
.lumen-cursor {{
    display: inline-block;
    width: 2px;
    height: 1em;
    background: var(--forest);
    margin-left: 1px;
    vertical-align: text-bottom;
    animation: lumen-blink 0.7s step-end infinite;
}}
@keyframes lumen-blink {{ 50% {{ opacity: 0; }} }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: var(--white) !important;
    border-right: 1px solid var(--line) !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem;
}}
.lumen-sidebar-title {{
    font-family: 'IBM Plex Serif', Georgia, serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 1.25rem;
}}
.lumen-sidebar-section {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {C["ink_faint"]};
    margin: 1.25rem 0 0.5rem 0;
}}

/* ── Controls ── */
.stButton > button {{
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border: 1px solid var(--line) !important;
    background: var(--white) !important;
    color: var(--ink) !important;
    transition: background 0.15s, border-color 0.15s !important;
}}
.stButton > button:hover {{
    background: var(--forest-tint) !important;
    border-color: #B7E4C7 !important;
    color: var(--forest-dark) !important;
}}
.stButton > button[kind="primary"] {{
    background: var(--forest) !important;
    border-color: var(--forest) !important;
    color: white !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: var(--forest-dark) !important;
    border-color: var(--forest-dark) !important;
    color: white !important;
}}

[data-testid="stFileUploader"] {{
    background: {C["paper"]};
    border: 2px dashed {C["line_strong"]};
    border-radius: 12px;
    padding: 1.5rem;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: var(--forest);
    background: var(--forest-tint);
}}

[data-testid="stChatInput"] {{
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    background: var(--white) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}}
[data-testid="stChatInput"] textarea {{
    color: var(--ink) !important;
    font-size: 1rem !important;
}}

[data-testid="stSidebarCollapseButton"] {{
    color: var(--forest) !important;
}}

@media (max-width: 768px) {{
    .lumen-steps {{ grid-template-columns: 1fr; }}
    .lumen-hero h1 {{ font-size: 2rem; }}
    .lumen-topbar {{ padding: 0.85rem 1rem; flex-direction: column; gap: 0.75rem; align-items: flex-start; }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "vector_store": None,
        "raw_text": "",
        "chat_history": [],
        "qa_pairs": [],
        "pdf_meta": {},
        "doc_loaded": False,
        "tts_lang": "en",
        "prefill_question": "",
        "auto_submit": False,
        "font_size": 18,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def confidence_tag(score: float) -> str:
    pct = int(score * 100)
    if pct > 60:
        return f'<span class="lumen-tag lumen-tag-high">Strong match · {pct}%</span>'
    if pct > 35:
        return f'<span class="lumen-tag lumen-tag-med">Partial match · {pct}%</span>'
    return f'<span class="lumen-tag lumen-tag-low">Weak match · {pct}%</span>'


def typewriter(container, text: str, speed: float = 0.02):
    words = text.split(" ")
    shown = ""
    for i, word in enumerate(words):
        shown += word + (" " if i < len(words) - 1 else "")
        cursor = '<span class="lumen-cursor"></span>' if i < len(words) - 1 else ""
        container.markdown(shown + cursor, unsafe_allow_html=True)
        time.sleep(speed)
    container.markdown(shown)


def render_topbar(doc_name: str = "", pages: int = 0, chunks: int = 0):
    pill = ""
    if doc_name:
        pill = f'<span class="lumen-doc-pill">{doc_name} · {pages} pages · {chunks} sections</span>'
    st.markdown(
        f"""
<div class="lumen-topbar">
    <div class="lumen-logo">
        <div class="lumen-logo-mark"><i>T</i></div>
        <div>
            <p class="lumen-logo-text">Talk 2 PDF</p>
            <p class="lumen-logo-sub">Accessible document assistant</p>
        </div>
    </div>
    {pill}
</div>
        """,
        unsafe_allow_html=True,
    )


init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="lumen-sidebar-title">Settings</p>', unsafe_allow_html=True)

    st.markdown('<p class="lumen-sidebar-section">Text</p>', unsafe_allow_html=True)
    st.session_state.font_size = st.slider(
        "Font size",
        16, 28,
        st.session_state.font_size,
        help="Increase for easier reading",
    )

    st.markdown('<p class="lumen-sidebar-section">Speech</p>', unsafe_allow_html=True)
    lang_name = st.selectbox("Read-aloud language", list(SUPPORTED_LANGUAGES.keys()), label_visibility="collapsed")
    st.session_state.tts_lang = SUPPORTED_LANGUAGES[lang_name]
    slow_speech = st.checkbox("Slow speech", value=False)

    if st.session_state.doc_loaded:
        meta = st.session_state.pdf_meta
        st.markdown('<p class="lumen-sidebar-section">Document</p>', unsafe_allow_html=True)
        st.markdown(f"**{meta.get('filename', 'Document')}**")
        st.caption(f"{meta.get('pages', 0)} pages indexed")

        st.markdown('<p class="lumen-sidebar-section">Actions</p>', unsafe_allow_html=True)
        if st.button("Summarize", use_container_width=True):
            st.session_state.prefill_question = "Please summarize this document."
            st.session_state.auto_submit = True
            st.rerun()
        if st.button("Explain simply", use_container_width=True):
            st.session_state.prefill_question = "Explain this document simply, like I'm 5 years old."
            st.session_state.auto_submit = True
            st.rerun()
        if st.button("Clear & start over", type="primary", use_container_width=True):
            for k in ["vector_store", "raw_text", "chat_history", "qa_pairs", "pdf_meta", "doc_loaded", "prefill_question"]:
                st.session_state[k] = None if k == "vector_store" else (
                    [] if isinstance(st.session_state.get(k), list) else (
                        "" if isinstance(st.session_state.get(k), str) else False
                    )
                )
            st.rerun()

    if not TTS_AVAILABLE:
        st.markdown("---")
        st.info("Install gTTS for read-aloud: pip install gtts")

inject_css(st.session_state.font_size)

# ── Views ─────────────────────────────────────────────────────────────────────
if not st.session_state.doc_loaded:
    render_topbar()

    st.markdown(
        """
<div class="lumen-hero">
    <h1>Read any PDF.<br><em>Ask, listen, understand.</em></h1>
    <p class="lumen-lead">
        logoer helps visually impaired users explore documents through
        plain-language answers and text-to-speech — no visual clutter, no jargon.
    </p>
    <div class="lumen-steps">
        <div class="lumen-step">
            <div class="lumen-step-num">1</div>
            <h3>Upload</h3>
            <p>Drop a PDF. We extract and index the text automatically.</p>
        </div>
        <div class="lumen-step">
            <div class="lumen-step-num">2</div>
            <h3>Ask</h3>
            <p>Type questions in plain English. Get clear, short answers.</p>
        </div>
        <div class="lumen-step">
            <div class="lumen-step-num">3</div>
            <h3>Listen</h3>
            <p>Press read-aloud on any answer. Adjust speed and language.</p>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(
            '<div class="lumen-upload-card"><h2>Upload your document</h2><p style="color:#7A7A7A;margin-bottom:1rem;">PDF files only</p></div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Choose PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded:
            with st.spinner("Indexing document…"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    raw_text = extract_text_from_pdf(tmp_path)
                    meta = get_pdf_metadata(tmp_path)
                    os.unlink(tmp_path)

                    chunks = chunk_text(raw_text, chunk_size=500, chunk_overlap=100)
                    vs = FAISSVectorStore()
                    vs.build(chunks)

                    st.session_state.raw_text = raw_text
                    st.session_state.vector_store = vs
                    st.session_state.pdf_meta = meta
                    st.session_state.doc_loaded = True
                    st.session_state.qa_pairs.append({
                        "role": "assistant",
                        "content": (
                            f"I've read **{meta.get('filename', 'your document')}** "
                            f"({meta.get('pages', 0)} pages). What would you like to know?"
                        ),
                        "score": 1.0,
                        "is_greeting": True,
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load PDF: {e}")

else:
    meta = st.session_state.pdf_meta
    render_topbar(
        doc_name=meta.get("filename", "Document"),
        pages=meta.get("pages", 0),
        chunks=st.session_state.vector_store.chunk_count,
    )

    st.markdown('<div class="lumen-chat-wrap">', unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.qa_pairs):
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role, avatar="🧑" if role == "user" else "📖"):
            st.markdown(msg["content"])
            if role == "assistant" and not msg.get("is_greeting"):
                st.markdown(confidence_tag(msg.get("score", 0)), unsafe_allow_html=True)
            if role == "assistant" and TTS_AVAILABLE:
                if st.button("Read aloud", key=f"tts_{i}"):
                    with st.spinner("Preparing audio…"):
                        audio = text_to_speech_bytes(
                            msg["content"],
                            lang=st.session_state.tts_lang,
                            slow=slow_speech,
                        )
                        st.audio(audio, format="audio/mp3")

    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Type your question here…")
    auto = st.session_state.get("prefill_question", "")
    if st.session_state.get("auto_submit"):
        st.session_state.auto_submit = False
        st.session_state.prefill_question = ""
        prompt = auto

    if prompt:
        st.session_state.qa_pairs.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="📖"):
            slot = st.empty()
            slot.markdown('<span class="lumen-cursor"></span>', unsafe_allow_html=True)

            results = st.session_state.vector_store.search(prompt, top_k=6)
            top_score = results[0][1] if results else 0

            lower = prompt.lower()
            if "summarize this document" in lower:
                answer = summarise_document(st.session_state.raw_text, mode="standard")
            elif "explain this document simply" in lower:
                answer = summarise_document(st.session_state.raw_text, mode="simple")
            else:
                answer = answer_question(
                    question=prompt,
                    context_chunks=results,
                    chat_history=st.session_state.chat_history,
                )

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            typewriter(slot, answer)

        st.session_state.qa_pairs.append({
            "role": "assistant",
            "content": answer,
            "context": results,
            "score": top_score,
        })
        st.rerun()
