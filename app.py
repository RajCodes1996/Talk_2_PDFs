"""
app.py  —  RAG Accessibility Reader
AIML assistant for visually impaired users: upload PDFs, ask questions, listen aloud.
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Accessibility AI",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme tokens ──────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg": "#0D1117",
        "surface": "#161B22",
        "surface_2": "#21262D",
        "border": "#30363D",
        "text": "#F0F6FC",
        "text_muted": "#8B949E",
        "accent": "#58A6FF",
        "accent_soft": "rgba(88, 166, 255, 0.12)",
        "success": "#3FB950",
        "warning": "#D29922",
        "danger": "#F85149",
        "user_bubble": "#1F6FEB",
        "ai_bubble": "#21262D",
        "header_grad": "linear-gradient(135deg, #58A6FF 0%, #79C0FF 100%)",
    },
    "light": {
        "bg": "#F6F8FA",
        "surface": "#FFFFFF",
        "surface_2": "#F6F8FA",
        "border": "#D0D7DE",
        "text": "#1F2328",
        "text_muted": "#656D76",
        "accent": "#0969DA",
        "accent_soft": "rgba(9, 105, 218, 0.08)",
        "success": "#1A7F37",
        "warning": "#9A6700",
        "danger": "#CF222E",
        "user_bubble": "#0969DA",
        "ai_bubble": "#FFFFFF",
        "header_grad": "linear-gradient(135deg, #0969DA 0%, #218BFF 100%)",
    },
}


def theme_css(mode: str, font_size: int) -> str:
    t = THEMES[mode]
    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    #MainMenu, footer, header {{ visibility: hidden; }}

    [data-testid="stAppViewContainer"] {{
        background: {t["bg"]};
        color: {t["text"]};
        font-family: 'Inter', system-ui, sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background: {t["surface"]};
        border-right: 1px solid {t["border"]};
    }}
    [data-testid="stSidebar"] * {{
        color: {t["text"]} !important;
    }}
    html, body, [class*="css"] {{
        font-size: {font_size}px;
        line-height: 1.65;
    }}

    /* Header */
    .app-header {{
        background: {t["surface"]};
        border: 1px solid {t["border"]};
        border-radius: 16px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }}
    .app-brand {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}
    .brand-icon {{
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: {t["header_grad"]};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        flex-shrink: 0;
    }}
    .brand-title {{
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        background: {t["header_grad"]};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }}
    .brand-tagline {{
        font-size: 0.9rem;
        color: {t["text_muted"]};
        margin: 2px 0 0 0;
    }}
    .doc-badge {{
        background: {t["accent_soft"]};
        border: 1px solid {t["border"]};
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 0.85rem;
        color: {t["text_muted"]};
        text-align: right;
    }}
    .doc-badge strong {{ color: {t["text"]}; }}

    /* Hero / upload */
    .hero-section {{
        max-width: 720px;
        margin: 2rem auto 0;
        text-align: center;
    }}
    .hero-title {{
        font-size: 2.25rem;
        font-weight: 700;
        color: {t["text"]};
        margin-bottom: 0.75rem;
        letter-spacing: -0.03em;
    }}
    .hero-desc {{
        font-size: 1.1rem;
        color: {t["text_muted"]};
        margin-bottom: 2.5rem;
        line-height: 1.7;
    }}
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
        text-align: left;
    }}
    .feature-card {{
        background: {t["surface"]};
        border: 1px solid {t["border"]};
        border-radius: 14px;
        padding: 1.25rem;
    }}
    .feature-card h4 {{
        margin: 0 0 0.4rem 0;
        font-size: 1rem;
        color: {t["text"]};
    }}
    .feature-card p {{
        margin: 0;
        font-size: 0.88rem;
        color: {t["text_muted"]};
        line-height: 1.5;
    }}
    .feature-icon {{
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }}

    /* Chat messages */
    [data-testid="stChatMessage"] {{
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
    }}
    [data-testid="stChatMessageContent"] {{
        background: {t["ai_bubble"]} !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        color: {t["text"]} !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{
        background: {t["user_bubble"]} !important;
        border-color: {t["user_bubble"]} !important;
        color: #FFFFFF !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] * {{
        color: #FFFFFF !important;
    }}

    /* Confidence badge */
    .conf-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-top: 12px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid {t["border"]};
        background: {t["surface_2"]};
        color: {t["text_muted"]};
    }}
    .conf-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }}
    .conf-high {{ background: {t["success"]}; }}
    .conf-med  {{ background: {t["warning"]}; }}
    .conf-low  {{ background: {t["danger"]}; }}

    /* Typewriter cursor */
    .typewriter-cursor {{
        display: inline-block;
        width: 2px;
        height: 1.1em;
        background: {t["accent"]};
        margin-left: 2px;
        vertical-align: text-bottom;
        animation: blink 0.8s step-end infinite;
    }}
    @keyframes blink {{
        50% {{ opacity: 0; }}
    }}

    /* TTS row */
    .tts-row {{
        margin-top: 4px;
        margin-bottom: 8px;
    }}

    /* Inputs */
    .stTextInput > div > div > input,
    [data-testid="stChatInput"] textarea {{
        background: {t["surface"]} !important;
        color: {t["text"]} !important;
        border: 2px solid {t["border"]} !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
    }}
    .stTextInput > div > div > input:focus,
    [data-testid="stChatInput"] textarea:focus {{
        border-color: {t["accent"]} !important;
        box-shadow: 0 0 0 3px {t["accent_soft"]} !important;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
        border: 1px solid {t["border"]} !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {t["accent"]} !important;
        color: #FFFFFF !important;
        border-color: {t["accent"]} !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background: {t["surface"]};
        border: 2px dashed {t["border"]};
        border-radius: 16px;
        padding: 1.5rem;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: {t["accent"]};
        background: {t["accent_soft"]};
    }}

    /* Sidebar section labels */
    .sidebar-label {{
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {t["text_muted"]};
        margin-bottom: 0.5rem;
    }}

    @media (max-width: 768px) {{
        .feature-grid {{ grid-template-columns: 1fr; }}
        .app-header {{ flex-direction: column; align-items: flex-start; }}
    }}
</style>
"""


def init_state():
    defaults = {
        "vector_store": None,
        "raw_text": "",
        "summary": "",
        "chat_history": [],
        "qa_pairs": [],
        "pdf_meta": {},
        "doc_loaded": False,
        "tts_lang": "en",
        "prefill_question": "",
        "auto_submit": False,
        "theme": "dark",
        "font_size": 18,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def confidence_badge(score: float) -> str:
    pct = int(score * 100)
    if pct > 60:
        cls, label = "conf-high", "High confidence"
    elif pct > 35:
        cls, label = "conf-med", "Medium confidence"
    else:
        cls, label = "conf-low", "Low confidence"
    return f'<span class="conf-badge"><span class="conf-dot {cls}"></span>{label} · {pct}%</span>'


def typewriter_effect(container, text: str, speed: float = 0.028):
    """Reveal AI response word-by-word for easier listening/following."""
    words = text.split(" ")
    displayed = ""
    for i, word in enumerate(words):
        displayed += word + (" " if i < len(words) - 1 else "")
        cursor = '<span class="typewriter-cursor"></span>' if i < len(words) - 1 else ""
        container.markdown(displayed + cursor, unsafe_allow_html=True)
        time.sleep(speed)
    container.markdown(displayed)


init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-label">Appearance</p>', unsafe_allow_html=True)
    theme_choice = st.radio(
        "Color theme",
        options=["Dark (high contrast)", "Light (high contrast)"],
        index=0 if st.session_state.theme == "dark" else 1,
        label_visibility="collapsed",
    )
    st.session_state.theme = "dark" if "Dark" in theme_choice else "light"

    font_size = st.slider("Text size", 16, 28, st.session_state.font_size, help="Larger text is easier to read")
    st.session_state.font_size = font_size

    st.markdown("---")
    st.markdown('<p class="sidebar-label">Read aloud</p>', unsafe_allow_html=True)
    lang_name = st.selectbox("Language", options=list(SUPPORTED_LANGUAGES.keys()), label_visibility="collapsed")
    st.session_state.tts_lang = SUPPORTED_LANGUAGES[lang_name]
    slow_speech = st.checkbox("Slow speech", value=False, help="Speaks more slowly for easier following")

    st.markdown("---")
    if st.session_state.doc_loaded:
        meta = st.session_state.pdf_meta
        st.markdown('<p class="sidebar-label">Document</p>', unsafe_allow_html=True)
        st.markdown(f"**{meta.get('filename', 'Document')}**")
        st.caption(f"{meta.get('pages', 0)} pages · {st.session_state.vector_store.chunk_count} sections indexed")

        st.markdown("---")
        st.markdown('<p class="sidebar-label">Quick actions</p>', unsafe_allow_html=True)
        if st.button("Summarize document", use_container_width=True):
            st.session_state.prefill_question = "Please summarize this document."
            st.session_state.auto_submit = True
            st.rerun()
        if st.button("Explain simply (ELI5)", use_container_width=True):
            st.session_state.prefill_question = "Explain this document simply, like I'm 5 years old."
            st.session_state.auto_submit = True
            st.rerun()

        st.markdown("---")
        if st.button("Clear chat & document", type="primary", use_container_width=True):
            for k in ["vector_store", "raw_text", "summary", "chat_history", "qa_pairs", "pdf_meta", "doc_loaded", "prefill_question"]:
                st.session_state[k] = None if k == "vector_store" else (
                    [] if isinstance(st.session_state.get(k), list) else (
                        "" if isinstance(st.session_state.get(k), str) else False
                    )
                )
            st.rerun()

    if not TTS_AVAILABLE:
        st.markdown("---")
        st.warning("Text-to-speech unavailable. Install with: pip install gtts")

# Apply theme CSS
st.markdown(theme_css(st.session_state.theme, st.session_state.font_size), unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
if not st.session_state.doc_loaded:
    st.markdown(
        """
        <div class="hero-section">
            <div class="app-brand" style="justify-content:center;margin-bottom:1.5rem;">
                <div class="brand-icon">♿</div>
                <div>
                    <p class="brand-title" style="font-size:2rem;">Accessibility AI</p>
                    <p class="brand-tagline">Your reading assistant for PDF documents</p>
                </div>
            </div>
            <p class="hero-desc">
                Upload any PDF to ask questions, hear answers read aloud, and get
                plain-language summaries — designed for visually impaired users.
            </p>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">💬</div>
                    <h4>Ask anything</h4>
                    <p>Chat naturally about your document. Answers cite relevant sections.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔊</div>
                    <h4>Listen aloud</h4>
                    <p>Every answer can be read back in multiple languages at your pace.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📖</div>
                    <h4>Plain language</h4>
                    <p>Get summaries and ELI5 explanations in short, clear sentences.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF to get started",
        type=["pdf"],
        help="Drag and drop or click to browse",
    )
    if uploaded_file:
        with st.spinner("Reading and indexing your document…"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                raw_text = extract_text_from_pdf(tmp_path)
                meta = get_pdf_metadata(tmp_path)
                os.unlink(tmp_path)

                chunks = chunk_text(raw_text, chunk_size=500, chunk_overlap=100)
                vs = FAISSVectorStore()
                with st.spinner("Building search index…"):
                    vs.build(chunks)

                st.session_state.raw_text = raw_text
                st.session_state.vector_store = vs
                st.session_state.pdf_meta = meta
                st.session_state.doc_loaded = True

                st.session_state.qa_pairs.append({
                    "role": "assistant",
                    "content": (
                        f"Hello! I've loaded **{meta.get('filename', 'your document')}** "
                        f"({meta.get('pages', 0)} pages). Ask me anything about it, "
                        f"or use the sidebar to summarize or simplify."
                    ),
                    "context": [],
                    "score": 1.0,
                    "is_greeting": True,
                })
                st.rerun()

            except Exception as e:
                st.error(f"Could not load PDF: {e}")

else:
    meta = st.session_state.pdf_meta
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-brand">
                <div class="brand-icon">♿</div>
                <div>
                    <p class="brand-title">Accessibility AI</p>
                    <p class="brand-tagline">Reading assistant · AIML for visual accessibility</p>
                </div>
            </div>
            <div class="doc-badge">
                <strong>{meta.get('filename', 'Document')}</strong><br>
                {meta.get('pages', 0)} pages · {st.session_state.vector_store.chunk_count} sections
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Chat history ──────────────────────────────────────────────────────────
    for i, msg in enumerate(st.session_state.qa_pairs):
        role = "user" if msg["role"] == "user" else "assistant"
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            if role == "assistant" and not msg.get("is_greeting"):
                score = msg.get("score", 0)
                st.markdown(confidence_badge(score), unsafe_allow_html=True)
                if TTS_AVAILABLE:
                    if st.button("🔊 Read aloud", key=f"tts_{i}"):
                        with st.spinner("Generating audio…"):
                            audio = text_to_speech_bytes(
                                msg["content"],
                                lang=st.session_state.tts_lang,
                                slow=slow_speech,
                            )
                            st.audio(audio, format="audio/mp3")

    # ── Chat input ────────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask a question about your document…")

    auto_prompt = st.session_state.get("prefill_question", "")
    if st.session_state.get("auto_submit", False):
        st.session_state.auto_submit = False
        st.session_state.prefill_question = ""
        prompt = auto_prompt

    if prompt:
        st.session_state.qa_pairs.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown(
                '<span class="typewriter-cursor"></span>',
                unsafe_allow_html=True,
            )

            results = st.session_state.vector_store.search(prompt, top_k=6)
            top_score = results[0][1] if results else 0

            if "summarize this document" in prompt.lower():
                answer = summarise_document(st.session_state.raw_text, mode="standard")
            elif "explain this document simply" in prompt.lower():
                answer = summarise_document(st.session_state.raw_text, mode="simple")
            else:
                answer = answer_question(
                    question=prompt,
                    context_chunks=results,
                    chat_history=st.session_state.chat_history,
                )

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            typewriter_effect(placeholder, answer, speed=0.022)

        st.session_state.qa_pairs.append({
            "role": "assistant",
            "content": answer,
            "context": results,
            "score": top_score,
        })
        st.rerun()
