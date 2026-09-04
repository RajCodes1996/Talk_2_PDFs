"""
app.py  —  RAG Accessibility Reader
Streamlit frontend for visually impaired users.

Run:
    streamlit run app.py
"""

import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from src.pdf_loader import extract_text_from_pdf, get_pdf_metadata
from src.chunker import chunk_text
from src.vector_store import FAISSVectorStore
from src.llm import summarise_document, answer_question, simplify_passage

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
    page_title="RAG Accessibility Reader",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 18px !important; }

    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
    .subtitle   { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }

    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #16213e;
        padding: 0.5rem 0; border-bottom: 2px solid #0f3460; margin-bottom: 1rem;
    }
    .answer-box {
        background: #f0f7ff; border-left: 5px solid #0f3460;
        border-radius: 8px; padding: 1.2rem 1.5rem;
        font-size: 1.15rem; line-height: 1.9; color: #1a1a2e; margin-top: 0.5rem;
    }
    .summary-box {
        background: #f8fff8; border-left: 5px solid #2d6a4f;
        border-radius: 8px; padding: 1.2rem 1.5rem;
        font-size: 1.1rem; line-height: 1.85; color: #1a1a2e;
    }
    .confidence-bar {
        height: 6px; border-radius: 3px; background: #e0e0e0; margin: 4px 0 12px 0;
    }
    .confidence-fill {
        height: 6px; border-radius: 3px; background: linear-gradient(90deg, #2d6a4f, #52b788);
    }
    .chunk-card {
        background: #fafafa; border: 1px solid #e0e0e0;
        border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;
        font-size: 0.9rem; color: #444; line-height: 1.6;
    }
    .meta-badge {
        background: #e8f4fd; border-radius: 20px; padding: 4px 14px;
        font-size: 0.85rem; color: #0f3460; display: inline-block; margin: 2px;
    }
    .suggested-q {
        background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px;
        padding: 8px 14px; margin: 4px 0; font-size: 0.95rem; color: #5d4037;
        cursor: pointer;
    }
    .stButton > button { font-size: 1rem !important; padding: 0.55rem 1.4rem !important; border-radius: 8px !important; }
    .stTextInput > div > div > input { font-size: 1.1rem !important; }
    *:focus { outline: 3px solid #f4a261 !important; outline-offset: 2px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### 🌐 Read-aloud language")
    lang_name = st.selectbox("Language for text-to-speech", options=list(SUPPORTED_LANGUAGES.keys()), index=0)
    st.session_state.tts_lang = SUPPORTED_LANGUAGES[lang_name]
    slow_speech = st.checkbox("🐢 Slow speech (easier to follow)", value=False)

    st.markdown("---")
    st.markdown("### 📖 Text size")
    font_size = st.slider("Base font size (px)", 16, 32, 20)
    st.markdown(f"<style>html,body,[class*='css']{{font-size:{font_size}px !important}}</style>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧩 RAG settings")
    top_k = st.slider("Chunks retrieved per question", 3, 10, 6,
                      help="More chunks = more context for the LLM but slower")
    chunk_size = st.slider("Chunk size (chars)", 200, 800, 500, step=50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 50, 200, 100, step=25)

    st.markdown("---")
    if st.session_state.doc_loaded:
        st.success("✅ Document loaded")
        meta = st.session_state.pdf_meta
        st.markdown(f"**{meta.get('filename', '')}**")
        st.markdown(
            f"<span class='meta-badge'>📄 {meta.get('pages', 0)} pages</span>"
            f"<span class='meta-badge'>🧩 {st.session_state.vector_store.chunk_count} chunks</span>",
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear document"):
            for k in ["vector_store","raw_text","summary","chat_history","qa_pairs","pdf_meta","doc_loaded","prefill_question"]:
                st.session_state[k] = None if k == "vector_store" else \
                    ([] if isinstance(st.session_state[k], list) else \
                    ("" if isinstance(st.session_state[k], str) else False))
            st.rerun()

    if not TTS_AVAILABLE:
        st.markdown("---")
        st.warning("🔇 TTS unavailable\n\nActivate venv then run:\n```\npip install gtts\n```")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<p class='main-title'>👁️ Accessibility Reader</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload any PDF — get a plain-language summary, ask questions, or hear it read aloud.</p>", unsafe_allow_html=True)


# ── Upload ────────────────────────────────────────────────────────────────────
if not st.session_state.doc_loaded:
    st.markdown("<p class='section-header'>📂 Upload your document</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"],
                                     help="Upload any PDF — a letter, article, form, or book.")
    if uploaded_file:
        with st.spinner("📖 Reading and indexing the document..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                raw_text = extract_text_from_pdf(tmp_path)
                meta = get_pdf_metadata(tmp_path)
                os.unlink(tmp_path)

                chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                vs = FAISSVectorStore()
                with st.spinner(f"🔢 Embedding {len(chunks)} chunks into FAISS..."):
                    vs.build(chunks)

                st.session_state.raw_text = raw_text
                st.session_state.vector_store = vs
                st.session_state.pdf_meta = meta
                st.session_state.doc_loaded = True

                st.success(f"✅ Ready! {len(chunks)} chunks indexed from {meta['pages']} pages.")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error loading PDF: {e}")

else:
    tab_summary, tab_qa, tab_listen = st.tabs(["📝  Summary", "❓  Ask a question", "🔊  Listen"])

    # ── Tab 1: Summary ────────────────────────────────────────────────────────
    with tab_summary:
        st.markdown("<p class='section-header'>📝 Document Summary</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 Standard summary", use_container_width=True):
                with st.spinner("Summarising..."):
                    st.session_state.summary = summarise_document(st.session_state.raw_text, mode="standard")
        with col2:
            if st.button("🧒 Simplify (ELI5)", use_container_width=True):
                with st.spinner("Simplifying..."):
                    st.session_state.summary = summarise_document(st.session_state.raw_text, mode="simple")
        with col3:
            if st.button("• Key bullet points", use_container_width=True):
                with st.spinner("Extracting key points..."):
                    st.session_state.summary = summarise_document(st.session_state.raw_text, mode="bullet")

        if st.session_state.summary:
            st.markdown(
                f"<div class='summary-box'>{st.session_state.summary.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
            if TTS_AVAILABLE:
                if st.button("🔊 Read this summary aloud"):
                    with st.spinner("Generating audio..."):
                        audio = text_to_speech_bytes(st.session_state.summary,
                                                      lang=st.session_state.tts_lang, slow=slow_speech)
                        st.audio(audio, format="audio/mp3")
            else:
                st.warning("⚠️ TTS unavailable — activate venv and run `pip install gtts`, then restart.")
        else:
            st.info("👆 Choose a summary style above to get started.")


    # ── Tab 2: Q&A ────────────────────────────────────────────────────────────
    with tab_qa:
        st.markdown("<p class='section-header'>❓ Ask anything about the document</p>", unsafe_allow_html=True)

        # Suggested starter questions
        st.markdown("**💡 Try asking:**")
        starter_cols = st.columns(3)
        starters = [
            "What is this document about?",
            "What are the main points?",
            "Who is this document for?",
        ]
        for i, sq in enumerate(starters):
            with starter_cols[i]:
                if st.button(sq, key=f"starter_{i}", use_container_width=True):
                    st.session_state.prefill_question = sq
                    st.rerun()

        st.markdown("---")

        # Question input — prefilled if starter was clicked
        question = st.text_input(
            "Your question",
            value=st.session_state.prefill_question,
            placeholder="e.g. What is the main topic? What should I do next? Who wrote this?",
            label_visibility="collapsed",
            key="qa_input_field",
        )
        # Clear prefill after use
        if st.session_state.prefill_question:
            st.session_state.prefill_question = ""

        col_ask, col_clear = st.columns([2, 1])
        with col_ask:
            ask_btn = st.button("💬 Ask", use_container_width=True, type="primary")
        with col_clear:
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state.qa_pairs = []
                st.session_state.chat_history = []
                st.rerun()

        if ask_btn and question.strip():
            with st.spinner("🔍 Searching document and generating answer..."):
                results = st.session_state.vector_store.search(question, top_k=top_k)

                # Show a warning if retrieval confidence is low
                top_score = results[0][1] if results else 0

                answer = answer_question(
                    question=question,
                    context_chunks=results,
                    chat_history=st.session_state.chat_history,
                )

                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.qa_pairs.insert(0, (question, answer, results, top_score))

        # ── Display Q&A history ───────────────────────────────────────────────
        if not st.session_state.qa_pairs:
            st.info("Type a question above or click a suggestion to get started.")

        for q, a, ctx, top_score in st.session_state.qa_pairs:
            with st.expander(f"❓ {q}", expanded=True):

                # Confidence indicator
                pct = int(top_score * 100)
                conf_color = "#52b788" if pct > 60 else "#f4a261" if pct > 35 else "#e63946"
                conf_label = "High confidence" if pct > 60 else "Medium confidence" if pct > 35 else "Low confidence — answer may be incomplete"
                st.markdown(
                    f"<div style='font-size:0.82rem;color:{conf_color};margin-bottom:2px'>"
                    f"📊 {conf_label} ({pct}%)</div>"
                    f"<div class='confidence-bar'><div class='confidence-fill' style='width:{pct}%;background:{conf_color}'></div></div>",
                    unsafe_allow_html=True,
                )

                # Answer
                st.markdown(
                    f"<div class='answer-box'>{a.replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True,
                )

                # Action row
                act_col1, act_col2 = st.columns([1, 2])
                with act_col1:
                    if TTS_AVAILABLE:
                        if st.button("🔊 Read aloud", key=f"tts_{q[:25]}"):
                            with st.spinner("Generating audio..."):
                                audio = text_to_speech_bytes(a, lang=st.session_state.tts_lang, slow=slow_speech)
                                st.audio(audio, format="audio/mp3")

                # Retrieved context chunks — visible for transparency
                with st.expander("🔍 View retrieved document excerpts", expanded=False):
                    st.caption("These are the parts of the document used to answer your question.")
                    for i, (chunk, score) in enumerate(ctx, 1):
                        bar_w = int(score * 100)
                        st.markdown(
                            f"<div class='chunk-card'>"
                            f"<b>Excerpt {i}</b> &nbsp; "
                            f"<span style='color:#2d6a4f;font-size:0.85rem'>relevance {score:.2f}</span>"
                            f"<div class='confidence-bar'><div class='confidence-fill' style='width:{bar_w}%'></div></div>"
                            f"{chunk[:350]}{'...' if len(chunk) > 350 else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )


    # ── Tab 3: Listen ─────────────────────────────────────────────────────────
    with tab_listen:
        st.markdown("<p class='section-header'>🔊 Listen to the document</p>", unsafe_allow_html=True)
        st.markdown("Convert any part of the document to speech.")

        listen_mode = st.radio(
            "What to listen to",
            ["Summary (recommended)", "Custom text", "Specific passage from document"],
            horizontal=True,
        )

        text_to_speak = ""

        if listen_mode == "Summary (recommended)":
            if st.session_state.summary:
                text_to_speak = st.session_state.summary
                st.success("✅ Summary is ready to play.")
            else:
                st.warning("Generate a summary first from the Summary tab.")

        elif listen_mode == "Custom text":
            text_to_speak = st.text_area("Paste or type any text to hear it", height=150,
                                          placeholder="Type or paste any text here...")

        elif listen_mode == "Specific passage from document":
            st.caption("Enter a topic and we'll find and read the most relevant passage.")
            passage_query = st.text_input("Topic or keywords", placeholder="e.g. payment terms, side effects")
            if st.button("🔍 Find passage"):
                if passage_query:
                    results = st.session_state.vector_store.search(passage_query, top_k=2)
                    if results:
                        text_to_speak = " ".join([r[0] for r in results])
                        st.markdown(f"<div class='answer-box'>{text_to_speak}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("No matching passage found.")

        if text_to_speak:
            if TTS_AVAILABLE:
                if st.button("▶️ Generate & Play audio", type="primary", use_container_width=True):
                    with st.spinner("🎙️ Converting text to speech..."):
                        try:
                            audio = text_to_speech_bytes(text_to_speak,
                                                          lang=st.session_state.tts_lang, slow=slow_speech)
                            st.audio(audio, format="audio/mp3")
                            st.success("✅ Audio ready — press play above.")
                        except Exception as e:
                            st.error(f"Audio generation failed: {e}")
            else:
                st.warning("⚠️ TTS unavailable — activate venv and run `pip install gtts`, then restart.")
