"""
app.py  —  RAG Accessibility Reader
Streamlit frontend for visually impaired users.

Run:
    streamlit run app.py
"""

import streamlit as st
import tempfile
import os
import time
import random
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
    /* Clean Blue Accessible Theme */
    [data-testid="stAppViewContainer"] { background-color: #F0F4F8 !important; color: #102A43 !important; }
    [data-testid="stSidebar"] { background-color: #E2E8F0 !important; }
    html, body, [class*="css"] { font-size: 22px !important; color: #102A43 !important; }

    .main-title { font-size: 2.8rem; font-weight: 800; color: #003E6B; margin-bottom: 0.2rem; letter-spacing: 0.5px; }
    .subtitle   { font-size: 1.3rem; color: #334E68; margin-bottom: 1.5rem; }

    .section-header {
        font-size: 1.6rem; font-weight: 700; color: #005C9F;
        padding: 0.5rem 0; border-bottom: 3px solid #005C9F; margin-bottom: 1rem;
    }
    .answer-box {
        background: #FFFFFF; border-left: 6px solid #005C9F;
        border-radius: 12px; padding: 1.5rem 1.8rem;
        font-size: 1.3rem; line-height: 1.9; color: #102A43; margin-top: 0.5rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
        position: relative;
    }
    .ai-avatar-header {
        display: flex; align-items: center; margin-bottom: 12px;
        font-weight: 800; font-size: 1.15rem; color: #005C9F;
        border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;
    }
    .ai-avatar {
        font-size: 1.8rem; margin-right: 12px;
        background: #E0E8F5; border-radius: 50%; padding: 8px;
        display: flex; align-items: center; justify-content: center;
    }
    .summary-box {
        background: #FFFFFF; border-left: 6px solid #003E6B;
        border-radius: 8px; padding: 1.2rem 1.5rem;
        font-size: 1.25rem; line-height: 1.8; color: #102A43;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .confidence-bar {
        height: 8px; border-radius: 4px; background: #D9E2EC; margin: 6px 0 14px 0;
    }
    .confidence-fill {
        height: 8px; border-radius: 4px; background: linear-gradient(90deg, #1992D4, #005C9F);
    }
    .chunk-card {
        background: #F8FAFC; border: 1px solid #BCCCDC;
        border-radius: 8px; margin-bottom: 12px;
        overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .chunk-header {
        background: #E2E8F0; padding: 10px 16px;
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid #BCCCDC;
    }
    .chunk-title {
        font-weight: 700; color: #003E6B; font-size: 1.05rem;
    }
    .chunk-relevance {
        font-size: 0.9rem; color: #005C9F; background: #D9E2EC;
        padding: 4px 10px; border-radius: 12px; font-weight: 600;
    }
    .chunk-text {
        padding: 14px 16px; font-size: 1.05rem; color: #334E68;
        line-height: 1.6; background: #FFFFFF;
    }
    .meta-badge {
        background: #E0E8F5; border-radius: 20px; padding: 6px 16px;
        font-size: 1rem; color: #003E6B; display: inline-block; margin: 4px; border: 1px solid #82CFFF;
    }
    .suggested-q {
        background: #FFFFFF; border: 2px solid #82CFFF; border-radius: 8px;
        padding: 10px 16px; margin: 6px 0; font-size: 1.1rem; color: #005C9F;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton > button { 
        font-size: 1.2rem !important; padding: 0.7rem 1.5rem !important; 
        border-radius: 10px !important; font-weight: 700 !important;
        background-color: #005C9F !important; color: #FFFFFF !important; border: 2px solid #003E6B !important;
    }
    .stButton > button:hover {
        background-color: #003E6B !important; color: #FFFFFF !important;
        border-color: #003E6B !important;
    }
    .stTextInput > div > div > input { font-size: 1.3rem !important; background-color: #FFFFFF !important; color: #102A43 !important; border: 2px solid #005C9F !important; }
    *:focus { outline: 4px solid #1992D4 !important; outline-offset: 3px !important; }
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
        "auto_submit": False,
        "just_asked": False,
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
                    st.session_state.auto_submit = True
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

        # Check if we should auto-submit from starter questions
        auto_submit_trigger = st.session_state.get("auto_submit", False)
        if auto_submit_trigger:
            st.session_state.auto_submit = False

        if (ask_btn and question.strip()) or (auto_submit_trigger and question.strip()):
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
                st.session_state.just_asked = True

        # ── Display Q&A history ───────────────────────────────────────────────
        if not st.session_state.qa_pairs:
            st.info("Type a question above or click a suggestion to get started.")

        for i, (q, a, ctx, top_score) in enumerate(st.session_state.qa_pairs):
            with st.expander(f"❓ {q}", expanded=(i==0)):

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

                # Answer with typewriter effect for newly generated answers
                avatar_html = "<div class='ai-avatar-header'><div class='ai-avatar'>🤖</div> AI Assistant</div>"
                
                if i == 0 and st.session_state.just_asked:
                    placeholder = st.empty()
                    typed_text = ""
                    # Word-by-word realistic typewriter effect
                    words = a.split(" ")
                    for word in words:
                        typed_text += word + " "
                        placeholder.markdown(
                            f"<div class='answer-box'>{avatar_html}<div style='margin-top: 10px;'>{typed_text.replace(chr(10), '<br>')}▌</div></div>", 
                            unsafe_allow_html=True
                        )
                        time.sleep(random.uniform(0.015, 0.05)) # Random delay feels more natural
                    
                    placeholder.markdown(
                        f"<div class='answer-box'>{avatar_html}<div style='margin-top: 10px;'>{a.replace(chr(10), '<br>')}</div></div>", 
                        unsafe_allow_html=True
                    )
                    st.session_state.just_asked = False
                else:
                    st.markdown(
                        f"<div class='answer-box'>{avatar_html}<div style='margin-top: 10px;'>{a.replace(chr(10), '<br>')}</div></div>",
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
                            f"<div class='chunk-header'>"
                            f"<span class='chunk-title'>Excerpt {i}</span>"
                            f"<span class='chunk-relevance'>Relevance: {score:.2f}</span>"
                            f"</div>"
                            f"<div class='chunk-text'>{chunk[:350]}{'...' if len(chunk) > 350 else ''}</div>"
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
