"""
tts.py
Text-to-speech using gTTS (Google Text-to-Speech).
Converts any text to an MP3 and returns the bytes for Streamlit's audio player.
Supports multiple languages for multilingual accessibility.
"""

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

import io

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Mandarin": "zh",
    "Tamil": "ta",
    "Telugu": "te",
}


def text_to_speech_bytes(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """
    Convert text to MP3 audio bytes.
    Returns MP3 audio as bytes, ready for st.audio().
    """
    if not GTTS_AVAILABLE:
        raise ImportError(
            "gTTS is not installed in this environment.\n"
            "Fix: activate your venv then run:  pip install gtts"
        )

    if not text or not text.strip():
        raise ValueError("No text provided for speech.")

    chunks = _split_for_tts(text, max_chars=4000)
    audio_buffer = io.BytesIO()

    for chunk in chunks:
        tts = gTTS(text=chunk, lang=lang, slow=slow)
        chunk_buf = io.BytesIO()
        tts.write_to_fp(chunk_buf)
        audio_buffer.write(chunk_buf.getvalue())

    return audio_buffer.getvalue()


def _split_for_tts(text: str, max_chars: int = 4000) -> list:
    """Split long text at sentence boundaries for gTTS."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    sentences = text.replace("\n", " ").split(". ")
    current = ""

    for sentence in sentences:
        candidate = current + sentence + ". "
        if len(candidate) > max_chars:
            if current:
                chunks.append(current.strip())
            current = sentence + ". "
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks
