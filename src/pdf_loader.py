"""
src/pdf_loader.py — Robust PDF text extractor with multi-method fallback chain.

Extraction priority:
  1. PyMuPDF  (pymupdf / fitz)  — best Unicode handling, handles ligatures & special chars
  2. pdfplumber                  — good for tables and complex layouts
  3. PyPDF2                      — lightweight fallback
  4. pdfminer.six                — last resort; slowest but thorough

Raises ValueError if all methods yield less than MIN_CHARS of text,
which usually means the PDF is image-only / scanned.
"""

import os
from pathlib import Path

# Minimum characters to consider extraction successful
MIN_CHARS = 50


# ── Helper: clean extracted text ──────────────────────────────────────────────
def _clean(text: str) -> str:
    """Strip nulls, normalise whitespace, drop (cid:NNN) placeholder glyphs."""
    import re
    text = text.replace("\x00", "")
    text = re.sub(r"\(cid:\d+\)", "", text)   # pdfplumber cid artefacts
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse excessive blank lines
    return text.strip()


# ── Method 1: PyMuPDF ─────────────────────────────────────────────────────────
def _extract_pymupdf(path: str) -> str:
    try:
        import pymupdf as fitz  # modern import name
    except ImportError:
        import fitz              # legacy import name (older pymupdf)

    doc = fitz.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))   # plain text, best Unicode
    doc.close()
    return _clean("\n".join(pages))


# ── Method 2: pdfplumber ──────────────────────────────────────────────────────
def _extract_pdfplumber(path: str) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return _clean("\n".join(pages))


# ── Method 3: PyPDF2 ──────────────────────────────────────────────────────────
def _extract_pypdf2(path: str) -> str:
    try:
        from pypdf import PdfReader          # pypdf (modern fork)
    except ImportError:
        from PyPDF2 import PdfReader        # older PyPDF2

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return _clean("\n".join(pages))


# ── Method 4: pdfminer ────────────────────────────────────────────────────────
def _extract_pdfminer(path: str) -> str:
    from pdfminer.high_level import extract_text
    return _clean(extract_text(path) or "")


# ── Public API ────────────────────────────────────────────────────────────────
_METHODS = [
    ("PyMuPDF",     _extract_pymupdf),
    ("pdfplumber",  _extract_pdfplumber),
    ("PyPDF2",      _extract_pypdf2),
    ("pdfminer",    _extract_pdfminer),
]


def extract_text_from_pdf(path: str) -> str:
    """
    Try each extraction method in order, return the first result that
    exceeds MIN_CHARS. Raises ValueError if all methods fail.
    """
    errors = []
    best_text = ""

    for name, fn in _METHODS:
        try:
            text = fn(path)
            if len(text) > len(best_text):
                best_text = text          # keep the richest result so far
            if len(text) >= MIN_CHARS:
                return text               # good enough — return immediately
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if len(best_text) >= MIN_CHARS:
        return best_text

    # All methods failed or returned near-empty text
    detail = " | ".join(errors) if errors else "no text layer found"
    raise ValueError(
        f"Could not load PDF: No readable text found. "
        f"The PDF may be image-only (scanned). "
        f"Consider using OCR preprocessing. [{detail}]"
    )


def extract_text_and_metadata(path: str) -> tuple:
    """
    Fast path: extract text AND metadata in a single PyMuPDF pass.
    Returns (text: str, meta: dict).  Falls back to individual calls if needed.
    """
    filename = Path(path).name
    size_kb = round(os.path.getsize(path) / 1024, 1)

    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        doc = fitz.open(path)
        raw_meta = doc.metadata or {}
        pages_list = [page.get_text("text") for page in doc]
        page_count = doc.page_count
        doc.close()

        text = _clean("\n".join(pages_list))
        if len(text) >= MIN_CHARS:
            meta = {
                "filename": filename,
                "pages": page_count,
                "size_kb": size_kb,
                "title": raw_meta.get("title", ""),
                "author": raw_meta.get("author", ""),
            }
            return text, meta
    except Exception:
        pass

    # Fallback: separate calls for text + metadata
    text = extract_text_from_pdf(path)
    meta = get_pdf_metadata(path)
    return text, meta


def get_pdf_metadata(path: str) -> dict:
    """
    Return basic metadata: filename, page count, file size.
    Uses PyMuPDF when available, falls back to PyPDF2.
    """
    filename = Path(path).name
    size_kb = round(os.path.getsize(path) / 1024, 1)

    # Try PyMuPDF first
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        doc = fitz.open(path)
        meta = doc.metadata or {}
        pages = doc.page_count
        doc.close()
        return {
            "filename": filename,
            "pages": pages,
            "size_kb": size_kb,
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
        }
    except Exception:
        pass

    # Fallback: PyPDF2 / pypdf
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader

        reader = PdfReader(path)
        info = reader.metadata or {}
        return {
            "filename": filename,
            "pages": len(reader.pages),
            "size_kb": size_kb,
            "title": getattr(info, "title", "") or "",
            "author": getattr(info, "author", "") or "",
        }
    except Exception:
        pass

    # Bare minimum
    return {"filename": filename, "pages": 0, "size_kb": size_kb}
