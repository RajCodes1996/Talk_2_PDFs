"""
pdf_loader.py
Extracts and cleans text from PDF files using PyMuPDF.
Handles multi-column layouts, image-only pages, and noisy text.
"""

import fitz  # PyMuPDF
import re
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file.
    Attempts to preserve paragraph structure.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    pages_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # plain text extraction
        if text.strip():
            pages_text.append(f"[Page {page_num}]\n{text.strip()}")

    doc.close()

    if not pages_text:
        raise ValueError(
            "No readable text found. The PDF may be image-only (scanned). "
            "Consider using OCR preprocessing."
        )

    full_text = "\n\n".join(pages_text)
    return clean_text(full_text)


def clean_text(text: str) -> str:
    """
    Remove noise: multiple blank lines, header/footer artifacts,
    excessive whitespace, and non-printable characters.
    """
    # Remove non-printable characters except newlines and tabs
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def get_pdf_metadata(pdf_path: str) -> dict:
    """Return basic PDF metadata for display."""
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    page_count = len(doc)
    doc.close()
    return {
        "title": meta.get("title", "Unknown"),
        "author": meta.get("author", "Unknown"),
        "pages": page_count,
        "filename": Path(pdf_path).name,
    }
