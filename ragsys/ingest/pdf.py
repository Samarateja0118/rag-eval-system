"""PDF -> list of page texts. pypdfium2 (PDFium) is fast and permissively licensed."""

from __future__ import annotations

import re
from pathlib import Path

import pypdfium2 as pdfium

_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _NL.sub("\n\n", text).strip()


def extract_pages(path: str | Path) -> list[str]:
    """Return one cleaned string per page, index 0 = first physical page."""
    doc = pdfium.PdfDocument(str(path))
    try:
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            pages.append(clean(tp.get_text_range()))
            tp.close()
            page.close()
        return pages
    finally:
        doc.close()
