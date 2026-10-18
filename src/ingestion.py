"""Local PDF and plain-text extraction. No cloud OCR."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Hard limit to prevent OOM on very large documents.
MAX_DOCUMENT_CHARS = 300_000

# Strings that suggest prompt-injection attempts inside a document.
_INJECTION_SIGNALS = re.compile(
    r"ignore\s+(all\s+)?previous\s+instructions"
    r"|system\s+prompt"
    r"|you\s+are\s+now\s+a"
    r"|override\s+(all\s+)?instructions"
    r"|disregard\s+(your\s+)?instructions",
    re.IGNORECASE,
)


@dataclass
class ExtractedDocument:
    name: str
    pages: list[str]
    source_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)


def extract_file(path: str | Path) -> ExtractedDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".txt", ".md"}:
        return extract_text_file(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use PDF or .txt")


def extract_bytes(data: bytes, filename: str) -> ExtractedDocument:
    name = filename or "upload"
    lower = name.lower()
    if lower.endswith(".pdf"):
        return extract_pdf_bytes(data, name)
    if lower.endswith(".txt") or lower.endswith(".md"):
        text = data.decode("utf-8", errors="replace")
        pages = _split_plain_pages(text)
        return _apply_guards(ExtractedDocument(name=name, pages=pages))
    # Treat unlabeled bytes as UTF-8 text.
    text = data.decode("utf-8", errors="replace")
    return _apply_guards(ExtractedDocument(name=name, pages=_split_plain_pages(text)))


def extract_text_file(path: Path) -> ExtractedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    return ExtractedDocument(
        name=path.name,
        pages=_split_plain_pages(text),
        source_path=str(path),
    )


def extract_pdf(path: Path) -> ExtractedDocument:
    return extract_pdf_bytes(path.read_bytes(), path.name, source_path=str(path))


def extract_pdf_bytes(
    data: bytes,
    name: str,
    source_path: str | None = None,
) -> ExtractedDocument:
    warnings: list[str] = []
    pages = _extract_with_pypdf(data, warnings)
    if _too_sparse(pages):
        plumber_pages = _extract_with_pdfplumber(data, warnings)
        if not _too_sparse(plumber_pages):
            pages = plumber_pages
        elif plumber_pages and (not pages or len("".join(plumber_pages)) > len("".join(pages))):
            pages = plumber_pages
    if not any(p.strip() for p in pages):
        warnings.append(
            "No extractable text found. This PDF may be image-only / scanned. "
            "Cloud OCR is disabled. To process it locally: export to text using "
            "Tesseract OCR (tesseract file.pdf out -l eng) or paste the text directly."
        )
        pages = [""]
    doc = ExtractedDocument(
        name=name,
        pages=pages,
        source_path=source_path,
        warnings=warnings,
    )
    return _apply_guards(doc)


def _extract_with_pypdf(data: bytes, warnings: list[str]) -> list[str]:
    try:
        from io import BytesIO

        from pypdf import PdfReader
    except ImportError:
        warnings.append("pypdf is not installed.")
        return []
    try:
        reader = PdfReader(BytesIO(data))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return pages
    except Exception as exc:  # noqa: BLE001 — keep triage running
        warnings.append(f"pypdf failed: {exc}")
        return []


def _extract_with_pdfplumber(data: bytes, warnings: list[str]) -> list[str]:
    try:
        from io import BytesIO

        import pdfplumber
    except ImportError:
        warnings.append("pdfplumber is not installed.")
        return []
    try:
        pages = []
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return pages
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pdfplumber failed: {exc}")
        return []


def _too_sparse(pages: list[str]) -> bool:
    joined = "".join(pages).strip()
    return len(joined) < 40


def _split_plain_pages(text: str) -> list[str]:
    parts = text.split("\f")
    pages = [p.strip("\n") for p in parts if p.strip() or len(parts) == 1]
    return pages or [""]


def _apply_guards(doc: "ExtractedDocument") -> "ExtractedDocument":
    """
    Apply safety guards to an already-extracted document:
    1. Truncate oversized documents to MAX_DOCUMENT_CHARS.
    2. Strip null bytes / binary control characters from page text.
    3. Detect prompt-injection signals and attach a warning.
    """
    # --- sanitize control chars ---
    cleaned_pages = [
        re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", page)
        for page in doc.pages
    ]

    # --- size guard ---
    total = sum(len(p) for p in cleaned_pages)
    if total > MAX_DOCUMENT_CHARS:
        kept: list[str] = []
        budget = MAX_DOCUMENT_CHARS
        for page in cleaned_pages:
            if budget <= 0:
                break
            kept.append(page[:budget])
            budget -= len(page)
        cleaned_pages = kept
        doc.warnings.append(
            f"Document truncated to {MAX_DOCUMENT_CHARS:,} characters to prevent memory issues. "
            "The full document was not analysed. A qualified lawyer should review the complete file."
        )

    # --- injection detection ---
    full_text = "\n".join(cleaned_pages)
    if _INJECTION_SIGNALS.search(full_text):
        doc.warnings.append(
            "Possible prompt-injection text detected inside the document. "
            "The system will ignore any embedded instructions."
        )

    doc.pages[:] = cleaned_pages
    return doc
