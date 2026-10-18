"""Local PDF and plain-text extraction. No cloud OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
        return ExtractedDocument(name=name, pages=pages)
    # Treat unlabeled bytes as UTF-8 text.
    text = data.decode("utf-8", errors="replace")
    return ExtractedDocument(name=name, pages=_split_plain_pages(text))


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
            "No extractable text. This PDF may be scanned; cloud OCR is disabled. "
            "Export text locally or paste the text instead."
        )
        pages = [""]
    return ExtractedDocument(
        name=name,
        pages=pages,
        source_path=source_path,
        warnings=warnings,
    )


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
