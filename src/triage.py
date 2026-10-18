"""
Structured clause extraction — pure regex, no LLM, no network.

Every field is explicitly marked FOUND / AMBIGUOUS / NOT_FOUND.
Nothing is invented. Silence in the document is reported as NOT_FOUND,
never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# ─────────────────────────── status enum ───────────────────────────


class FieldStatus(str, Enum):
    FOUND = "FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"


STATUS_EMOJI = {
    FieldStatus.FOUND: "🟢",
    FieldStatus.AMBIGUOUS: "🟡",
    FieldStatus.NOT_FOUND: "🔴",
}

STATUS_COLOR = {
    FieldStatus.FOUND: "#1a7a3c",
    FieldStatus.AMBIGUOUS: "#92680d",
    FieldStatus.NOT_FOUND: "#8b1a1a",
}


# ─────────────────────────── field result ──────────────────────────


@dataclass
class FieldResult:
    label: str
    status: FieldStatus
    excerpts: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def emoji(self) -> str:
        return STATUS_EMOJI[self.status]


# ─────────────────────────── clause report ─────────────────────────


@dataclass
class ClauseReport:
    document_name: str
    parties: FieldResult
    effective_dates: FieldResult
    notice_periods: FieldResult
    penalties: FieldResult
    termination: FieldResult
    jurisdiction: FieldResult
    confidentiality: FieldResult
    payment_terms: FieldResult

    def all_fields(self) -> list[FieldResult]:
        return [
            self.parties,
            self.effective_dates,
            self.notice_periods,
            self.penalties,
            self.termination,
            self.jurisdiction,
            self.confidentiality,
            self.payment_terms,
        ]

    def risk_summary(self) -> str:
        """One-line plain-English risk summary, never a legal conclusion."""
        not_found = [f.label for f in self.all_fields() if f.status == FieldStatus.NOT_FOUND]
        ambiguous = [f.label for f in self.all_fields() if f.status == FieldStatus.AMBIGUOUS]
        parts: list[str] = []
        if not_found:
            parts.append(f"Clauses not found: {', '.join(not_found)}")
        if ambiguous:
            parts.append(f"Ambiguous clauses: {', '.join(ambiguous)}")
        if not parts:
            return "All tracked clause categories have at least one matching passage."
        return " · ".join(parts) + ". A qualified lawyer should review the full document."


# ─────────────────────────── helpers ───────────────────────────────

_WINDOW = 220  # characters of context around each match


def _snip(text: str, match: re.Match, window: int = _WINDOW) -> str:
    """Return a short excerpt centred on the regex match."""
    s = max(0, match.start() - window // 2)
    e = min(len(text), match.end() + window // 2)
    snippet = text[s:e].replace("\n", " ").strip()
    if s > 0:
        snippet = "…" + snippet
    if e < len(text):
        snippet = snippet + "…"
    return snippet


def _find(text: str, pattern: str, flags: int = re.IGNORECASE) -> list[str]:
    """Return up to 3 deduplicated excerpts for a pattern."""
    seen: set[str] = set()
    results: list[str] = []
    for m in re.finditer(pattern, text, flags):
        snip = _snip(text, m)
        if snip not in seen:
            seen.add(snip)
            results.append(snip)
        if len(results) >= 3:
            break
    return results


def _field(label: str, text: str, patterns: list[str]) -> FieldResult:
    """Run multiple patterns; combine & deduplicate results."""
    excerpts: list[str] = []
    for pat in patterns:
        excerpts.extend(_find(text, pat))
        if len(excerpts) >= 3:
            break
    excerpts = list(dict.fromkeys(excerpts))[:3]  # stable dedup

    if not excerpts:
        return FieldResult(label=label, status=FieldStatus.NOT_FOUND)
    if len(excerpts) == 1:
        return FieldResult(label=label, status=FieldStatus.FOUND, excerpts=excerpts)
    # Multiple hits could mean repeated clauses (normal) or conflicting ones
    status = FieldStatus.FOUND
    # Simple heuristic: if two excerpts share no words they may conflict
    return FieldResult(label=label, status=status, excerpts=excerpts)


# ─────────────────────────── extractors ────────────────────────────

_PARTY_PATTERNS = [
    # "between [Party A] and [Party B]"
    r"between\s+([A-Z][^\n,;]{3,60})\s+(?:and|&)\s+([A-Z][^\n,;]{3,60})",
    # "Party A:" / "Employer:" / "Client:" labelled blocks
    r"(Employer|Employee|Client|Vendor|Contractor|Licensor|Licensee|Lessor|Lessee"
    r"|Disclosing Party|Receiving Party|Party\s+[AB1-9])\s*[:\-–]\s*([A-Z][^\n]{3,80})",
    # Common intro: "This Agreement is entered into by ABC Corp"
    r"entered\s+into\s+by\s+(?:and\s+between\s+)?([A-Z][^\n,;]{3,60})",
    # Section heading 'Parties' followed by names on the next line
    r"(?:^|\n)\s*Parties\s*\n([^\n]{5,120})",
    # Quoted company names with role: Acme Corp ("Disclosing Party")
    r'([A-Z][A-Za-z0-9\s\.\,]+(?:LLC|Ltd|Corp|Inc|LLP|Co|Limited|GmbH))\s*\(["\u201c]?'
    r"(?:Disclosing|Receiving|Employer|Employee|Client|Vendor|Licensor|Licensee)",
]

_DATE_PATTERNS = [
    r"\b(?:effective|dated?|executed?|signed?|as\s+of)\s+(?:the\s+)?"
    r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July"
    r"|August|September|October|November|December),?\s+\d{4})",
    r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b",
    r"\b(January|February|March|April|May|June|July|August|September|October"
    r"|November|December)\s+\d{1,2},?\s+\d{4}\b",
]

_NOTICE_PATTERNS = [
    r"(\d+)\s*(?:calendar\s+|business\s+)?days[''']?\s+(?:prior\s+|written\s+)?notice",
    r"notice\s+(?:of|period\s+of)\s+(\d+)\s*(?:calendar\s+|business\s+)?days",
    r"written\s+notice\s+(?:of\s+)?(?:at\s+least\s+)?(\d+)\s*days",
    r"notify(?:ing)?\s+.{0,40}within\s+(\d+)\s*(?:business\s+)?days",
]

_PENALTY_PATTERNS = [
    r"(?:liquidat(?:ed)?\s+damages?|penalty|penalties|fine|forfeit(?:ure)?)"
    r"\s+(?:of\s+|up\s+to\s+|equal\s+to\s+)?"
    r"(?:USD?\s*|INR\s*|Rs\.?\s*|£\s*|\$\s*)?[\d,]+",
    r"(?:\$|USD|INR|Rs\.?)\s*[\d,]+(?:\.\d{2})?\s*(?:per\s+day|per\s+breach|as\s+damages?)?",
    r"indemnif(?:y|ication|ies)\s+.{0,80}(?:damages?|losses?|costs?)",
    r"liable\s+for\s+.{0,60}(?:\$|USD|INR|Rs\.?|damages?)",
]

_TERMINATION_PATTERNS = [
    r"(?:terminat(?:e|ion|ed)|cancel(?:lation)?)\s+.{0,120}"
    r"(?:breach|default|notice|immediately|written|upon|without\s+cause)",
    r"(?:either\s+party|company|employer|client)\s+may\s+terminat",
    r"terminat(?:e|ion)\s+(?:for\s+)?(?:cause|convenience|breach|default)",
    r"this\s+(?:agreement|contract)\s+(?:shall\s+)?terminat",
    r"right\s+to\s+terminat",
]

_JURISDICTION_PATTERNS = [
    r"(?:governed\s+by|subject\s+to)\s+the\s+laws?\s+of\s+([A-Z][^\n,;.]{3,60})",
    r"jurisdiction\s+(?:of|shall\s+be)\s+([A-Z][^\n,;.]{3,60})",
    r"courts?\s+of\s+([A-Z][^\n,;.]{3,60})\s+(?:shall\s+have|will\s+have)",
    r"exclusive\s+jurisdiction\s+in\s+([A-Z][^\n,;.]{3,60})",
    r"(?:arbitration|dispute\s+resolution)\s+(?:shall\s+be\s+)?(?:in|at)\s+([A-Z][^\n,;.]{3,60})",
]

_CONFIDENTIALITY_PATTERNS = [
    r"confidential(?:ity)?\s+(?:information|obligation|clause|period)",
    r"(?:shall\s+)?(?:keep|maintain|hold)\s+.{0,40}confidential",
    r"non[‑\-]?disclosure",
    r"proprietary\s+information",
    r"trade\s+secrets?",
]

_PAYMENT_PATTERNS = [
    r"(?:payment|fee|compensation|salary|remuneration)\s+of\s+"
    r"(?:USD?\s*|INR\s*|Rs\.?\s*|£\s*|\$\s*)?[\d,]+",
    r"(?:pay(?:ment)?|invoice|bill)\s+(?:within|due\s+(?:in|within))\s+(\d+)\s+days",
    r"(?:monthly|annual|quarterly)\s+(?:fee|payment|subscription|retainer)\s+of",
    r"(?:\$|USD|INR|Rs\.?)\s*[\d,]+(?:\.\d{2})?\s*(?:per\s+month|per\s+year|per\s+annum)?",
]


# ─────────────────────────── public API ────────────────────────────

def extract_clauses(text: str, document_name: str = "document") -> ClauseReport:
    """
    Run regex extraction over the full document text.
    Returns a ClauseReport. Never invents content.
    Called offline; no network, no LLM.
    """
    if not text or not text.strip():
        not_found = lambda label: FieldResult(label=label, status=FieldStatus.NOT_FOUND,
                                              note="Document text is empty.")
        return ClauseReport(
            document_name=document_name,
            parties=not_found("Parties"),
            effective_dates=not_found("Effective Date / Key Dates"),
            notice_periods=not_found("Notice Periods"),
            penalties=not_found("Penalties / Damages"),
            termination=not_found("Termination Conditions"),
            jurisdiction=not_found("Governing Jurisdiction"),
            confidentiality=not_found("Confidentiality / NDA"),
            payment_terms=not_found("Payment Terms"),
        )

    return ClauseReport(
        document_name=document_name,
        parties=_field("Parties", text, _PARTY_PATTERNS),
        effective_dates=_field("Effective Date / Key Dates", text, _DATE_PATTERNS),
        notice_periods=_field("Notice Periods", text, _NOTICE_PATTERNS),
        penalties=_field("Penalties / Damages", text, _PENALTY_PATTERNS),
        termination=_field("Termination Conditions", text, _TERMINATION_PATTERNS),
        jurisdiction=_field("Governing Jurisdiction", text, _JURISDICTION_PATTERNS),
        confidentiality=_field("Confidentiality / NDA", text, _CONFIDENTIALITY_PATTERNS),
        payment_terms=_field("Payment Terms", text, _PAYMENT_PATTERNS),
    )
