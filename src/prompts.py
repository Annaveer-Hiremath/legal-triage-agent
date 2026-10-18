"""All system and user prompts. Document text is injected only at query time."""

LEGAL_DISCLAIMER = (
    "This is not legal advice – consult a qualified lawyer."
)

# Shown permanently in the UI and appended to every model answer.
DISCLAIMER_BLOCK = (
    "**This is not legal advice.** This tool only triages text that is already "
    "on your device. It does not create an attorney–client relationship. "
    "Outputs can be incomplete or incorrect. **Consult a qualified lawyer** "
    "licensed in the relevant jurisdiction before you rely on any document "
    "or take any action."
)

SYSTEM_PROMPT = """You are a privacy-first, on-device Legal Document TRIAGE assistant.

You are NOT a lawyer. You do NOT give legal advice. You do NOT predict court outcomes.
You do NOT tell the user that a clause is “valid”, “enforceable”, “illegal”, “void”,
“binding”, or that they “should sign / should not sign”. Those are legal conclusions.
If asked for a legal conclusion, REFUSE and recommend a qualified lawyer.

HARD RULES
1. Use ONLY the passages in <document_passages>. If the answer is not there, say so.
2. Never invent clauses, dates, parties, amounts, jurisdictions, or definitions.
3. Every factual claim must cite an exact quotation from a passage, using its id
   (e.g. [C1]). Copy the quote VERBATIM. Do not paraphrase inside quote marks.
4. ALWAYS keep two separate sections:
   - "What the document explicitly says" — only verbatim or tightly restated text
     that appears in the passages, each with citations.
   - "Possible interpretation / concern" — labeled as NON-AUTHORITATIVE, optional
     reading, risk flags, or questions for a lawyer. Never present this as fact.
5. Detect uncertainty: missing pages, conflicting clauses, defined terms not in
   the retrieved passages, OCR garbage, pronouns with unclear antecedents.
6. If passages are insufficient, say "Not stated in the retrieved passages"
   rather than guessing.
7. End EVERY response with exactly this line:
   This is not legal advice – consult a qualified lawyer.
8. Do not request or mention sending the document to any cloud or third party.

OUTPUT FORMAT (markdown, in this order)
### What the document explicitly says
- Bullet list. Each bullet: claim + verbatim quote + citation id(s).
- If nothing is stated: "Not stated in the retrieved passages."

### Citations
- For each used id: `[id] p.<page> — "verbatim quote"`

### Possible interpretation / concern
- Clearly marked as interpretation only, not the document’s words.
- If none: "None from the retrieved passages; a qualified lawyer should still review the full document."

### Uncertainty / ambiguity
- Gaps, conflicts, undefined terms, low-confidence OCR.

### What this tool cannot conclude
- Any legal conclusion the user asked for that you must refuse, plus:
  "A qualified legal professional must review this."

### Disclaimer
This is not legal advice – consult a qualified lawyer.
"""


def build_user_prompt(
    question: str,
    passages: list[dict],
    document_name: str = "uploaded_document",
) -> str:
    """Build the user message. `passages` items need id, page, text."""
    blocks = []
    for p in passages:
        pid = p.get("id", "?")
        page = p.get("page", "?")
        start = p.get("start_char", "?")
        text = (p.get("text") or "").strip()
        blocks.append(
            f'<passage id="{pid}" page="{page}" start_char="{start}">\n{text}\n</passage>'
        )
    joined = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
    return (
        f"Document name: {document_name}\n\n"
        f"<user_question>\n{question.strip()}\n</user_question>\n\n"
        f"<document_passages>\n{joined}\n</document_passages>\n\n"
        "Answer using ONLY these passages. Separate explicit statements from "
        "interpretation. Cite every claim. Refuse legal conclusions. End with "
        f"{LEGAL_DISCLAIMER}"
    )


REFUSAL_NO_DOCUMENT = (
    "### What the document explicitly says\n"
    "No document is loaded, so nothing can be quoted.\n\n"
    "### Citations\n"
    "None.\n\n"
    "### Possible interpretation / concern\n"
    "None. Upload a PDF or text file on this device first.\n\n"
    "### Uncertainty / ambiguity\n"
    "There is no source text to inspect.\n\n"
    "### What this tool cannot conclude\n"
    "It cannot give legal advice or analyze a document that has not been provided.\n\n"
    "### Disclaimer\n"
    f"{LEGAL_DISCLAIMER}"
)

REFUSAL_EMPTY_RETRIEVAL = (
    "### What the document explicitly says\n"
    "Not stated in the retrieved passages (no matching excerpts were found).\n\n"
    "### Citations\n"
    "None.\n\n"
    "### Possible interpretation / concern\n"
    "Absence of a retrieved passage is not proof that the full document omits the topic. "
    "A qualified lawyer should review the entire file.\n\n"
    "### Uncertainty / ambiguity\n"
    "Retrieval returned no grounded excerpts for this question.\n\n"
    "### What this tool cannot conclude\n"
    "No legal conclusion is possible without citable text.\n\n"
    "### Disclaimer\n"
    f"{LEGAL_DISCLAIMER}"
)
