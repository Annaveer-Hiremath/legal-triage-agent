"""Streamlit UI — fully on-device. Permanent disclaimer. Data never leaves this machine."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import streamlit as st

from src.gemma_agent import DEFAULT_MODEL, GemmaTriageAgent
from src.ingestion import extract_bytes
from src.prompts import DISCLAIMER_BLOCK, LEGAL_DISCLAIMER
from src.triage import ClauseReport, FieldStatus, STATUS_EMOJI

# ─────────────────────── CSS ──────────────────────────────────────


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Sidebar privacy badge */
        .privacy-badge {
            background: linear-gradient(135deg, #0d4f2c 0%, #1a7a3c 100%);
            border-radius: 10px;
            padding: 12px 16px;
            color: #e8f8ee;
            font-size: 0.82rem;
            margin-bottom: 12px;
            border: 1px solid #2ecc71;
        }
        .privacy-badge b { color: #2ecc71; }

        /* Fact section — blue left border */
        .fact-block {
            border-left: 4px solid #1a73e8;
            background: #f0f6ff;
            padding: 10px 14px;
            border-radius: 0 8px 8px 0;
            margin: 8px 0;
        }

        /* Interpretation section — amber left border */
        .interp-block {
            border-left: 4px solid #f9a825;
            background: #fffde7;
            padding: 10px 14px;
            border-radius: 0 8px 8px 0;
            margin: 8px 0;
        }

        /* Clause card row */
        .clause-row {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 8px 12px;
            border-radius: 8px;
            margin: 4px 0;
            font-size: 0.88rem;
            border: 1px solid rgba(0,0,0,0.08);
        }
        .clause-found   { background: #f0fdf4; }
        .clause-ambig   { background: #fffbeb; }
        .clause-missing { background: #fff1f0; }
        .clause-label   { font-weight: 600; min-width: 200px; }

        /* Metric pill */
        .metric-pill {
            display: inline-block;
            background: #f1f3f4;
            border-radius: 12px;
            padding: 2px 10px;
            font-size: 0.78rem;
            color: #555;
            margin-right: 6px;
        }

        /* Disclaimer pin at bottom */
        .disclaimer-footer {
            border-top: 2px solid #d32f2f;
            background: #fff8f8;
            padding: 10px 14px;
            border-radius: 0 0 8px 8px;
            color: #b71c1c;
            font-size: 0.82rem;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────── helpers ─────────────────────────────────


def _agent() -> GemmaTriageAgent:
    if "agent" not in st.session_state:
        st.session_state.agent = GemmaTriageAgent()
    return st.session_state.agent


def _clause_css_class(status: FieldStatus) -> str:
    return {
        FieldStatus.FOUND: "clause-found",
        FieldStatus.AMBIGUOUS: "clause-ambig",
        FieldStatus.NOT_FOUND: "clause-missing",
    }[status]


def _render_clause_report(report: ClauseReport) -> None:
    st.markdown("#### 📋 Document Clause Overview")
    st.caption(
        "Auto-extracted via regex — no LLM cost. "
        "🟢 Found · 🟡 Ambiguous · 🔴 Not found. "
        "This is **not legal advice**."
    )
    for f in report.all_fields():
        css = _clause_css_class(f.status)
        excerpt_html = ""
        if f.excerpts:
            q = f.excerpts[0].replace("<", "&lt;").replace(">", "&gt;")
            excerpt_html = f'<span style="color:#444;font-style:italic">"{q}"</span>'
        elif f.status == FieldStatus.NOT_FOUND:
            excerpt_html = '<span style="color:#999">Not found in document text</span>'
        st.markdown(
            f'<div class="clause-row {css}">'
            f'<span class="clause-label">{f.emoji} {f.label}</span>'
            f"{excerpt_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
    risk = report.risk_summary()
    if "Not found" in risk or "Ambiguous" in risk:
        st.warning(f"⚠️ {risk}")


def _render_answer(answer: str) -> None:
    """Render with fact/interpretation visual separation."""
    lines = answer.split("\n")
    in_fact = False
    in_interp = False
    buffer: list[str] = []

    def flush() -> None:
        nonlocal in_fact, in_interp, buffer
        if buffer:
            content = "\n".join(buffer).strip()
            if in_fact:
                st.markdown(
                    f'<div class="fact-block">{content}</div>',
                    unsafe_allow_html=True,
                )
            elif in_interp:
                st.markdown(
                    f'<div class="interp-block">{content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(content)
        buffer = []

    for line in lines:
        lower = line.lower().strip()
        if "what the document explicitly says" in lower:
            flush()
            in_fact = True
            in_interp = False
            buffer.append(line)
        elif "possible interpretation" in lower or "possible concern" in lower:
            flush()
            in_fact = False
            in_interp = True
            buffer.append(line)
        elif lower.startswith("### "):
            flush()
            in_fact = False
            in_interp = False
            buffer.append(line)
        else:
            buffer.append(line)
    flush()


def _ram_info() -> str:
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        used_gb = (mem.total - mem.available) / 1e9
        total_gb = mem.total / 1e9
        return f"RAM {used_gb:.1f} / {total_gb:.1f} GB"
    except ImportError:
        return "RAM: psutil not installed"


# ─────────────────────── main ─────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Legal Triage Agent — On-Device",
        layout="wide",
        page_icon="⚖️",
    )
    _inject_css()

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="privacy-badge">'
            "<b>🔒 PRIVACY VERIFIED</b><br>"
            "✅ Wi-Fi-off safe<br>"
            "✅ No cloud calls<br>"
            "✅ Localhost Ollama only<br>"
            "✅ No telemetry<br>"
            "✅ No embeddings sent to cloud"
            "</div>",
            unsafe_allow_html=True,
        )
        st.subheader("Local Runtime")
        st.write(f"🤖 Model: `{DEFAULT_MODEL}`")
        st.write("🔗 Ollama: `http://127.0.0.1:11434`")
        st.write("🗂️ Retriever: In-memory TF-IDF (no model download)")
        st.write(f"💾 {_ram_info()}")
        st.markdown("---")
        st.markdown(
            "**How to use:**\n"
            "1. Upload a **PDF** or **.txt** contract\n"
            "2. Review the **Clause Overview** panel\n"
            "3. Ask a question grounded in the document\n"
            "4. Read **explicit statements** (🔵) and **interpretations** (🟡) separately\n"
            "5. **Consult a qualified lawyer** for any action"
        )
        st.markdown("---")
        st.caption("Powered by Google Gemma via local Ollama · MIT licence")

    # ── Header ───────────────────────────────────────────────────
    st.title("⚖️ Legal Document Triage Agent")
    st.caption(
        "Fully on-device · Google Gemma via local Ollama · "
        "**Not a lawyer — not legal advice**"
    )
    st.error(DISCLAIMER_BLOCK)

    # ── Upload ───────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a document *(processed only in this session, on this device)*",
        type=["pdf", "txt", "md"],
        help="The file bytes are processed entirely in local memory. Nothing is sent anywhere.",
    )

    if uploaded is not None:
        file_key = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_loaded_key") != file_key:
            data = uploaded.getvalue()
            with st.spinner("Extracting text locally…"):
                doc = extract_bytes(data, uploaded.name)
                n_chunks = _agent().load_document(doc)
                report = _agent().extract_clauses_from_doc()
            st.session_state._loaded_key = file_key
            st.session_state.doc_name = doc.name
            st.session_state.n_chunks = n_chunks
            st.session_state.n_pages = len(doc.pages)
            st.session_state.clause_report = report
            st.session_state.doc_warnings = list(doc.warnings)
            # Clear previous Q&A when new doc is loaded
            st.session_state.pop("last_result", None)

        # Show doc metadata pills
        n_chunks = st.session_state.get("n_chunks", 0)
        n_pages = st.session_state.get("n_pages", 0)
        st.markdown(
            f'<span class="metric-pill">📄 {st.session_state.doc_name}</span>'
            f'<span class="metric-pill">📑 {n_pages} page(s)</span>'
            f'<span class="metric-pill">🧩 {n_chunks} chunk(s)</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        for w in st.session_state.get("doc_warnings", []):
            st.warning(w)

        # Clause overview
        report: ClauseReport | None = st.session_state.get("clause_report")
        if report is not None:
            with st.expander("📋 Document Clause Overview (auto-extracted)", expanded=True):
                _render_clause_report(report)

        # Raw text preview
        with st.expander("🔍 Extracted text preview (stays on device)", expanded=False):
            agent = _agent()
            if agent._doc:
                preview = agent._doc.text[:4000]
                st.text(preview + ("…" if len(agent._doc.text) > 4000 else ""))

    # ── Q&A ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Ask the Document")

    question = st.text_area(
        "Question about the uploaded document",
        placeholder=(
            "e.g. What does the document say about the confidentiality period?\n"
            "e.g. What are the termination conditions?\n"
            "e.g. What penalties does the document mention?"
        ),
        height=110,
    )

    col_btn, col_tip = st.columns([1, 3])
    with col_btn:
        run = st.button("⚡ Triage locally", type="primary", use_container_width=True)
    with col_tip:
        st.caption(
            "Answers are grounded only in document passages. "
            "The agent cannot give legal advice."
        )

    if run:
        if uploaded is None:
            st.warning("Upload a local PDF or text file first.")
        elif not question.strip():
            st.warning("Enter a question grounded in the document.")
        else:
            with st.spinner(f"Querying local {DEFAULT_MODEL}…"):
                t_start = time.perf_counter()
                try:
                    result = _agent().ask(question.strip())
                    elapsed = time.perf_counter() - t_start
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        f"Local Ollama call failed. Confirm `ollama serve` is running "
                        f"and `{DEFAULT_MODEL}` is pulled.\n\n{exc}"
                    )
                    st.stop()
            result._elapsed = elapsed  # type: ignore[attr-defined]
            st.session_state.last_result = result

    result = st.session_state.get("last_result")
    if result is not None:
        elapsed = getattr(result, "_elapsed", None)
        if elapsed is not None:
            st.markdown(
                f'<span class="metric-pill">⏱️ Inference: {elapsed:.1f}s</span>'
                f'<span class="metric-pill">🤖 {result.model}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

        st.markdown("### 📄 Analysis")
        _render_answer(result.answer)

        st.markdown(
            '<div class="disclaimer-footer">'
            f"⚠️ {LEGAL_DISCLAIMER}"
            "</div>",
            unsafe_allow_html=True,
        )

        # Citation cards
        if result.citations:
            with st.expander(
                f"📎 Retrieved source passages ({len(result.citations)})", expanded=False
            ):
                for p in result.citations:
                    score = p.get("score", 0)
                    score_pct = int(score * 100)
                    with st.expander(
                        f"[{p['id']}] Page {p['page']} · Relevance {score_pct}%",
                        expanded=False,
                    ):
                        st.markdown(
                            f"**Document:** {p.get('document_name', 'N/A')}  \n"
                            f"**Chars:** {p.get('start_char', '?')} – {p.get('end_char', '?')}"
                        )
                        st.code(p["text"], language=None)

        if result.warnings:
            st.caption("⚠️ Ingestion warnings: " + "; ".join(result.warnings))


if __name__ == "__main__":
    main()
