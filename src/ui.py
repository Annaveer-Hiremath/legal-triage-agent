"""Streamlit UI. Permanent disclaimer. All work stays on this device."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import streamlit as st

from src.gemma_agent import DEFAULT_MODEL, GemmaTriageAgent
from src.ingestion import extract_bytes
from src.prompts import DISCLAIMER_BLOCK, LEGAL_DISCLAIMER


def _agent() -> GemmaTriageAgent:
    if "agent" not in st.session_state:
        st.session_state.agent = GemmaTriageAgent()
    return st.session_state.agent


def main() -> None:
    st.set_page_config(page_title="Legal Document Triage (On-Device)", layout="wide")
    st.title("Legal Document Triage Agent")
    st.caption("Fully on-device · Google Gemma via local Ollama · not a lawyer")

    st.error(DISCLAIMER_BLOCK)
    st.markdown(f"**{LEGAL_DISCLAIMER}**")

    with st.sidebar:
        st.subheader("Local runtime")
        st.write(f"Model: `{DEFAULT_MODEL}`")
        st.write("Ollama: `http://127.0.0.1:11434`")
        st.write("Retriever: in-memory lexical (default)")
        st.info(
            "Private file content, chunks, and answers never leave this machine. "
            "No remote LLM, OCR, or embedding APIs."
        )
        st.markdown(
            "- Upload a **PDF** or **.txt**\n"
            "- Ask what the text **explicitly says**\n"
            "- Treat **interpretation** as non-authoritative\n"
            "- **Consult a qualified lawyer** for advice"
        )

    uploaded = st.file_uploader(
        "Upload a document (processed only in this session, on this device)",
        type=["pdf", "txt", "md"],
    )

    if uploaded is not None:
        data = uploaded.getvalue()
        doc = extract_bytes(data, uploaded.name)
        n_chunks = _agent().load_document(doc)
        st.session_state.doc_name = doc.name
        st.success(
            f"Loaded **{doc.name}** · {len(doc.pages)} page(s) · {n_chunks} local chunk(s)."
        )
        for w in doc.warnings:
            st.warning(w)
        with st.expander("Extracted text preview (stays on device)", expanded=False):
            preview = doc.text[:4000]
            st.text(preview + ("…" if len(doc.text) > 4000 else ""))

    question = st.text_area(
        "Question about the uploaded document",
        placeholder="Example: What does the document say about the confidentiality period?",
        height=100,
    )

    if st.button("Triage locally", type="primary"):
        if uploaded is None:
            st.warning("Upload a local PDF or text file first.")
        elif not question.strip():
            st.warning("Enter a question grounded in the document.")
        else:
            with st.spinner(f"Querying local {DEFAULT_MODEL}…"):
                try:
                    result = _agent().ask(question.strip())
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        "Local Ollama call failed. Confirm `ollama serve` is running "
                        f"and `{DEFAULT_MODEL}` is pulled. {exc}"
                    )
                    return
            st.session_state.last_result = result

    result = st.session_state.get("last_result")
    if result is not None:
        st.markdown(result.answer)
        st.markdown("---")
        st.markdown(f"**{LEGAL_DISCLAIMER}**")
        with st.expander("Retrieved passages used as grounding", expanded=False):
            for p in result.citations:
                st.markdown(
                    f"**[{p['id']}]** page {p['page']} · score {p.get('score', 0):.3f}"
                )
                st.text(p["text"])
        if result.warnings:
            st.caption("Ingestion warnings: " + "; ".join(result.warnings))


if __name__ == "__main__":
    main()
