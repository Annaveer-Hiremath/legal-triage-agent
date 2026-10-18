"""Local Gemma via Ollama. Document content stays on this machine."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

from src.chunking import chunk_document
from src.ingestion import ExtractedDocument, extract_file
from src.prompts import (
    LEGAL_DISCLAIMER,
    REFUSAL_EMPTY_RETRIEVAL,
    REFUSAL_NO_DOCUMENT,
    SYSTEM_PROMPT,
    TRIAGE_EXTRACTION_PROMPT,
    build_extraction_prompt,
    build_user_prompt,
)
from src.retrieval import LocalStore
from src.triage import ClauseReport, extract_clauses

DEFAULT_MODEL = os.environ.get("LEGAL_TRIAGE_MODEL", "gemma2:2b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def _lock_localhost() -> None:
    """Refuse non-loopback Ollama hosts so document text cannot leave the device."""
    host = OLLAMA_HOST.lower().replace("http://", "").replace("https://", "")
    allowed = host.startswith("127.0.0.1") or host.startswith("localhost")
    if not allowed:
        raise RuntimeError(
            f"Refusing OLLAMA_HOST={OLLAMA_HOST!r}. Use http://127.0.0.1:11434 only."
        )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@dataclass
class TriageResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    warnings: list[str] = field(default_factory=list)


class GemmaTriageAgent:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST):
        _lock_localhost()
        self.model = model
        self.host = host if host.startswith("http") else f"http://{host}"
        if "127.0.0.1" not in self.host and "localhost" not in self.host:
            raise RuntimeError("Ollama host must be localhost.")
        self._store: LocalStore | None = None
        self._doc: ExtractedDocument | None = None

    def load_document(self, doc: ExtractedDocument) -> int:
        self._doc = doc
        chunks = chunk_document(doc)
        self._store = LocalStore(chunks)
        return len(chunks)

    def load_path(self, path: str) -> int:
        return self.load_document(extract_file(path))

    def extract_clauses_from_doc(self) -> ClauseReport | None:
        """Run regex-based clause extraction (zero LLM cost, instant)."""
        if self._doc is None:
            return None
        return extract_clauses(self._doc.text, document_name=self._doc.name)

    def ask_extraction(self) -> str:
        """
        One-shot LLM extraction pass on the loaded document.
        Returns raw model output (key: value lines) or an error string.
        Falls back gracefully if Ollama is unavailable.
        """
        if self._doc is None:
            return "No document loaded."
        user_prompt = build_extraction_prompt(self._doc.text, self._doc.name)
        try:
            raw = self._complete_with_system(TRIAGE_EXTRACTION_PROMPT, user_prompt)
            return raw
        except Exception as exc:  # noqa: BLE001
            return f"LLM extraction unavailable: {exc}"

    def ask(self, question: str, k: int = 6) -> TriageResult:
        if self._doc is None or self._store is None:
            return TriageResult(answer=REFUSAL_NO_DOCUMENT, warnings=["no_document"])

        hits = self._store.query(question, k=k)
        if not hits:
            return TriageResult(
                answer=REFUSAL_EMPTY_RETRIEVAL,
                warnings=list(self._doc.warnings) + ["empty_retrieval"],
            )

        passages = [h.as_dict() for h in hits]
        user_prompt = build_user_prompt(
            question=question,
            passages=passages,
            document_name=self._doc.name,
        )
        raw = self._complete(user_prompt)
        answer = _ensure_disclaimer(raw)
        return TriageResult(
            answer=answer,
            citations=passages,
            model=self.model,
            warnings=list(self._doc.warnings),
        )

    def _complete(self, user_prompt: str) -> str:
        return self._complete_with_system(SYSTEM_PROMPT, user_prompt)

    def _complete_with_system(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("The ollama package is required. pip install -r requirements.txt") from exc

        client = ollama.Client(host=self.host)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.1,
                "num_ctx": 4096,
            },
        )
        message = response.get("message") or {}
        return (message.get("content") or "").strip()


def _ensure_disclaimer(text: str) -> str:
    if LEGAL_DISCLAIMER.lower() in text.lower():
        return text
    return text.rstrip() + "\n\n### Disclaimer\n" + LEGAL_DISCLAIMER


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="On-device legal document triage (not legal advice)."
    )
    parser.add_argument("path", help="Local PDF or .txt")
    parser.add_argument("question", help="Question about the document")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    agent = GemmaTriageAgent(model=args.model)
    agent.load_path(args.path)
    result = agent.ask(args.question)
    sys.stdout.write(result.answer + "\n")
    if result.warnings:
        sys.stderr.write("warnings: " + "; ".join(result.warnings) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
