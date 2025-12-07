"""Rule extraction chain for parsing PDF style guides into rule configs."""

from __future__ import annotations

import copy
import io
import json
import os
import math
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:  # Optional dependency; handled gracefully if unavailable
    from langchain_community.document_loaders import PyPDFLoader  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    try:  # Backwards compatibility with older langchain package layout
        from langchain.document_loaders import PyPDFLoader  # type: ignore
    except Exception:  # pragma: no cover - optional dependency guard
        PyPDFLoader = None  # type: ignore

try:  # Optional dependency; handled gracefully if unavailable
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    RecursiveCharacterTextSplitter = None  # type: ignore

try:  # Secondary fallback when langchain PDF loader is missing
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    PdfReader = None  # type: ignore

try:  # Streamlit is only available at runtime inside the app
    import streamlit as st
except Exception:  # pragma: no cover - optional dependency guard
    st = None  # type: ignore

try:  # Optional OpenAI dependency – same guard pattern as in app.py
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = None  # type: ignore

try:  # Optional sentence-transformers embedder
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    SentenceTransformer = None  # type: ignore

from langchain_core.runnables import RunnableLambda

from core.content_rules import DEFAULT_RULES

# Avoid importing Streamlit; accept generic uploaded content types instead.
UploadedContent = Union[bytes, "UploadedFileLike"]


@dataclass
class RuleExtraction:
    """Structured payload returned after rule extraction."""

    rules: Dict[str, Any]
    messages: List[str]
    source: str = "Built-in defaults"


class UploadedFileLike:  # pragma: no cover - lightweight protocol substitute
    """Duck-typed minimal interface for Streamlit's UploadedFile."""

    def getvalue(self) -> bytes:  # noqa: D401 - documentation via class docstring
        raise NotImplementedError

    def read(self) -> bytes:  # noqa: D401
        raise NotImplementedError


RULE_PROMPT = """You are an expert Amazon style guide analyst.
Read the provided excerpt from a Pet Supplies style guide and infer the critical
rules for PDP content quality checks.
Return a compact JSON object with exactly this structure:
{
  "title": {
    "max_chars": <int>,
    "brand_required": <bool>,
    "capitalize_words": <bool>,
    "no_all_caps": <bool>,
    "no_promo": <bool>,
    "allow_pack_of": <bool>
  },
  "bullets": {
    "max_count": <int>,
    "start_capital": <bool>,
    "sentence_fragments": <bool>,
    "no_end_punct": <bool>,
    "no_promo_or_seller_info": <bool>,
    "numbers_as_numerals": <bool>,
    "semicolons_ok": <bool>
  },
  "description": {
    "max_chars": <int>,
    "no_promo": <bool>,
    "no_seller_info": <bool>,
    "sentence_caps": <bool>,
    "truthful_claims": <bool>
  },
  "images": {
    "min_count": <int>,
    "preferred_min_px": <int>,
    "white_bg_required": <bool>,
    "no_text_watermarks": <bool>
  }
}
Use integers for counts/limits and booleans for true/false statements. If a rule
is not explicitly covered, make a reasonable assumption based on Amazon Pet
Supplies best practices.
Respond with JSON only."""


def _read_uploaded_content(file_obj: UploadedContent) -> bytes:
    """Return raw bytes from either a Streamlit upload or direct byte input."""
    if isinstance(file_obj, bytes):
        return file_obj
    for attr in ("getvalue", "read"):
        if hasattr(file_obj, attr):
            try:
                data = getattr(file_obj, attr)()
                if isinstance(data, bytes) and data:
                    return data
            except Exception:
                continue
    return b""


def _extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[List[str], List[str]]:
    """Best-effort PDF text extraction with langchain/PyPDF fallbacks."""
    errors: List[str] = []
    if not pdf_bytes:
        errors.append("Uploaded PDF is empty or unreadable")
        return [], errors

    # Preferred path: use langchain loader + splitter when available.
    if PyPDFLoader is not None and RecursiveCharacterTextSplitter is not None:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            if not docs:
                errors.append("No text extracted from PDF")
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1800, chunk_overlap=200
                )
                combined = "\n\n".join(
                    doc.page_content for doc in docs if doc.page_content
                )
                if combined.strip():
                    chunks = splitter.split_text(combined)
                    return chunks, errors
                errors.append("PDF text content is empty")
        except Exception as exc:  # pragma: no cover - depends on runtime PDFs
            errors.append(f"Langchain PDF parse failed: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    else:
        errors.append("langchain PDF loader not available; using basic parser")

    # Fallback: use PyPDF to extract raw text if langchain path unavailable/failed.
    if PdfReader is None:
        errors.append("PyPDF fallback unavailable; using default rules")
        return [], errors

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for idx, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - depends on PDF content
                errors.append(f"Failed to read page {idx + 1}: {exc}")
                page_text = ""
            if page_text:
                pages_text.append(page_text)
        combined = "\n\n".join(pages_text).strip()
        if not combined:
            errors.append("PDF text content is empty")
            return [], errors

        if RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1800, chunk_overlap=200
            )
            chunks = splitter.split_text(combined)
        else:
            chunk_size = 1800
            chunks = [
                combined[i : i + chunk_size]
                for i in range(0, len(combined), chunk_size)
            ]

        return chunks, errors
    except Exception as exc:  # pragma: no cover - depends on runtime PDFs
        errors.append(f"Failed to parse PDF with PyPDF: {exc}")
        return [], errors


def _coerce_bool(value: Any, default: bool) -> bool:
    """Convert loosely-typed values into booleans with a default fallback."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "y"}:
            return True
        if lowered in {"false", "no", "0", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_int(value: Any, default: int) -> int:
    """Cast a value to int, returning the default on failure."""
    try:
        return int(value)
    except Exception:
        return default


def _merge_rules(
    default_rules: Dict[str, Any], candidate: Dict[str, Any]
) -> Dict[str, Any]:
    """Overlay parsed rule data on top of defaults while preserving types."""
    merged = copy.deepcopy(default_rules)
    for section, defaults in default_rules.items():
        cand_section = candidate.get(section)
        if not isinstance(defaults, dict) or not isinstance(cand_section, dict):
            continue
        for key, default_value in defaults.items():
            if isinstance(default_value, bool):
                merged[section][key] = _coerce_bool(
                    cand_section.get(key), default_value
                )
            elif isinstance(default_value, int):
                merged[section][key] = _coerce_int(cand_section.get(key), default_value)
            else:
                merged[section][key] = cand_section.get(key, default_value)
    return merged


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors with graceful fallbacks."""

    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _select_context_chunks(
    chunks: List[str], embedder: Optional[Callable[[List[str]], List[List[float]]]], errors: List[str]
) -> List[str]:
    """Return the most relevant chunks for domain queries using embeddings."""

    if not embedder or not chunks:
        return []

    domain_queries = [
        "rules for bullet points",
        "title content requirements",
        "product description guidelines",
        "image requirements and standards",
    ]

    try:
        chunk_embeddings = embedder(chunks)
        query_embeddings = embedder(domain_queries)
    except Exception as exc:  # pragma: no cover - embedder runtime dependent
        errors.append(f"Embedding generation failed: {exc}")
        return []

    if not chunk_embeddings or not query_embeddings:
        return []

    scored: Dict[int, float] = {}
    for query_emb in query_embeddings:
        best_idx: Optional[int] = None
        best_score = -1.0
        for idx, chunk_emb in enumerate(chunk_embeddings):
            score = _cosine_similarity(query_emb, chunk_emb)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None:
            scored[best_idx] = max(best_score, scored.get(best_idx, -1.0))

    ranked_indices = [
        idx for idx, _ in sorted(scored.items(), key=lambda item: item[1], reverse=True)
    ]
    return [chunks[idx] for idx in ranked_indices[:5]]


def extract_rules_config(
    uploaded: UploadedContent,
    llm_client: Any,
    default_rules: Dict[str, Any],
    embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Parse a PDF and synthesize rule configuration."""

    errors: List[str] = []
    pdf_bytes = _read_uploaded_content(uploaded)
    if not pdf_bytes:
        errors.append("Could not read uploaded PDF bytes")
        return copy.deepcopy(default_rules), errors

    chunks, parse_errors = _extract_text_from_pdf(pdf_bytes)
    errors.extend(parse_errors)
    if not chunks:
        return copy.deepcopy(default_rules), errors

    selected_chunks = _select_context_chunks(chunks, embedder, errors)
    context_chunks = selected_chunks if selected_chunks else chunks[:5]
    context = "\n\n".join(context_chunks)

    if llm_client is None:
        errors.append("LLM client unavailable; falling back to default rules")
        return copy.deepcopy(default_rules), errors

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You convert style guides into machine-readable JSON.",
                },
                {
                    "role": "user",
                    "content": f"{RULE_PROMPT}\n\nGuide excerpt:\n{context}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        if not response.choices:
            raise ValueError("LLM returned no choices")
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty response")
        parsed = json.loads(content)
    except Exception as exc:  # pragma: no cover - network/LLM dependent
        errors.append(f"LLM extraction failed: {exc}")
        return copy.deepcopy(default_rules), errors

    if not isinstance(parsed, dict):
        errors.append("LLM did not return a JSON object")
        return copy.deepcopy(default_rules), errors

    merged = _merge_rules(default_rules, parsed)
    return merged, errors


class _RuleExtractorRunnable:
    """LangChain runnable wrapper that orchestrates rule extraction."""

    def __init__(self, default_rules: Optional[Dict[str, Any]] = None):
        self._default_rules = copy.deepcopy(default_rules or DEFAULT_RULES)

    def __call__(self, inputs: Dict[str, Any]) -> RuleExtraction:  # type: ignore[override]
        rules_file = inputs.get("rules_file") if isinstance(inputs, dict) else None
        llm_client = self._get_llm_client()
        embedder = self._get_embedder()
        rules, messages = extract_rules_config(
            rules_file, llm_client, self._default_rules, embedder
        )

        # Determine the rules provenance for downstream display.
        using_defaults = rules == self._default_rules
        if rules_file is None:
            source = "Built-in defaults"
        elif using_defaults:
            source = "Uploaded rules (fallback to defaults)"
        else:
            source = "Uploaded rules"

        return RuleExtraction(rules=rules, messages=list(messages), source=source)

    def _get_api_key(self) -> Optional[str]:
        """Return the OpenAI API key from Streamlit or environment."""

        if st is not None:
            return st.session_state.get("OPENAI_API_KEY_UI") or os.getenv(
                "OPENAI_API_KEY"
            )
        return os.getenv("OPENAI_API_KEY")

    def _get_llm_client(self) -> Any:
        """Best-effort construction of an OpenAI client, mirroring app logic."""

        if OpenAI is None:
            return None

        api_key = self._get_api_key()
        if not api_key:
            return None

        try:
            return OpenAI(api_key=api_key)
        except Exception:
            return None

    def _get_embedder(self) -> Optional[Callable[[List[str]], List[List[float]]]]:
        """Return an embedding callable if optional dependencies permit."""

        if SentenceTransformer is not None:
            try:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                return model.encode
            except Exception:
                pass

        if OpenAI is None:
            return None

        api_key = self._get_api_key()
        if not api_key:
            return None

        try:
            client = OpenAI(api_key=api_key)
        except Exception:
            return None

        def _embed(texts: List[str]) -> List[List[float]]:
            response = client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            return [data.embedding for data in response.data]

        return _embed


def create_rule_extractor() -> RunnableLambda:
    """Return a runnable that loads rules and returns a :class:`RuleExtraction`."""

    return RunnableLambda(_RuleExtractorRunnable())


__all__ = ["RuleExtraction", "extract_rules_config", "create_rule_extractor"]
