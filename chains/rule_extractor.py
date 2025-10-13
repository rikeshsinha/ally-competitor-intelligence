"""Rule extraction chain for parsing PDF style guides into rule configs."""

from __future__ import annotations

import copy
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Tuple, Union

try:  # Optional dependency; handled gracefully if unavailable
    from langchain.document_loaders import PyPDFLoader  # type: ignore
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    PyPDFLoader = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore

try:  # Secondary fallback when langchain PDF loader is missing
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    PdfReader = None  # type: ignore

# Avoid importing Streamlit; accept generic uploaded content types instead.
UploadedContent = Union[bytes, "UploadedFileLike"]


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


def _extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[str, List[str]]:
    errors: List[str] = []
    if not pdf_bytes:
        errors.append("Uploaded PDF is empty or unreadable")
        return "", errors

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
                splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200)
                combined = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
                if combined.strip():
                    chunks = splitter.split_text(combined)
                    context = "\n\n".join(chunks[:5])
                    return context, errors
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
        return "", errors

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
            return "", errors

        if RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200)
            chunks = splitter.split_text(combined)
        else:
            chunk_size = 1800
            chunks = [combined[i : i + chunk_size] for i in range(0, len(combined), chunk_size)]

        context = "\n\n".join(chunks[:5])
        return context, errors
    except Exception as exc:  # pragma: no cover - depends on runtime PDFs
        errors.append(f"Failed to parse PDF with PyPDF: {exc}")
        return "", errors


def _coerce_bool(value: Any, default: bool) -> bool:
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
    try:
        return int(value)
    except Exception:
        return default


def _merge_rules(default_rules: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(default_rules)
    for section, defaults in default_rules.items():
        cand_section = candidate.get(section)
        if not isinstance(defaults, dict) or not isinstance(cand_section, dict):
            continue
        for key, default_value in defaults.items():
            if isinstance(default_value, bool):
                merged[section][key] = _coerce_bool(cand_section.get(key), default_value)
            elif isinstance(default_value, int):
                merged[section][key] = _coerce_int(cand_section.get(key), default_value)
            else:
                merged[section][key] = cand_section.get(key, default_value)
    return merged


def extract_rules_config(
    uploaded: UploadedContent,
    llm_client: Any,
    default_rules: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Parse a PDF and synthesize rule configuration."""

    errors: List[str] = []
    pdf_bytes = _read_uploaded_content(uploaded)
    if not pdf_bytes:
        errors.append("Could not read uploaded PDF bytes")
        return copy.deepcopy(default_rules), errors

    context, parse_errors = _extract_text_from_pdf(pdf_bytes)
    errors.extend(parse_errors)
    if not context:
        return copy.deepcopy(default_rules), errors

    if llm_client is None:
        errors.append("LLM client unavailable; falling back to default rules")
        return copy.deepcopy(default_rules), errors

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You convert style guides into machine-readable JSON."},
                {"role": "user", "content": f"{RULE_PROMPT}\n\nGuide excerpt:\n{context}"},
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


__all__ = ["extract_rules_config"]
