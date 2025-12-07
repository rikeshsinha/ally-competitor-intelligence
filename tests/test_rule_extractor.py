import json
import types

import pytest

from chains import rule_extractor


class _DummyLLM:
    def __init__(self):
        self.captured_prompt = None
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, *, model, messages, response_format, temperature):  # noqa: D401 - signature parity
        self.captured_prompt = messages[-1]["content"]
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(rule_extractor.DEFAULT_RULES)
                    )
                )
            ]
        )


def _fake_embedder(texts):
    def _vector(text):
        lowered = text.lower()
        return [
            int("title" in lowered),
            int("bullet" in lowered),
            int("description" in lowered),
            int("image" in lowered),
            max(len(text), 1),
        ]

    return [_vector(text) for text in texts]


def test_embeddings_select_relevant_chunks(monkeypatch):
    chunks = [
        "General introduction",  # low relevance
        "Title must include brand and size",  # title
        "Bullets should be fragments without punctuation",  # bullets
        "Description should focus on benefits",  # description
        "Images require white background",  # images
    ]

    monkeypatch.setattr(
        rule_extractor, "_extract_text_from_pdf", lambda _: (chunks, [])
    )

    llm = _DummyLLM()
    rules, errors = rule_extractor.extract_rules_config(
        uploaded=b"stub",
        llm_client=llm,
        default_rules=rule_extractor.DEFAULT_RULES,
        embedder=_fake_embedder,
    )

    assert not errors
    assert rules == rule_extractor.DEFAULT_RULES

    prompt = llm.captured_prompt or ""
    assert "Title must include brand" in prompt
    assert "Bullets should be fragments" in prompt
    assert "Description should focus" in prompt
    assert "Images require white background" in prompt
    assert "General introduction" not in prompt


def test_no_embedding_falls_back_to_default_chunks(monkeypatch):
    chunks = ["One", "Two", "Three", "Four", "Five", "Six"]
    monkeypatch.setattr(
        rule_extractor, "_extract_text_from_pdf", lambda _: (chunks, [])
    )

    llm = _DummyLLM()
    rules, errors = rule_extractor.extract_rules_config(
        uploaded=b"stub",
        llm_client=llm,
        default_rules=rule_extractor.DEFAULT_RULES,
        embedder=None,
    )

    assert not errors
    assert rules == rule_extractor.DEFAULT_RULES

    prompt = llm.captured_prompt or ""
    for expected in chunks[:5]:
        assert expected in prompt
    assert "Six" not in prompt
