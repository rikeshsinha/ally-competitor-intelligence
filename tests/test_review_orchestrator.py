import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chains.review_orchestrator import classify_review_followup


class _DummyCompletions:
    def __init__(self, payload, *, raises=False):
        self._payload = payload
        self._raises = raises

    def create(self, **_):
        if self._raises:
            raise RuntimeError("boom")
        return type(
            "Resp",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {"content": self._payload},
                            )()
                        },
                    )()
                ]
            },
        )()


class _DummyChat:
    def __init__(self, payload, *, raises=False):
        self.completions = _DummyCompletions(payload, raises=raises)


class _DummyClient:
    def __init__(self, payload, *, raises=False):
        self.chat = _DummyChat(payload, raises=raises)


def test_heuristic_generate_edits():
    action = classify_review_followup("summary", "Yes, please generate the edits now.")
    assert action == "generate_edits"


def test_heuristic_select_competitor():
    action = classify_review_followup("summary", "Let's pick a different competitor brand.")
    assert action == "select_competitor"


def test_heuristic_stop():
    action = classify_review_followup("summary", "No thanks, that's all")
    assert action == "stop"


def test_heuristic_clarify_for_question():
    action = classify_review_followup("summary", "What else do you need?")
    assert action == "clarify"


def test_llm_result_overrides_heuristic():
    payload = json.dumps({"action": "select_competitor"})
    client = _DummyClient(payload)
    action = classify_review_followup("summary", "Maybe", client=client)
    assert action == "select_competitor"


def test_llm_failure_falls_back_to_heuristics():
    client = _DummyClient("{}", raises=True)
    action = classify_review_followup("summary", "Yes, go ahead", client=client)
    assert action == "generate_edits"


def test_llm_invalid_action_uses_heuristic():
    payload = json.dumps({"action": "unknown"})
    client = _DummyClient(payload)
    action = classify_review_followup("summary", "Yes, please", client=client)
    assert action == "generate_edits"
