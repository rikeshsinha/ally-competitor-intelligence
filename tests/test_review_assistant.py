import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chains.review_assistant import classify_review_followup


class _DummyCompletions:
    def __init__(self, payload, *, raises=False):
        self._payload = payload
        self._raises = raises
        self.last_kwargs = None

    def create(self, **kwargs):
        if self._raises:
            raise RuntimeError("boom")
        self.last_kwargs = kwargs
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
    action = classify_review_followup(
        "summary", "Let's pick a different competitor brand."
    )
    assert action == "select_competitor"


def test_heuristic_stop():
    action = classify_review_followup("summary", "No thanks, that's all")
    assert action == "stop"


def test_heuristic_answer_question_for_question():
    action = classify_review_followup("summary", "What else do you need?")
    assert action == "answer_question"


def test_heuristic_answer_question_for_rule_request():
    action = classify_review_followup(
        "summary",
        "Please share the bullet rules again.",
    )
    assert action == "answer_question"


def test_llm_result_overrides_heuristic():
    payload = json.dumps({"action": "select_competitor"})
    client = _DummyClient(payload)
    action = classify_review_followup("summary", "Maybe", client=client)
    assert action == "select_competitor"


def test_llm_answer_question_override():
    payload = json.dumps({"action": "clarify"})
    client = _DummyClient(payload)
    action = classify_review_followup("summary", "What are the rules?", client=client)
    assert action == "answer_question"


def test_llm_failure_falls_back_to_heuristics():
    client = _DummyClient("{}", raises=True)
    action = classify_review_followup("summary", "Yes, go ahead", client=client)
    assert action == "generate_edits"


def test_llm_invalid_action_uses_heuristic():
    payload = json.dumps({"action": "unknown"})
    client = _DummyClient(payload)
    action = classify_review_followup("summary", "Yes, please", client=client)
    assert action == "generate_edits"


def test_llm_failure_falls_back_to_answer_question():
    client = _DummyClient("{}", raises=True)
    action = classify_review_followup(
        "summary",
        "What is the bullet limit?",
        client=client,
    )
    assert action == "answer_question"


def test_llm_payload_includes_additional_products():
    payload = json.dumps({"action": "clarify"})
    client = _DummyClient(payload)
    extra_products = [{"brand": "Brand B", "row_index": 5}]
    classify_review_followup(
        "summary",
        "Maybe",
        client=client,
        additional_products=extra_products,
    )
    kwargs = client.chat.completions.last_kwargs
    assert kwargs is not None
    message_payload = json.loads(kwargs["messages"][1]["content"])
    assert message_payload["additional_products"] == extra_products


def test_llm_payload_omits_additional_products_when_missing():
    payload = json.dumps({"action": "clarify"})
    client = _DummyClient(payload)
    classify_review_followup(
        "summary",
        "Maybe",
        client=client,
    )
    kwargs = client.chat.completions.last_kwargs
    assert kwargs is not None
    message_payload = json.loads(kwargs["messages"][1]["content"])
    assert "additional_products" not in message_payload
