import inspect
from unittest.mock import MagicMock
import json

import pytest

from voice_agent.services.llm.base import (
    GenerationOptions,
    LLMContextBudgetExceededError,
    LLMDecodeError,
)
from voice_agent.services.llm.medgemma_llamacpp import MedGemmaLlamaCppService


def test_llamacpp_accepts_generation_options_parameter():
    signature = inspect.signature(MedGemmaLlamaCppService.generate)
    assert "options" in signature.parameters


def test_llamacpp_forwards_stop_sequences_from_options():
    service = MedGemmaLlamaCppService.__new__(MedGemmaLlamaCppService)
    service.config = MagicMock(max_new_tokens=128, n_ctx=2048, context_margin=64)
    captured = {}
    service._estimate_prompt_tokens = lambda _: 10  # type: ignore[method-assign]

    def mock_client(prompt, **kwargs):
        captured["kwargs"] = kwargs
        return {"choices": [{"text": "answer\nUser: trailing"}]}

    service.client = mock_client
    options = GenerationOptions(stop=["\nUser:"], max_new_tokens=33, temperature=0.2)

    response = service.generate("prompt", options=options)

    assert captured["kwargs"]["stop"] == ["\nUser:"]
    assert captured["kwargs"]["max_tokens"] == 33
    assert captured["kwargs"]["temperature"] == 0.2
    assert response == "answer"


def test_llamacpp_builds_and_forwards_json_schema_grammar(monkeypatch):
    service = MedGemmaLlamaCppService.__new__(MedGemmaLlamaCppService)
    service.config = MagicMock(max_new_tokens=128, n_ctx=2048, context_margin=64)
    service._estimate_prompt_tokens = lambda _: 20  # type: ignore[method-assign]
    captured: dict[str, object] = {}

    def mock_client(prompt, **kwargs):
        captured["kwargs"] = kwargs
        return {"choices": [{"text": '{"ok": true}'}]}

    class DummyGrammar:
        @staticmethod
        def from_json_schema(schema: str, verbose: bool = False):
            captured["schema"] = json.loads(schema)
            captured["verbose"] = verbose
            return "GRAMMAR_OBJECT"

    monkeypatch.setattr(
        "llama_cpp.LlamaGrammar",
        DummyGrammar,
    )
    service.client = mock_client

    response = service.generate(
        "prompt",
        options=GenerationOptions(json_schema={"type": "object", "properties": {}}),
    )

    assert response == '{"ok": true}'
    assert captured["schema"] == {"type": "object", "properties": {}}
    assert captured["kwargs"]["grammar"] == "GRAMMAR_OBJECT"


def test_llamacpp_uses_chat_completion_when_messages_provided():
    service = MedGemmaLlamaCppService.__new__(MedGemmaLlamaCppService)
    service.config = MagicMock(max_new_tokens=128, n_ctx=2048, context_margin=64)
    service._estimate_prompt_tokens = lambda _: 20  # type: ignore[method-assign]

    class ChatClient:
        def __call__(self, prompt, **kwargs):
            raise AssertionError("prompt completion path should not be called")

        def create_chat_completion(self, messages, **kwargs):
            assert messages[0]["role"] == "system"
            assert messages[-1]["content"] == "Current symptom"
            return {"choices": [{"message": {"content": "<question>Next question?</question>"}}]}

    service.client = ChatClient()
    response = service.generate(
        "fallback prompt",
        options=GenerationOptions(
            max_new_tokens=64,
            messages=[
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "Current symptom"},
            ],
        ),
    )

    assert "Next question?" in response


def test_llamacpp_maps_context_overflow_errors():
    service = MedGemmaLlamaCppService.__new__(MedGemmaLlamaCppService)
    service.config = MagicMock(max_new_tokens=128, n_ctx=2048, context_margin=64)
    service._estimate_prompt_tokens = lambda _: 5  # type: ignore[method-assign]

    def failing_client(*args, **kwargs):
        raise RuntimeError("Requested tokens (4097) exceed context window of 2048")

    service.client = failing_client

    with pytest.raises(LLMContextBudgetExceededError):
        service.generate("prompt", options=GenerationOptions(max_new_tokens=100))


def test_llamacpp_maps_decode_failures():
    service = MedGemmaLlamaCppService.__new__(MedGemmaLlamaCppService)
    service.config = MagicMock(max_new_tokens=128, n_ctx=2048, context_margin=64)
    service._estimate_prompt_tokens = lambda _: 5  # type: ignore[method-assign]

    def failing_client(*args, **kwargs):
        raise RuntimeError("llama_decode returned -1")

    service.client = failing_client

    with pytest.raises(LLMDecodeError):
        service.generate("prompt", options=GenerationOptions(max_new_tokens=100))
