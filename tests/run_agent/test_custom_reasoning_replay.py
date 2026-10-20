"""Preserve real reasoning for configured endpoints, independent of model names."""

import copy
import json

import httpx
import pytest
import yaml
from openai import OpenAI

from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_cli.config import (
    _custom_provider_entry_to_provider_config,
    get_custom_provider_reasoning_replay,
)
from run_agent import AIAgent


BASE_URL = "http://inference.example:35996/v1"
MODELS = ["glm-5.3-flash", "kimi-k3", "deepseek-v4.1-flash", "aliased-model"]
REAL_REASONING = "先核对工具返回的两个数，再计算乘积。\n保持原始推理字节。"


def make_agent(model, base_url=BASE_URL, provider="custom"):
    agent = object.__new__(AIAgent)
    agent.provider, agent.model, agent.base_url = provider, model, base_url
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


@pytest.fixture
def configured_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CUSTOM_BASE_URL", BASE_URL)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "base_url": "${CUSTOM_BASE_URL}",
                  "reasoning_replay": "preserve", "context_length": 131072},
    }), encoding="utf-8")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tool_turn", [True, False])
def test_saved_reasoning_survives_both_request_passes(configured_endpoint, model, tool_turn):
    agent = make_agent(model)
    source = {"role": "assistant", "content": "", "reasoning_content": REAL_REASONING}
    if tool_turn:
        source["tool_calls"] = [{"id": "call_1", "type": "function",
                                 "function": {"name": "calculate", "arguments": "{}"}}]
    original = copy.deepcopy(source)
    wire = copy.deepcopy(source)
    agent._copy_reasoning_content_for_api(source, wire)
    agent._reapply_reasoning_echo_for_provider([wire])
    assert wire["reasoning_content"] == REAL_REASONING
    assert source == original
    assert agent._reapply_reasoning_echo_for_provider([wire]) == 0


@pytest.mark.parametrize("value", [None, "", 123])
def test_custom_preservation_does_not_invent_thinking(configured_endpoint, value):
    agent = make_agent("aliased-model")
    source = {"role": "assistant", "reasoning_content": value}
    wire = dict(source)
    agent._copy_reasoning_content_for_api(source, wire)
    agent._reapply_reasoning_echo_for_provider([wire])
    if value == "":
        assert wire["reasoning_content"] == ""
    else:
        assert "reasoning_content" not in wire


def test_legacy_normalized_reasoning_is_preserved(configured_endpoint):
    agent = make_agent("glm-5.3-flash")
    source = {"role": "assistant", "reasoning": REAL_REASONING,
              "tool_calls": [{"id": "call_1"}]}
    wire = {"role": "assistant"}
    agent._copy_reasoning_content_for_api(source, wire)
    assert wire["reasoning_content"] == REAL_REASONING


def test_switching_destination_rechecks_policy_and_keeps_saved_history(configured_endpoint):
    agent = make_agent("kimi-k3")
    source = {"role": "assistant", "reasoning_content": REAL_REASONING}
    assert agent._should_preserve_reasoning_content()
    agent.provider, agent.model, agent.base_url = "mistral", "mistral-small", "https://api.mistral.ai/v1"
    wire = dict(source)
    agent._reapply_reasoning_echo_for_provider([wire])
    assert "reasoning_content" not in wire
    assert source["reasoning_content"] == REAL_REASONING
    agent.provider, agent.model, agent.base_url = "custom", "kimi-k3", BASE_URL
    agent._copy_reasoning_content_for_api(source, wire)
    agent._reapply_reasoning_echo_for_provider([wire])
    assert wire["reasoning_content"] == REAL_REASONING


@pytest.mark.parametrize("legacy", [True, False])
def test_provider_configuration_survives_normalization_and_migration(legacy):
    entry = {"name": "self-hosted", "base_url": BASE_URL, "reasoning_replay": "preserve"}
    modern = _custom_provider_entry_to_provider_config(entry)
    assert modern["reasoning_replay"] == "preserve"
    config = {"custom_providers": [entry]} if legacy else {"providers": {"self-hosted": modern}}
    assert get_custom_provider_reasoning_replay(BASE_URL + "/", config) == "preserve"
    assert get_custom_provider_reasoning_replay(BASE_URL + "/other", config) == "auto"


def test_explicit_strict_override_wins_over_model_setting():
    config = {"model": {"base_url": BASE_URL, "reasoning_replay": "preserve"},
              "providers": {"strict": {"api": BASE_URL, "reasoning_replay": "strip"}}}
    assert get_custom_provider_reasoning_replay(BASE_URL, config) == "strip"


def test_provider_without_policy_does_not_override_model_capability():
    config = {"model": {"base_url": BASE_URL, "reasoning_replay": "preserve"},
              "providers": {"inference": {"api": BASE_URL}}}
    assert get_custom_provider_reasoning_replay(BASE_URL, config) == "preserve"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True])
def test_sdk_roundtrip_replays_original_reasoning_on_wire(configured_endpoint, model, stream):
    """Real SDK parsing, assistant storage, both replay passes and serialization."""
    captured = []

    def serve(request):
        body = json.loads(request.content)
        captured.append(body)
        if len(captured) == 1:
            response = {"id": "one", "object": "chat.completion", "created": 1, "model": model,
                        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                            "role": "assistant", "content": None, "reasoning_content": REAL_REASONING,
                            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                                "name": "calculate", "arguments": "{}"}}]}}]}
            return httpx.Response(200, json=response)
        assert body["messages"][1]["reasoning_content"] == REAL_REASONING
        assert body["messages"][2]["tool_call_id"] == "call_1"
        if stream:
            chunk = {"id": "two", "object": "chat.completion.chunk", "created": 1, "model": model,
                     "choices": [{"index": 0, "delta": {"content": "60"}, "finish_reason": None}]}
            return httpx.Response(200, text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "two", "object": "chat.completion", "created": 1,
            "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": "60"}}]})

    agent = make_agent(model)
    with OpenAI(api_key="test-only", base_url=BASE_URL,
                http_client=httpx.Client(transport=httpx.MockTransport(serve))) as client:
        history = [{"role": "user", "content": "Calculate the product."}]
        first = client.chat.completions.create(model=model, messages=history)
        stored = agent._build_assistant_message(first.choices[0].message, "tool_calls")
        # Simulate persisted-session restore rather than forwarding the SDK object.
        stored = json.loads(json.dumps(stored, ensure_ascii=False))
        wire = dict(stored)
        agent._copy_reasoning_content_for_api(stored, wire)
        for key in ("reasoning", "finish_reason"):
            wire.pop(key, None)
        history += [wire, {"role": "tool", "tool_call_id": "call_1", "content": "60"}]
        agent._reapply_reasoning_echo_for_provider(history)
        history = ChatCompletionsTransport().convert_messages(history, model=model)
        second = client.chat.completions.create(model=model, messages=history, stream=stream)
        if stream:
            assert "".join(c.choices[0].delta.content or "" for c in second) == "60"
        else:
            assert second.choices[0].message.content == "60"
    assert len(captured) == 2


@pytest.mark.parametrize("model", MODELS)
def test_real_conversation_loop_keeps_restored_reasoning(configured_endpoint, monkeypatch, model):
    """Exercise actual initialization and loop, mocking only the network/tool catalog."""
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda *a, **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda *a, **kw: {})
    # Model catalog discovery is outside the replay path and must stay offline.
    def no_catalog_network(*args, **kwargs):
        raise RuntimeError("offline model catalog")
    monkeypatch.setattr("requests.sessions.Session.request", no_catalog_network)
    captured = []

    def serve(request):
        body = json.loads(request.content)
        captured.append(body)
        assistants = [m for m in body["messages"] if m["role"] == "assistant"]
        assert any(m.get("reasoning_content") == REAL_REASONING for m in assistants)
        choice = {"index": 0, "delta": {"role": "assistant", "content": "60"}, "finish_reason": "stop"}
        chunk = {"id": "reply", "object": "chat.completion.chunk", "created": 1,
                 "model": model, "choices": [choice]}
        if body.get("stream"):
            return httpx.Response(200, text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                                  headers={"content-type": "text/event-stream"})
        choice.pop("delta")
        choice["message"] = {"role": "assistant", "content": "60"}
        chunk["object"] = "chat.completion"
        return httpx.Response(200, json=chunk)

    def make_client(self, kwargs, **options):
        return OpenAI(api_key="test-only", base_url=BASE_URL,
                      http_client=httpx.Client(transport=httpx.MockTransport(serve)))
    monkeypatch.setattr(AIAgent, "_create_openai_client", make_client)
    agent = AIAgent(api_key="test-only", base_url=BASE_URL, provider="custom", model=model,
                    enabled_toolsets=[], quiet_mode=True, skip_memory=True,
                    skip_context_files=True, max_iterations=2, max_tokens=256)
    restored = [
        {"role": "user", "content": "计算 12 乘 5"},
        {"role": "assistant", "content": "", "reasoning_content": REAL_REASONING,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "calculate", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "60"},
    ]
    original = copy.deepcopy(restored)
    try:
        result = agent.run_conversation("根据工具结果回答。", conversation_history=restored)
        assert result["final_response"] == "60"
        assert captured
        assert restored == original
    finally:
        agent.client.close()
