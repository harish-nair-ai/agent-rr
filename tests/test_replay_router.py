from agent_rr.proxy.replay_router import apply_overrides, request_digest


def test_model_override() -> None:
    assert apply_overrides("openai", {"model": "a", "x": 1}, model="b", system_prompt=None) == {
        "model": "b",
        "x": 1,
    }


def test_openai_system_message_replaced() -> None:
    body = {"messages": [{"role": "system", "content": "old"}, {"role": "user", "content": "u"}]}
    out = apply_overrides("openai", body, model=None, system_prompt="new")
    assert out["messages"] == [
        {"role": "system", "content": "new"},
        {"role": "user", "content": "u"},
    ]
    assert body["messages"][0]["content"] == "old"  # pure


def test_openai_system_message_inserted_when_absent() -> None:
    out = apply_overrides("openai", {"messages": [{"role": "user", "content": "u"}]}, None, "new")
    assert out["messages"][0] == {"role": "system", "content": "new"}


def test_anthropic_top_level_system_replaced() -> None:
    out = apply_overrides("anthropic", {"system": "old", "messages": []}, None, "new")
    assert out == {"system": "new", "messages": []}
    assert apply_overrides("anthropic", {"messages": []}, None, "s") == {
        "system": "s",
        "messages": [],
    }


def test_no_overrides_is_identity() -> None:
    body = {"model": "a", "messages": []}
    assert apply_overrides("anthropic", body, None, None) == body


def test_request_digest_is_canonical() -> None:
    assert request_digest({"b": 1, "a": [1, 2]}) == request_digest({"a": [1, 2], "b": 1})
    assert request_digest({"a": 1}) != request_digest({"a": 2})
    assert len(request_digest({})) == 64
