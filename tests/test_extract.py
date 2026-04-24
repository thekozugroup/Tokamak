from tokamak.extract import find_reasoning_spans


def test_finds_thinking_tag_inside_assistant_content():
    trace = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<thinking>internal step</thinking>\nAnswer."},
        ]
    }
    spans = find_reasoning_spans(trace)
    assert len(spans) == 1
    assert spans[0].source == "tag:thinking"
    assert "internal step" in spans[0].text


def test_finds_reasoning_tag_alongside_thinking():
    trace = {
        "messages": [
            {"role": "assistant", "content": "<reasoning>step a</reasoning> answer <thinking>step b</thinking>"},
        ]
    }
    spans = find_reasoning_spans(trace)
    sources = sorted(s.source for s in spans)
    assert sources == ["tag:reasoning", "tag:thinking"]


def test_anthropic_thinking_block():
    trace = {
        "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hidden chain"},
                {"type": "text", "text": "visible answer"},
            ]},
        ]
    }
    spans = find_reasoning_spans(trace)
    assert any(s.source == "block:thinking" for s in spans)


def test_apply_rewrites_in_place():
    trace = {"messages": [{"role": "assistant", "content": "<thinking>orig</thinking>\nA"}]}
    spans = find_reasoning_spans(trace)
    spans[0].apply("compressed")
    assert "<thinking>compressed</thinking>" in trace["messages"][0]["content"]
    assert "<thinking>orig</thinking>" not in trace["messages"][0]["content"]
