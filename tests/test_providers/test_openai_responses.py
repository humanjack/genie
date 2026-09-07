"""Responses streaming contract tests; scripted SDK, no network."""

from types import SimpleNamespace as NS

import pytest

from genie.providers.base import ChatMessage
from genie.providers.openai_client import OpenAIClient


def event(kind, **kwargs):
    return NS(type=kind, **kwargs)


def completed(id="resp_1", **kwargs):
    return event("response.completed", response=NS(id=id, usage=None, **kwargs))


class Stream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aiter__(self):
        for item in self.events:
            if isinstance(item, BaseException):
                raise item
            yield item

    async def close(self):
        self.closed = True


class SDK:
    def __init__(self, *turns):
        self.streams = [Stream(turn) for turn in turns]
        self.requests = []
        self.responses = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.streams[len(self.requests) - 1]


async def drain(client, messages, **kwargs):
    return [chunk async for chunk in client.stream(messages, [], **kwargs)]


async def test_text_state_and_request_options():
    sdk = SDK([event("response.output_text.delta", delta="hello"), completed()], [completed("r2")])
    client = OpenAIClient("test", client=sdk, api="responses")
    messages = [ChatMessage("user", "hi")]
    chunks = await drain(client, messages, system="sys", max_tokens=20, temperature=0.5)
    assert chunks[0].delta_text == "hello"
    assert chunks[-1].finish_reason == "stop"
    assert sdk.requests[0] == dict(
        model="test",
        input=[{"role": "user", "content": "hi"}],
        instructions="sys",
        max_output_tokens=20,
        temperature=0.5,
        stream=True,
        store=True,
    )
    messages.extend([ChatMessage("assistant", "hello"), ChatMessage("user", "again")])
    await drain(client, messages, system="sys")
    assert sdk.requests[1]["previous_response_id"] == "resp_1"
    assert sdk.requests[1]["input"] == [{"role": "user", "content": "again"}]
    assert sdk.requests[1]["instructions"] == "sys"
    assert all(stream.closed for stream in sdk.streams)


def opening(index, call_id, name):
    return event(
        "response.output_item.added",
        output_index=index,
        item=NS(type="function_call", call_id=call_id, name=name, arguments=""),
    )


def arguments(index, delta):
    return event("response.function_call_arguments.delta", output_index=index, delta=delta)


async def test_interleaved_tools_roundtrip_and_usage():
    done = completed()
    done.response.usage = NS(
        input_tokens=30, output_tokens=10, input_tokens_details=NS(cached_tokens=12)
    )
    sdk = SDK(
        [
            event("response.output_item.added", output_index=0, item=NS(type="reasoning")),
            opening(1, "call_a", "read"),
            opening(3, "call_b", "run"),
            arguments(3, '{"cmd":'),
            arguments(1, '{"path": "a"}'),
            arguments(3, '"ls"}'),
            event(
                "response.function_call_arguments.done", output_index=3, arguments='{"cmd":"ls"}'
            ),
            done,
        ],
        [completed("r2")],
    )
    client = OpenAIClient("test", client=sdk, api="responses")
    history = [ChatMessage("user", "go")]
    chunks = [
        chunk
        async for chunk in client.stream(
            history, [{"name": "read", "description": "read", "input_schema": {"type": "object"}}]
        )
    ]
    slots = {}
    for chunk in chunks:
        if chunk.tool_call_delta:
            delta = chunk.tool_call_delta
            slot = slots.setdefault(delta["index"], {"arguments": ""})
            slot["arguments"] += delta["arguments_delta"]
            slot.update({k: v for k, v in delta.items() if k in ("id", "name")})
    assert slots == {
        1: {"id": "call_a", "name": "read", "arguments": '{"path": "a"}'},
        3: {"id": "call_b", "name": "run", "arguments": '{"cmd":"ls"}'},
    }
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].usage == dict(input_tokens=30, output_tokens=10, cache_read=12, cache_write=0)
    assert sdk.requests[0]["tools"] == [
        {
            "type": "function",
            "name": "read",
            "description": "read",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]
    history.extend(
        [
            ChatMessage(
                "assistant",
                "",
                [
                    {"id": "call_a", "name": "read", "arguments": {"path": "a"}},
                    {"id": "call_b", "name": "run", "arguments": {"cmd": "ls"}},
                ],
            ),
            ChatMessage("tool", "contents", tool_call_id="call_a"),
            ChatMessage("tool", "done", tool_call_id="call_b"),
        ]
    )
    await drain(client, history)
    assert sdk.requests[1]["previous_response_id"] == "resp_1"
    assert sdk.requests[1]["input"] == [
        {"type": "function_call_output", "call_id": "call_a", "output": "contents"},
        {"type": "function_call_output", "call_id": "call_b", "output": "done"},
    ]


@pytest.mark.parametrize("mode", ["reset", "edit", "different_assistant", "truncate"])
async def test_modified_history_does_not_reuse_server_state(mode):
    sdk = SDK([event("response.output_text.delta", delta="ok"), completed()], [completed("r2")])
    client = OpenAIClient("test", client=sdk, api="responses")
    history = [ChatMessage("user", "hi")]
    await drain(client, history)
    history.extend([ChatMessage("assistant", "ok"), ChatMessage("user", "next")])
    if mode == "reset":
        history = [ChatMessage("user", "new session")]
    elif mode == "edit":
        history[0].content = "changed"
    elif mode == "different_assistant":
        history[1].content = "changed"
    else:
        history.pop(0)
    await drain(client, history)
    assert "previous_response_id" not in sdk.requests[1]


@pytest.mark.parametrize(
    "failure",
    [
        event("response.failed", response=NS(error="failed")),
        event("response.incomplete", response=NS(incomplete_details="max_output_tokens")),
        event("error", message="bad request"),
        RuntimeError("network"),
    ],
)
async def test_failed_turn_never_advances_state_or_leaves_stream_open(failure):
    sdk = SDK(
        [event("response.output_text.delta", delta="ok"), completed("good")],
        [failure],
        [completed("retried")],
    )
    client = OpenAIClient("test", client=sdk, api="responses")
    history = [ChatMessage("user", "hi")]
    await drain(client, history)
    history += [ChatMessage("assistant", "ok"), ChatMessage("user", "next")]
    with pytest.raises(RuntimeError):
        await drain(client, history)
    await drain(client, history)
    assert sdk.requests[2]["previous_response_id"] == "good"
    assert sdk.streams[1].closed


async def test_missing_terminal_is_an_error():
    sdk = SDK([event("response.output_text.delta", delta="partial")])
    with pytest.raises(RuntimeError, match=r"without response\.completed"):
        await drain(OpenAIClient("test", client=sdk, api="responses"), [])
    assert sdk.streams[0].closed


async def test_close_and_concurrent_guard():
    sdk = SDK(
        [event("response.output_text.delta", delta="partial"), completed()], [completed("fresh")]
    )
    client = OpenAIClient("test", client=sdk, api="responses")
    stream = client.stream([ChatMessage("user", "hi")], [])
    assert (await anext(stream)).delta_text == "partial"
    with pytest.raises(RuntimeError, match="Concurrent"):
        await drain(client, [])
    await stream.aclose()
    assert sdk.streams[0].closed
    await drain(client, [ChatMessage("user", "hi")])
    assert "previous_response_id" not in sdk.requests[1]


async def test_task_cancellation_closes_stream_and_releases_guard():
    import asyncio

    started = asyncio.Event()

    class WaitingStream(Stream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield completed()

    sdk = SDK([], [completed()])
    sdk.streams[0] = WaitingStream([])
    client = OpenAIClient("test", client=sdk, api="responses")
    task = asyncio.create_task(drain(client, [ChatMessage("user", "hi")]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sdk.streams[0].closed
    await drain(client, [])
    assert "previous_response_id" not in sdk.requests[1]


async def test_refusal_and_independent_clients():
    sdk = SDK([event("response.refusal.delta", delta="Cannot help"), completed()], [completed()])
    first = OpenAIClient("test", client=sdk, api="responses")
    second = OpenAIClient("test", client=sdk, api="responses")
    history = [ChatMessage("user", "hi")]
    assert (await drain(first, history))[0].delta_text == "Cannot help"
    await drain(
        second, [*history, ChatMessage("assistant", "Cannot help"), ChatMessage("user", "next")]
    )
    assert "previous_response_id" not in sdk.requests[1]


async def test_fresh_tool_history_translates_without_previous_id():
    sdk = SDK([completed()])
    client = OpenAIClient("test", client=sdk, api="responses")
    await drain(
        client,
        [
            ChatMessage("system", "sys"),
            ChatMessage(
                "assistant",
                "checking",
                [{"id": "c", "function": {"name": "read", "arguments": "{}"}}],
            ),
            ChatMessage("tool", "result", tool_call_id="c"),
        ],
    )
    assert sdk.requests[0]["input"] == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "checking"},
        {"type": "function_call", "call_id": "c", "name": "read", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": "result"},
    ]


async def test_content_blocks_are_translated_without_mutating_history():
    sdk = SDK([completed()])
    client = OpenAIClient("test", client=sdk, api="responses")
    blocks = [
        {"type": "text", "text": "see this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "low"}},
    ]
    await drain(
        client,
        [
            ChatMessage("user", blocks),
            ChatMessage("tool", [{"type": "text", "text": "done"}], tool_call_id="c"),
        ],
    )
    assert sdk.requests[0]["input"][0]["content"] == [
        {"type": "input_text", "text": "see this"},
        {"type": "input_image", "image_url": "https://example.com/a.png", "detail": "low"},
    ]
    assert sdk.requests[0]["input"][1]["output"] == [{"type": "input_text", "text": "done"}]
    assert blocks[0]["type"] == "text"
    assert isinstance(blocks[1]["image_url"], dict)


async def test_request_failure_releases_concurrency_guard():
    class FailingSDK(SDK):
        async def create(self, **kwargs):
            raise RuntimeError("request failed")

    client = OpenAIClient("test", client=FailingSDK(), api="responses")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="request failed"):
            await drain(client, [])


async def test_malformed_arguments_are_preserved_for_validation():
    sdk = SDK([completed()])
    client = OpenAIClient("test", client=sdk, api="responses")
    await drain(
        client, [ChatMessage("assistant", "", [{"id": "c", "name": "read", "arguments": "{"}])]
    )
    assert sdk.requests[0]["input"][0]["arguments"] == "{"


async def test_real_sdk_serializes_request_and_parses_sse():
    import json

    import httpx
    from openai import AsyncOpenAI

    requests = []
    responses = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        events = [
            {
                "type": "response.output_item.added",
                "sequence_number": 0,
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "read",
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 1,
                "output_index": 2,
                "item_id": "fc_1",
                "delta": '{"path":"a"}',
            },
            {
                "type": "response.completed",
                "sequence_number": 2,
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "output": [],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                        "input_tokens_details": {"cached_tokens": 3},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            },
        ]
        body = "".join(f"event: {item['type']}\ndata: {json.dumps(item)}\n\n" for item in events)
        response = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        responses.append(response)
        return response

    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as sdk:
        client = OpenAIClient("gpt-4o-mini", client=sdk, api="responses")
        chunks = [
            chunk
            async for chunk in client.stream(
                [ChatMessage("user", [{"type": "text", "text": "read a"}])], [{"name": "read"}]
            )
        ]
    assert requests[0]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "read a"}]}
    ]
    assert requests[0]["tools"] == [
        {"type": "function", "name": "read", "parameters": {}, "strict": False}
    ]
    assert chunks[0].tool_call_delta == {
        "index": 2,
        "id": "call_1",
        "name": "read",
        "arguments_delta": "",
    }
    assert chunks[1].tool_call_delta == {"index": 2, "arguments_delta": '{"path":"a"}'}
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].usage == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_read": 3,
        "cache_write": 0,
    }
    assert responses[0].is_closed
