"""观察 Agents SDK 在多步工具调用中，是否把上一步的 reasoning 回传给模型。

这个脚本不需要真实的 API key，也不需要 GPU。
它用 httpx.MockTransport 伪造一个 "vLLM 风格" 的 OpenAI 兼容服务器，
每次请求都返回预先写好的回复，同时把 SDK 发出的请求体完整记录下来。
因为请求体由 SDK 的真实代码路径（Runner + 模型适配器 + 转换器）构造，
所以打印出来的内容就是 SDK 在下一步解码前真正发给服务器的上下文。

运行方式（在仓库根目录）：

    uv run python tutorial/01-reasoning-in-trajectory/probe_reasoning_replay.py

可选参数 `--dump <path>` 会把所有场景的原始请求体写成 JSON 文件。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx2 as httpx
from openai import AsyncOpenAI

from agents import (
    Agent,
    Model,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
)
from agents.decorators import tool
from agents.run import CallModelData, ModelInputData

MODEL_NAME = "Qwen/Qwen3-8B"


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴，25°C"


class FakeServer:
    """按顺序返回预设回复，并记录收到的每一个请求体。"""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self._replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, json=self._replies.pop(0))

    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="EMPTY",
            base_url="http://fake-vllm:8000/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
        )


# ---------------------------------------------------------------------------
# 预设回复：Responses API（/v1/responses）。
# ---------------------------------------------------------------------------


def _response(resp_id: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": resp_id,
        "object": "response",
        "created_at": 0,
        "model": MODEL_NAME,
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 20,
        },
    }


def _reasoning(item_id: str, text: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }


def _message(item_id: str, text: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


RESPONSES_REPLIES = [
    _response(
        "resp_1",
        [
            _reasoning("rs_1", "用户问上海天气，我需要调用 get_weather。"),
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "上海"}',
                "status": "completed",
            },
        ],
    ),
    _response(
        "resp_2",
        [
            _reasoning("rs_2", "工具返回晴 25°C，可以直接回答。"),
            _message("msg_2", "上海今天晴，25°C。"),
        ],
    ),
    _response(
        "resp_3",
        [
            _reasoning("rs_3", "用户追问北京，但这里直接给出示例回答。"),
            _message("msg_3", "北京我还没查，需要的话我可以再调用工具。"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# 预设回复：Chat Completions（/v1/chat/completions）。
# ---------------------------------------------------------------------------


def _chat_completion(
    resp_id: str,
    *,
    reasoning_field: str,
    reasoning: str,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    message[reasoning_field] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": 0,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": message,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def chat_replies(reasoning_field: str) -> list[dict[str, Any]]:
    return [
        _chat_completion(
            "chatcmpl_1",
            reasoning_field=reasoning_field,
            reasoning="用户问上海天气，我需要调用 get_weather。",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "上海"}'},
                }
            ],
        ),
        _chat_completion(
            "chatcmpl_2",
            reasoning_field=reasoning_field,
            reasoning="工具返回晴 25°C，可以直接回答。",
            content="上海今天晴，25°C。",
        ),
        _chat_completion(
            "chatcmpl_3",
            reasoning_field=reasoning_field,
            reasoning="用户追问北京，但这里直接给出示例回答。",
            content="北京我还没查，需要的话我可以再调用工具。",
        ),
    ]


# ---------------------------------------------------------------------------
# 把请求体压缩成便于阅读的一行一条的形式。
# ---------------------------------------------------------------------------


def _short(value: Any, limit: int = 36) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summarize_responses_input(body: dict[str, Any]) -> list[str]:
    lines = []
    for item in body["input"]:
        item_type = item.get("type", "message")
        if item_type == "message":
            content = item["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            lines.append(f"{item['role']:<9} message         {_short(content)}")
        elif item_type == "reasoning":
            texts = [c["text"] for c in item.get("content") or []]
            texts += [s["text"] for s in item.get("summary") or []]
            lines.append(f"assistant reasoning ({item.get('id')})  {_short(' '.join(texts))}")
        elif item_type == "function_call":
            lines.append(f"assistant function_call   {item['name']}({item['arguments']})")
        elif item_type == "function_call_output":
            lines.append(f"tool      function_call_output  {_short(item['output'])}")
        else:
            lines.append(f"{item_type}")
    return lines


def summarize_chat_messages(body: dict[str, Any]) -> list[str]:
    lines = []
    for msg in body["messages"]:
        role = msg["role"]
        parts = []
        if msg.get("content"):
            parts.append(f"content={_short(msg['content'])!r}")
        for key in ("reasoning", "reasoning_content"):
            if msg.get(key):
                parts.append(f"{key}={_short(msg[key])!r}")
        for call in msg.get("tool_calls") or []:
            parts.append(f"tool_call={call['function']['name']}")
        lines.append(f"{role:<9} " + ", ".join(parts))
    return lines


# ---------------------------------------------------------------------------
# 场景。
# ---------------------------------------------------------------------------


async def run_two_user_turns(model: Model, config: RunConfig) -> None:
    """第 1 个用户回合包含一次工具调用（两次模型请求），第 2 个用户回合再请求一次。"""
    agent = Agent(
        name="weather",
        instructions="你是天气助手。",
        model=model,
        tools=[get_weather],
    )
    first = await Runner.run(agent, "上海天气怎么样？", run_config=config)
    # 第二个用户回合：把第一轮的完整轨迹（to_input_list）接上新问题。
    await Runner.run(
        agent,
        first.to_input_list() + [{"role": "user", "content": "那北京呢？"}],
        run_config=config,
    )


def keep_only_current_turn_reasoning(data: CallModelData[Any]) -> ModelInputData:
    """在客户端模拟 "current_turn"：删掉最后一条 user 消息之前的 reasoning item。"""
    items = data.model_data.input
    user_indexes = [
        i for i, item in enumerate(items) if isinstance(item, dict) and item.get("role") == "user"
    ]
    last_user_index = user_indexes[-1] if user_indexes else -1
    kept = [
        item
        for i, item in enumerate(items)
        if not (isinstance(item, dict) and item.get("type") == "reasoning" and i < last_user_index)
    ]
    return ModelInputData(input=kept, instructions=data.model_data.instructions)


@dataclass
class Scenario:
    title: str
    make_model: Callable[[AsyncOpenAI], Model]
    replies: list[dict[str, Any]]
    kind: Literal["responses", "chat"]
    run_config: RunConfig = field(default_factory=lambda: RunConfig(tracing_disabled=True))


SCENARIOS: list[Scenario] = [
    Scenario(
        "A. Responses API（/v1/responses），默认配置",
        lambda client: OpenAIResponsesModel(MODEL_NAME, client),
        RESPONSES_REPLIES,
        "responses",
    ),
    Scenario(
        "B. Chat Completions，服务器返回 message.reasoning（新版 vLLM 字段），默认配置",
        lambda client: OpenAIChatCompletionsModel(MODEL_NAME, client),
        chat_replies("reasoning"),
        "chat",
    ),
    Scenario(
        "C. Chat Completions，服务器返回 message.reasoning_content（旧字段），默认配置",
        lambda client: OpenAIChatCompletionsModel(MODEL_NAME, client),
        chat_replies("reasoning_content"),
        "chat",
    ),
    Scenario(
        "D. 同 C，但用 should_replay_reasoning_content 钩子主动开启回传",
        lambda client: OpenAIChatCompletionsModel(
            MODEL_NAME, client, should_replay_reasoning_content=lambda ctx: True
        ),
        chat_replies("reasoning_content"),
        "chat",
    ),
    Scenario(
        "E. 同 A，但用 call_model_input_filter 只保留当前用户回合的 reasoning",
        lambda client: OpenAIResponsesModel(MODEL_NAME, client),
        RESPONSES_REPLIES,
        "responses",
        RunConfig(tracing_disabled=True, call_model_input_filter=keep_only_current_turn_reasoning),
    ),
]


async def main() -> dict[str, list[dict[str, Any]]]:
    dumped: dict[str, list[dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        server = FakeServer(scenario.replies)
        await run_two_user_turns(scenario.make_model(server.client()), scenario.run_config)
        dumped[scenario.title] = server.requests
        print("=" * 78)
        print(scenario.title)
        print("=" * 78)
        labels = [
            "请求 1：用户提问",
            "请求 2：返回工具结果后（同一用户回合内）",
            "请求 3：下一个用户回合",
        ]
        for label, body in zip(labels, server.requests, strict=True):
            print(f"\n--- {label} ---")
            summarize = (
                summarize_responses_input
                if scenario.kind == "responses"
                else summarize_chat_messages
            )
            for line in summarize(body):
                print("  " + line)
        print()
    return dumped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default=None, help="把原始请求体写入这个 JSON 文件。")
    dump_path = parser.parse_args().dump
    all_requests = asyncio.run(main())
    if dump_path:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(all_requests, f, ensure_ascii=False, indent=2)
        print(f"原始请求体已写入 {dump_path}")
