"""观察一个自定义 schema 的 computer-use agent 在多步之后，SDK 发给服务器的上下文长什么样。

和第 01 课一样，这个脚本不需要 API key，也不需要 GPU。
它用 httpx2.MockTransport 伪造一个 OpenAI 兼容服务器（可以想象成 vLLM），
按顺序返回预先写好的工具调用，并记录 SDK 每一次发出的请求体。

我们自己设计的工具 schema（而不是 OpenAI 托管的 computer 工具）：

- `screenshot()`：返回当前屏幕截图。
- `click(x, y)`：点击坐标，返回一段文字和点击后的截图。
- `type_text(text)`：输入文字，返回一段文字和输入后的截图。

截图用 `ToolOutputImage(image_url="data:image/png;base64,...")` 返回。
为了让输出可读，脚本生成的是很小的纯色 PNG；真实截图通常是 1280x720 或更大。

运行方式（在仓库根目录）：

    uv run python tutorial/02-cua-multimodal-context/probe_cua_context.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import httpx2 as httpx
from openai import AsyncOpenAI

from agents import (
    Agent,
    Model,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
    ToolOutputImage,
    ToolOutputText,
)
from agents.decorators import tool
from agents.run import CallModelData, ModelInputData

MODEL_NAME = "moonshotai/Kimi-K3"


# ---------------------------------------------------------------------------
# 生成 "截图"：每一步一个不同颜色的小 PNG。
# ---------------------------------------------------------------------------


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


SCREENS = {
    "S1": _data_url(_png(32, 18, (240, 240, 240))),
    "S2": _data_url(_png(32, 18, (200, 220, 255))),
    "S3": _data_url(_png(32, 18, (210, 255, 210))),
}
LABEL_BY_URL = {url: label for label, url in SCREENS.items()}


class FakeDesktop:
    """一个假的桌面：每次动作之后屏幕换成下一张截图。"""

    def __init__(self) -> None:
        self._order = ["S1", "S2", "S3"]
        self._index = 0

    def next_screen(self) -> ToolOutputImage:
        label = self._order[min(self._index, len(self._order) - 1)]
        self._index += 1
        return ToolOutputImage(image_url=SCREENS[label], detail="auto")


def build_tools(desktop: FakeDesktop) -> list[Any]:
    @tool
    def screenshot() -> ToolOutputImage:
        """截取当前屏幕。"""
        return desktop.next_screen()

    @tool
    def click(x: int, y: int) -> list[ToolOutputText | ToolOutputImage]:
        """在屏幕坐标 (x, y) 处单击，返回点击后的截图。"""
        return [ToolOutputText(text=f"已点击 ({x}, {y})"), desktop.next_screen()]

    @tool
    def type_text(text: str) -> list[ToolOutputText | ToolOutputImage]:
        """在当前焦点处输入文字，返回输入后的截图。"""
        return [ToolOutputText(text=f"已输入 {text!r}"), desktop.next_screen()]

    return [screenshot, click, type_text]


# ---------------------------------------------------------------------------
# 伪造的服务器。
# ---------------------------------------------------------------------------

STEPS: list[tuple[str, dict[str, Any]] | str] = [
    ("screenshot", {}),
    ("click", {"x": 640, "y": 88}),
    ("type_text", {"text": "vLLM\n"}),
    "搜索框里已经输入 vLLM 并回车。",
]


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


def responses_replies() -> list[dict[str, Any]]:
    replies = []
    for i, step in enumerate(STEPS, start=1):
        if isinstance(step, str):
            output: dict[str, Any] = {
                "id": f"msg_{i}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": step, "annotations": []}],
            }
        else:
            name, args = step
            output = {
                "id": f"fc_{i}",
                "type": "function_call",
                "call_id": f"call_{i}",
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
                "status": "completed",
            }
        replies.append(
            {
                "id": f"resp_{i}",
                "object": "response",
                "created_at": 0,
                "model": MODEL_NAME,
                "status": "completed",
                "output": [output],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 0,
                },
            }
        )
    return replies


def chat_replies() -> list[dict[str, Any]]:
    replies = []
    for i, step in enumerate(STEPS, start=1):
        message: dict[str, Any] = {"role": "assistant", "content": None}
        if isinstance(step, str):
            message["content"] = step
        else:
            name, args = step
            message["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            ]
        replies.append(
            {
                "id": f"chatcmpl_{i}",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop" if isinstance(step, str) else "tool_calls",
                        "message": message,
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )
    return replies


# ---------------------------------------------------------------------------
# 两个 call_model_input_filter 示例。
# ---------------------------------------------------------------------------


def _as_dict(item: Any) -> dict[str, Any] | None:
    return cast(dict[str, Any], item) if isinstance(item, dict) else None


def _image_parts(entry: dict[str, Any] | None) -> list[dict[str, Any]]:
    if entry is None or entry.get("type") != "function_call_output":
        return []
    output = entry.get("output")
    if not isinstance(output, list):
        return []
    return [part for part in output if isinstance(part, dict) and part.get("type") == "input_image"]


def screenshots_as_user_messages(data: CallModelData[Any]) -> ModelInputData:
    """把工具结果里的截图挪到紧随其后的 user 消息中，工具结果只留文字。"""
    new_input: list[Any] = []
    for item in data.model_data.input:
        entry = _as_dict(item)
        images = _image_parts(entry)
        if entry is None or not images:
            new_input.append(item)
            continue
        texts = [
            str(part.get("text", ""))
            for part in entry["output"]
            if isinstance(part, dict) and part.get("type") == "input_text"
        ]
        new_input.append({**entry, "output": " ".join(texts) or "截图见下一条消息。"})
        new_input.append({"role": "user", "content": images})
    return ModelInputData(input=new_input, instructions=data.model_data.instructions)


def keep_last_screenshot_only(data: CallModelData[Any]) -> ModelInputData:
    """只保留最近一张截图，更早的截图替换成一句文字。"""
    items = data.model_data.input
    image_positions = [i for i, item in enumerate(items) if _image_parts(_as_dict(item))]
    stale = set(image_positions[:-1])
    new_input: list[Any] = []
    for i, item in enumerate(items):
        entry = _as_dict(item)
        if i not in stale or entry is None:
            new_input.append(item)
            continue
        kept = [
            part
            for part in entry["output"]
            if not (isinstance(part, dict) and part.get("type") == "input_image")
        ]
        kept.append({"type": "input_text", "text": "[旧截图已省略]"})
        new_input.append({**entry, "output": kept})
    return ModelInputData(input=new_input, instructions=data.model_data.instructions)


# ---------------------------------------------------------------------------
# 把请求体压缩成便于阅读的形式。
# ---------------------------------------------------------------------------


def _describe_parts(parts: Any) -> str:
    if isinstance(parts, str):
        return repr(parts)
    shown = []
    for part in parts:
        kind = part.get("type")
        if kind in ("input_image", "image_url"):
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            shown.append(f"[图片 {LABEL_BY_URL.get(url, '?')}]")
        elif kind in ("input_text", "text"):
            shown.append(repr(part.get("text")))
        else:
            shown.append(f"[{kind}]")
    return " + ".join(shown)


def summarize_responses_input(body: dict[str, Any]) -> list[str]:
    lines = []
    for item in body["input"]:
        item_type = item.get("type", "message")
        if item_type == "message":
            lines.append(
                f"{item['role']:<9} message               {_describe_parts(item['content'])}"
            )
        elif item_type == "function_call":
            lines.append(f"assistant function_call         {item['name']}({item['arguments']})")
        elif item_type == "function_call_output":
            lines.append(f"tool      function_call_output  {_describe_parts(item['output'])}")
        else:
            lines.append(item_type)
    return lines


def summarize_chat_messages(body: dict[str, Any]) -> list[str]:
    lines = []
    for msg in body["messages"]:
        parts = []
        if msg.get("content"):
            parts.append(_describe_parts(msg["content"]))
        for call in msg.get("tool_calls") or []:
            parts.append(f"tool_call {call['function']['name']}({call['function']['arguments']})")
        lines.append(f"{msg['role']:<9} " + ", ".join(parts))
    return lines


def _items(body: dict[str, Any]) -> list[Any]:
    return cast(list[Any], body["input"] if "input" in body else body["messages"])


def prefix_report(requests: list[dict[str, Any]]) -> list[str]:
    """检查第 k+1 次请求是否以第 k 次请求的全部输入作为前缀。"""
    lines = []
    for k in range(1, len(requests)):
        prev, cur = _items(requests[k - 1]), _items(requests[k])
        diverge = next(
            (i for i, (a, b) in enumerate(zip(prev, cur, strict=False)) if a != b),
            None,
        )
        if diverge is None:
            lines.append(f"请求 {k + 1} 以请求 {k} 的全部 {len(prev)} 条输入为前缀：是")
        else:
            lines.append(f"请求 {k + 1} 以请求 {k} 的输入为前缀：否，第 {diverge + 1} 条开始不同")
    return lines


# ---------------------------------------------------------------------------
# 场景。
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    title: str
    make_model: Callable[[AsyncOpenAI], Model]
    replies: Callable[[], list[dict[str, Any]]]
    kind: Literal["responses", "chat"]
    run_config: RunConfig = field(default_factory=lambda: RunConfig(tracing_disabled=True))


SCENARIOS: list[Scenario] = [
    Scenario(
        "A. Responses API，默认配置",
        lambda client: OpenAIResponsesModel(MODEL_NAME, client),
        responses_replies,
        "responses",
    ),
    Scenario(
        "B. Chat Completions，默认配置",
        lambda client: OpenAIChatCompletionsModel(MODEL_NAME, client),
        chat_replies,
        "chat",
    ),
    Scenario(
        "C. Chat Completions，用 call_model_input_filter 把截图挪到 user 消息",
        lambda client: OpenAIChatCompletionsModel(MODEL_NAME, client),
        chat_replies,
        "chat",
        RunConfig(tracing_disabled=True, call_model_input_filter=screenshots_as_user_messages),
    ),
    Scenario(
        "D. Responses API，用 call_model_input_filter 只保留最近一张截图",
        lambda client: OpenAIResponsesModel(MODEL_NAME, client),
        responses_replies,
        "responses",
        RunConfig(tracing_disabled=True, call_model_input_filter=keep_last_screenshot_only),
    ),
]


async def run_scenario(scenario: Scenario) -> list[dict[str, Any]]:
    server = FakeServer(scenario.replies())
    agent = Agent(
        name="computer-use",
        instructions="你是一个操作电脑的助手。每次动作之后都会收到新的屏幕截图。",
        model=scenario.make_model(server.client()),
        tools=build_tools(FakeDesktop()),
    )
    await Runner.run(
        agent,
        "在浏览器的搜索框里输入 vLLM 并回车。",
        run_config=scenario.run_config,
    )
    return server.requests


async def main() -> dict[str, list[dict[str, Any]]]:
    dumped: dict[str, list[dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        requests = await run_scenario(scenario)
        dumped[scenario.title] = requests
        print("=" * 78)
        print(scenario.title)
        print("=" * 78)
        summarize = (
            summarize_responses_input if scenario.kind == "responses" else summarize_chat_messages
        )
        for k, body in enumerate(requests, start=1):
            print(f"\n--- 请求 {k} ---")
            for line in summarize(body):
                print("  " + line)
        print()
        for line in prefix_report(requests):
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
