# Agent Harness 学习笔记（中文）

这个目录是一套以问答形式组织的中文教程，用来理解 "agent harness"：也就是包在大模型外面、负责组织上下文、调用工具、维护对话轨迹（trajectory）的那一层运行时。

教程以本仓库的 [OpenAI Agents SDK](../src/agents/) 为主线，同时对照 [vLLM](https://github.com/vllm-project/vllm) 的 OpenAI 兼容服务器（`/v1/chat/completions` 与 `/v1/responses`），看清楚一条消息从 SDK 出发、经过 HTTP 协议、最后被渲染成 token 送进模型的完整路径。

每一课都尽量做到三件事：

- 给出结论，并标明结论属于哪一层（SDK、协议/服务器、chat template）。
- 给出源码位置（相对仓库根目录的路径和行号），方便你自己去读。
- 给出一个不需要 API key、不需要 GPU 就能跑的小脚本，用来亲眼验证结论。

## 目录

| 课 | 问题 | 内容 |
| --- | --- | --- |
| 01 | 模型输出的 reasoning 会不会保留在轨迹里，并在下一步解码时出现在上下文中？ | [01-reasoning-in-trajectory](01-reasoning-in-trajectory/README.md) |
| 02 | computer-use agent 的截图工具结果在上下文里长什么样？图片怎么变成嵌入？vLLM 的多模态缓存怎样命中？ | [02-cua-multimodal-context](02-cua-multimodal-context/README.md) |

## 如何运行示例脚本

在仓库根目录执行：

```bash
make sync
uv run python tutorial/01-reasoning-in-trajectory/probe_reasoning_replay.py
uv run python tutorial/02-cua-multimodal-context/probe_cua_context.py
```

这些脚本用一个伪造的 OpenAI 兼容服务器记录 SDK 发出的真实请求体，所以不会产生任何网络请求或费用。

## 说明

- 源码行号基于编写教程时的提交，后续代码变动后可能会有偏移，用函数名搜索即可定位。
- 关于 OpenAI 平台行为（例如 `reasoning.context`、`store`、`encrypted_content`）的描述，引用自 `openai` Python 包中由 OpenAPI 规范生成的类型注释。
