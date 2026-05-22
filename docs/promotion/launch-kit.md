# WebLLM Gateway 开源推广发布包

这份发布包面向 OpenClaw、Hermes、Claude Code、Codex 和其它 Agent runtime 玩家。第一波传播不要泛泛讲“API 网关”，要讲清楚：网页登录模型也可以进入标准工具调用链路。

## 核心定位

WebLLM Gateway 是一个给 Agent 工具链使用的网页登录模型 API 网关。它把 Qwen Web、DeepSeek Web 和已验证的网页登录模型包装成 OpenAI / Anthropic 兼容、可工具调用的标准 API，让 OpenClaw、Hermes、Claude Code、Codex 等客户端像调用原生模型 API 一样使用网页账号。

## 一句话版本

把 Qwen / DeepSeek 网页账号接进 OpenClaw、Hermes、Claude Code、Codex，让网页登录模型也能跑标准工具调用。

## 30 秒版本

我开源了一个小项目 WebLLM Gateway：它不是聊天壳子，而是给 Agent 工具链用的网页登录模型 API 网关。

登录 Qwen / DeepSeek 网页账号后，Gateway 会自动检测可用模型，并暴露 OpenAI / Anthropic 兼容 API。OpenClaw、Hermes、Claude Code、Codex 这类客户端可以直接填 `base_url`、`api_key`、`model` 来调用。重点是它处理了工具调用桥接：OpenAI `tool_calls` 和 Anthropic `tool_use/tool_result` 都会被转换成网页登录模型能理解的协议，再转回标准结构。

Qwen 3.7 系列已经调通，适合 Agent 玩家拿来做编程、项目分析和工具调用测试。

## GitHub 仓库描述

```text
WebLLM Gateway：让 Qwen/DeepSeek 网页模型接入 OpenAI/Anthropic 兼容 Agent 工具链，支持 OpenClaw、Hermes、Claude Code、Codex 工具调用
```

## 推荐 GitHub Topics

```text
agent-tools
tool-calling
openai-compatible
anthropic-compatible
claude-code
openclaw
hermes
codex
qwen
deepseek
web-llm
ai-gateway
llm-gateway
webai2api
ds2api
```

## 微信群 / 社群文案

我开源了一个 Agent 工具链网关项目：WebLLM Gateway。

它可以把 Qwen / DeepSeek 这类网页登录模型包装成 OpenAI / Anthropic 兼容 API，给 OpenClaw、Hermes、Claude Code、Codex 等客户端调用。

重点不是简单转发，而是做了工具调用桥接：OpenAI `tool_calls`、Anthropic `tool_use/tool_result` 都能转成网页登录模型可理解的协议，再转回标准结构。Qwen 3.7 系列已经调通。

适合想折腾 Agent runtime、低成本编程模型、网页登录模型工具调用的人试试。

项目地址：
https://github.com/Kriswd/webllm-gateway

## 朋友圈文案

最近把 WebLLM Gateway 开源了。

一句话：把 Qwen / DeepSeek 网页账号接进 OpenClaw、Hermes、Claude Code、Codex，让网页登录模型也能跑标准工具调用。

这个项目主要解决的是 Agent 工具链里的一个痛点：很多网页登录模型本身没有稳定标准 API，更别说 OpenAI / Anthropic 工具调用协议。Gateway 会负责登录态检测、模型目录、协议适配和 ToolBridge，真正的文件读写、命令执行、MCP 权限仍然交给下游客户端，不越界。

Qwen 3.7 系列已经调通，欢迎 Agent 玩家试用、提 issue、给 star。

https://github.com/Kriswd/webllm-gateway

## GitHub / 论坛标题

```text
WebLLM Gateway: OpenAI/Anthropic-compatible gateway for Qwen and DeepSeek web accounts, built for Agent tool calling
```

中文标题：

```text
开源 WebLLM Gateway：把 Qwen / DeepSeek 网页模型接进 OpenClaw、Hermes、Claude Code、Codex 工具调用链路
```

## X / Twitter 文案

```text
I open-sourced WebLLM Gateway.

It turns Qwen / DeepSeek web accounts into OpenAI & Anthropic-compatible APIs for agent runtimes like OpenClaw, Hermes, Claude Code and Codex.

The core part is ToolBridge: web models can participate in standard tool calling flows without executing local tools inside the gateway.

Qwen 3.7 is already wired up.

https://github.com/Kriswd/webllm-gateway
```

## 60 秒 demo 脚本

1. 打开 Gateway 首页，展示 `Qwen 3.7 系列已调通`。
2. 点开 Qwen 平台，展示已授权和模型列表。
3. 复制 `base_url`、`api_key`、`model`。
4. 打开 OpenClaw、Hermes、Claude Code 或 Codex 客户端配置页。
5. 填入 `http://127.0.0.1:8610/v1`、`local-dev-key`、`qwen-web/qwen3.7-max-preview`。
6. 下发任务：读取 README 并总结 Agent 工具调用卖点。
7. 展示客户端触发工具调用。
8. 展示最终回答基于真实文件内容。
9. 收尾：Gateway 只做协议桥接，不执行本地工具，权限仍由客户端控制。

## 首批投放渠道

- OpenClaw、Hermes、Claude Code、Codex、MCP 相关群。
- Qwen / DeepSeek 网页模型折腾群。
- GitHub Trending / 开源项目推荐社群。
- V2EX、掘金、知乎想法、即刻、小红书、B站动态。
- 给 WebAI2API、ds2api、OpenClaw、Hermes 相关讨论区发“兼容/接入经验”，少讲广告，多讲真实 demo。

## 冷启动 Checklist

- GitHub topics 已配置。
- README 首屏能在 10 秒内看懂 Agent 工具调用价值。
- 首页截图能看到 Qwen 3.7 和授权流程。
- 至少准备 1 个真实 Agent 工具调用录屏。
- 发帖时附上 demo 文档，不只贴仓库链接。
- 置顶一个 issue：欢迎反馈 OpenClaw / Hermes / Claude Code / Codex 接入问题。
- 每收到一个用户报错，都沉淀成 FAQ 或 troubleshooting 文档。

## 风险提示

传播时不要承诺“永久免费”“永不封号”“绕过限制”。建议统一表述为：本项目仅做本地学习、研究和技术验证，使用者需要遵守对应网站和模型服务条款。
