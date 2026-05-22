# Qwen 3.7 接入 Agent 工具调用链路 Demo

这个 demo 面向 OpenClaw、Hermes、Claude Code、Codex 等 Agent 工具玩家，用来证明 WebLLM Gateway 不是普通反向代理，而是把网页登录模型接进标准工具调用协议的 Gateway。

## Demo 目标

用 Qwen Web 登录账号作为上游模型，通过 WebLLM Gateway 暴露 OpenAI / Anthropic 兼容 API，让下游 Agent 客户端完成一次真实工具调用任务。

验证点：

- Qwen 3.7 模型出现在 Gateway 模型目录里。
- 下游客户端能使用 `http://127.0.0.1:8610/v1` 和 `local-dev-key` 接入。
- 模型需要工具时，Gateway 返回标准 OpenAI `tool_calls` 或 Anthropic `tool_use`。
- 工具执行仍由下游 Agent 客户端完成，Gateway 不执行本地命令、不读取文件、不绕过权限系统。
- 工具结果回传后，模型能基于真实 observation 继续回答。

## 准备工作

1. 启动 Gateway：

```text
start_webai_gateway.bat
```

2. 打开控制台：

```text
http://127.0.0.1:8610/
```

3. 选择 `Qwen / 通义千问国际版`，点击“打开授权浏览器”，完成网页账号登录。

4. 点击“刷新模型”或“检测模型”，确认可用模型里出现：

```text
qwen-web/qwen3.7-max-preview
qwen-web/qwen3.7-max
qwen-web/qwen3.7-plus-preview
```

## 客户端接入参数

OpenAI-compatible 客户端：

```text
base_url = http://127.0.0.1:8610/v1
api_key = local-dev-key
model = qwen-web/qwen3.7-max-preview
```

Anthropic-compatible 客户端：

```text
base_url = http://127.0.0.1:8610/v1
api_key = local-dev-key
model = qwen-web/qwen3.7-max-preview
endpoint = /v1/messages
```

Claude Code 可用环境变量示例：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8610/v1",
    "ANTHROPIC_AUTH_TOKEN": "local-dev-key",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen-web/qwen3.7-plus-preview",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen-web/qwen3.7-max",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen-web/qwen3.7-max-preview"
  }
}
```

## 推荐录屏任务

在 Agent 客户端里执行一个需要真实工具调用的任务，例如：

```text
请读取当前项目 README，找出它面向 Agent 工具玩家的核心卖点，并给出 3 条可以改进首屏转化率的建议。需要先查看文件再回答。
```

理想执行过程：

1. Agent 客户端把读文件工具 schema 通过 OpenAI / Anthropic 协议发给 Gateway。
2. Gateway 把工具 schema 转成严格 ToolBridge prompt，发给 Qwen Web。
3. Qwen 返回工具调用意图。
4. Gateway 转回标准 `tool_calls` / `tool_use`。
5. Agent 客户端按自己的权限系统读取 README。
6. 工具结果回传 Gateway。
7. Gateway 把 observation 转成网页登录模型可理解的上下文。
8. Qwen 基于真实 README 内容给出最终答案。

## 录屏证据清单

发布 demo 时建议展示这些画面：

- Gateway 首页显示 `Qwen 3.7 系列已调通`。
- 模型列表包含 `qwen-web/qwen3.7-max-preview`。
- 客户端配置里的 `base_url`、`api_key`、`model`。
- Agent 客户端日志里出现工具调用和工具结果。
- Gateway 请求诊断里没有 `x-webai-tool-bridge-error`。
- 最终回答引用了真实读取到的 README 内容。

## 一句话说明

WebLLM Gateway 的边界是协议适配：它把网页登录模型整理成 Agent 工具链能理解的 OpenAI / Anthropic 兼容 API，但不会替客户端执行本地工具，也不会绕过客户端权限系统。
