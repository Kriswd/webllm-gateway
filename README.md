# WebAI Gateway

WebAI Gateway 是一个独立的网页登录模型 API 网关。它把 Qwen Web、DeepSeek Web、WebAI2API 等网页登录模型包装成稳定的 OpenAI 兼容接口，并提供严格工具调用适配层，方便 KrisAI、OpenClaw、Hermes、Claude Code 等客户端接入。

网关只做协议适配和格式修复，不执行本地文件、终端、浏览器或 MCP 工具。工具和 MCP 仍由客户端自己的权限系统执行。

## 启动

双击启动：

```text
start_webai_gateway.bat
```

打开统一控制台：

```text
http://127.0.0.1:8610/
```

停止服务：

```text
stop_webai_gateway.bat
```

## 登录授权

打开统一控制台后，先进入“网页登录向导”：

- DeepSeek Web、Qwen / 通义千问国际版：点击“打开授权浏览器”，在弹出的浏览器里登录。网关检测到真实登录态后才会显示已授权。
- ChatGPT、Gemini、LMArena、豆包、Sora 等 WebAI2API 站点：在 Gateway 控制台点击 WebAI2API 原生管理入口完成授权。
- 授权完成后，在“可用模型”里复制模型 ID，填到 KrisAI、OpenClaw、Hermes 或 Claude Code。
- “接入 KrisAI”区域可以复制 OpenAI 兼容地址和 API Key，也可以重新生成本地网关令牌。

Gateway 首页优先展示统一接入、状态、授权和复制配置。WebAI2API 原来的状态概览、工作池、适配器、浏览器、虚拟显示器、缓存、日志、请求历史和接口测试页面保留在控制台的“WebAI2API 原生管理台”高级入口中。

## 工具调用适配层

默认启用 `ToolBridgeV2` 严格模式：

- 下游客户端可以按原生 OpenAI `tools/tool_choice/tool_calls` 调用。
- 网关不会把原生 `tools` 发送给网页登录模型，而是注入严格的 `tool_json` 协议提示。
- 模型需要工具时只能输出一个 fenced `tool_json` 块，结构为 `{ "calls": [{ "id", "name", "input" }] }`。
- 网关校验工具名、参数对象、重复 ID、调用数量，并把合法输出转换成标准 `tool_calls`。
- JSON 损坏时会自动发起一次 repair；仍失败时返回诊断头 `x-webai-tool-bridge-error`。
- OpenAI `role=tool` 或 Anthropic `tool_result` 会转换成网页登录模型可读的 observation 文本，过长结果会自动压缩。

网关还提供 Claude Code 可用的最小 Anthropic 兼容接口：

```text
POST /v1/messages
POST /v1/messages/count_tokens
```

Anthropic 兼容接口支持文本、流式事件、`tool_use` / `tool_result`、`x-api-key` 或 `Authorization: Bearer` 鉴权、近似 token 计数，以及图片/文档块的标准化转换。Qwen Web 直连目前只声明文本能力；收到多模态附件时会明确拒绝，避免伪装成已经完成真实上传。

## 客户端配置

KrisAI 或其他 OpenAI 兼容客户端：

```text
base_url = http://127.0.0.1:8610/v1
api_key = local-dev-key
model = qwen-web/qwen3.6-max-preview
```

Claude Code / Anthropic 兼容客户端可使用：

```text
base_url = http://127.0.0.1:8610/v1
api_key = local-dev-key
model = qwen-web/qwen3.6-max-preview
endpoint = /v1/messages
```

Claude Code Best 可在 `/login` 里选择 `Anthropic Compatible`，或写入 `~/.claude/settings.json`：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8610/v1",
    "ANTHROPIC_AUTH_TOKEN": "local-dev-key",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen-web/qwen3.5-plus",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen-web/qwen3.6-plus",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen-web/qwen3.6-max-preview"
  }
}
```

如果希望 Skill、MCP、Bash、Read、Edit、Write、TodoWrite、computer-use 等 Claude Code 工具都能被 Qwen 请求，请在网关配置中设置：

```json
{
  "tool_bridge": {
    "mode": "strict",
    "activationPolicy": "always",
    "exposurePolicy": "all"
  }
}
```

默认 `tool_bridge.activationPolicy=auto` 会把普通问答和联网问答直接交给网页登录 provider，自带联网能力的 provider 可通过 `providerRuntime.nativeWebSearchPolicy=auto` 在“联网、最新、官网、网址”等问题里启用原生搜索；`/init`、读文件、项目分析、MCP/Skill 等本地 agent 任务仍走 ToolBridgeV2。这个策略是按任务意图和 provider capability 分流，不针对某个客户端或某个模型写特判。

网关仍不会执行任何本地工具。所有 Skill 展开、MCP 调用、文件编辑、终端命令和权限确认都由下游客户端自己完成；网关只负责把网页模型输出转换成标准 `tool_use` / `tool_calls`，再把 `tool_result` 转回网页登录模型可理解的 observation。

推荐让客户端继续负责 agent loop、工具执行、MCP 权限和文件系统访问；网关只负责“网页登录模型文本 ↔ 标准工具协议”的转换。

如果某类工具结果会把上下文撑爆，例如 `Glob("*.md")` 返回大量 `node_modules`、`.pnpm`、`dist`、`build` 路径，应优先调整通用的 `tool_bridge.observationPolicy`，不要为某个模型或客户端写专用分支。

网页登录模型的单次请求超时由通用 `providerRuntime.requestTimeoutSeconds` 控制，默认 180 秒。工具调用 JSON 一旦完整返回仍会提前结束请求，因此提高该值主要保护 `/init`、项目总结、长文档归纳等慢任务，不会刻意拖慢快速工具选择。

网页登录模型的输入预算由 `providerRuntime.promptMaxChars` 控制，默认 12000 字符。超过预算时会保留开头、WebAI Gateway 工具协议和最后的用户任务，压缩中间的大段技能列表、规则列表或历史上下文，避免网页模型在第一轮请求里长时间无输出。

## 模型目录

Qwen / 通义千问国际版本地直连模型：

```text
qwen-web/qwen3.6-max-preview
qwen-web/qwen3.6-plus
qwen-web/qwen3.5-plus
qwen-web/qwen3-max
```

DeepSeek Web 已改为通过本地 `ds2api` sidecar 接入。完成浏览器网页登录授权后，前端和 `/v1/models` 只展示已经做过端到端验证的模型：

```text
deepseek-v4-pro
```

默认 sidecar 地址是 `http://127.0.0.1:9331/v1`，可在控制台设置页通过 `providerRuntime.deepseekDs2apiBaseUrl` 调整。

WebAI2API 支持的站点和模型会继续透传并合并到 `/v1/models`。模型元数据里会包含非标准字段：

```json
{
  "capabilities": {
    "tool_bridge": true,
    "supports_native_tools": false,
    "preferred_protocol": "openai"
  }
}
```

## 配置示例

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8610,
    "apiKey": "local-dev-key"
  },
  "upstream": {
    "baseUrl": "http://127.0.0.1:8500/v1",
    "apiKey": "",
    "model": "webai2api-model",
    "toolMode": "prompt"
  },
  "providerRuntime": {
    "requestTimeoutSeconds": 180,
    "promptMaxChars": 12000,
    "nativeWebSearchPolicy": "auto",
    "responseLanguage": "zh-CN",
    "deepseekDs2apiBaseUrl": "http://127.0.0.1:9331/v1"
  },
  "tool_bridge": {
    "mode": "strict",
    "maxToolsInPrompt": 32,
    "maxCallsPerTurn": 1,
    "maxReadonlyCallsPerTurn": 4,
    "toolPromptMaxChars": 8000,
    "observationMaxChars": 4000,
    "exposurePolicy": "safe",
    "allowedToolNames": [],
    "observationPolicy": {
      "summarizePathLists": true,
      "excludedPathParts": [
        ".cache",
        ".git",
        ".pnpm",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "vendor"
      ],
      "excludedPathGlobs": [],
      "pathListMaxItems": 80
    }
  }
}
```

## API

- `GET /`
- `GET /health`
- `GET /admin/*`
- `GET /api/admin/config`
- `PUT /api/admin/config`
- `POST /api/admin/token/rotate`
- `GET /api/admin/web-auth/providers`
- `POST /api/admin/web-auth/browser/start`
- `POST /api/admin/web-auth/jobs`
- `GET /api/admin/web-auth/jobs/{job_id}`
- `GET /api/admin/web-auth/credentials`
- `DELETE /api/admin/web-auth/credentials/{provider_id}`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

## 开发验证

```powershell
python -m pytest -q
cd webui
pnpm build
python -m webai_gateway
```
