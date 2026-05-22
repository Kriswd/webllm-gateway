# WebLLM Gateway

WebLLM Gateway 是一个独立的网页登录模型 API 网关：把 Qwen Web、DeepSeek Web 和已验证的网页登录模型包装成 OpenAI / Anthropic 兼容、可工具调用的标准 API，让 OpenClaw、Hermes、Claude Code、Codex 等客户端像调用原生模型 API 一样使用网页账号。

> Turn web AI accounts into OpenAI/Anthropic-compatible, tool-callable APIs.

项目地址：[github.com/Kriswd/webllm-gateway](https://github.com/Kriswd/webllm-gateway)

一句话：登录网页登录模型账号，自动检测可用模型，把地址、Key 和模型 ID 填到兼容 OpenAI / Anthropic API 的客户端里，就能把网页模型接入工具调用工作流。

它的核心价值不是简单转发，而是把“不稳定的网页对话”整理成“可验证、可接入、可工具调用”的标准 API：

- 一键网页登录授权，自动检测真实登录态和可用模型，不要求用户手动找 cookie、bearer 或 session。
- 同时提供 OpenAI `/v1/chat/completions` 和 Anthropic `/v1/messages`，覆盖常见兼容客户端。
- `ToolBridgeV2` 将 OpenAI `tools/tool_calls` 与 Anthropic `tool_use/tool_result` 统一桥接到网页登录模型，返回标准工具调用结构。
- Gateway 只做协议适配、错误修复、模型目录和登录管理，不执行本地文件、终端、浏览器或 MCP 工具；权限和副作用仍由下游客户端控制。
- Qwen direct、DeepSeek via ds2api、WebAI2API sidecar 都被抽象为 provider/runtime 能力，避免为单一客户端写临时特判。

项目本体是 Gateway：API、ToolBridge、网页登录授权、模型目录、Qwen direct 和客户端接入配置。WebAI2API 与 ds2api 不作为源码混入本仓库，只作为按需启用的外部 adapter runtime；未安装它们时，Gateway 核心服务仍可启动。

## 界面预览

![WebLLM Gateway 核心界面](docs/assets/webllm-gateway-home.png)

## 项目优势

- **协议稳定**：优先保持 OpenAI / Anthropic 兼容响应结构，工具调用返回标准 `tool_calls` 或 `tool_use`，不把网页模型的自然语言幻想当成真实工具结果。
- **边界清楚**：Gateway 不接管 agent loop、不执行本地工具、不绕过客户端权限系统，适合作为 OpenClaw、Hermes、Claude Code、Codex 等客户端前面的统一协议层。
- **可对标验证**：DeepSeek 链路以 ds2api 作为外部 oracle 对照；Qwen 链路持续用真实请求和 parity/oracle 测试回归。
- **面向小白**：前端优先展示授权、模型检测、接入地址、API Key 和模型 ID；高级 adapter runtime 页面只作为诊断入口。
- **可开源维护**：第三方 runtime 不混入源码树，许可证边界、敏感数据边界和发布检查写在文档里。

## 支持作者

如果 WebAI Gateway 帮你少踩坑，可以通过作者的小店支持，有稳定可靠有质保的plus成品账号、Claude代充等，欢迎选购。
[支持作者 / 服务咨询](https://pay.ldxp.cn/shop/FTIWLFHQ)。交流群：1105908706。

## 免责声明

本项目仅供学习、研究和本地技术验证使用，不提供任何官方模型服务、账号服务或规避平台限制的保证。

WebLLM Gateway 不是 OpenAI、Anthropic、Qwen、DeepSeek、ChatGPT、WebAI2API 或 ds2api 的官方项目，也不代表这些平台或项目的立场。使用者应自行确认并遵守所接入网站、模型服务和第三方 runtime 的服务条款、账号规则、访问限制以及所在地法律法规。

请勿将本项目用于绕过付费限制、规避风控、批量注册或滥用账号、未经授权的数据采集、攻击测试，或任何违法违规场景。因使用、部署、分发或二次开发本项目产生的账号风险、服务限制、数据合规风险或法律责任，由使用者自行承担。

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

### Docker 部署

如果希望跨 Windows / macOS / Linux 使用同一套运行方式，可以直接用 Docker Compose 启动 Gateway：

```powershell
docker compose up -d --build
```

启动后打开：

```text
http://127.0.0.1:8610/
```

Docker 版本会把 `config.json`、授权凭据、账号检测缓存和运行日志保存在 `webai-gateway-data` volume 里，不会写进镜像。首次启动会自动生成容器内配置，并把 Gateway 监听地址设为 `0.0.0.0:8610`，对外仍通过 `http://127.0.0.1:8610/` 访问。

WebAI2API / ds2api 仍然是可选第三方 adapter runtime，不会被混入本项目镜像。`docker-compose.yml` 默认通过 `host.docker.internal` 访问宿主机上的：

```text
http://host.docker.internal:8500/v1
http://host.docker.internal:9331/v1
```

如果这两个 runtime 跑在别的机器或别的容器里，请修改 `WEBAI_UPSTREAM_BASE_URL` 和 `WEBAI_DEEPSEEK_DS2API_BASE_URL`。网页登录授权推荐在宿主机启动带 CDP 的 Chrome / Edge，再让容器访问 `http://host.docker.internal:9222`；`WEBAI_DEFAULT_CDP_URL` 已在 compose 中给出默认值。

Windows 宿主机可参考：

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-address=0.0.0.0 --remote-debugging-port=9222 --user-data-dir="$env:TEMP\webai-gateway-cdp"
```

更多安装、依赖和启动排障见 [docs/installation.md](docs/installation.md)。

## 登录授权

打开统一控制台后，先进入“网页登录向导”：

- 首页默认只展示当前已经实际验证可用的网页登录通路，避免把未验证站点当成可用产品能力。
- DeepSeek Web、Qwen / 通义千问国际版、Qwen Coder：点击“打开授权浏览器”，在弹出的浏览器里登录。网关检测到真实登录态后才会显示已授权。
- NAS、Docker 或远程机器部署时，推荐在你正在使用的电脑上启动带 CDP 的 Chrome / Edge，把控制台里的 CDP 地址改成 `http://电脑IP:9222` 后再检测登录态；如果你拿到的是包含 `cookie`、`bearer`、`session_token` 等凭据的授权回调 URL，也可以点“远程/NAS 授权”粘贴导入。普通登录页 URL 不包含登录态，Gateway 会明确拒绝并提示改用远程 CDP。
- ChatGPT：通过可选 WebAI2API adapter runtime 接入图片生成；点击“登录或修复账号”后 Gateway 会自动准备隔离 worker，完成网页登录授权并恢复 API 后即可检测和调用已验证模型，不需要手动进入外部后台配置工作池。
- 授权完成后，在“可用模型”里复制模型 ID，填到 OpenClaw、Hermes、Claude Code、Codex 或其它兼容客户端。
- “接入客户端”区域可以复制 OpenAI / Anthropic 兼容地址和 API Key，也可以重新生成本地网关令牌。

Gateway 首页优先展示统一接入、状态、授权和复制配置。外部 adapter runtime 的状态概览、工作池、浏览器、缓存、日志、请求历史和接口测试页面只保留为高级诊断入口，不作为主产品界面。

## 文档索引

- 安装和单入口启动：[docs/installation.md](docs/installation.md)
- 图片生成：[docs/media-generation.md](docs/media-generation.md)
- 可选 adapter runtime 说明：[docs/third-party-runtime.md](docs/third-party-runtime.md)
- 架构和边界：[docs/architecture.md](docs/architecture.md)
- 贡献规范：[CONTRIBUTING.md](CONTRIBUTING.md)
- 安全策略：[SECURITY.md](SECURITY.md)
- 第三方声明：[NOTICE.md](NOTICE.md)
- 第三方许可证清单：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

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

OpenAI 兼容客户端：

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

网页登录模型的单次请求超时由通用 `providerRuntime.requestTimeoutSeconds` 控制，默认 300 秒。工具调用 JSON 一旦完整返回仍会提前结束请求，因此提高该值主要保护 `/init`、项目总结、长文档归纳等慢任务，不会刻意拖慢快速工具选择。

网页登录模型的输入预算由 `providerRuntime.promptMaxChars` 控制，默认 32000 字符。超过预算时会保留开头、WebLLM Gateway 工具协议和最后的用户任务，压缩中间的大段技能列表、规则列表或历史上下文，避免网页模型在第一轮请求里长时间无输出。

## 图片生成

Gateway 暴露 OpenAI-compatible 媒体接口：

```text
POST /v1/images/generations
```

图片生成建议优先使用 `gpt-image-2`，兼容已验证的 `gpt-image-1.5`。Gateway 会把参考图参数转成 WebAI2API 多模态消息；首页“图片生成测试”可以直接执行一次 smoke test，并预览返回图片。

未通过真实链路验证的媒体模型当前先从用户入口和模型目录关闭，不作为开源首发能力宣传。

## 模型目录

Qwen / 通义千问国际版本地直连模型：

```text
qwen-web/qwen3.6-max-preview
qwen-web/qwen3.6-plus
qwen-web/qwen3.5-plus
qwen-web/qwen3-max
```

DeepSeek Web 已改为通过本地 `ds2api` sidecar 接入。完成浏览器网页登录授权后，需要点击“检测模型”做一次真实请求验证；检测通过后，前端和 `/v1/models` 才应作为可用模型使用：

```text
deepseek-v4-pro
```

默认 sidecar 地址是 `http://127.0.0.1:9331/v1`，可在控制台设置页通过 `providerRuntime.deepseekDs2apiBaseUrl` 调整。

WebAI2API 支持且已验证开放的站点和模型会继续透传并合并到 `/v1/models`。未验证的媒体/视频模型会被过滤，避免用户误用。模型元数据里会包含非标准字段：

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
    "requestTimeoutSeconds": 300,
    "promptMaxChars": 32000,
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
- `POST /api/admin/web-auth/callback-url`
- `POST /api/admin/web-auth/jobs`
- `GET /api/admin/web-auth/jobs/{job_id}`
- `GET /api/admin/web-auth/credentials`
- `DELETE /api/admin/web-auth/credentials/{provider_id}`
- `GET /v1/models`
- `POST /v1/images/generations`
- `POST /v1/chat/completions`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`

## 开发验证

```powershell
python -m pytest -q
cd webui
corepack pnpm build
python -m webai_gateway
```

修改 provider、工具桥、模型目录、授权流程或 ds2api 相关链路前，请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，并检查当前锁定的 ds2api oracle 是否仍是最新上游源码。
