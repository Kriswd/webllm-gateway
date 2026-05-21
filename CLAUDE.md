# WebLLM Gateway Claude Code Guide

## 项目定位

WebLLM Gateway 是一个独立的网页登录模型 API 网关，位于 Claude Code、KrisAI、OpenClaw、Hermes 等下游客户端与 Qwen Web、DeepSeek Web、WebAI2API 等网页模型提供方之间。

核心目标是把不稳定的网页登录模型文本交互包装成稳定的 OpenAI / Anthropic 兼容 API，并通过严格工具桥把网页模型输出转换为标准 `tool_calls` / `tool_use`。

适配层核心原则是尽可能 100% 复刻 `ds2api` 的协议行为、错误形态、模型目录、请求历史、工具调用往返和网页登录体验；任何偏差都必须抽象为通用 provider/model capability 或 Gateway 配置，并用 parity/oracle 测试证明。

每次修改适配器、工具桥、provider、请求历史压缩、错误恢复、模型目录或网页登录流程前，都必须主动对标当前锁定的 `ds2api` oracle 版本和相关源码；完成后必须补充 parity/oracle 或等价回归测试来证明仍然复刻 `ds2api` 成功做法。禁止在没有 ds2api 对照和测试证据的情况下把一次性观察写成长期行为。

核心产品价值是让网页大模型能够在 Claude Code 中承担编程模型角色，也能够被小龙虾、Hermes 等 OpenAI / Anthropic 兼容客户端直接使用，并实现稳定、可验证、标准化的工具调用。

## 架构边界

- Gateway 只做协议适配、工具调用格式修复、网页登录模型交互和错误降级。
- Gateway 永远不执行本地工具，不读取或写入用户项目文件，不运行终端命令，不连接 MCP，不接管下游权限系统。
- 下游客户端负责 agent loop、工具注册、MCP、Skill、文件系统、终端权限和用户确认。
- 不为单一下游客户端写业务分支；兼容差异应抽象为标准协议能力、provider capability、model capability 或通用配置。
- 不保存、不输出 cookie、bearer、session token、API key 或其他敏感凭证。
- 所有项目用户数据、登录态、凭证目录、浏览器 profile、WebAI2API / ds2api runtime 数据、`data/`、`credentials/`、`.webai-gateway/` 等默认不读取、不修改、不移动、不清空、不删除；健康检查所需的非敏感 metadata 除外。任何清理、重置、迁移或覆盖用户数据的操作都必须先逐项说明风险并获得用户明确确认。

## 关键模块

- `webai_gateway/app.py`: FastAPI 应用、OpenAI / Anthropic 路由、WebAI2API 代理、direct provider 入口。
- `webai_gateway/openai_api.py`: OpenAI-compatible 请求构造、响应解析、工具桥集成。
- `webai_gateway/anthropic_api.py`: Anthropic `/v1/messages` 与 OpenAI chat/completions 的双向转换。
- `webai_gateway/tool_bridge.py`: ToolBridgeV2 严格 prompt 协议、工具 schema 过滤、工具 JSON 解析/repair、tool_result observation 压缩。
- `webai_gateway/qwen_web.py`: Qwen Web 国际版直连登录态调用、SSE 解析、超时和多模态拒绝。
- `webai_gateway/deepseek_web.py`: DeepSeek Web 直连适配。
- `webai_gateway/web_auth.py`: 网页登录授权、provider/model catalog、credential store。
- `webui/`: Vue 3 控制台前端。

## 常用命令

```powershell
python -m pytest -q
```

```powershell
cd webui
pnpm build
```

```powershell
python -m webai_gateway
```

控制台默认地址：

```text
http://127.0.0.1:8610/
```

API 默认地址：

```text
http://127.0.0.1:8610/v1
```

## 工具桥规范

- 默认使用 `tool_bridge.mode=strict`。
- 不把原生 `tools` / `tool_choice` 直接发给网页登录模型。
- 网页模型需要工具时只能输出一个 fenced `tool_json` block：

```json
{
  "calls": [
    {
      "id": "call_1",
      "name": "tool_name",
      "input": {}
    }
  ]
}
```

- Gateway 必须校验工具名、输入对象、重复 call id 和单轮调用数量。
- 坏 JSON 可 repair 一次；repair 仍失败时返回普通 assistant 响应并附带诊断头。
- 不允许把自然语言里的“我已读取文件/工具返回”当作真实工具结果。
- 大型 `tool_result` 要压缩；路径列表通过 `tool_bridge.observationPolicy` 过滤依赖/构建/cache 目录，例如 `node_modules`、`.pnpm`、`.git`、`dist`、`build`、`.venv`、`__pycache__`。
- 工具太多或 schema 太大时，通过 `tool_bridge.toolPromptMaxChars` 压缩工具 manifest：保留全部真实工具名，压缩 description/schema，不允许发明不存在的 ToolSearch。

## Qwen Web 注意事项

- `qwen-web/*` 直连走网页登录态，不支持原生工具协议；普通问答可直连 provider，只有本地 agent 任务经 ToolBridgeV2。
- Qwen Web 直连目前只声明文本能力；遇到图片/文档附件应明确拒绝，不伪造上传成功。
- Qwen SSE 要逐行消费；拿到完整 `tool_json` 后应尽早返回，不等待上游关闭流。
- Qwen Web 超时应带 `x-should-retry: false`，避免下游对网页登录模型超时进行长时间自动重试。

## 开发约束

- 新增或修改用户可见界面默认使用简体中文。
- 所有代码开发任务完成并通过端到端验证后，必须在确认不会覆盖用户改动的前提下更新本地 `main` 和远程 `origin/main`，然后启动本机最新 Gateway 代码供用户验收；如果仓库没有远程、推送失败或存在无法安全合并的用户改动，必须明确报告阻塞原因和当前分支状态。
- 修改协议、工具桥、provider、授权或前端关键流程时必须补充测试。
- 优先做小而可验证的改动，不做无关重构。
- 文档、测试 fixture、日志和错误信息不得包含真实 token、cookie、bearer 或 session。
- 修改配置项时同步考虑 `config.example.json`、`config_to_public`、`config_to_admin`、`load_config`、`update_config` 和前端展示。
- 每次兼容、加速或降级方案都必须符合产品边界和通用性：抽象为标准协议能力、provider/model capability 或可配置 Gateway 策略，不写模型名、客户端名或一次性任务专用分支。
- 网页登录 provider 的请求超时走 `providerRuntime.requestTimeoutSeconds`，默认 180 秒；不要在 Qwen、DeepSeek 或某个下游路径里硬编码短超时。
- 网页登录 provider 的输入预算走 `providerRuntime.promptMaxChars`，默认 12000 字符；超出时保留协议和最新任务、压缩中间上下文，不写 `/init` 专用逻辑。
- 工具桥激活走 `tool_bridge.activationPolicy`，默认 `auto`：普通问答和联网问答优先 provider 原生能力，本地文件/命令/MCP/Skill/agent loop 任务才注入 ToolBridge prompt。
- provider 原生联网走 `providerRuntime.nativeWebSearchPolicy`，默认 `auto`；Gateway 只传递 provider capability，不自己联网搜索或执行下游工具。

## 验证清单

- 后端协议或工具桥改动后运行 `python -m pytest -q`。
- 前端改动后运行 `cd webui; pnpm build`。
- 涉及网页登录模型时，至少验证普通聊天、工具请求、工具结果回传后继续回答。
- 声称修复完成前，必须使用新进程或 fresh command 重新验证。
