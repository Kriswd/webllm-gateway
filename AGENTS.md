# WebAI Gateway Agent Guidelines

## 项目定位

WebAI Gateway 是一个独立的网页登录模型 API 网关。它位于 KrisAI、OpenClaw、Hermes、Claude Code 等下游客户端和 Qwen Web、DeepSeek Web、WebAI2API 等网页模型提供方之间。

本项目的核心目标是：把“不稳定的网页模型文本交互”包装成“稳定的 OpenAI / Anthropic 兼容 API”，让下游客户端像使用原生模型 API 一样使用网页登录模型。

当前适配层的核心原则是尽可能 100% 复刻 `ds2api` 的协议行为、错误形态、模型目录、请求历史、工具调用往返和网页登录体验；任何偏差都必须抽象为通用 provider/model capability 或 Gateway 配置，并用 parity/oracle 测试证明。

每次修改适配器、工具桥、provider、请求历史压缩、错误恢复、模型目录或网页登录流程前，都必须主动对标当前锁定的 `ds2api` oracle 版本和相关源码；完成后必须补充 parity/oracle 或等价回归测试来证明仍然复刻 `ds2api` 成功做法。禁止在没有 ds2api 对照和测试证据的情况下把一次性观察写成长期行为。

当前核心产品价值是让网页大模型能够在 Claude Code 中承担编程模型角色，也能够被小龙虾、Hermes 等 OpenAI / Anthropic 兼容客户端直接使用，并实现稳定、可验证、标准化的工具调用。

## 最高优先级边界

- 不为任何单一下游客户端写定制逻辑。不要出现 “KrisAI 专用”、“OpenClaw 专用”、“Hermes 专用”、“Claude Code 专用” 的业务分支；如确实需要兼容差异，必须抽象成标准协议能力、模型能力或可配置的 Gateway 通用策略。
- 不修改下游客户端代码来修复 Gateway 适配问题。KrisAI、OpenClaw、Hermes、Claude Code 的职责是注册工具、执行工具、管理权限和跑 agent loop；Gateway 的职责是协议适配、工具调用格式修复和网页模型交互。
- Gateway 永远不执行本地工具。不得在 Gateway 中执行文件读写、终端命令、浏览器自动化、MCP、插件、技能、代码解释器或任何本地副作用操作。
- Gateway 不接管下游权限系统。文件系统访问、终端权限、MCP 权限、用户确认、审计日志都应留在下游客户端。
- Gateway 不保存或输出敏感凭证。日志、测试输出、错误信息、前端提示都不得暴露 cookie、bearer、session token、API key 或加密后的密钥正文。
- 所有项目用户数据、登录态、凭证目录、浏览器 profile、WebAI2API / ds2api runtime 数据、`data/`、`credentials/`、`.webai-gateway/` 等默认不读取、不修改、不移动、不清空、不删除；健康检查所需的非敏感 metadata 除外。任何清理、重置、迁移或覆盖用户数据的操作都必须先逐项说明风险并获得用户明确确认。
- 每次兼容、加速或降级方案都必须符合产品边界和通用性：优先抽象为标准协议行为、provider/model capability 或可配置 Gateway 策略，禁止为单一模型、单一下游客户端或一次性任务写特判。

## 架构原则

- Gateway 是协议层和适配层，不是 agent runtime。
- 下游客户端发送标准 OpenAI `tools/tool_choice/tool_calls` 或 Anthropic `tools/tool_use/tool_result`。
- 网页模型上游通常不支持原生工具协议，因此 Gateway 使用严格 prompt bridge，把工具 schema 注入为文本协议，并把模型输出转换回标准工具调用结构。
- 工具执行结果由下游客户端以 OpenAI `role=tool` 或 Anthropic `tool_result` 回传，Gateway 只负责压缩、格式化为网页模型可理解的 observation 文本。
- 对网页模型的错误、坏 JSON、未知工具、重复 call id、非 object 参数等情况，要在 Gateway 层校验和修复，不能让下游客户端猜测网页模型文本。
- 如果需要兼容新提供方，优先添加 provider capability、model capability 或通用配置，不要在业务路径中写下游客户端名称特判。

## 工具调用产品规范

- 默认使用 `ToolBridgeV2` 严格模式。
- 不把原生 `tools` 和 `tool_choice` 直接发给网页模型上游。
- 工具调用协议只允许模型输出一个 fenced `tool_json` block。
- 一旦需要工具，assistant content 必须为空；不要混合自然语言解释和工具调用。
- `tool_json` 标准结构为：

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

- Gateway 必须校验工具名在当前允许列表中、`input` 是 object、call id 不重复、每轮调用数量不超过配置。
- 坏 JSON 可 repair 一次；repair 仍失败时返回普通 assistant 响应并附带诊断信息，例如 `x-webai-tool-bridge-error`。
- 不允许模型伪造工具结果。如果文本里出现 `<tool_result>`、`工具返回`、`已读取文件` 等暗示已执行工具的内容，但没有真实标准 tool call，应按普通文本返回并记录 warning metadata。
- 大型 observation 要压缩，保留长度、前后片段和可再次读取提示，避免把上下文撑爆。
- 路径列表、搜索结果、日志片段等 observation 归一化必须走通用配置（例如 `tool_bridge.observationPolicy`），默认可过滤依赖/构建/cache 路径，但不得写成 Qwen、Claude Code 或 `/init` 专用逻辑。

## 工具暴露策略

- Gateway 可以对“暴露给网页模型看的工具 schema”做通用安全过滤，但不能替下游执行权限判断。
- prompt bridge 默认隐藏本地运行时类危险工具，例如 `terminal`、`shell`、`powershell`、`bash`、`cmd`、`python`、`python_repl`、`process`、`write_file`、`edit_file`、`apply_patch`、`browser`、`computer`、`generate_image`、`generate_video`。
- 这些工具被隐藏的含义是：不要把它们作为网页模型可直接选择的 tool schema 注入 prompt。下游客户端仍可拥有这些工具，但网页模型应该通过更窄、更可控的读类工具或上层安全工具完成任务。
- 优先暴露只读、检索、获取信息类工具，例如 `fetch_url`、`web_search`、`read_file`、`list_dir`、`search_*`、`get_*`、`query_*`。
- 如果某个客户端需要写文件或跑命令，应该由下游 agent loop 在自己的权限系统中决定，而不是让网页模型直接选择底层运行时工具。
- 新增过滤规则必须是 Gateway 通用规则，不能以某个下游客户端名称作为条件。

## OpenAI 兼容接口

- `/v1/chat/completions` 是主要 OpenAI-compatible endpoint。
- 保持请求和响应结构尽量兼容 OpenAI Chat Completions。
- 对工具调用响应，返回标准 `message.tool_calls`，`finish_reason` 使用 `tool_calls`。
- 流式模式可以先缓冲完整网页模型输出，再发送标准 tool call chunk，优先保证协议稳定，不追求半截 JSON 的实时透传。
- 不支持的字段应明确降级或报错，不能静默产生误导性行为。

## Anthropic 兼容接口

- `/v1/messages` 是 Claude Code 等 Anthropic-compatible 客户端的最小闭环接口。
- 支持文本消息、`tools`、`tool_use`、`tool_result`、`tool_choice`、`max_tokens`、`stream=false`。
- 暂不支持 image blocks、computer-use blocks、server-side thinking 等复杂能力；遇到不支持字段要返回明确 400。
- Anthropic 工具协议最终仍复用同一套 ToolBridgeV2，不维护两套工具解析逻辑。

## WebAI2API 和网页登录提供方

- WebAI2API 应作为可复用 sidecar 或 upstream 能力接入，优先复用其已有登录管理、状态页、工作池、适配器、缓存、日志和请求历史体验。
- Qwen Web、DeepSeek Web 等 direct provider 的网页登录授权只负责获取真实网页登录态，不把工具执行能力塞进 provider。
- 授权完成判定必须严格。访客 cookie、空 bearer、空 session token 不能被视为已授权。
- `/v1/models` 可以返回非标准 `capabilities` metadata，但不能破坏 OpenAI 兼容客户端读取模型列表。

## 前端和用户体验

- 与用户的所有对话默认使用简体中文，包括进度更新、问题澄清、最终回复、验证报告和错误说明；除非用户明确要求使用其他语言。
- 所有新增或修改的用户可见界面默认使用简体中文。
- 保留产品名、协议名、API path、模型 ID、配置键等英文技术标识。
- 面向小白用户的流程要优先“一键授权浏览器登录 -> 自动检测真实登录态 -> 展示可用模型 -> 复制接入信息”。
- 不要求用户手动找 cookie、bearer 或 session token。
- 对 WebAI2API 已经做得好的界面和流程，优先复用或贴近原流程，不重复造轮子。
- 错误提示必须说明下一步该做什么，但不得暴露敏感凭证。

## 配置原则

- 默认配置应该能在本机直接启动：
  - 控制台：`http://127.0.0.1:8610/`
  - OpenAI API：`http://127.0.0.1:8610/v1`
- `tool_bridge.mode` 支持 `off | prompt | strict`，默认应为 `strict`。
- `tool_bridge.activationPolicy` 支持 `auto | always | off`，默认应为 `auto`：普通问答和 provider 原生联网问答不注入工具桥，本地 agent 任务才注入 ToolBridge prompt。
- `providerRuntime.requestTimeoutSeconds` 是网页登录 provider 的通用单次请求超时，默认偏向任务成功；不要在 provider 代码里硬编码短超时或为某个模型名写专用超时。
- `providerRuntime.promptMaxChars` 是网页登录 provider 的通用输入预算；超过预算时应压缩中间上下文、保留协议指令和最新用户任务，不能把一次性 `/init`、某个模型或某个客户端写成特判。
- `providerRuntime.nativeWebSearchPolicy` 支持 `auto | force | off`，默认应为 `auto`；只能作为 provider capability 传递，Gateway 不自行联网搜索或执行搜索工具。
- `tool_bridge.toolPromptMaxChars` 是工具 manifest 注入预算；工具过多时要压缩 schema/description 但保留全部真实工具名，不得发明 ToolSearch 或隐藏 `exposurePolicy=all` 下的真实工具。
- 新增配置必须同时考虑 `config.example.json`、`config_to_public`、`config_to_admin`、`load_config`、`update_config` 和前端展示。
- 不要把真实 token、cookie、bearer、session 写入示例配置、文档或测试 fixture。

## 测试要求

- 修改协议、工具桥、provider、授权或前端关键流程时，必须补充或调整测试。
- Gateway 后端最小验证命令：

```powershell
python -m pytest -q
```

- 前端有构建链路时，运行：

```powershell
cd webui
pnpm build
```

- 涉及真实网页登录模型时，至少用一个无敏感内容的小请求验证：
  - 普通聊天能返回文本。
  - 工具请求能返回标准 `tool_calls`。
  - 工具结果回传后能继续回答。
- 验证日志和最终报告中不得包含 cookie、bearer、session token、API key。

## 开发流程约束

- 这个目录目前可能不是独立 git 仓库；改动前先检查 `git status`，不要假设可以提交。
- 所有代码开发任务完成并通过端到端验证后，必须在确认不会覆盖用户改动的前提下更新本地 `main` 和远程 `origin/main`，然后启动本机最新 Gateway 代码供用户验收；如果仓库没有远程、推送失败或存在无法安全合并的用户改动，必须明确报告阻塞原因和当前分支状态。
- 每天首次开发、验收或排查 ds2api 相关链路前，必须检查一次 `CJackHwang/ds2api` 是否有新源码：先执行 `git -C .tmp/ds2api fetch origin main`（GitHub 网络超时时按本文件代理规则重试），再对比 `origin/main` 与 `webai_gateway/ds2api_oracle.py` 中锁定的 commit/version；如果有更新，必须先更新本地 `.tmp/ds2api` 源码和 oracle commit/version，运行 parity/oracle 与 Gateway 回归测试通过后再继续。不得清理或覆盖 ds2api runtime 数据、登录态、凭证目录。
- 优先改 Gateway 自己的模块和测试，不要跨目录修改 KrisAI、OpenClaw、Hermes 或 Claude Code。
- 如果必须改下游示例或文档，只能作为接入说明，不得改变下游运行时代码。
- 修改文本文件使用 `apply_patch`，避免脚本化批量重写造成编码和无关 diff。
- Windows 下运行命令前设置 UTF-8，避免中文输出和文件内容乱码。
- 每次完成 Gateway 代码、配置、启动脚本或关键文档改动并完成验证后，必须重启本机 Gateway 服务，让 `http://127.0.0.1:8610/` 运行最新代码供用户验收。重启后必须检查 `/health` 返回 `ok=true`、`runtime.sourceFresh=true`、`runtime.sourceStale=false`，并在最终回复中报告新进程 PID 和健康检查结果。不得只说“已修复”而不重启正在运行的服务。
- 不做无关重构，不移动目录，不改端口、不改默认 API key、不改启动脚本行为，除非任务明确要求。

## 典型反模式

- 为 KrisAI 写一个 Gateway 分支，检测到 KrisAI 就改变工具行为。
- 为了让某个任务跑通，在下游客户端里隐藏或改写工具 schema。
- 在 Gateway 里直接执行 `terminal`、读取本地文件、连接 MCP 或调用浏览器自动化。
- 把网页模型返回的自然语言“我已读取文件”当成真实工具结果。
- 把所有工具无差别注入网页模型 prompt，导致模型选择底层 shell 命令。
- 在错误日志中打印完整请求头、cookie、bearer 或本地 API key。

## 正确改法示例

- 如果网页模型经常选错底层工具：在 Gateway 的通用工具暴露策略中收紧 schema，而不是改某个下游客户端。
- 如果某个 provider 输出工具 JSON 不稳定：增强 ToolBridgeV2 parser、repair prompt 或 provider capability，而不是让下游解析网页模型文本。
- 如果 Claude Code 需要 Anthropic 结构：在 `/v1/messages` 做标准协议转换，而不是让 Claude Code 使用 OpenAI 私有格式。
- 如果 KrisAI 需要 MCP：在 KrisAI 的 tool registry 接 MCP，Gateway 仍只看见标准 tool schema 和 tool result。
