# Open Source Readiness

这份清单用于把 WebAI Gateway 从本地实验项目整理成可公开发布的仓库。

## 当前状态

- 核心边界清晰：Gateway 只做协议适配、工具桥和网页登录 provider 交互，不执行本地工具。
- 后端已有较厚测试覆盖：OpenAI、Anthropic、ToolBridge、provider、ds2api parity、replay fixtures。
- 运行体验已收敛为单入口：`start_webai_gateway.bat` 通过 Gateway runtime supervisor 托管 WebAI2API / ds2api 内部 runtime，前端默认只暴露网页登录向导、模型可用性和接入信息。
- 本地运行态已在 `.gitignore` 中排除：`config.json`、`data/`、`credentials/`、`.webai-gateway/`、日志和 `.codex-logs/`。
- README 已补齐安装、登录、客户端配置、媒体接口、第三方 runtime、贡献和安全入口。
- 已补齐 `LICENSE`、`NOTICE.md`、`CONTRIBUTING.md`、`SECURITY.md`、`.github/workflows/ci.yml`。
- 首页已加入 `gpt-image-2` 图片生成 smoke test，便于开源后快速验证 ChatGPT / WebAI2API 媒体链路。

## 开源前必须完成

- 已选择并添加 MIT `LICENSE`；第三方 runtime 授权差异记录在 `NOTICE.md` 和 [third-party-runtime.md](third-party-runtime.md)。
- 决定 `webui/dist/` 是否继续跟踪。如果希望开箱即用，可保留构建产物；如果希望源码发布更干净，应从仓库移除并在 README 中要求 `pnpm build`。
- 清理或归档 `docs/superpowers/plans/` 中的内部开发计划。它们对开发有价值，但包含大量本地路径、历史决策和临时语境；公开仓库建议迁移到 `docs/dev-history/` 或删减。
- 将示例路径从 `E:/ProjectX/...`、`C:\Users\...` 泛化为 fixture 路径，至少避免在 README 和正式文档中出现个人机器路径。测试 fixture 中的路径可保留为协议样本，但应说明是匿名化测试数据。
- 已明确 WebAI2API sidecar 依赖方式：作为外部 prerequisite 或本地 sidecar 目录，详见 [installation.md](installation.md) 和 [third-party-runtime.md](third-party-runtime.md)。
- 已给 ds2api oracle 增加公开可复现说明：如何拉取 `.tmp/ds2api`、如何更新 oracle commit、如何运行 parity 测试，详见 [third-party-runtime.md](third-party-runtime.md)。
- 已增加贡献指南 `CONTRIBUTING.md`：编码规范、测试命令、敏感信息规则、ds2api parity 要求、provider 改动要求。
- 已增加安全说明 `SECURITY.md`：凭证存储位置、日志脱敏、漏洞报告方式、网页登录账号风险。
- 已增加 `.env.example`，说明真实 cookie/bearer/API key 不应写入仓库。

## 发布前验证矩阵

后端：

```powershell
python -m pytest -q
```

前端：

```powershell
cd webui
corepack pnpm build
```

启动脚本：

```powershell
.\start_webai_gateway.bat
Invoke-RestMethod http://127.0.0.1:8610/health
```

网页登录 E2E：

- `http://127.0.0.1:8610/health` 显示 `runtime.supervisor.singleEntry=true`，并能看到 WebAI2API / ds2api 内部 runtime 状态。
- WebAI2API 内部 runtime 已启动且只存在一个监听 `8500` 的进程树。
- DeepSeek ds2api 内部 runtime 已启动且监听 `9331`。
- `/v1/models` 能看到已授权 provider 的模型。
- `POST /v1/chat/completions` 普通文本请求成功。
- 带工具的 OpenAI 请求返回标准 `message.tool_calls`。
- Anthropic `/v1/messages` 返回标准 text 或 `tool_use` block。
- `POST /v1/images/generations` 使用 `gpt-image-2` 返回图片数据。
- `POST /v1/videos` 返回 video id，`GET /v1/videos/{id}/content` 可取回内容。

## 架构整理建议

短期不要为了开源做大重构。当前 `app.py` 偏大，但测试覆盖高、行为稳定；开源前优先文档化和补齐发布材料。

后续可按低风险顺序拆分：

1. `media_api.py`：迁移 `/v1/images/generations`、`/v1/videos` 和媒体缓存 helper。
2. `admin_api.py`：迁移管理 API、onboarding、diagnostics。
3. `providers/`：统一 direct provider、WebAI2API、ds2api sidecar 的路由入口。
4. `security.py`：集中鉴权、脱敏、localhost admin guard。

拆分原则：先保持公开行为不变，再用现有测试证明无回归。不要在拆分时同时调整工具桥或 provider 语义。

## 风险点

- Web 登录模型 UI 变化会导致 provider 适配失效，需要清晰的错误信息和可观测日志。
- ToolBridge 的兼容逻辑复杂，任何“看起来简化”的改动都可能破坏 Claude Code / OpenAI / Anthropic 客户端闭环。
- 真实网页登录凭证属于高风险本地状态，开源文档必须反复强调不要上传 `credentials/`、`data/`、`.webai-gateway/`。
- ds2api parity 是当前 DeepSeek 稳定性的关键，不能只靠本地一次成功替代 oracle 测试。
- 生图链路依赖 WebAI2API sidecar 的 ChatGPT UI 适配；Gateway 只负责 OpenAI-compatible 包装和错误透传。
