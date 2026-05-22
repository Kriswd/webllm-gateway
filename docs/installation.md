# Installation

这份文档描述从干净机器启动 WebLLM Gateway 的最小路径。

## Prerequisites

- Windows 10/11 或等价本机开发环境。
- Python 3.12+。
- Node.js 22+，并启用 Corepack。
- Git。

仅使用 Qwen direct、本地 OpenAI / Anthropic 兼容协议和 ToolBridge 时，不需要安装 WebAI2API 或 ds2api。

可选 adapter：

- WebAI2API sidecar：用于 ChatGPT、LMArena 等已验证或自行配置的网页登录站点。本地默认目录：`../WebAI2API-sidecar`。
- ds2api runtime：用于 DeepSeek Web 兼容链路和 ds2api parity/oracle 测试。本地默认可执行文件：`.tmp/ds2api/.tmp-bin/ds2api.exe`。
- Go 工具链：仅在运行 ds2api parity/oracle 测试时需要。

WebLLM Gateway 不会把 WebAI2API 或 ds2api 源码作为本仓库的一部分发布。开源使用者可以只运行 Gateway 核心能力；需要对应站点时再自行准备可选 runtime，或者按自己的发布方式提供下载脚本。

## Setup

```powershell
git clone https://github.com/Kriswd/webllm-gateway.git
cd webllm-gateway

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cd webui
corepack enable
corepack pnpm install --frozen-lockfile
corepack pnpm build
cd ..

Copy-Item config.example.json config.json
```

## Docker

Docker 部署适合在不同系统上快速启动同一套 Gateway：

```powershell
docker compose up -d --build
```

默认映射端口：

```text
http://127.0.0.1:8610/
```

容器内的配置、网页登录凭据、账号验证缓存和日志保存在 `webai-gateway-data` volume。不要把宿主机已有的 `credentials/`、`.webai-gateway/` 或浏览器 profile 复制进镜像；如需迁移，请先确认里面没有 cookie、bearer、session token 或 API key 泄露风险。

WebAI2API / ds2api 是外部可选 adapter runtime。Compose 默认使用 `host.docker.internal` 访问宿主机运行的 runtime：

```text
WEBAI_UPSTREAM_BASE_URL=http://host.docker.internal:8500/v1
WEBAI_DEEPSEEK_DS2API_BASE_URL=http://host.docker.internal:9331/v1
```

如果在 Linux 上直接使用 Docker Engine，`docker-compose.yml` 已配置 `host.docker.internal:host-gateway`。如果 runtime 跑在另一台机器或另一个 compose network，请改成对应服务地址。

网页登录授权需要 Gateway 能连到 Chrome / Edge 的 CDP 端口。Docker Compose 默认设置：

```text
WEBAI_DEFAULT_CDP_URL=http://host.docker.internal:9222
```

也就是说浏览器建议在宿主机上启动并开放 `9222` 调试端口，容器只连接它并捕获登录态。
如果 Gateway 跑在 NAS 或远程服务器，而你在另一台电脑上完成登录，请把控制台里的 CDP 地址改成 `http://电脑IP:9222`。只有当你拿到的授权回调 URL 本身包含 `cookie`、`bearer` 或 `session_token` 等凭据时，才可以使用首页的“远程/NAS 授权”粘贴导入；普通登录页 URL 不包含登录态，无法靠复制地址完成授权。

Windows 宿主机示例：

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-address=0.0.0.0 --remote-debugging-port=9222 --user-data-dir="$env:TEMP\webai-gateway-cdp"
```

## Optional Runtime Layout

如果启用 WebAI2API adapter，推荐布局：

```text
ProjectX/
  webllm-gateway/
  WebAI2API-sidecar/
```

如果 WebAI2API sidecar 不在默认位置，可在启动前设置：

```powershell
$env:WEBAI2API_SIDECAR_DIR="D:\path\to\WebAI2API-sidecar"
```

如果启用 DeepSeek adapter 且 ds2api 可执行文件不在默认位置，可设置：

```powershell
$env:WEBAI_DEEPSEEK_DS2API_EXE="D:\path\to\ds2api.exe"
```

## Start

推荐使用唯一入口：

```powershell
.\start_webai_gateway.bat
```

也可以手动启动：

```powershell
python -m webai_gateway.runtime_supervisor --config config.json --ensure
python -m webai_gateway
```

打开控制台：

```text
http://127.0.0.1:8610/
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8610/health
```

期望看到：

- `ok=true`
- `runtime.sourceFresh=true`
- `runtime.sourceStale=false`
- `runtime.supervisor.singleEntry=true`
- Gateway service `running`
- 未安装的 WebAI2API / ds2api adapter 可以显示为 `missing` 或 `stopped`，但应带 `optional=true`，不会阻止 Gateway 核心能力启动。
- 如果已经启用对应 adapter，则对应 runtime 应显示 `running`。

## Login

1. 打开 `http://127.0.0.1:8610/`。
2. 在网页登录向导中选择平台。
3. 点击授权按钮，在弹出的浏览器中完成真实网页登录。
4. 回到 Gateway，刷新模型并复制模型 ID。
5. 下游客户端使用 `http://127.0.0.1:8610/v1` 和本地 API key。

不要手动复制 cookie、bearer 或 session token 到文档、Issue、PR 或日志里。
