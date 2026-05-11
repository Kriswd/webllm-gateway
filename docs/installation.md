# Installation

这份文档描述从干净机器启动 WebAI Gateway 的最小路径。

## Prerequisites

- Windows 10/11 或等价本机开发环境。
- Python 3.12+。
- Node.js 22+，并启用 Corepack。
- Git。
- Go 工具链，用于运行 ds2api parity/oracle 测试。
- WebAI2API sidecar，本地默认目录：`../WebAI2API-sidecar`。
- ds2api runtime，本地默认可执行文件：`.tmp/ds2api/.tmp-bin/ds2api.exe`。

WebAI Gateway 不会把 WebAI2API 或 ds2api 源码作为本仓库的一部分发布。开源使用者需要自行准备这两个 runtime，或者按自己的发布方式提供下载脚本。

## Setup

```powershell
git clone <your-fork-or-upstream-url> webai-gateway
cd webai-gateway

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

## Runtime layout

默认布局：

```text
ProjectX/
  webai-gateway/
  WebAI2API-sidecar/
```

如果 WebAI2API sidecar 不在默认位置，可在启动前设置：

```powershell
$env:WEBAI2API_SIDECAR_DIR="D:\path\to\WebAI2API-sidecar"
```

如果 ds2api 可执行文件不在默认位置，可设置：

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
- WebAI2API runtime `running`
- ds2api runtime `running`

## Login

1. 打开 `http://127.0.0.1:8610/`。
2. 在网页登录向导中选择平台。
3. 点击授权按钮，在弹出的浏览器中完成真实网页登录。
4. 回到 Gateway，刷新模型并复制模型 ID。
5. 下游客户端使用 `http://127.0.0.1:8610/v1` 和本地 API key。

不要手动复制 cookie、bearer 或 session token 到文档、Issue、PR 或日志里。
