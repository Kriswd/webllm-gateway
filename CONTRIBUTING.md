# Contributing

感谢你改进 WebLLM Gateway。这个项目的核心目标是把网页登录模型稳定包装成 OpenAI / Anthropic 兼容 API，而不是实现一个 agent runtime。

## 开发前检查

1. 不要提交 `config.json`、`credentials/`、`data/`、`.webai-gateway/`、`.tmp/` 或日志。
2. 每天首次开发 ds2api 相关链路前执行：

```powershell
git -C .tmp/ds2api fetch origin main
git -C .tmp/ds2api rev-parse origin/main
```

然后对比 `webai_gateway/ds2api_oracle.py` 中的 `DS2API_ORACLE_COMMIT`。

3. provider、工具桥、请求历史、模型目录或授权流程改动必须对照 ds2api parity/oracle 行为，不能把一次观察写成长期特判。

## 必跑验证

```powershell
python -m pytest -q
cd webui
corepack pnpm install --frozen-lockfile
corepack pnpm build
```

修改启动脚本或 runtime supervisor 后，还要启动本机服务并检查：

```powershell
python -m webai_gateway.runtime_supervisor --config config.json --ensure
python -m webai_gateway
Invoke-RestMethod http://127.0.0.1:8610/health
```

`/health` 应返回 `ok=true`、`runtime.sourceFresh=true`、`runtime.sourceStale=false`，并显示 WebAI2API / ds2api runtime 状态。

## 设计边界

- Gateway 不执行本地工具，不读写用户项目文件，不调用 MCP，不接管下游权限。
- 工具执行、权限确认、审计、文件系统和终端能力属于下游客户端。
- 新兼容逻辑必须抽象成 provider capability、model capability 或 Gateway 配置，禁止写成某个客户端专用分支。
- 错误、日志、测试输出和前端提示不得暴露 cookie、bearer、session token 或 API key。

## 前端改动

主界面只保留网页登录授权、模型可用性、接入信息和必要的故障恢复入口。WebAI2API 的完整后台能力只作为隐藏诊断入口保留。

前端用户可见文案默认使用简体中文。改动后必须运行 `corepack pnpm build`，并确认首页仍可通过 `http://127.0.0.1:8610/` 打开。
