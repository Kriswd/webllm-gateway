# Optional Adapter Runtimes

WebLLM Gateway 的核心项目是协议网关、网页登录授权管理、direct provider、ToolBridge 和 OpenAI / Anthropic 兼容接口。WebAI2API 和 ds2api 不属于项目本体；它们是按 provider 启用的可选 adapter runtime。

开源仓库不应把这些 runtime 的源码、运行态、凭证、浏览器 profile 或构建缓存提交进来。未安装可选 runtime 时，Gateway 仍应能启动并提供核心接口；只是在对应 provider 上显示不可用或需要安装/授权。

## WebAI2API

- 用途：ChatGPT、LMArena 等已验证或自行配置的网页登录授权、模型目录和网页调用。
- 默认地址：`http://127.0.0.1:8500/v1`。
- 默认目录：`../WebAI2API-sidecar`，可用 `WEBAI2API_SIDECAR_DIR` 覆盖。
- 本地 package metadata 声明 license 为 MIT，author 为 `foxhui`。

如果你分发包含 WebAI2API 的安装包，需要同时分发 WebAI2API 的许可证和 notice，并遵守它的上游项目条款。不捆绑时，Gateway 仓库只需要说明这是可选外部 runtime。

## ds2api

- 用途：DeepSeek Web / qwen ds2api 后端链路、协议行为 oracle、parity 测试。
- 默认地址：`http://127.0.0.1:9331/v1`。
- 默认可执行文件：`.tmp/ds2api/.tmp-bin/ds2api.exe`，可用 `WEBAI_DEEPSEEK_DS2API_EXE` 覆盖。
- 当前 Gateway oracle commit 写在 `webai_gateway/ds2api_oracle.py`。
- 本地 ds2api checkout 声明 license 为 AGPL-3.0。

如果你发布、修改、托管或捆绑 ds2api，需要遵守 AGPL-3.0。尤其是网络服务场景下，AGPL-3.0 可能要求向用户提供对应源代码。因此公开发布 Gateway 时，不建议把 ds2api 二进制或源码直接混入 Gateway release，除非同时处理好 AGPL 义务。

## Updating ds2api oracle

```powershell
git -C .tmp/ds2api fetch origin main
git -C .tmp/ds2api rev-parse origin/main
```

如果远端 commit 和 `webai_gateway/ds2api_oracle.py` 不一致：

1. 更新本地 `.tmp/ds2api` checkout。
2. 更新 `DS2API_ORACLE_COMMIT` 和版本号。
3. 运行 ds2api parity/oracle 测试。
4. 再继续 provider、工具桥或请求历史相关改动。

不要清理或覆盖 ds2api runtime 数据、网页登录态或凭证目录。
