# Third-party Runtime Dependencies

WebAI Gateway 把第三方网页登录 runtime 当作本机可选依赖使用。开源仓库不应把这些 runtime 的运行态、凭证、浏览器 profile 或构建缓存提交进来。

## WebAI2API

- 用途：ChatGPT、Gemini、LMArena、Sora、豆包等网页登录授权、模型目录和网页调用。
- 默认地址：`http://127.0.0.1:8500/v1`。
- 默认目录：`../WebAI2API-sidecar`，可用 `WEBAI2API_SIDECAR_DIR` 覆盖。
- 本地 package metadata 声明 license 为 MIT，author 为 `foxhui`。

如果你分发包含 WebAI2API 的安装包，需要同时分发 WebAI2API 的许可证和 notice，并遵守它的上游项目条款。

## ds2api

- 用途：DeepSeek Web / qwen ds2api 后端链路、协议行为 oracle、parity 测试。
- 默认地址：`http://127.0.0.1:9331/v1`。
- 默认可执行文件：`.tmp/ds2api/.tmp-bin/ds2api.exe`，可用 `WEBAI_DEEPSEEK_DS2API_EXE` 覆盖。
- 当前 Gateway oracle commit 写在 `webai_gateway/ds2api_oracle.py`。
- 本地 ds2api checkout 声明 license 为 AGPL-3.0。

如果你发布、修改、托管或捆绑 ds2api，需要遵守 AGPL-3.0。尤其是网络服务场景下，AGPL-3.0 可能要求向用户提供对应源代码。

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
