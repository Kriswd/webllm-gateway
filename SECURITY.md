# Security Policy

WebLLM Gateway 处理网页登录态和本地 API token，默认只应在用户自己的机器上运行。

## 支持范围

当前项目仍处于本地网关阶段。安全修复优先覆盖 `main` 分支最新代码。

## 敏感数据

不要提交或分享以下目录和文件：

- `config.json`
- `.env*`
- `credentials/`
- `data/`
- `.webai-gateway/`
- `.tmp/`
- `*.log`

这些路径可能包含 cookie、bearer、session token、浏览器 profile、请求历史、运行日志或本地 API key。仓库已通过 `.gitignore` 排除它们；发布前仍应执行 `git status --short` 和 `git ls-files` 检查。

## 运行边界

- 默认只绑定 `127.0.0.1:8610`。
- Gateway 不执行本地工具，不接管 MCP、终端、文件系统或浏览器自动化权限。
- Gateway 日志和错误返回必须脱敏，不得输出 cookie、bearer、session token 或 API key。
- WebAI2API 和 ds2api 是内部 runtime。公开部署、转发公网或共享账号都可能违反上游服务条款，请自行评估风险。

## 漏洞报告

如果你发现凭证泄露、鉴权绕过、日志脱敏失败或远程可利用问题，请不要公开贴出敏感样本。建议提供：

- 受影响版本或 commit
- 复现步骤
- 影响范围
- 已脱敏的日志片段

在公开仓库启用 Issues 后，可先提交不含敏感信息的安全问题；涉及凭证或可利用细节时，请改用维护者指定的私下渠道。
