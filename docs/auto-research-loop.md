# WebAI Gateway Auto-Research Loop

WebAI Gateway 的 auto-research loop 用于把 Claude Code / 网页模型适配失败沉淀为可回放的 ToolBridge fixture。它不会在线修改 Gateway，也不会执行本地工具；它只做离线采集、脱敏、fixture 生成和 replay 验证。

## 采集失败样本

从一个或多个 Claude JSONL transcript 采集可行动失败样本：

```powershell
python -m webai_gateway.auto_research collect `
  <path-to-claude-transcript.jsonl> `
  --output tests\fixtures\tool_bridge_replays
```

采集器会：

- 解析 `assistant` 的 `tool_use`、`user` 的 `tool_result` 和最近任务文本。
- 脱敏 cookie、bearer、session、token、api key、secret 等字段。
- 截断大段代码或工具结果，避免 fixture 膨胀。
- 只保留 ToolBridge 能明确归因的失败，例如 `unsafe_local_shell_command`、`write_after_failed_read_without_discovery`、`write_after_failed_path_without_discovery`。

## 验证 replay corpus

```powershell
python -m webai_gateway.auto_research report --fixtures tests\fixtures\tool_bridge_replays
python -m pytest tests\test_tool_bridge_replay.py tests\test_auto_research.py -q
```

`report` 返回 JSON：

```json
{
  "total": 25,
  "passed": 25,
  "failed": 0,
  "failures": []
}
```

## 合并门禁

新的失败样本进入 `tests/fixtures/tool_bridge_replays` 后，必须满足：

- `python -m webai_gateway.auto_research report --fixtures tests\fixtures\tool_bridge_replays` 通过。
- `python -m pytest -q` 通过。
- fixture 不包含 cookie、bearer、session token、API key 或其他凭证正文。

## 当前边界

第一版只做离线“失败样本 -> replay fixture -> 报告”。自动生成修复补丁应作为后续独立阶段实现，并且必须经过测试和人工审查后才能合并。
