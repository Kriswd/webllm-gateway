# Strict Final Gate 设计规格

## 背景

Claude Code 通过 WebLLM Gateway 使用网页登录模型时，工具调用协议经常被网页模型绕开。近期失败样本显示，模型不一定输出坏 JSON，也不一定直接说工具不存在，而是把未执行动作伪装成普通最终文本，例如：

- `Need current codebase overview first.`
- `Use different file path for project structure analysis.`
- `Fix: change to ... [next step].`
- `需直接查看文件内容识别问题。下一步检查项目结构和核心文件。`

如果 Gateway 把这些纯文本作为 Anthropic `end_turn` 返回，Claude Code 会停止 agent loop，用户看到的像是“任务完成或进入下一步”，但实际没有任何下游工具执行。

## 目标

把活动本地 agent 工具循环中的纯文本返回从“默认放行”改成“必须证明是完整最终回答”。证明不了时，Gateway 必须返回 repairable protocol error，让上游模型重新发标准 `tool_json`。

## 非目标

- Gateway 不执行本地工具。
- Gateway 不替 Claude Code 做权限判断。
- 不为 Claude Code、Qwen Coder 或某个项目写专用业务分支。
- 本轮不把真实 LLM-as-judge 接入请求链路；先预留配置和 shadow 数据结构，避免把不稳定模型作为安全边界。

## 设计

### Strict Final Gate

当同时满足以下条件时触发：

- 当前是 strict ToolBridge 解析；
- `context.has_tool_loop` 为真；
- 当前允许工具列表非空；
- 上游没有输出任何可解析工具调用；
- 任务看起来是本地 agent / 本地代码 / 工具驱动任务。

Gate 对纯文本做三类处理：

- **允许放行**：文本看起来是完整最终回答，例如明确的审查结论、总结、无后续动作的结果说明。
- **保守 repair**：文本包含未完成动作、下一步、检查、查看、读取、搜索、修改、修复、补丁、验证、换路径、继续分析等语义。
- **已有专用错误优先**：安全类错误、隐藏 shell、坏工具、代码补丁片段等保持现有更具体的 error kind。

新增通用错误：

- `unproven_final_answer_without_tool_call`

该错误表示模型输出纯文本，但在活动工具循环中无法证明这是最终答案。repair prompt 要求模型二选一：

- 发一个可用工具的 fenced `tool_json`；
- 或给出完整最终回答，且不能包含“下一步/需要继续/将要检查/建议改成但未执行”等未完成动作。

### Semantic Judge 预留

新增配置字段：

```json
{
  "tool_bridge": {
    "semanticFinalJudge": "off"
  }
}
```

允许值：

- `off`：默认，不调用语义判别。
- `shadow`：只记录本地启发式分类结果和未来 judge 结果字段，不影响返回。
- `enforce`：未来版本可启用 LLM-as-judge；低置信度或判别失败时仍走 Strict Final Gate 的保守路径。

本轮只实现配置透传和本地分类元数据，不接入真实二次模型请求。

### Auto Research

所有新增 gate 样本都进入 `tests/fixtures/tool_bridge_replays/`。`auto_research report` 必须覆盖这些样本，并显示总样本数与通过率。

## 验收标准

- 新增失败样本先红后绿。
- 所有 replay 通过。
- `python -m pytest -q` 通过。
- 服务重启后 `/health` 返回：
  - `ok=true`
  - `runtime.sourceFresh=true`
  - `runtime.sourceStale=false`

## 风险与约束

最主要风险是误杀正常最终回答。控制方式：

- Gate 只在活动工具循环中启用；
- 明确最终总结、审查结论、无后续动作的结果说明允许放行；
- 现有 read-only review plan 放行逻辑不直接删除；
- 新增测试覆盖正常 final answer 不被拦截。
