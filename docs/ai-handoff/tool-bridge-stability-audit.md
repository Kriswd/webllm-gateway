# Tool Bridge Stability Audit

## Product Boundary

- Gateway remains a protocol adapter. It never executes local tools, MCP, skills, shell commands, browser automation, or file operations.
- Compatibility fixes must be expressed as standard protocol behavior, provider/model capability, or configurable Gateway policy.
- Do not add Qwen-only, Claude Code-only, or single slash-command branches for protocol behavior.

## Routing Invariants

- Local agent tasks, slash commands, explicit `tool_choice`, and existing tool loops must use `ToolBridgeV2` when tools are present.
- Ordinary current/web questions may use provider-native web search only when `providerRuntime.nativeWebSearchPolicy` allows it and no local tool loop is active.
- Direct provider requests that are not bridged must not forward OpenAI/Anthropic `tools` or `tool_choice` fields to web-model providers unless the provider explicitly supports native tools.
- Empty direct provider replies must become a visible diagnostic text response, not a blank assistant message.

## Tool Parsing Invariants

- Preferred format remains exactly one fenced `tool_json` block with no prose.
- The parser accepts provider deviations only when the payload is clearly tool-call shaped: `{"calls":[...]}`, a list of tool-call objects, or an object with a tool `name` plus `input`/`args`/`arguments`.
- Embedded tool-call JSON after short prose is treated as a tool call if the tool name is allowed in the current request.
- Parsed tool calls always produce empty assistant content and standard downstream `tool_calls` / `tool_use` structures.
- Unknown tools, invalid inputs, duplicate ids, and excessive call counts are rejected by Gateway; Gateway does not invent tools or execute them.
- Unknown tool-call JSON should trigger one repair attempt against the real allowed-tool list. If the model repeats the unknown tool, Gateway should make one recovery attempt that forces either real allowed tools or a direct final answer; if that still fails, return a diagnostic instead of leaking raw tool JSON as normal assistant text.

## Observed Failure Classes Covered

- Provider says allowed tools do not exist instead of emitting a tool call.
- Provider says it will inspect/search/read later without emitting a tool call.
- Provider emits provider-native search markup when only downstream search tools are available.
- Provider emits only a web-search placeholder instead of a final answer.
- Provider emits valid tool-call JSON with prose before it, causing downstream clients to render raw JSON instead of executing the tool.
- Provider emits a short preparation prelude such as "我来设计/让我研究/I'll draft" and stops before a final answer or tool call.
- Provider emits a plausible but unavailable tool name such as `Task`; Gateway must ask it to use only the currently exposed tools instead of passing the raw JSON downstream.

## Verification Checklist

- Run targeted bridge tests after parser/routing changes:
  `python -m pytest tests\test_gateway.py -q -k "tool_bridge or qwen_web_auto_activation or native_search or deferred_research or embedded_json or incomplete_prelude"`
- Run full backend tests:
  `python -m pytest -q`
- For real provider smoke tests, use non-sensitive prompts only and verify:
  normal chat returns text, tool request returns standard tool calls, and tool result continuation returns a final answer.
