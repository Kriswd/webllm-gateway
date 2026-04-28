from __future__ import annotations

import json
import fnmatch
import html
import re
import shlex
from dataclasses import dataclass
from typing import Any

from webai_gateway.config import ToolBridgeConfig


_FENCED_TOOL_RE = re.compile(r"```tool_json\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_XML_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_PROVIDER_SEARCH_RE = re.compile(r"<search\b[^>]*>(?P<body>.*?)</search>", re.IGNORECASE | re.DOTALL)
_PROVIDER_SEARCH_QUERY_RE = re.compile(r"<query\b[^>]*>(?P<query>.*?)</query>", re.IGNORECASE | re.DOTALL)
_TOOL_SUMMARY_HEADER_RE = re.compile(r"assistant\s+requested\s+tool\s+calls\s*:", re.IGNORECASE)
_TOOL_SUMMARY_CALL_RE = re.compile(r"^\s*(?:[-*•]\s*)?(?P<name>[A-Za-z_][\w.:-]*)\s*\((?P<input>\{.*\})\)\s*$")
_TOOL_RESULT_CLAIM_RE = re.compile(r"(<tool_result\b|工具返回|已读取文件|already read|tool result)", re.IGNORECASE)
_TOOL_DENIAL_RE = re.compile(
    r"\b(?:tool|function|capability)\s+[`'\"“”]?(?P<name>[A-Za-z_][\w.:-]*)[`'\"“”]?\s+"
    r"(?:does\s+not\s+exists?|doesn't\s+exist|is\s+not\s+available|is\s+unavailable|"
    r"is\s+not\s+accessible|not\s+found|cannot\s+be\s+used|can't\s+be\s+used)\b",
    re.IGNORECASE,
)
_CJK_TOOL_DENIAL_RE = re.compile(
    r"(?:工具|函数|能力)\s*[`'\"“”]?(?P<name>[A-Za-z_][\w.:-]*)[`'\"“”]?\s*"
    r"(?:不存在|不可用|无法使用|不能使用|找不到|未注册)"
)
_TOOL_ENV_DENIAL_RE = re.compile(
    r"((?:cannot|can't|unable\s+to)\s+(?:directly\s+)?(?:access|use|execute|run)\s+"
    r"(?:the\s+)?(?:filesystem|file\s+system|tools?|commands?|shell|bash|terminal)|"
    r"no\s+access\s+to\s+(?:the\s+)?(?:filesystem|file\s+system|tools?|commands?)|"
    r"(?:无法|不能|不可|没有权限|无权限)\s*(?:直接)?\s*(?:访问|使用|执行|运行|操作)?\s*"
    r"(?:您的|本地|任何)?\s*(?:文件系统|文件|工具|操作工具|命令|git命令|终端|shell|bash)|"
    r"(?:系统|环境).{0,24}(?:限制|禁止).{0,24}(?:文件系统|工具|操作工具|命令|git|bash))",
    re.IGNORECASE,
)
_DEFERRED_TOOL_ACTION_RE = re.compile(
    r"("
    r"(?:我(?:来|会|将|先|可以)?|让我|首先|先|需要|准备|接下来).{0,40}"
    r"(?:研究|查看|读取|检查|搜索|检索|分析|了解|调查|梳理|打开|访问)"
    r"|(?:i(?:'ll| will)|i’ll|let me|first|next|i need to|i am going to|i'm going to).{0,60}"
    r"(?:inspect|read|search|check|review|analy[sz]e|look up|investigate|open|access)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_QUOTED_WINDOWS_PATH_RE = re.compile(r"(?P<quote>[\"'])(?P<path>[A-Za-z]:\\[^\"'\r\n]+)(?P=quote)")
_UNQUOTED_WINDOWS_PATH_RE = re.compile(r"(?<![\w/])(?P<path>[A-Za-z]:\\[^\s&|;<>]*)")
_SHELL_COMMAND_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;)\s*")
_READ_ONLY_PREFIXES = ("read", "list", "search", "fetch", "get", "show", "find", "query")
_READ_ONLY_NAMES = frozenset(
    {
        "dir",
        "glob",
        "grep",
        "ls",
        "tree",
        "toolsearch",
        "web_fetch",
        "web_search",
        "webfetch",
        "websearch",
    }
)
_PROMPT_BRIDGE_BLOCKED_TOOL_NAMES = frozenset(
    {
        "terminal",
        "shell",
        "powershell",
        "bash",
        "bashoutput",
        "killbash",
        "cmd",
        "python",
        "python_repl",
        "pythonrepl",
        "process",
        "write",
        "write_file",
        "writefile",
        "edit",
        "edit_file",
        "editfile",
        "multiedit",
        "notebookedit",
        "todowrite",
        "apply_patch",
        "applypatch",
        "browser",
        "computer",
        "generate_image",
        "generateimage",
        "generate_video",
        "generatevideo",
    }
)
_PROMPT_BRIDGE_BLOCKED_TOOL_PARTS = frozenset(
    {
        "applypatch",
        "bash",
        "browser",
        "cmd",
        "computer",
        "delete",
        "edit",
        "move",
        "patch",
        "powershell",
        "process",
        "python",
        "remove",
        "rename",
        "shell",
        "terminal",
        "write",
    }
)
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool = False


@dataclass(frozen=True)
class ToolCallDraft:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class BridgeError:
    kind: str
    message: str
    repairable: bool = False


@dataclass(frozen=True)
class BridgeResult:
    content: str
    tool_calls: list[ToolCallDraft]
    error: BridgeError | None = None
    warning: str | None = None
    raw_content: str = ""


@dataclass(frozen=True)
class ToolBridgeContext:
    enabled: bool
    mode: str
    tools: list[ToolSpec]
    options: ToolBridgeConfig

    @property
    def allowed_names(self) -> set[str]:
        return {tool.name for tool in self.tools}


def should_bridge_tools(
    tools: Any,
    tool_mode: str,
    *,
    activation_policy: str = "auto",
    messages: Any = None,
    tool_choice: Any = None,
    provider_native_web_search: bool = False,
) -> bool:
    mode = (tool_mode or "strict").strip().lower()
    if not (isinstance(tools, list) and bool(tools) and mode in {"prompt", "strict"}):
        return False
    policy = (activation_policy or "auto").strip().lower()
    if policy == "off":
        return False
    if policy == "always":
        return True
    if policy != "auto":
        return True
    if _tool_choice_requires_tool(tool_choice):
        return True
    if _messages_have_tool_loop(messages):
        return True
    latest = _latest_user_text(messages)
    if _looks_like_local_agent_task(latest):
        return True
    if provider_native_web_search:
        return False
    return True


def should_enable_native_web_search(messages: Any, policy: str) -> bool:
    normalized = (policy or "auto").strip().lower()
    if normalized == "off":
        return False
    if normalized == "force":
        return True
    if normalized != "auto":
        return False
    latest = _latest_user_text(messages)
    if _looks_like_local_agent_task(latest):
        return False
    if _looks_like_web_search_task(latest):
        return True
    if _looks_like_continuation_task(latest):
        previous = _previous_conversation_text(messages)
        if _looks_like_local_agent_task(previous):
            return False
        return _looks_like_web_search_task(previous)
    return False


def build_context(
    tools: Any,
    options: ToolBridgeConfig | None = None,
    *,
    mode: str | None = None,
    model: str | None = None,
) -> ToolBridgeContext:
    cfg = options or ToolBridgeConfig()
    bridge_mode = (mode or cfg.mode or "strict").strip().lower()
    specs = normalize_openai_tools(tools, max_tools=0)
    specs = _filter_prompt_bridge_tools(specs, cfg)
    specs = _limit_prompt_bridge_tools(specs, cfg)
    return ToolBridgeContext(enabled=bool(specs) and bridge_mode != "off", mode=bridge_mode, tools=specs, options=cfg)


def prefer_local_tools_for_local_agent_task(context: ToolBridgeContext, messages: Any) -> ToolBridgeContext:
    if not context.enabled or not _looks_like_local_agent_task(_latest_user_text(messages)):
        return context
    local_tools = [tool for tool in context.tools if _provider_search_tool_score(tool) <= 0]
    if not local_tools or len(local_tools) == len(context.tools):
        return context
    return ToolBridgeContext(
        enabled=bool(local_tools) and context.mode != "off",
        mode=context.mode,
        tools=local_tools,
        options=context.options,
    )


def normalize_openai_tools(tools: Any, *, max_tools: int = 32) -> list[ToolSpec]:
    if not isinstance(tools, list):
        return []
    specs: list[ToolSpec] = []
    for item in tools:
        fn = item.get("function") if isinstance(item, dict) else None
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        schema = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object", "properties": {}}
        specs.append(
            ToolSpec(
                name=name,
                description=str(fn.get("description") or ""),
                input_schema=schema,
                read_only=_is_read_only_tool(name, fn),
            )
        )
    if max_tools <= 0 or len(specs) <= max_tools:
        return specs
    return specs[:max_tools]


def normalize_anthropic_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(item.get("description") or ""),
                    "parameters": item.get("input_schema") if isinstance(item.get("input_schema"), dict) else {"type": "object"},
                },
            }
        )
    return out


def _filter_prompt_bridge_tools(specs: list[ToolSpec], options: ToolBridgeConfig) -> list[ToolSpec]:
    policy = (options.exposure_policy or "safe").strip().lower()
    if policy == "all":
        return specs
    if policy == "allowlist":
        allowed = {name.strip().lower() for name in options.allowed_tool_names if name.strip()}
        if allowed:
            return [spec for spec in specs if spec.name.strip().lower() in allowed]
    return [spec for spec in specs if not _is_prompt_bridge_blocked_tool(spec.name)]


def _limit_prompt_bridge_tools(specs: list[ToolSpec], options: ToolBridgeConfig) -> list[ToolSpec]:
    policy = (options.exposure_policy or "safe").strip().lower()
    if policy == "all":
        return specs
    max_tools = int(options.max_tools_in_prompt or 0)
    if max_tools <= 0 or len(specs) <= max_tools:
        return specs
    return specs[:max_tools]


def _is_prompt_bridge_blocked_tool(name: str) -> bool:
    lowered = (name or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if lowered in _PROMPT_BRIDGE_BLOCKED_TOOL_NAMES or compact in _PROMPT_BRIDGE_BLOCKED_TOOL_NAMES:
        return True
    parts = [part for part in re.split(r"[^a-z0-9]+", lowered) if part]
    if any(part in _PROMPT_BRIDGE_BLOCKED_TOOL_PARTS for part in parts):
        return True
    return any(compact.endswith(part) for part in {"edit", "write", "applypatch"})


def prepare_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ctx = build_context(tools)
    return prepare_openai_messages(messages, ctx)


def prepare_openai_messages(messages: list[dict[str, Any]], context: ToolBridgeContext) -> list[dict[str, Any]]:
    if not context.enabled:
        return [message for message in messages if isinstance(message, dict)]
    prompt = build_tool_prompt(context.tools, context.options)
    call_names = _assistant_tool_call_names(messages)
    converted = [_convert_message(message, call_names=call_names, options=context.options) for message in messages if isinstance(message, dict)]
    if converted and converted[0].get("role") == "system":
        converted[0] = {**converted[0], "content": f"{_as_text(converted[0].get('content')).strip()}\n\n{prompt}".strip()}
        return converted
    return [{"role": "system", "content": prompt}, *converted]


def build_tool_prompt(tools: list[ToolSpec] | list[dict[str, Any]], options: ToolBridgeConfig | None = None) -> str:
    cfg = options or ToolBridgeConfig()
    specs = [_coerce_spec(item) for item in tools]
    functions = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
            "read_only": spec.read_only,
        }
        for spec in specs
        if spec.name
    ]
    tool_manifest = _tool_manifest_json(functions, max_chars=max(2000, int(cfg.tool_prompt_max_chars or 12000)))
    return (
        "You are using WebAI Gateway's strict tool bridge. You are allowed to request the downstream client to execute listed tools; "
        "do not confuse this with executing tools inside the web model runtime yourself.\n\n"
        "Available tools (allowed names only):\n"
        f"{tool_manifest}\n\n"
        "Decision rules:\n"
        "- If the task can be answered from the conversation or prior Tool result messages, answer normally.\n"
        "- If a tool is needed, output exactly one fenced tool_json block and no natural language outside it.\n"
        "- Never use provider-native markup such as <search>, <tool>, <tool_call>, <function_call>, or browsing tags. Use only fenced tool_json.\n"
        "- Never output summaries like 'Assistant requested tool calls'. Those are history text, not the required protocol.\n"
        "- Never say an available tool does not exist or that you lack permission to access files, run commands, use Bash/Git, or update the project. The downstream client owns permissions and executes allowed tools after you request them with tool_json.\n"
        "- After a Tool result message, use the observation to answer or request another allowed tool. Do not wait, do not claim that you executed a tool yourself, and do not repeat a failed identical call.\n"
        "- If a Tool result says is_error: true, do not treat it as successful data. Choose a different allowed tool/input if recovery is possible; otherwise explain the failure briefly.\n"
        "- For Glob/file-discovery tools, avoid repository-wide recursive patterns such as **/*, **/*.ext, or **/package.json unless the input also scopes the search to a narrow path. Prefer Read for known files, LS/list tools for directory overviews, or scoped patterns like src/**/*.py.\n"
        "- For public GitHub repository or source-code URLs, prefer machine-readable endpoints such as https://api.github.com/repos/<owner>/<repo>, the GitHub contents API, or raw.githubusercontent.com files instead of interactive HTML pages.\n\n"
        "Required tool-call format:\n"
        "```tool_json\n"
        "{\"calls\":[{\"id\":\"call_1\",\"name\":\"tool_name\",\"input\":{\"arg\":\"value\"}}]}\n"
        "```\n"
        f"Default maximum tool calls per turn: {cfg.max_calls_per_turn}; read-only tools may use up to {cfg.max_readonly_calls_per_turn}. "
        "All arguments must be inside the input object."
    )


def _tool_manifest_json(functions: list[dict[str, Any]], *, max_chars: int) -> str:
    full = json.dumps(functions, ensure_ascii=False, indent=2)
    if len(full) <= max_chars:
        return full

    compact = [
        {
            "name": item.get("name"),
            "description": _shorten(str(item.get("description") or ""), 160),
            "input_schema": _compact_tool_schema(item.get("input_schema")),
            "read_only": bool(item.get("read_only")),
        }
        for item in functions
        if item.get("name")
    ]
    compact_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    prefix = (
        "Tool prompt manifest was compacted to fit the web model prompt budget. "
        "All listed names remain allowed; request exact names only.\n"
    )
    if len(prefix) + len(compact_json) <= max_chars:
        return prefix + compact_json

    names_only = [
        {
            "name": item.get("name"),
            "read_only": bool(item.get("read_only")),
        }
        for item in functions
        if item.get("name")
    ]
    return prefix + json.dumps(names_only, ensure_ascii=False, separators=(",", ":"))


def _compact_tool_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object"}
    out: dict[str, Any] = {"type": str(schema.get("type") or "object")}
    required = schema.get("required")
    if isinstance(required, list):
        out["required"] = [str(item) for item in required[:12]]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        compact_properties: dict[str, Any] = {}
        for index, (name, value) in enumerate(properties.items()):
            if index >= 12:
                break
            if isinstance(value, dict):
                compact_properties[str(name)] = {
                    "type": str(value.get("type") or "string"),
                    "description": _shorten(str(value.get("description") or ""), 80),
                }
            else:
                compact_properties[str(name)] = {"type": "string"}
        out["properties"] = compact_properties
    return out


def _shorten(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def extract_tool_calls(text: str, allowed_tool_names: set[str] | None = None) -> tuple[str, list[dict[str, Any]]]:
    specs = [
        ToolSpec(name=name, description="", input_schema={"type": "object"}, read_only=_is_read_only_name(name))
        for name in sorted(allowed_tool_names or [])
    ]
    context = ToolBridgeContext(enabled=True, mode="strict", tools=specs, options=ToolBridgeConfig())
    result = parse_tool_response(text, context)
    return result.content, [
        {"name": call.name, "args": call.input, "id": call.id}
        for call in result.tool_calls
    ]


def parse_tool_response(text: str, context: ToolBridgeContext) -> BridgeResult:
    raw = text or ""
    candidates: list[Any] = []
    marker_seen = bool(_FENCED_TOOL_RE.search(raw) or _XML_TOOL_RE.search(raw))
    malformed_seen = False
    for match in _FENCED_TOOL_RE.finditer(raw):
        item = _loads(match.group(1))
        if item is None:
            malformed_seen = True
        else:
            candidates.append(item)
    for match in _XML_TOOL_RE.finditer(raw):
        item = _loads(match.group(1))
        if item is None:
            malformed_seen = True
        else:
            candidates.append(item)
    stripped = raw.strip()
    used_bare = False
    if not candidates and (stripped.startswith("{") or stripped.startswith("[")):
        item = _loads(stripped)
        if item is None:
            malformed_seen = True
        else:
            candidates.append(item)
            used_bare = True
    used_embedded_json = False
    if not candidates:
        embedded_candidates = _extract_embedded_tool_json_candidates(raw)
        if embedded_candidates:
            candidates.extend(embedded_candidates)
            used_embedded_json = True
    used_summary = False
    if not candidates:
        summary_candidates = _extract_tool_summary_candidates(raw, context)
        if summary_candidates:
            candidates.extend(summary_candidates)
            used_summary = True
    used_provider_search = False
    if not candidates:
        search_candidates, search_query = _extract_provider_search_candidates(raw, context)
        if search_candidates:
            candidates.extend(search_candidates)
            used_provider_search = True
        elif search_query is not None:
            return BridgeResult(
                content=_provider_search_fallback(search_query),
                tool_calls=[],
                warning="provider_search_markup_without_search_tool",
                raw_content=raw,
            )
    if not candidates:
        warning = "tool_result_claim_without_tool_call" if _TOOL_RESULT_CLAIM_RE.search(raw) else None
        if _is_allowed_tool_denial(raw, context):
            return BridgeResult(
                content=raw,
                tool_calls=[],
                error=BridgeError("tool_denial_without_call", "模型把下游允许工具误判为不可用", repairable=True),
                warning=warning,
                raw_content=raw,
            )
        if context.allowed_names and _DEFERRED_TOOL_ACTION_RE.search(raw):
            return BridgeResult(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "deferred_tool_action_without_call",
                    "模型承诺研究、读取、搜索或检查，但没有发起工具调用",
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if marker_seen or malformed_seen:
            return BridgeResult(
                content=raw,
                tool_calls=[],
                error=BridgeError("malformed_json", "工具调用 JSON 无效", repairable=True),
                warning=warning,
                raw_content=raw,
            )
        return BridgeResult(content=raw, tool_calls=[], warning=warning, raw_content=raw)

    normalized, error = _normalize_candidates(candidates, context)
    if error is not None:
        return BridgeResult(content=raw, tool_calls=[], error=error, raw_content=raw)
    if used_bare or used_embedded_json or marker_seen or used_summary or used_provider_search:
        return BridgeResult(content="", tool_calls=normalized, raw_content=raw)
    clean = _FENCED_TOOL_RE.sub("", raw)
    clean = _XML_TOOL_RE.sub("", clean)
    return BridgeResult(content=clean.strip() if not normalized else "", tool_calls=normalized, raw_content=raw)


def to_openai_tool_calls(tool_calls: list[dict[str, Any]] | list[ToolCallDraft]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        if isinstance(call, ToolCallDraft):
            call_id = call.id
            name = call.name
            args = call.input
        elif isinstance(call, dict):
            call_id = str(call.get("id") or "")
            name = str(call.get("name") or "").strip()
            args = call.get("args") if isinstance(call.get("args"), dict) else call.get("input")
            args = args if isinstance(args, dict) else {}
        else:
            continue
        if not name:
            continue
        out.append(
            {
                "id": call_id or f"call_web_{index + 1}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    return out


def _extract_tool_summary_candidates(text: str, context: ToolBridgeContext) -> list[dict[str, Any]]:
    if not _TOOL_SUMMARY_HEADER_RE.search(text or ""):
        return []
    allowed_lower = {name.lower() for name in context.allowed_names}
    candidates: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        match = _TOOL_SUMMARY_CALL_RE.match(line)
        if not match:
            continue
        name = match.group("name").strip()
        if name.lower() not in allowed_lower:
            continue
        args = _loads_summary_input(match.group("input"))
        candidates.append({"id": f"toolu_summary_{len(candidates) + 1}", "name": name, "input": args})
    return candidates


def _extract_provider_search_candidates(text: str, context: ToolBridgeContext) -> tuple[list[dict[str, Any]], str | None]:
    query = _extract_provider_search_query(text)
    if query is None:
        return [], None
    if not query:
        return [], query
    tool = _select_provider_search_tool(context)
    if tool is None:
        return [], query
    return [{"id": "toolu_search_1", "name": tool.name, "input": _provider_search_input(tool, query)}], query


def _extract_provider_search_query(text: str) -> str | None:
    match = _PROVIDER_SEARCH_RE.search(text or "")
    if not match:
        return None
    body = match.group("body") or ""
    query_match = _PROVIDER_SEARCH_QUERY_RE.search(body)
    raw_query = query_match.group("query") if query_match else re.sub(r"<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", html.unescape(raw_query or "")).strip()


def _select_provider_search_tool(context: ToolBridgeContext) -> ToolSpec | None:
    scored: list[tuple[int, int, ToolSpec]] = []
    for index, tool in enumerate(context.tools):
        score = _provider_search_tool_score(tool)
        if score > 0:
            scored.append((score, -index, tool))
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def _provider_search_tool_score(tool: ToolSpec) -> int:
    name = (tool.name or "").strip()
    lowered = name.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    description = (tool.description or "").lower()
    external_hints = ("web", "internet", "online", "browser", "联网", "网页", "网络")
    exact_web_search_names = {
        "websearch",
        "internetsearch",
        "browsersearch",
        "googlesearch",
        "bingsearch",
        "searchweb",
        "searchinternet",
    }
    if compact in exact_web_search_names or compact.endswith(("websearch", "internetsearch", "browsersearch", "googlesearch", "bingsearch")):
        return 100
    if "web" in compact and "search" in compact:
        return 90
    if "search" in compact and any(hint in description for hint in external_hints):
        return 80
    return 0


def _provider_search_input(tool: ToolSpec, query: str) -> dict[str, Any]:
    properties = tool.input_schema.get("properties") if isinstance(tool.input_schema, dict) else None
    props = properties if isinstance(properties, dict) else {}
    for key in ("query", "q", "search_query", "searchQuery"):
        if key in props:
            return {key: query}
    string_props = [
        key
        for key, value in props.items()
        if isinstance(key, str) and (not isinstance(value, dict) or str(value.get("type") or "string") == "string")
    ]
    if len(string_props) == 1:
        return {string_props[0]: query}
    return {"query": query}


def _provider_search_fallback(query: str) -> str:
    if query:
        return (
            "模型请求联网搜索，但当前允许工具中没有可用的搜索工具。\n"
            f"搜索词：{query}\n"
            "请在下游客户端暴露标准搜索工具（例如 WebSearch 或 web_search），或直接提供搜索结果后继续。"
        )
    return (
        "模型请求联网搜索，但没有提供有效搜索词，且当前允许工具中没有可用的搜索工具。"
        "请重新提问或在下游客户端暴露标准搜索工具。"
    )


def _tool_choice_requires_tool(tool_choice: Any) -> bool:
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() in {"required", "any"}
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type") or "").strip().lower()
        return choice_type in {"function", "tool", "required", "any"}
    return False


def _messages_have_tool_loop(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            return True
        if isinstance(message.get("tool_calls"), list) and message.get("tool_calls"):
            return True
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and str(block.get("type") or "") in {"tool_use", "tool_result"}:
                    return True
    return False


def _latest_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or str(message.get("role") or "") != "user":
            continue
        return _content_to_text(message.get("content"))
    return ""


def _previous_conversation_text(messages: Any, *, max_messages: int = 8) -> str:
    if not isinstance(messages, list):
        return ""
    skipped_latest_user = False
    parts: list[str] = []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "user" and not skipped_latest_user:
            skipped_latest_user = True
            continue
        text = _content_to_text(message.get("content"))
        if text.strip():
            parts.append(text)
        if len(parts) >= max_messages:
            break
    return "\n".join(reversed(parts))


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("text") is not None:
                    parts.append(str(item.get("text")))
                elif item.get("content") is not None:
                    parts.append(str(item.get("content")))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _looks_like_continuation_task(text: str) -> bool:
    value = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not value:
        return False
    exact_markers = {
        "继续",
        "接着",
        "继续说",
        "继续讲",
        "继续查",
        "继续搜索",
        "展开",
        "更多",
        "还有呢",
        "然后呢",
        "往下说",
        "接着说",
        "continue",
        "go on",
        "keep going",
        "more",
        "next",
    }
    if value in exact_markers:
        return True
    prefixes = ("继续 ", "接着 ", "continue ", "go on ", "more ", "next ")
    return any(value.startswith(prefix) for prefix in prefixes)


def _looks_like_local_agent_task(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if lowered.startswith("/"):
        return True
    local_markers = (
        "readme",
        "claude.md",
        "package.json",
        "pyproject",
        "requirements",
        "代码",
        "项目",
        "仓库",
        "文件",
        "目录",
        "读取",
        "查看",
        "打开",
        "修改",
        "编辑",
        "写入",
        "创建",
        "运行",
        "执行",
        "测试",
        "命令",
        "终端",
        "本地",
        "授权",
        "登录",
        "接入",
        "实现",
        "计划",
        "设计",
        "现有",
        "研究",
        "网页授权",
        "mcp",
        "skill",
        "tool",
        "repo",
        "repository",
        "codebase",
        "file",
        "folder",
        "directory",
        "shell",
        "bash",
        "terminal",
        "test",
        "lint",
    )
    if any(marker in lowered for marker in local_markers):
        return True
    return bool(re.search(r"\b[\w.-]+\.(?:py|js|ts|tsx|json|md|toml|yaml|yml|txt)\b", lowered))


def _looks_like_web_search_task(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    markers = (
        "联网",
        "搜索",
        "搜",
        "查一下",
        "查找",
        "最新",
        "当前",
        "现在",
        "发布",
        "官网",
        "网址",
        "地址",
        "体验",
        "web",
        "internet",
        "online",
        "search",
        "browse",
        "lookup",
        "latest",
        "current",
        "official",
        "url",
        "website",
        "released",
    )
    return any(marker in lowered for marker in markers)


def _is_allowed_tool_denial(text: str, context: ToolBridgeContext) -> bool:
    if not context.allowed_names:
        return False
    raw = text or ""
    allowed_lower = {name.lower() for name in context.allowed_names}
    for match in _TOOL_DENIAL_RE.finditer(raw):
        if match.group("name").strip().lower() in allowed_lower:
            return True
    for match in _CJK_TOOL_DENIAL_RE.finditer(raw):
        if match.group("name").strip().lower() in allowed_lower:
            return True
    if not _TOOL_ENV_DENIAL_RE.search(raw):
        return False
    lowered = raw.lower()
    return any(name.lower() in lowered for name in context.allowed_names)


def _loads_summary_input(raw: str) -> dict[str, Any]:
    variants = [raw]
    if '\\"' in raw:
        variants.append(raw.replace('\\"', '"'))
    for item in list(variants):
        escaped = _escape_invalid_json_backslashes(item)
        if escaped not in variants:
            variants.append(escaped)
    for item in variants:
        parsed = _loads(item)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _escape_invalid_json_backslashes(raw: str) -> str:
    out: list[str] = []
    index = 0
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    while index < len(raw):
        char = raw[index]
        if char == "\\" and (index + 1 >= len(raw) or raw[index + 1] not in valid_escapes):
            out.append("\\\\")
        else:
            out.append(char)
        index += 1
    return "".join(out)


def _extract_embedded_tool_json_candidates(text: str) -> list[Any]:
    raw = text or ""
    stripped = raw.strip()
    if not stripped or stripped.startswith("{") or stripped.startswith("["):
        return []

    decoder = json.JSONDecoder()
    starts_checked = 0
    for index, char in enumerate(raw):
        if char not in "{[":
            continue
        starts_checked += 1
        if starts_checked > 128:
            break
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except ValueError:
            continue
        if _looks_like_tool_json_candidate(value):
            return [value]
    return []


def _looks_like_tool_json_candidate(value: Any) -> bool:
    if isinstance(value, list):
        return any(_looks_like_tool_json_candidate(item) for item in value)
    if not isinstance(value, dict):
        return False

    calls = value.get("calls")
    if isinstance(calls, list):
        return any(_looks_like_tool_json_candidate(item) for item in calls)

    fn = value.get("function") if isinstance(value.get("function"), dict) else {}
    name = value.get("name") or value.get("tool") or value.get("tool_name") or fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return False
    return any(key in value for key in ("input", "args", "arguments")) or "arguments" in fn


def build_repair_messages(messages: list[dict[str, Any]], bad_text: str, error: BridgeError | None) -> list[dict[str, Any]]:
    reason = error.message if error else "tool call format is invalid"
    repair_instruction = (
        "Previous tool JSON was invalid.\n"
        f"Error: {reason}.\n"
        "The listed tools are available through the downstream client, not your own runtime. "
        "Do not say tools do not exist. Do not say you cannot access files or commands. "
        "If your previous answer said you would research, inspect, read, search, check, or analyze something, it was incomplete because it did not request a tool. "
        "没有发起工具调用时，不要说“我先研究/让我查看/正在搜索”。 "
        "If the task needs a listed tool, request it using the protocol below.\n"
        "Rewrite only one valid tool_json fenced block. Do not output explanation text. Required format:\n"
        "```tool_json\n"
        "{\"calls\":[{\"id\":\"call_1\",\"name\":\"tool_name\",\"input\":{}}]}\n"
        "```"
    )
    return [
        *messages,
        {"role": "assistant", "content": (bad_text or "")[:4000]},
        {"role": "user", "content": repair_instruction},
    ]


def _normalize_candidates(candidates: list[Any], context: ToolBridgeContext) -> tuple[list[ToolCallDraft], BridgeError | None]:
    out: list[ToolCallDraft] = []
    seen_ids: set[str] = set()
    allowed = context.allowed_names
    canonical_by_lower: dict[str, str] = {}
    for tool in context.tools:
        canonical_by_lower.setdefault(tool.name.lower(), tool.name)
    specs_by_name = {tool.name: tool for tool in context.tools}
    for candidate in candidates:
        items = _candidate_items(candidate)
        if not items:
            return [], BridgeError("empty_tool_call", "工具调用为空")
        for item in items:
            normalized = _normalize_item(item)
            if normalized is None:
                return [], BridgeError("invalid_tool_call", "工具调用必须是对象")
            canonical_name = normalized.name if normalized.name in allowed else canonical_by_lower.get(normalized.name.lower())
            if not canonical_name:
                replacement = _safe_replacement_for_cli_tool(normalized, context.tools)
                if replacement:
                    normalized = replacement
                    canonical_name = replacement.name
                else:
                    return [], BridgeError("unknown_tool", f"未知工具：{normalized.name}", repairable=True)
            if not isinstance(normalized.input, dict):
                return [], BridgeError("invalid_input", f"工具 {normalized.name} 的 input 必须是对象")
            replacement = _safe_replacement_for_expensive_glob(normalized, canonical_name, context.tools)
            if replacement:
                normalized = replacement
                canonical_name = replacement.name
            else:
                expensive_glob = _expensive_glob_message(canonical_name, normalized.input)
                if expensive_glob:
                    return [], BridgeError("expensive_tool_input", expensive_glob, repairable=True)
            normalized = _normalize_shell_tool_command(normalized, canonical_name, context.tools)
            shell_error = _shell_tool_command_error(normalized, canonical_name, context.tools)
            if shell_error:
                return [], shell_error
            call_id = normalized.id or f"call_web_{len(out) + 1}"
            if call_id in seen_ids:
                return [], BridgeError("duplicate_tool_call_id", f"重复的工具调用 id：{call_id}")
            seen_ids.add(call_id)
            out.append(ToolCallDraft(id=call_id, name=canonical_name, input=normalized.input))
    if not out:
        return [], BridgeError("empty_tool_call", "工具调用为空")
    max_calls = context.options.max_calls_per_turn
    if all(specs_by_name.get(call.name, ToolSpec(call.name, "", {}, False)).read_only for call in out):
        max_calls = context.options.max_readonly_calls_per_turn
    if len(out) > max_calls:
        return [], BridgeError("too_many_tool_calls", f"本轮工具调用过多：{len(out)} > {max_calls}")
    return out, None


def _safe_replacement_for_cli_tool(call: ToolCallDraft, tools: list[ToolSpec]) -> ToolCallDraft | None:
    if not isinstance(call.input, dict) or not _is_cli_tool_name(call.name):
        return None
    bash_tool = _select_bash_tool(tools)
    if not bash_tool:
        return None
    command = _cli_tool_command(call.name, call.input)
    if not command:
        return None
    return ToolCallDraft(
        id=call.id,
        name=bash_tool.name,
        input={_shell_command_key(bash_tool): _normalize_windows_paths_for_bash(command)},
    )


def _normalize_shell_tool_command(call: ToolCallDraft, canonical_name: str, tools: list[ToolSpec]) -> ToolCallDraft:
    if not _is_bash_tool_name(canonical_name):
        return call
    bash_tool = next((tool for tool in tools if tool.name == canonical_name), None)
    command_key = _shell_command_key(bash_tool) if bash_tool else "command"
    command = call.input.get(command_key)
    if not isinstance(command, str):
        return call
    normalized = _normalize_windows_paths_for_bash(command)
    if normalized == command:
        return call
    updated = dict(call.input)
    updated[command_key] = normalized
    return ToolCallDraft(id=call.id, name=call.name, input=updated)


def _shell_tool_command_error(call: ToolCallDraft, canonical_name: str, tools: list[ToolSpec]) -> BridgeError | None:
    if not _is_bash_tool_name(canonical_name):
        return None
    bash_tool = next((tool for tool in tools if tool.name == canonical_name), None)
    command_key = _shell_command_key(bash_tool) if bash_tool else "command"
    command = call.input.get(command_key)
    if not isinstance(command, str) or not command.strip():
        return BridgeError("incomplete_shell_command", "Bash command is empty; provide a complete command string.", repairable=True)
    return _incomplete_shell_command_error(command)


def _is_cli_tool_name(name: str) -> bool:
    lowered = (name or "").strip().lower()
    return lowered in {
        "bun",
        "cargo",
        "deno",
        "docker",
        "docker-compose",
        "gh",
        "git",
        "go",
        "kubectl",
        "mypy",
        "node",
        "npm",
        "npx",
        "pip",
        "pip3",
        "pnpm",
        "pytest",
        "python",
        "python3",
        "ruff",
        "rustc",
        "shell",
        "terminal",
        "uv",
        "yarn",
    }


def _select_bash_tool(tools: list[ToolSpec]) -> ToolSpec | None:
    for tool in tools:
        if _is_bash_tool_name(tool.name):
            return tool
    return None


def _is_bash_tool_name(name: str) -> bool:
    lowered = (name or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    return lowered == "bash" or compact == "bash"


def _cli_tool_command(name: str, input_value: dict[str, Any]) -> str:
    command_name = name.strip()
    for key in ("command", "cmd"):
        value = input_value.get(key)
        if isinstance(value, str):
            command = value.strip()
            if not command:
                return command_name
            if _is_shell_wrapper_tool_name(command_name):
                return command
            return command if _command_starts_with_cli_name(command, command_name) else f"{command_name} {command}"
    args = input_value.get("args")
    if args is None:
        args = input_value.get("arguments")
    if args is None:
        args = input_value.get("argv")
    if isinstance(args, str):
        return f"{command_name} {args.strip()}".strip()
    if isinstance(args, list):
        parts = [command_name]
        for arg in args:
            if arg is None:
                continue
            parts.append(shlex.quote(str(arg)))
        return " ".join(parts).strip()
    if not input_value:
        return command_name
    return ""


def _is_shell_wrapper_tool_name(name: str) -> bool:
    return (name or "").strip().lower() in {"shell", "terminal"}


def _command_starts_with_cli_name(command: str, name: str) -> bool:
    command = command.strip()
    lowered = command.lower()
    name_lower = name.strip().lower()
    return lowered == name_lower or lowered.startswith(f"{name_lower} ")


def _shell_command_key(tool: ToolSpec) -> str:
    properties = tool.input_schema.get("properties") if isinstance(tool.input_schema, dict) else None
    if isinstance(properties, dict):
        for key in ("command", "cmd"):
            if key in properties:
                return key
    return "command"


def _normalize_windows_paths_for_bash(command: str) -> str:
    def replace_quoted(match: re.Match[str]) -> str:
        return f"{match.group('quote')}{_windows_path_to_bash_path(match.group('path'))}{match.group('quote')}"

    def replace_unquoted(match: re.Match[str]) -> str:
        return _windows_path_to_bash_path(match.group("path"))

    normalized = _QUOTED_WINDOWS_PATH_RE.sub(replace_quoted, command)
    return _UNQUOTED_WINDOWS_PATH_RE.sub(replace_unquoted, normalized)


def _windows_path_to_bash_path(path: str) -> str:
    return path.replace("\\", "/")


def _incomplete_shell_command_error(command: str) -> BridgeError | None:
    for segment in _SHELL_COMMAND_SPLIT_RE.split(command):
        stripped = segment.strip()
        if not stripped:
            return BridgeError(
                "incomplete_shell_command",
                "Bash command contains an empty shell segment; provide a complete command.",
                repairable=True,
            )
        words = _shell_words(stripped)
        if not words:
            continue
        if len(words) == 1 and words[0] in {"cd", "git"}:
            return BridgeError(
                "incomplete_shell_command",
                f"Bash command segment {stripped!r} is incomplete; include the required arguments.",
                repairable=True,
            )
        if len(words) >= 2 and words[0] == "git" and words[1] == "clone":
            meaningful_args = _meaningful_git_clone_args(words[2:])
            if not meaningful_args:
                return BridgeError(
                    "incomplete_shell_command",
                    "git clone is missing the repository URL. Include the repo URL and destination path when needed.",
                    repairable=True,
                )
    return None


def _shell_words(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return [part for part in re.split(r"\s+", segment.strip()) if part]


def _meaningful_git_clone_args(args: list[str]) -> list[str]:
    meaningful: list[str] = []
    options_with_values = {
        "-b",
        "--branch",
        "-c",
        "--config",
        "--depth",
        "-j",
        "--jobs",
        "-o",
        "--origin",
        "--reference",
        "--reference-if-able",
        "--recurse-submodules",
        "--server-option",
        "--shallow-exclude",
        "--shallow-since",
        "--template",
        "--upload-pack",
    }
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if not arg:
            continue
        if arg in options_with_values:
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        meaningful.append(arg)
    return meaningful


def _safe_replacement_for_expensive_glob(
    call: ToolCallDraft,
    canonical_name: str,
    tools: list[ToolSpec],
) -> ToolCallDraft | None:
    if not _is_glob_tool_name(canonical_name):
        return None
    pattern = _glob_pattern(call.input)
    if not pattern or not _is_repository_wide_glob(pattern):
        return None
    scope = _glob_scope(call.input)
    if scope and scope not in {".", "./", "\\", "/"}:
        return None

    if _is_all_files_glob(pattern):
        list_tool = _select_directory_list_tool(tools)
        if list_tool:
            return ToolCallDraft(
                id=call.id,
                name=list_tool.name,
                input={_directory_path_key(list_tool): scope or "."},
            )

    shallow_pattern = _shallow_glob_pattern(pattern)
    if not shallow_pattern or shallow_pattern == pattern:
        return None
    sanitized = dict(call.input)
    for key in ("pattern", "glob", "file_pattern"):
        if key in sanitized:
            sanitized[key] = shallow_pattern
            break
    else:
        sanitized["pattern"] = shallow_pattern
    return ToolCallDraft(id=call.id, name=canonical_name, input=sanitized)


def _expensive_glob_message(name: str, input_value: dict[str, Any]) -> str | None:
    if not _is_glob_tool_name(name):
        return None
    pattern = _glob_pattern(input_value)
    if not pattern:
        return None
    if not _is_repository_wide_glob(pattern):
        return None
    scope = _glob_scope(input_value)
    if scope and scope not in {".", "./", "\\", "/"}:
        return None
    return (
        f"Glob pattern {pattern!r} is repository-wide and likely to time out on large workspaces. "
        "Use Read for known files, LS/list tools for directory overviews, or a scoped Glob pattern with a narrow path."
    )


def _is_glob_tool_name(name: str) -> bool:
    lowered = (name or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    return lowered == "glob" or compact.endswith("glob")


def _glob_pattern(input_value: dict[str, Any]) -> str:
    for key in ("pattern", "glob", "file_pattern"):
        value = input_value.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _glob_scope(input_value: dict[str, Any]) -> str:
    for key in ("path", "cwd", "root", "directory", "base_path"):
        value = input_value.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_repository_wide_glob(pattern: str) -> bool:
    normalized = pattern.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized in {"**", "**/*", "**/*.*"} or normalized.startswith("**/")


def _is_all_files_glob(pattern: str) -> bool:
    normalized = pattern.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized in {"**", "**/*", "**/*.*"}


def _shallow_glob_pattern(pattern: str) -> str:
    normalized = pattern.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if _is_all_files_glob(normalized):
        return "*"
    if normalized.startswith("**/"):
        return normalized[3:] or "*"
    return normalized


def _select_directory_list_tool(tools: list[ToolSpec]) -> ToolSpec | None:
    preferred_names = ("ls", "list", "list_dir", "listdir", "list_files", "listfiles")
    for name in preferred_names:
        for tool in tools:
            lowered = tool.name.strip().lower()
            compact = re.sub(r"[^a-z0-9]+", "", lowered)
            if lowered == name or compact == name:
                return tool
    for tool in tools:
        lowered = tool.name.strip().lower()
        compact = re.sub(r"[^a-z0-9]+", "", lowered)
        if compact.startswith("list") and _is_read_only_name(tool.name):
            return tool
    return None


def _directory_path_key(tool: ToolSpec) -> str:
    properties = tool.input_schema.get("properties") if isinstance(tool.input_schema, dict) else None
    if isinstance(properties, dict):
        for key in ("path", "directory", "dir_path", "folder"):
            if key in properties:
                return key
    return "path"


def _candidate_items(candidate: Any) -> list[Any]:
    if isinstance(candidate, list):
        return candidate
    if isinstance(candidate, dict) and "calls" in candidate:
        calls = candidate.get("calls")
        return calls if isinstance(calls, list) else []
    return [candidate]


def _normalize_item(item: Any) -> ToolCallDraft | None:
    if not isinstance(item, dict):
        return None
    fn = item.get("function") if isinstance(item.get("function"), dict) else {}
    name = item.get("name") or item.get("tool") or item.get("tool_name") or fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    args = item.get("input")
    if args is None:
        args = item.get("args")
    if args is None:
        args = item.get("arguments")
    if args is None:
        args = fn.get("arguments")
    if isinstance(args, str) and args.strip():
        parsed = _loads(args)
        args = parsed if isinstance(parsed, dict) else {}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return ToolCallDraft(id=str(item.get("id") or item.get("tool_call_id") or ""), name=name.strip(), input=args)  # type: ignore[arg-type]
    return ToolCallDraft(id=str(item.get("id") or item.get("tool_call_id") or ""), name=name.strip(), input=args)


def _convert_message(message: dict[str, Any], *, call_names: dict[str, str], options: ToolBridgeConfig) -> dict[str, Any]:
    role = str(message.get("role") or "")
    if role == "tool":
        call_id = str(message.get("tool_call_id") or "")
        name = str(message.get("name") or call_names.get(call_id) or "tool")
        raw_content = _as_text(message.get("content"))
        is_error = _tool_message_is_error(message, raw_content)
        content = compress_observation(raw_content, options)
        next_step = (
            "The tool call failed. Do not treat this as successful data; choose a different allowed tool or input if one can recover the task, otherwise explain the failure briefly."
            if is_error
            else "Use this tool result to continue the task."
        )
        return {
            "role": "user",
            "content": (
                f"Tool result for {name} (call id: {call_id}, is_error: {str(is_error).lower()}):\n"
                f"{content}\n\n"
                f"{next_step}"
            ).strip(),
        }
    if role == "assistant" and isinstance(message.get("tool_calls"), list) and message.get("tool_calls"):
        lines: list[str] = []
        for item in message["tool_calls"]:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") if isinstance(item.get("function"), dict) else {}
            name = str(fn.get("name") or item.get("name") or "").strip()
            args = fn.get("arguments") or item.get("args") or item.get("input") or {}
            if not name:
                continue
            lines.append(f"- {name}({args})")
        suffix = "\n".join(lines)
        return {"role": "assistant", "content": f"{_as_text(message.get('content')).strip()}\n\nAssistant requested tool calls:\n{suffix}".strip()}
    return {"role": role, "content": message.get("content", "")}


def _tool_message_is_error(message: dict[str, Any], content: str) -> bool:
    for key in ("is_error", "isError", "error"):
        value = message.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "error", "failed", "failure"}:
            return True

    parsed = _loads(content.strip()) if isinstance(content, str) and content.strip() else None
    if _parsed_tool_result_is_error(parsed):
        return True

    lowered = (content or "").strip().lower()
    if not lowered:
        return False
    failure_markers = (
        "network_error",
        "runtime_error",
        "permission_error",
        "timeout_error",
        "request failed",
        "tool failed",
        "traceback (most recent call last)",
    )
    return any(marker in lowered for marker in failure_markers)


def _parsed_tool_result_is_error(value: Any) -> bool:
    if isinstance(value, dict):
        for key in ("is_error", "isError"):
            flag = value.get(key)
            if isinstance(flag, bool):
                return flag
            if isinstance(flag, str) and flag.strip().lower() in {"true", "1", "yes"}:
                return True
        for key in ("ok", "success"):
            flag = value.get(key)
            if isinstance(flag, bool) and not flag:
                return True
        error_value = value.get("error")
        if isinstance(error_value, bool):
            return error_value
        if error_value not in (None, "", False):
            return True
        status = str(value.get("status") or value.get("statusText") or "").strip().lower()
        if status in {"error", "failed", "failure", "timeout"}:
            return True
        return any(_parsed_tool_result_is_error(item) for item in value.values() if isinstance(item, (dict, list)))
    if isinstance(value, list):
        return any(_parsed_tool_result_is_error(item) for item in value)
    return False


def compress_observation(text: str, options: ToolBridgeConfig) -> str:
    raw = text or ""
    limit = max(200, int(options.observation_max_chars or 4000))
    path_summary = _compress_path_list_observation(raw, limit=limit, options=options)
    if path_summary is not None:
        return path_summary
    if len(raw) <= limit:
        return raw
    half = max(80, (limit - 220) // 2)
    return (
        "Tool result was too long and has been compressed.\n"
        f"Original length: {len(raw)} characters.\n"
        "Leading excerpt:\n"
        f"{raw[:half]}\n\n"
        "Trailing excerpt:\n"
        f"{raw[-half:]}\n\n"
        "If the complete content is needed, request a narrower or more specific read range."
    )


def _compress_path_list_observation(raw: str, *, limit: int, options: ToolBridgeConfig) -> str | None:
    policy = options.observation_policy
    if not policy.summarize_path_lists:
        return None
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    path_like = [line for line in lines if _is_path_like_line(line)]
    if len(path_like) < 2 or len(path_like) < max(2, len(lines) // 2):
        return None

    kept = [line for line in lines if not (_is_path_like_line(line) and _is_excluded_observation_path(line, options))]
    omitted = len(lines) - len(kept)
    if omitted <= 0:
        return raw if len(raw) <= limit else None
    if not kept:
        return (
            "Only dependency/build/cache paths were returned; they were omitted before sending to the web model.\n"
            f"Omitted paths: {omitted}.\n"
            "Please request a narrower pattern that excludes dependency/build/cache directories, such as root-level "
            "README, package, pyproject, requirements, src, tests, or docs files."
        )

    header = (
        "Path list was summarized before sending to the web model.\n"
        f"{omitted} dependency/build paths were omitted.\n"
        "Relevant paths:\n"
    )
    body_limit = max(80, limit - len(header) - 120)
    max_items = max(1, int(policy.path_list_max_items or 80))
    visible = kept[:max_items]
    hidden_relevant = len(kept) - len(visible)
    body = "\n".join(visible)
    if hidden_relevant > 0:
        body = f"{body}\n... omitted {hidden_relevant} additional relevant paths ..."
    if len(body) > body_limit:
        half = max(40, body_limit // 2)
        body = (
            f"{body[:half]}\n"
            f"... omitted {len(body) - (half * 2)} characters from relevant paths ...\n"
            f"{body[-half:]}"
        )
    return header + body


def _is_path_like_line(line: str) -> bool:
    value = (line or "").strip()
    if not value or " " in value:
        return False
    lowered = value.lower().replace("\\", "/")
    return "/" in lowered or bool(re.search(r"\.[a-z0-9]{1,8}$", lowered))


def _is_excluded_observation_path(line: str, options: ToolBridgeConfig) -> bool:
    policy = options.observation_policy
    parts = [part for part in re.split(r"[\\/]+", (line or "").strip().lower()) if part]
    excluded_parts = {str(part).strip().lower() for part in policy.excluded_path_parts if str(part).strip()}
    if excluded_parts and any(part in excluded_parts for part in parts):
        return True
    normalized = (line or "").strip().replace("\\", "/")
    normalized_lower = normalized.lower()
    for pattern in policy.excluded_path_globs:
        lowered_pattern = str(pattern).strip().replace("\\", "/").lower()
        if lowered_pattern and fnmatch.fnmatch(normalized_lower, lowered_pattern):
            return True
    return False


def _assistant_tool_call_names(messages: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("tool_calls"), list):
            continue
        for item in message["tool_calls"]:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") if isinstance(item.get("function"), dict) else {}
            call_id = str(item.get("id") or "").strip()
            name = str(fn.get("name") or item.get("name") or "").strip()
            if call_id and name:
                out[call_id] = name
    return out


def _coerce_spec(item: ToolSpec | dict[str, Any]) -> ToolSpec:
    if isinstance(item, ToolSpec):
        return item
    fn = item.get("function") if isinstance(item, dict) else None
    if isinstance(fn, dict):
        name = str(fn.get("name") or "").strip()
        return ToolSpec(
            name=name,
            description=str(fn.get("description") or ""),
            input_schema=fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object"},
            read_only=_is_read_only_tool(name, fn),
        )
    return ToolSpec(name="", description="", input_schema={"type": "object"})


def _is_read_only_tool(name: str, fn: dict[str, Any]) -> bool:
    annotations = fn.get("annotations") if isinstance(fn.get("annotations"), dict) else {}
    if isinstance(annotations.get("readOnlyHint"), bool):
        return bool(annotations["readOnlyHint"])
    return _is_read_only_name(name)


def _is_read_only_name(name: str) -> bool:
    lowered = (name or "").strip().lower()
    return lowered.startswith(_READ_ONLY_PREFIXES) or lowered in _READ_ONLY_NAMES


def _loads(raw: str) -> Any | None:
    try:
        return json.loads((raw or "").strip())
    except Exception:
        return None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)
