from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any


_DEFAULT_PROTOCOL_MARKER = "You are using WebAI Gateway's strict tool bridge."
_REQUIRED_TOOL_FORMAT_MARKER = "Required tool-call format:"
_DS2API_HISTORY_TITLE = "# DS2API_HISTORY.txt"
_DS2API_HISTORY_SUMMARY = "Prior conversation history and tool progress."
PRESERVED_TASK_STATE_MARKER = "# WebAI Gateway preserved task state"
LAYERED_HISTORY_MARKER = "[Layered history compaction]"
_LAYERED_HISTORY_STRATEGY = "ds2api_layered_history"
_CURRENT_USER_REQUEST_MARKER = "=== CURRENT USER REQUEST (highest priority) ==="
_DS2API_HISTORY_CONTINUATION = (
    "Continue from the latest state in the provided DS2API_HISTORY.txt context. "
    "Treat it as the current working state and answer the latest user request directly."
)
_CURRENT_USER_REQUEST_INSTRUCTION = (
    "Use the prior DS2API_HISTORY only as context. Answer this latest user request; "
    "do not continue unrelated earlier tasks unless this request explicitly asks to. "
    "Do not summarize DS2API_HISTORY.txt or the current state unless the latest user request asks for a status summary."
)
STATELESS_WEB_API_GUARD = (
    "You are serving a stateless WebAI Gateway API request. Ignore any previous website chat, "
    "account memory, profile memory, or project context that is not included in this request. "
    "Follow only the messages below. "
    "这是一次无状态 WebAI Gateway API 请求；不要引用网页端旧会话、账号记忆、个人资料记忆或本次请求以外的项目上下文。"
)


def compact_web_prompt(
    prompt: str,
    *,
    max_chars: int,
    protocol_marker: str = _DEFAULT_PROTOCOL_MARKER,
) -> str:
    limit = max(1000, int(max_chars or 12000))
    raw = prompt or ""
    if len(raw) <= limit:
        return raw
    notice = (
        "\n\n[Prompt content was compacted by WebAI Gateway for a web-model prompt budget. "
        f"Original length: {len(raw)} characters. Earlier middle content omitted.]\n\n"
    )
    head_len = min(2000, max(80, limit // 10))
    marker_index = raw.find(protocol_marker)
    if marker_index < 0:
        return _compact_head_tail(raw, limit=limit, notice=notice, head_len=head_len)
    return _compact_preserving_protocol(raw, limit=limit, notice=notice, head_len=head_len, marker_index=marker_index)


def _compact_head_tail(raw: str, *, limit: int, notice: str, head_len: int) -> str:
    tail_len = max(200, limit - head_len - len(notice))
    compacted = raw[:head_len].rstrip() + notice + raw[-tail_len:].lstrip()
    if len(compacted) > limit:
        compacted = compacted[:limit]
    return compacted


def _compact_preserving_protocol(raw: str, *, limit: int, notice: str, head_len: int, marker_index: int) -> str:
    protocol_label = "[Preserved strict tool bridge protocol]\n"
    tail_label = "\n\n[Latest conversation tail]\n"
    head = raw[:head_len].rstrip()
    fixed_len = len(head) + len(notice) + len(protocol_label) + len(tail_label)
    remaining = max(240, limit - fixed_len)
    protocol_budget = min(len(raw) - marker_index, max(220, min(18000, int(limit * 0.55), remaining - 160)))
    protocol_excerpt = _protocol_excerpt(raw[marker_index:], budget=protocol_budget)
    tail_budget = limit - (len(head) + len(notice) + len(protocol_label) + len(protocol_excerpt) + len(tail_label))
    if tail_budget < 120 and len(protocol_excerpt) > 260:
        trim = 120 - tail_budget
        protocol_excerpt = protocol_excerpt[: max(260, len(protocol_excerpt) - trim)].rstrip()
        tail_budget = limit - (len(head) + len(notice) + len(protocol_label) + len(protocol_excerpt) + len(tail_label))
    tail_slice = max(80, tail_budget)
    tail_start = max(marker_index, len(raw) - tail_slice)
    tail = raw[tail_start:].lstrip()
    compacted = head + notice + protocol_label + protocol_excerpt + tail_label + tail
    if len(compacted) <= limit:
        return compacted
    overflow = len(compacted) - limit
    if overflow < len(tail):
        tail = tail[overflow:].lstrip()
        compacted = head + notice + protocol_label + protocol_excerpt + tail_label + tail
    return compacted[:limit]


def _protocol_excerpt(protocol_source: str, *, budget: int) -> str:
    source = protocol_source.strip()
    if len(source) <= budget:
        return source
    required_index = source.find(_REQUIRED_TOOL_FORMAT_MARKER)
    if required_index < 0 or required_index < budget:
        return source[:budget].rstrip()
    omission = "\n...[tool manifest middle omitted during prompt compaction]...\n"
    first_budget = max(180, budget // 2)
    last_budget = max(180, budget - first_budget - len(omission))
    return (source[:first_budget].rstrip() + omission + source[required_index : required_index + last_budget].strip()).strip()


def compact_role_messages_as_ds2api_history(
    entries: Iterable[tuple[str, str]],
    *,
    max_chars: int,
    protocol_marker: str = _DEFAULT_PROTOCOL_MARKER,
    current_user_override: str | None = None,
) -> str:
    """Render long web prompts using ds2api's current-input-file transcript shape.

    Qwen direct mode does not have a verified file-upload path, so the transcript
    is embedded into the live prompt instead of uploaded. The visible structure and
    continuation instruction mirror ds2api's DS2API_HISTORY.txt flow.
    """
    limit = max(1000, int(max_chars or 12000))
    live_limit = _target_live_prompt_limit(limit)
    entry_list = [(role, text) for role, text in entries]
    history_transcript = build_ds2api_history_transcript(entry_list)
    latest_user = _normalized_current_user_override(current_user_override) or _latest_user_request(entry_list)
    snapshot = build_preserved_task_state_snapshot(
        entry_list,
        max_chars=max(600, min(1800, live_limit // 3)),
    )
    transcript = (snapshot.rstrip() + "\n\n" + history_transcript) if snapshot else history_transcript
    continuation = _history_continuation_prompt(latest_user, max_user_chars=max(240, min(1800, live_limit // 5)))
    if not transcript:
        return continuation[:live_limit]
    prompt = transcript.rstrip() + "\n\n" + continuation
    if len(prompt) <= live_limit:
        return prompt
    layered = _compact_role_messages_layered(
        entry_list,
        snapshot=snapshot,
        max_chars=live_limit,
        protocol_marker=protocol_marker,
        latest_user=latest_user,
    )
    if layered:
        return layered[:limit]
    if snapshot:
        snapshot_block = snapshot.rstrip() + "\n\n"
        transcript_limit = max(400, live_limit - len(snapshot_block) - len(continuation) - 2)
        compacted_transcript = compact_web_prompt(
            history_transcript,
            max_chars=transcript_limit,
            protocol_marker=protocol_marker,
        )
        prompt = snapshot_block + compacted_transcript.rstrip() + "\n\n" + continuation
        if len(prompt) <= limit:
            return prompt
        overflow = len(prompt) - limit
        if overflow < len(compacted_transcript):
            compacted_transcript = compacted_transcript[overflow:].lstrip()
            prompt = snapshot_block + compacted_transcript.rstrip() + "\n\n" + continuation
        return prompt[:limit]
    transcript_limit = max(400, live_limit - len(continuation) - 2)
    compacted_transcript = compact_web_prompt(transcript, max_chars=transcript_limit, protocol_marker=protocol_marker)
    prompt = compacted_transcript.rstrip() + "\n\n" + continuation
    if len(prompt) <= limit:
        return prompt
    overflow = len(prompt) - limit
    if overflow < len(compacted_transcript):
        compacted_transcript = compacted_transcript[overflow:].lstrip()
        prompt = compacted_transcript.rstrip() + "\n\n" + continuation
    return prompt[:limit]


def _target_live_prompt_limit(limit: int) -> int:
    if limit >= 24000:
        return max(12000, int(limit * 0.75))
    return limit


def _compact_role_messages_layered(
    entries: list[tuple[str, str]],
    *,
    snapshot: str,
    max_chars: int,
    protocol_marker: str,
    latest_user: str = "",
) -> str:
    limit = max(1000, int(max_chars or 12000))
    history_entries = _render_history_entries(entries)
    if not history_entries:
        return ""

    latest_entries = _latest_history_window(history_entries)
    protocol_excerpt = _protocol_excerpt_from_entries(
        entries,
        budget=max(800, min(9000, int(limit * 0.28))),
        protocol_marker=protocol_marker,
    )
    continuation = _history_continuation_prompt(latest_user, max_user_chars=max(240, min(1800, limit // 5)))
    header_lines = [
        _DS2API_HISTORY_TITLE,
        _DS2API_HISTORY_SUMMARY,
        LAYERED_HISTORY_MARKER,
        "Prompt content was compacted by WebAI Gateway using layered DS2API_HISTORY live context.",
        _layered_history_summary_line(
            original_count=len(history_entries),
            latest_count=len(latest_entries),
            has_snapshot=bool(snapshot),
        ),
    ]

    blocks: list[str] = []
    if snapshot:
        blocks.append(snapshot.rstrip())
    blocks.append("\n".join(header_lines).rstrip())
    if protocol_excerpt:
        blocks.append("=== PRESERVED SYSTEM AND TOOL PROTOCOL ===\n" + protocol_excerpt.strip())

    fixed = "\n\n".join(blocks + ["=== LATEST CONVERSATION TAIL ===", continuation])
    tail_budget = max(500, limit - len(fixed) - 4)
    tail = _fit_history_entries(latest_entries, max_chars=tail_budget)
    prompt = "\n\n".join(blocks + ["=== LATEST CONVERSATION TAIL ===", tail.rstrip(), continuation])
    if len(prompt) <= limit:
        return prompt

    overflow = len(prompt) - limit
    if overflow < len(tail):
        tail = tail[overflow:].lstrip()
        prompt = "\n\n".join(blocks + ["=== LATEST CONVERSATION TAIL ===", tail.rstrip(), continuation])
    return prompt[:limit]


def _layered_history_summary_line(*, original_count: int, latest_count: int, has_snapshot: bool) -> str:
    guidance = (
        "use the preserved task state, latest tail, and current user request."
        if has_snapshot
        else "use the latest tail and current user request; older tool history is reference context only."
    )
    return (
        f"Original entries: {original_count}. Latest entries retained: {latest_count}. "
        f"Older bulk observations were omitted from the live prompt; {guidance}"
    )


def _latest_user_request(entries: list[tuple[str, str]]) -> str:
    for role, text in reversed(entries):
        if _history_role_label(role) != "USER":
            continue
        content = (text or "").strip()
        if not content or _looks_like_current_request_control_text(content):
            continue
        task_text = _current_request_task_text(content)
        if task_text and not _looks_like_current_request_control_text(task_text):
            return task_text
    return ""


def _normalized_current_user_override(text: str | None) -> str:
    content = (text or "").strip()
    if not content or _looks_like_current_request_control_text(content):
        return ""
    task_text = _current_request_task_text(content)
    if task_text and not _looks_like_current_request_control_text(task_text):
        return task_text
    return ""


def _current_request_task_text(text: str) -> str:
    content = (text or "").strip()
    command = _SLASH_COMMAND_WITH_ARGS_RE.match(content)
    if command:
        return _one_line(command.group("body") or "", 4000)
    return content


def _looks_like_current_request_control_text(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    if not compact:
        return True
    if looks_like_gateway_tool_observation(text):
        return True
    control_markers = (
        "do not request the same skill again",
        "use the loaded skill instructions already in the conversation",
        "continue the original user task",
        "use this tool result to continue the task",
        "loaded skill instructions",
        "launching skill:",
        "skill loaded",
        "loaded using-superpowers",
        "不要再次请求同一",
        "继续原始用户任务",
        "已加载技能",
    )
    return any(marker in compact for marker in control_markers)


def _history_continuation_prompt(latest_user: str, *, max_user_chars: int) -> str:
    prompt = f"User: {_DS2API_HISTORY_CONTINUATION}"
    current = _current_user_request_block(latest_user, max_chars=max_user_chars)
    if current:
        prompt += "\n\n" + current
    return prompt


def _current_user_request_block(latest_user: str, *, max_chars: int) -> str:
    text = (latest_user or "").strip()
    if not text:
        return ""
    body = _compact_entry_content(text, max(120, max_chars))
    return f"{_CURRENT_USER_REQUEST_MARKER}\n{body}\n\nInstruction: {_CURRENT_USER_REQUEST_INSTRUCTION}"


def _render_history_entries(entries: list[tuple[str, str]]) -> list[tuple[int, str, str]]:
    rendered: list[tuple[int, str, str]] = []
    index = 0
    for role, text in entries:
        content = (text or "").strip()
        if not content:
            continue
        index += 1
        label = _history_role_label(role)
        rendered.append((index, label, content))
    return rendered


def _latest_history_window(entries: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    if len(entries) <= 10:
        return entries
    latest_user_position = -1
    for position, (_, role, _) in enumerate(entries):
        if role == "USER":
            latest_user_position = position
    if latest_user_position >= 0:
        start = max(0, latest_user_position - 3)
        return entries[start:]
    return entries[-8:]


def _protocol_excerpt_from_entries(
    entries: list[tuple[str, str]],
    *,
    budget: int,
    protocol_marker: str,
) -> str:
    candidates: list[str] = []
    for role, text in entries:
        content = (text or "").strip()
        if not content:
            continue
        if protocol_marker in content or _REQUIRED_TOOL_FORMAT_MARKER in content:
            candidates.append(content)
    if not candidates:
        for role, text in entries:
            if str(role or "").strip().lower() == "system" and (text or "").strip():
                candidates.append((text or "").strip())
                break
    if not candidates:
        return ""
    source = "\n\n".join(candidates)
    marker_index = source.find(protocol_marker)
    if marker_index >= 0:
        source = source[marker_index:]
    return _protocol_excerpt(source, budget=max(200, budget))


def _fit_history_entries(entries: list[tuple[int, str, str]], *, max_chars: int) -> str:
    limit = max(400, int(max_chars or 1200))
    if not entries:
        return ""
    rendered = [_format_history_entry(index, role, content) for index, role, content in entries]
    joined = "\n\n".join(rendered)
    if len(joined) <= limit:
        return joined

    per_entry = max(120, (limit - (len(entries) * 28)) // max(1, len(entries)))
    compacted = [_format_history_entry(index, role, _compact_entry_content(content, per_entry)) for index, role, content in entries]
    joined = "\n\n".join(compacted)
    if len(joined) <= limit:
        return joined

    kept: list[str] = []
    for item in reversed(compacted):
        candidate = "\n\n".join([item, *kept])
        if len(candidate) > limit - len("\n\n...[latest conversation tail truncated]"):
            break
        kept.insert(0, item)
    if len(kept) < len(compacted):
        kept.insert(0, "...[latest conversation tail truncated]")
    return "\n\n".join(kept)[:limit]


def _format_history_entry(index: int, role: str, content: str) -> str:
    return f"=== {index}. {role} ===\n{content.strip()}"


def _compact_entry_content(content: str, max_chars: int) -> str:
    text = str(content or "").strip()
    if len(text) <= max_chars:
        return text
    omission = "\n...[entry middle omitted during layered history compaction]...\n"
    head_chars = max(60, int(max_chars * 0.45))
    tail_chars = max(60, max_chars - head_chars - len(omission))
    if head_chars + tail_chars + len(omission) >= len(text):
        return text
    return text[:head_chars].rstrip() + omission + text[-tail_chars:].lstrip()


def build_ds2api_history_transcript(entries: Iterable[tuple[str, str]]) -> str:
    parts = [_DS2API_HISTORY_TITLE, _DS2API_HISTORY_SUMMARY, ""]
    index = 0
    for role, text in entries:
        content = (text or "").strip()
        if not content:
            continue
        index += 1
        parts.append(f"=== {index}. {_history_role_label(role)} ===")
        parts.append(content)
        parts.append("")
    if index == 0:
        return ""
    return "\n".join(parts).rstrip() + "\n"


def _history_role_label(role: str) -> str:
    normalized = (role or "").strip().lower()
    if normalized == "function":
        normalized = "tool"
    return (normalized or "unknown").upper()


_INVOKE_RE = re.compile(
    r"<\s*(?:\|DSML\|)?invoke\b[^>]*\bname\s*=\s*['\"](?P<name>[^'\"]+)['\"][^>]*>"
    r"(?P<body>.*?)"
    r"</\s*(?:\|DSML\|)?invoke\s*>",
    re.IGNORECASE | re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<\s*(?:\|DSML\|)?parameter\b[^>]*\bname\s*=\s*['\"](?P<name>[^'\"]+)['\"][^>]*>"
    r"(?P<body>.*?)"
    r"</\s*(?:\|DSML\|)?parameter\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ITEM_RE = re.compile(r"<\s*item\b[^>]*>(?P<body>.*?)</\s*item\s*>", re.IGNORECASE | re.DOTALL)
_CDATA_RE = re.compile(r"^\s*<!\[CDATA\[(?P<body>.*)\]\]>\s*$", re.DOTALL)
_RESULT_SIGNAL_RE = re.compile(
    r"\b(?:file does not exist|no such file|not found|is_error:\s*true|"
    r"unchanged content|reported unchanged|already available in history|missing content)\b",
    re.IGNORECASE,
)
_SLASH_COMMAND_WITH_ARGS_RE = re.compile(
    r"^\s*/[A-Za-z0-9][A-Za-z0-9:_\.-]*(?:\s+(?P<body>.+))?\s*$",
    re.DOTALL,
)
_GATEWAY_TOOL_OBSERVATION_RE = re.compile(
    r"^\s*(?:Tool result for|Tool call failed|Function result for)\b",
    re.IGNORECASE,
)
_TASK_LIST_PARAM_NAMES = {"todos", "tasks", "items", "tasklist", "todolist"}


def web_prompt_history_role(role: str, text: str) -> str:
    normalized = str(role or "").strip().lower() or "user"
    if normalized == "function":
        return "tool"
    if normalized == "user" and looks_like_gateway_tool_observation(text):
        return "tool"
    return normalized


def looks_like_gateway_tool_observation(text: str) -> bool:
    return bool(_GATEWAY_TOOL_OBSERVATION_RE.match(str(text or "")))


def build_preserved_task_state_snapshot(entries: Iterable[tuple[str, str]], *, max_chars: int = 1200) -> str:
    entry_list = [(role, text or "") for role, text in entries]
    tasks = _extract_task_updates(entry_list)
    if not tasks:
        return ""
    recent_calls = _extract_recent_tool_call_summaries(entry_list)
    result_signals = _extract_tool_result_signals(entry_list)

    lines = [
        PRESERVED_TASK_STATE_MARKER,
        (
            "Generated before prompt compaction. Treat this as the active task ledger and recent tool evidence; "
            "verify contradictory file evidence before marking work complete."
        ),
    ]
    if tasks:
        lines.append("Tasks:")
        for task_id in _sorted_task_ids(tasks):
            task = tasks[task_id]
            status = task.get("status") or "unknown"
            description = task.get("description") or "no description"
            lines.append(f"- Task {task_id}: {status} - {_one_line(description, 180)}")
    if recent_calls:
        lines.append("Recent tool calls:")
        lines.extend(f"- {summary}" for summary in recent_calls[-10:])
    if result_signals:
        lines.append("Tool result signals:")
        lines.extend(f"- {_one_line(signal, 220)}" for signal in result_signals[-6:])
    return _fit_snapshot("\n".join(lines), max_chars=max_chars)


def prompt_preserved_task_state_diagnostics(prompt: str) -> dict[str, Any]:
    text = prompt or ""
    strategy = _prompt_compaction_strategy(text)
    history_entry_count, latest_entry_count = _layered_history_counts(text)
    index = text.find(PRESERVED_TASK_STATE_MARKER)
    if index < 0:
        return {
            "prompt_task_state_preserved": False,
            "prompt_task_state_chars": 0,
            "prompt_task_count": 0,
            "prompt_recent_tool_call_count": 0,
            "prompt_compaction_strategy": strategy,
            "prompt_history_entry_count": history_entry_count,
            "prompt_latest_entry_count": latest_entry_count,
        }
    end = text.find("\n\n" + _DS2API_HISTORY_TITLE, index)
    section = text[index:] if end < 0 else text[index:end]
    return {
        "prompt_task_state_preserved": True,
        "prompt_task_state_chars": len(section),
        "prompt_task_count": len(re.findall(r"^- Task\s+", section, re.MULTILINE)),
        "prompt_recent_tool_call_count": len(
            re.findall(r"^- [A-Za-z_][A-Za-z0-9_:\.-]*\(", section, re.MULTILINE)
        ),
        "prompt_compaction_strategy": strategy,
        "prompt_history_entry_count": history_entry_count,
        "prompt_latest_entry_count": latest_entry_count,
    }


def _prompt_compaction_strategy(text: str) -> str:
    if LAYERED_HISTORY_MARKER in text:
        return _LAYERED_HISTORY_STRATEGY
    if _DS2API_HISTORY_TITLE in text:
        return "ds2api_history"
    return "none"


def _layered_history_counts(text: str) -> tuple[int, int]:
    match = re.search(r"Original entries:\s*(\d+)\.\s*Latest entries retained:\s*(\d+)\.", text or "")
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _extract_task_updates(entries: list[tuple[str, str]]) -> dict[str, dict[str, str]]:
    tasks: dict[str, dict[str, str]] = {}
    for _, text in entries:
        for name, params in _iter_invocations(text):
            if _compact_name(name) != "taskupdate":
                continue
            items = _extract_taskupdate_items(params)
            if items:
                for index, item in enumerate(items, start=1):
                    task_id = (
                        item.get("task_id")
                        or item.get("taskId")
                        or item.get("id")
                        or item.get("task")
                        or str(index)
                    )
                    _merge_task_update(
                        tasks,
                        task_id=task_id,
                        status=item.get("status"),
                        description=(
                            item.get("description")
                            or item.get("title")
                            or item.get("content")
                            or item.get("summary")
                            or item.get("activeForm")
                        ),
                    )
                continue
            task_id = (
                params.get("task_id")
                or params.get("taskId")
                or params.get("id")
                or params.get("task")
                or str(len(tasks) + 1)
            )
            _merge_task_update(
                tasks,
                task_id=task_id,
                status=params.get("status"),
                description=(
                    params.get("description")
                    or params.get("title")
                    or params.get("content")
                    or params.get("summary")
                    or params.get("activeForm")
                ),
            )
    return tasks


def _merge_task_update(
    tasks: dict[str, dict[str, str]],
    *,
    task_id: str,
    status: str | None,
    description: str | None,
) -> None:
    compact_id = _one_line(task_id, 48)
    current = dict(tasks.get(compact_id) or {})
    if status:
        current["status"] = _one_line(status, 64)
    if description:
        current["description"] = _one_line(description, 220)
    tasks[compact_id] = current


def _extract_taskupdate_items(params: dict[str, str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for name, value in params.items():
        if _compact_name(name) not in _TASK_LIST_PARAM_NAMES:
            continue
        items.extend(_extract_taskupdate_items_from_value(value))
    return items


def _extract_taskupdate_items_from_value(value: str) -> list[dict[str, str]]:
    raw = value or ""
    xml_items = [
        item
        for item in (_taskupdate_item_from_xml(match.group("body")) for match in _ITEM_RE.finditer(raw))
        if item
    ]
    if xml_items:
        return xml_items
    return _taskupdate_items_from_json(raw)


def _taskupdate_item_from_xml(body: str) -> dict[str, str]:
    item: dict[str, str] = {}
    for field in (
        "id",
        "task_id",
        "taskId",
        "task",
        "status",
        "content",
        "description",
        "title",
        "summary",
        "activeForm",
    ):
        value = _xmlish_child_text(body, field)
        if value:
            item[field] = value
    return item


def _xmlish_child_text(body: str, field: str) -> str:
    match = re.search(
        rf"<\s*{re.escape(field)}\b[^>]*>(?P<body>.*?)</\s*{re.escape(field)}\s*>",
        body or "",
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return _decode_parameter_value(match.group("body"))


def _taskupdate_items_from_json(value: str) -> list[dict[str, str]]:
    raw = _strip_outer_cdata(value or "")
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if isinstance(data, dict):
        for key in ("todos", "tasks", "items"):
            nested = data.get(key)
            if isinstance(nested, list):
                data = nested
                break
        else:
            data = [data]
    if not isinstance(data, list):
        return []
    return [item for item in (_normalize_taskupdate_json_item(value) for value in data) if item]


def _normalize_taskupdate_json_item(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        return {"description": _one_line(value, 1000)}
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, item_value in value.items():
        text = _json_value_to_text(item_value)
        if text:
            normalized[str(key)] = text
    return normalized


def _json_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _one_line(value, 1000)
    if isinstance(value, (int, float, bool)):
        return str(value)
    return _one_line(json.dumps(value, ensure_ascii=False, separators=(",", ":")), 1000)


def _strip_outer_cdata(value: str) -> str:
    raw = (value or "").strip()
    cdata = _CDATA_RE.match(raw)
    if cdata:
        return cdata.group("body").strip()
    return raw


def _extract_recent_tool_call_summaries(entries: list[tuple[str, str]]) -> list[str]:
    summaries: list[str] = []
    for _, text in entries:
        for name, params in _iter_invocations(text):
            pieces: list[str] = []
            for key in (
                "task_id",
                "taskId",
                "status",
                "file_path",
                "path",
                "pattern",
                "query",
                "command",
                "description",
            ):
                value = params.get(key)
                if value:
                    pieces.append(f"{key}={_one_line(value, 120)}")
                if len(pieces) >= 4:
                    break
            summaries.append(f"{name}({', '.join(pieces)})" if pieces else f"{name}()")
    return summaries


def _extract_tool_result_signals(entries: list[tuple[str, str]]) -> list[str]:
    signals: list[str] = []
    for role, text in entries:
        if str(role or "").strip().lower() not in {"tool", "user"}:
            continue
        for line in (text or "").splitlines():
            if _RESULT_SIGNAL_RE.search(line):
                signals.append(line.strip())
                break
    return signals


def _iter_invocations(text: str) -> Iterable[tuple[str, dict[str, str]]]:
    for match in _INVOKE_RE.finditer(text or ""):
        params = {
            param.group("name"): _decode_parameter_value(param.group("body"))
            for param in _PARAM_RE.finditer(match.group("body") or "")
        }
        yield match.group("name").strip(), params


def _decode_parameter_value(value: str) -> str:
    raw = (value or "").strip()
    cdata = _CDATA_RE.match(raw)
    if cdata:
        raw = cdata.group("body")
    return _one_line(raw, 1000)


def _compact_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").strip().lower())


def _sorted_task_ids(tasks: dict[str, dict[str, str]]) -> list[str]:
    def key(value: str) -> tuple[int, int | str]:
        if value.isdigit():
            return (0, int(value))
        return (1, value)

    return sorted(tasks, key=key)


def _one_line(value: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _fit_snapshot(snapshot: str, *, max_chars: int) -> str:
    limit = max(400, int(max_chars or 1200))
    if len(snapshot) <= limit:
        return snapshot
    lines = snapshot.splitlines()
    kept: list[str] = []
    for line in lines:
        candidate = "\n".join([*kept, line])
        if len(candidate) > limit - len("\n...[preserved task state truncated]"):
            break
        kept.append(line)
    if len(kept) < len(lines):
        kept.append("...[preserved task state truncated]")
    return "\n".join(kept)[:limit]
