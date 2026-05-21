from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable
from typing import Any


_DEFAULT_PROTOCOL_MARKER = "You are using WebLLM Gateway's strict tool bridge."
_REQUIRED_TOOL_FORMAT_MARKER = "Required tool-call format:"
_DS2API_HISTORY_TITLE = "# DS2API_HISTORY.txt"
_DS2API_HISTORY_SUMMARY = "Prior conversation history and tool progress."
_DS2API_TOOLS_TITLE = "# DS2API_TOOLS.txt"
_DS2API_TOOLS_SUMMARY = "Available tool descriptions and parameter schemas."
PRESERVED_TASK_STATE_MARKER = "# WebLLM Gateway preserved task state"
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
_ERROR_RECOVERY_CONTROL_MARKERS = (
    "system-generated error correction",
    "system generated error correction",
    "previous answer went off task",
    "went off task into agent",
    "environment configuration advice",
    "continue the original local-agent task",
    "continue the original local agent task",
    "original task is missing",
    "valid dsml tool block",
    "latest user request is a system",
    "\u7cfb\u7edf\u751f\u6210\u7684\u9519\u8bef\u7ea0\u6b63",
    "\u9519\u8bef\u5904\u7406\u6307\u4ee4",
    "\u4e4b\u524d\u7684\u56de\u7b54\u504f\u79bb\u4e86\u4efb\u52a1",
    "\u7ee7\u7eed\u539f\u59cb\u672c\u5730\u4ee3\u7406\u4efb\u52a1",
    "\u539f\u59cb\u4efb\u52a1\u7f3a\u5931",
    "\u6ca1\u6709\u539f\u59cb\u4efb\u52a1",
)
STATELESS_WEB_API_GUARD = (
    "You are serving a stateless WebLLM Gateway API request. Ignore any previous website chat, "
    "account memory, profile memory, or project context that is not included in this request. "
    "Follow only the messages below. "
    "这是一次无状态 WebLLM Gateway API 请求；不要引用网页端旧会话、账号记忆、个人资料记忆或本次请求以外的项目上下文。"
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
        "\n\n[Prompt content was compacted by WebLLM Gateway for a web-model prompt budget. "
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
        max_chars=max(800, min(18000, int(live_limit * 0.45))),
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
        "Prompt content was compacted by WebLLM Gateway using layered DS2API_HISTORY live context.",
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
        blocks.append(_ds2api_tools_context_from_protocol(protocol_excerpt))

    fixed = "\n\n".join(blocks + ["=== LATEST CONVERSATION TAIL ===", continuation])
    if len(fixed) >= limit:
        return _fit_layered_fixed_prompt(blocks, continuation, limit=limit)
    tail_budget = max(500, limit - len(fixed) - 4)
    tail = _fit_history_entries(latest_entries, max_chars=tail_budget)
    prompt = "\n\n".join(blocks + ["=== LATEST CONVERSATION TAIL ===", tail.rstrip(), continuation])
    if len(prompt) <= limit:
        return prompt

    overflow = len(prompt) - limit
    if overflow < len(tail):
        tail = tail[overflow:].lstrip()
        prompt = "\n\n".join(blocks + ["=== LATEST CONVERSATION TAIL ===", tail.rstrip(), continuation])
    if len(prompt) <= limit:
        return prompt
    return _fit_layered_fixed_prompt(blocks, continuation, limit=limit)


def _fit_layered_fixed_prompt(blocks: list[str], continuation: str, *, limit: int) -> str:
    label = "=== LATEST CONVERSATION TAIL ==="
    fixed = "\n\n".join(blocks + [label, continuation])
    if len(fixed) <= limit:
        return fixed
    snapshot = blocks[0] if blocks and blocks[0].startswith(PRESERVED_TASK_STATE_MARKER) else ""
    remaining_blocks = blocks[1:] if snapshot else blocks
    if snapshot:
        snapshot_budget = limit - len(label) - len(continuation) - 4
        if snapshot_budget > 0:
            compact_snapshot = _fit_snapshot(snapshot, max_chars=snapshot_budget)
            if len(compact_snapshot) > snapshot_budget:
                compact_snapshot = compact_snapshot[:snapshot_budget].rstrip()
            prompt = "\n\n".join([compact_snapshot, label, continuation])
            if len(prompt) <= limit:
                return prompt
    without_snapshot = "\n\n".join(remaining_blocks + [label, continuation])
    if snapshot and len(without_snapshot) < limit:
        snapshot_budget = max(120, limit - len(without_snapshot) - 4)
        compact_snapshot = _fit_snapshot(snapshot, max_chars=snapshot_budget)
        prompt = "\n\n".join([compact_snapshot] + remaining_blocks + [label, continuation])
        if len(prompt) <= limit:
            return prompt
    return fixed[-limit:].lstrip()


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
    candidates = _user_request_candidates(entries)
    if not candidates:
        return ""
    latest = candidates[0]
    local_context = _recent_local_project_context_lines(entries, latest, prior_task=candidates[1] if len(candidates) > 1 else "")
    if local_context:
        return "\n".join(local_context + [f"Latest user request: {latest.strip()}"]).strip()
    if not (_looks_like_referential_followup_request(latest) or _looks_like_search_followup_request(latest)):
        return latest
    for prior in candidates[1:]:
        if prior.strip() and not (_looks_like_referential_followup_request(prior) or _looks_like_search_followup_request(prior)):
            return f"{prior.strip()}\n{latest.strip()}".strip()
    return latest


def _user_request_candidates(entries: list[tuple[str, str]]) -> list[str]:
    candidates: list[str] = []
    for role, text in reversed(entries):
        if _history_role_label(role) != "USER":
            continue
        content = (text or "").strip()
        if not content or _looks_like_current_request_control_text(content):
            continue
        task_text = _current_request_task_text(content)
        if task_text and not _looks_like_current_request_control_text(task_text):
            candidates.append(task_text)
    return candidates


def _looks_like_referential_followup_request(text: str) -> bool:
    value = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not value or len(value) > 260:
        return False
    reference_markers = (
        "\u521a\u624d",
        "\u521a\u521a",
        "\u4e4b\u524d",
        "\u524d\u9762",
        "\u4e0a\u9762",
        "\u90a3\u4e2a",
        "\u8fd9\u4e2a",
        "\u8fd9\u4ef6\u4e8b",
        "\u8fd9\u4e2a\u4efb\u52a1",
        "\u4e0a\u4e00\u6b21",
        "\u63d0\u4f9b\u8fc7",
        "\u94fe\u63a5",
        "url",
    )
    english_reference_markers = (
        "previous",
        "earlier",
        "above",
        "same",
        "that",
        "it",
        "provided",
    )
    action_markers = (
        "\u7ee7\u7eed",
        "\u63a5\u7740",
        "\u4fee\u590d",
        "\u5904\u7406",
        "\u6267\u884c",
        "\u91cd\u8bd5",
        "\u518d\u8bd5",
        "\u81ea\u5df1",
        "\u6293\u53d6",
        "\u67e5",
        "\u770b",
        "continue",
        "fix",
        "retry",
        "use",
        "do it",
        "handle",
    )
    has_reference = any(marker in value for marker in reference_markers) or any(
        re.search(rf"\b{re.escape(marker)}\b", value) for marker in english_reference_markers
    )
    if not has_reference:
        return False
    if any(marker in value for marker in action_markers):
        return True
    return ("\u94fe\u63a5" in value or "url" in value) and (
        "\u63d0\u4f9b\u8fc7" in value or "provided" in value or "same" in value
    )


def _looks_like_search_followup_request(text: str) -> bool:
    value = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not value or len(value) > 160:
        return False
    markers = (
        "\u4f60\u81ea\u5df1\u641c",
        "\u81ea\u5df1\u641c",
        "\u641c\u554a",
        "\u641c\u4e00\u4e0b",
        "\u53bb\u641c",
        "\u5e2e\u6211\u641c",
        "\u4f60\u81ea\u5df1\u67e5",
        "\u81ea\u5df1\u67e5",
        "\u67e5\u4e00\u4e0b",
        "\u4e0a\u7f51\u67e5",
        "\u4e0a\u7f51\u641c",
        "\u8054\u7f51\u67e5",
        "\u8054\u7f51\u641c",
        "search it",
        "search yourself",
        "look it up",
        "google it",
        "browse for it",
    )
    if any(marker in value for marker in markers):
        return True
    return len(value) <= 80 and ("\u81ea\u5df1" in value or "\u4f60" in value) and any(
        marker in value
        for marker in (
            "\u8054\u7f51",
            "\u641c\u7d22",
            "\u641c",
            "\u67e5\u4e00\u4e0b",
            "\u67e5\u627e",
            "search",
            "browse",
            "lookup",
        )
    )


def _recent_local_project_context_lines(
    entries: list[tuple[str, str]],
    latest_user_text: str,
    *,
    prior_task: str = "",
) -> list[str]:
    if _looks_like_project_context_reset_request(latest_user_text):
        return []
    if not _looks_like_local_project_reference_request(latest_user_text):
        return []
    project_path = _recent_local_project_path(entries)
    if not project_path:
        return []
    lines = [f"Project path: {project_path}"]
    if prior_task:
        lines.append(f"Previous user task: {_one_line(prior_task, 240)}")
    lines.append("Use local project files and tool evidence before public web search when this request asks about this project.")
    return lines


def _looks_like_local_project_reference_request(text: str) -> bool:
    value = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not value or len(value) > 500:
        return False
    if _looks_like_project_context_reset_request(value):
        return False
    markers = (
        "\u8fd9\u4e2a\u9879\u76ee",
        "\u8be5\u9879\u76ee",
        "\u5f53\u524d\u9879\u76ee",
        "\u672c\u9879\u76ee",
        "\u8fd9\u4e2a\u4ed3\u5e93",
        "\u8be5\u4ed3\u5e93",
        "\u5f53\u524d\u4ed3\u5e93",
        "\u672c\u4ed3\u5e93",
        "\u9879\u76ee",
        "\u4ed3\u5e93",
        "this project",
        "this repo",
        "this repository",
        "current project",
        "current repo",
        "current repository",
        "local project",
        "local repo",
        "local repository",
    )
    if any(marker in value for marker in markers):
        return True
    return bool(re.search(r"\b[\w.-]+\s+(?:project|repo|repository)\b", value))


def _looks_like_project_context_reset_request(text: str) -> bool:
    value = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not value or len(value) > 500:
        return False
    markers = (
        "\u53e6\u4e00\u4e2a\u9879\u76ee",
        "\u53e6\u4e00\u4e2a\u4ed3\u5e93",
        "\u53e6\u5916\u4e00\u4e2a\u9879\u76ee",
        "\u53e6\u5916\u4e00\u4e2a\u4ed3\u5e93",
        "\u5916\u90e8\u9879\u76ee",
        "\u5916\u90e8\u4ed3\u5e93",
        "\u72ec\u7acb\u9879\u76ee",
        "\u72ec\u7acb\u4ed3\u5e93",
        "\u4e0d\u540c\u9879\u76ee",
        "\u4e0d\u540c\u4ed3\u5e93",
        "\u4e0d\u662f\u8fd9\u4e2a\u9879\u76ee",
        "\u4e0d\u662f\u8fd9\u4e2a\u4ed3\u5e93",
        "\u4e0d\u662f\u5f53\u524d\u9879\u76ee",
        "\u4e0d\u662f\u5f53\u524d\u4ed3\u5e93",
        "another project",
        "another repo",
        "another repository",
        "different project",
        "different repo",
        "different repository",
        "external project",
        "external repo",
        "external repository",
        "separate project",
        "separate repo",
        "separate repository",
        "not this project",
        "not this repo",
        "not this repository",
        "not the current project",
        "not the current repo",
        "not the current repository",
    )
    return any(marker in value for marker in markers)


def _recent_local_project_path(entries: list[tuple[str, str]]) -> str:
    seen: set[str] = set()
    for _, text in reversed(entries):
        for path in _local_project_paths_from_text(text):
            normalized = _normalize_path_for_compare(path)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            if _windows_path_looks_like_directory(path):
                return path
    return ""


def _local_project_paths_from_text(text: str) -> list[str]:
    paths: list[str] = []
    for match in _WINDOWS_DRIVE_PATH_RE.finditer(text or ""):
        path = _canonical_local_project_path(match.group("path"))
        if path:
            paths.append(path)
    return paths


def _canonical_local_project_path(path: str) -> str:
    return (path or "").strip().strip("\"'`").rstrip(".,;:，。；：)]}").replace("\\", "/").rstrip("/")


def _normalize_path_for_compare(path: str) -> str:
    return _canonical_local_project_path(path).lower()


def _windows_path_looks_like_directory(path: str) -> bool:
    normalized = _canonical_local_project_path(path)
    leaf = normalized.rsplit("/", 1)[-1]
    return not bool(re.search(r"\.[A-Za-z0-9]{1,12}$", leaf))


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
    if _STRUCTURAL_CONTROL_MESSAGE_RE.match(str(text or "")):
        return True
    compact = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    if not compact:
        return True
    if compact in {"(no content)", "no content", "[no content]", "(empty)", "empty"}:
        return True
    if (
        "<system-reminder" in compact
        or "</system-reminder" in compact
        or "<system_reminder" in compact
        or "</system_reminder" in compact
        or "<time-reminder" in compact
        or "<time_reminder" in compact
        or compact.startswith("this is a system reminder")
    ):
        return True
    if looks_like_gateway_tool_observation(text):
        return True
    control_markers = (
        "system-reminder",
        "system_reminder",
        "time-reminder",
        "time_reminder",
        "do not mention this message",
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
    if any(marker in compact for marker in control_markers):
        return True
    return _looks_like_error_recovery_control_text(compact)


def _looks_like_error_recovery_control_text(compact: str) -> bool:
    if not compact:
        return False
    has_history_context = "ds2api_history" in compact or "latest user request" in compact
    if not has_history_context:
        return False
    return any(marker in compact for marker in _ERROR_RECOVERY_CONTROL_MARKERS)


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


def _ds2api_tools_context_from_protocol(protocol_excerpt: str) -> str:
    body = (protocol_excerpt or "").strip()
    if not body:
        return ""
    return "\n".join(
        [
            _DS2API_TOOLS_TITLE,
            _DS2API_TOOLS_SUMMARY,
            "",
            body,
        ]
    ).rstrip()


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
_PLAIN_TOOL_CALL_HEADER_RE = re.compile(
    r"\b(?:Assistant requested tool calls|requested tool calls|tool calls requested)\b",
    re.IGNORECASE,
)
_PLAIN_TOOL_CALL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?P<name>[A-Za-z_][A-Za-z0-9_:\.-]*)\((?P<args>[^)]{0,1200})\)\s*$"
)
_PLAIN_TOOL_ARG_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_:-]*)\s*=\s*(?P<value>.*?)(?=,\s*[A-Za-z_][A-Za-z0-9_:-]*\s*=|$)"
)
_WINDOWS_DRIVE_PATH_RE = re.compile(r"(?<![\w/])(?P<path>[A-Za-z]:[\\/][^\s\"'&|;<>]*)")
_RESULT_SIGNAL_RE = re.compile(
    r"\b(?:file does not exist|no such file|not found|is_error:\s*true|"
    r"unchanged content|reported unchanged|already available in history|missing content|"
    r"browser is already running|use --isolated|profile is already running|already running)\b",
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
_STRUCTURAL_CONTROL_MESSAGE_RE = re.compile(
    r"^\s*(?:<\s*/?\s*)?(?:system[-_ ]?reminder|time[-_ ]?reminder)\b",
    re.IGNORECASE,
)


def message_entries_for_ds2api_prompt(message: dict[str, Any], content_text: str) -> list[tuple[str, str]]:
    """Return prompt-visible role entries using ds2api's Claude tool history shape."""
    if not isinstance(message, dict):
        return []
    role = str(message.get("role") or "user").strip().lower() or "user"
    text = str(content_text or "").strip()
    entries: list[tuple[str, str]] = []

    if role == "assistant":
        if text:
            entries.append(("assistant", text))
        rendered_calls = format_tool_calls_for_ds2api_prompt(_prompt_visible_tool_calls_from_message(message))
        if rendered_calls:
            entries.append(("assistant", rendered_calls))
        return entries

    if role == "user":
        tool_results = _prompt_visible_tool_results_from_content(message.get("content"))
        if text:
            entries.append(("user", text))
        entries.extend(("tool", result) for result in tool_results if result.strip())
        return entries

    if role == "tool":
        rendered = _format_tool_result_for_ds2api_prompt(message, text)
        return [("tool", rendered)] if rendered else []
    return [(role, text)] if text else []


def format_tool_calls_for_ds2api_prompt(tool_calls: Iterable[Any]) -> str:
    call_lines: list[str] = []
    for item in tool_calls:
        name, args = _tool_call_name_and_args(item)
        if not name:
            continue
        call_lines.append(f'  <|DSML|invoke name="{html.escape(name, quote=True)}">')
        for key, value in args.items():
            call_lines.extend(_render_dsml_parameter(str(key), value, indent="    "))
        call_lines.append("  </|DSML|invoke>")
    if not call_lines:
        return ""
    return "\n".join(["<|DSML|tool_calls>", *call_lines, "</|DSML|tool_calls>"])


def looks_like_current_request_control_text(text: str) -> bool:
    return _looks_like_current_request_control_text(text)


def _prompt_visible_tool_calls_from_message(message: dict[str, Any]) -> list[Any]:
    calls: list[Any] = []
    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        calls.extend(raw_calls)
    content = message.get("content")
    blocks = content if isinstance(content, list) else [content] if isinstance(content, dict) else []
    for block in blocks:
        if not isinstance(block, dict) or str(block.get("type") or "") != "tool_use":
            continue
        name = str(block.get("name") or "").strip()
        if not name:
            continue
        args = block.get("input")
        if not isinstance(args, dict):
            args = {}
        calls.append(
            {
                "id": str(block.get("id") or block.get("tool_use_id") or ""),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
    return calls


def _tool_call_name_and_args(item: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(item, dict):
        return "", {}
    fn = item.get("function") if isinstance(item.get("function"), dict) else {}
    name = str(fn.get("name") or item.get("name") or "").strip()
    args: Any = fn.get("arguments") if fn else None
    if args is None:
        args = item.get("arguments") or item.get("args") or item.get("input") or {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except Exception:
            parsed = {}
        args = parsed if isinstance(parsed, dict) else {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _render_dsml_parameter(name: str, value: Any, *, indent: str) -> list[str]:
    escaped = html.escape(name, quote=True)
    if isinstance(value, dict):
        lines = [f'{indent}<|DSML|parameter name="{escaped}">']
        for key, child in value.items():
            lines.extend(_render_dsml_child(str(key), child, indent=indent + "  "))
        lines.append(f"{indent}</|DSML|parameter>")
        return lines
    if isinstance(value, list):
        lines = [f'{indent}<|DSML|parameter name="{escaped}">']
        for item in value:
            lines.extend(_render_dsml_child("item", item, indent=indent + "  "))
        lines.append(f"{indent}</|DSML|parameter>")
        return lines
    if isinstance(value, str):
        return [f'{indent}<|DSML|parameter name="{escaped}">{_dsml_cdata(value)}</|DSML|parameter>']
    return [f'{indent}<|DSML|parameter name="{escaped}">{json.dumps(value, ensure_ascii=False)}</|DSML|parameter>']


def _render_dsml_child(name: str, value: Any, *, indent: str) -> list[str]:
    escaped = html.escape(name, quote=True)
    if isinstance(value, dict):
        lines = [f"{indent}<{escaped}>"]
        for key, child in value.items():
            lines.extend(_render_dsml_child(str(key), child, indent=indent + "  "))
        lines.append(f"{indent}</{escaped}>")
        return lines
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            lines.extend(_render_dsml_child(name, item, indent=indent))
        return lines
    if isinstance(value, str):
        return [f"{indent}<{escaped}>{_dsml_cdata(value)}</{escaped}>"]
    return [f"{indent}<{escaped}>{json.dumps(value, ensure_ascii=False)}</{escaped}>"]


def _dsml_cdata(text: str) -> str:
    return "<![CDATA[" + (text or "").replace("]]>", "]]]]><![CDATA[>") + "]]>"


def _prompt_visible_tool_results_from_content(content: Any) -> list[str]:
    blocks = content if isinstance(content, list) else [content] if isinstance(content, dict) else []
    results: list[str] = []
    for block in blocks:
        if not isinstance(block, dict) or str(block.get("type") or "") != "tool_result":
            continue
        text = _tool_result_content_to_text(block.get("content"))
        if text.strip():
            results.append(_format_tool_result_for_ds2api_prompt(block, text))
    return results


def _format_tool_result_for_ds2api_prompt(message: dict[str, Any], text: str) -> str:
    content = str(text or "").strip()
    parts: list[str] = []
    name = str(message.get("name") or "").strip()
    tool_call_id = str(message.get("tool_call_id") or message.get("tool_use_id") or "").strip()
    if name:
        parts.append(f"name={name}")
    if tool_call_id:
        parts.append(f"tool_call_id={tool_call_id}")
    if not parts:
        return content
    header = "[" + " ".join(parts) + "]"
    return f"{header}\n{content}" if content else header


def _tool_result_content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_tool_result_content_part_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return _tool_result_content_part_to_text(value)
    return "" if value is None else str(value)


def _tool_result_content_part_to_text(value: Any) -> str:
    if not isinstance(value, dict):
        return "" if value is None else str(value)
    if str(value.get("type") or "") == "text":
        return str(value.get("text") or "")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
    snapshot_limit = int(max_chars or 1200)
    tasks = _extract_task_updates(entry_list)
    raw_recent_calls = _extract_recent_tool_call_summaries(entry_list)
    recent_calls = _latest_unique(raw_recent_calls, limit=10)
    no_progress_signals = _extract_no_progress_loop_signals(raw_recent_calls)
    result_signals = _latest_unique(_extract_tool_result_signals(entry_list), limit=6)
    read_excerpts = _extract_recent_read_result_excerpts(
        entry_list,
        limit=8 if snapshot_limit >= 9000 else 5 if snapshot_limit >= 3000 else 3,
        max_excerpt_chars=max(240, min(2600, snapshot_limit // 3)),
    )
    user_candidates = _user_request_candidates(entry_list)
    local_project_context = _recent_local_project_context_lines(
        entry_list,
        user_candidates[0] if user_candidates else "",
        prior_task=user_candidates[1] if len(user_candidates) > 1 else "",
    )
    if not tasks and not (recent_calls or no_progress_signals or result_signals or read_excerpts or local_project_context):
        return ""
    if not tasks and not local_project_context and not no_progress_signals and not _has_active_recent_tool_evidence(entry_list):
        return ""

    lines = [
        PRESERVED_TASK_STATE_MARKER,
        (
            "Generated before prompt compaction. Treat this as the active task ledger and recent tool evidence; "
            "verify contradictory file evidence before marking work complete."
        )
        if tasks
        else (
            "Generated before prompt compaction. Treat this as recent tool evidence; use it only when relevant "
            "to the current user request and verify contradictory file evidence before marking work complete."
        ),
    ]
    if tasks:
        lines.append("Tasks:")
        for task_id in _sorted_task_ids(tasks):
            task = tasks[task_id]
            status = task.get("status") or "unknown"
            description = task.get("description") or "no description"
            lines.append(f"- Task {task_id}: {status} - {_one_line(description, 180)}")
    if local_project_context:
        lines.append("Local project context:")
        lines.extend(f"- {line}" for line in local_project_context)
    if result_signals and not tasks:
        lines.append("Tool result signals:")
        lines.extend(f"- {_one_line(signal, 220)}" for signal in result_signals)
    if read_excerpts:
        lines.append("Read result excerpts:")
        lines.append(
            "- Cache rule: if a later Read result says unchanged or already available in history, reuse these "
            "cached excerpts as current file context. Do not Read/Glob the same path only to recover content."
        )
        for path, excerpt in read_excerpts:
            alias_text = _read_path_alias_text(path)
            suffix = f" ({alias_text})" if alias_text else ""
            lines.append(f"- {_one_line(path, 180)}{suffix}:")
            lines.extend(_indent_snapshot_block(excerpt))
    if no_progress_signals:
        lines.append("No-progress tool loop signals:")
        lines.extend(f"- {signal}" for signal in no_progress_signals)
    if recent_calls:
        lines.append("Recent tool calls:")
        lines.extend(f"- {summary}" for summary in recent_calls)
    if result_signals and tasks:
        lines.append("Tool result signals:")
        lines.extend(f"- {_one_line(signal, 220)}" for signal in result_signals)
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


def _has_active_recent_tool_evidence(entries: list[tuple[str, str]]) -> bool:
    skipped_latest_user = False
    for role, text in reversed(entries):
        content = (text or "").strip()
        if not content:
            continue
        label = _history_role_label(role)
        if label == "TOOL":
            if skipped_latest_user:
                if _RESULT_SIGNAL_RE.search(content):
                    return True
                continue
            return True
        if label == "ASSISTANT" and (
            next(iter(_iter_invocations(content)), None) is not None or _plain_tool_call_summaries(content)
        ):
            if skipped_latest_user:
                return _assistant_has_read_like_tool_call(content)
            return True
        if label == "USER" and (
            looks_like_gateway_tool_observation(content) or _looks_like_current_request_control_text(content)
        ):
            return True
        if label == "USER" and not skipped_latest_user:
            skipped_latest_user = True
            continue
        return False
    return False


def _assistant_has_read_like_tool_call(text: str) -> bool:
    for name, _ in _iter_invocations(text):
        if _is_read_like_tool_name(name):
            return True
    for summary in _plain_tool_call_summaries(text):
        name = summary.split("(", 1)[0]
        if _is_read_like_tool_name(name):
            return True
    return False


def _is_read_like_tool_name(name: str) -> bool:
    return _compact_name(name) in {"read", "readfile", "fileread"}


def _latest_unique(values: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for value in reversed(values):
        key = _recent_evidence_dedupe_key(value)
        if key in seen:
            continue
        seen.add(key)
        kept.insert(0, value)
        if len(kept) >= limit:
            break
    return kept


def _recent_evidence_dedupe_key(value: str) -> str:
    key = _one_line(value, 1000)
    key = re.sub(r"\bcall id:\s*[^,\):]+", "call id:<id>", key, flags=re.IGNORECASE)
    key = re.sub(r"\btoolu(?:_call)?_[A-Za-z0-9_:-]+", "toolu_<id>", key)
    return key


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
    for role, text in entries:
        for name, params in _iter_invocations(text):
            summaries.append(_tool_call_summary(name, params))
        if _history_role_label(role) == "ASSISTANT":
            summaries.extend(_plain_tool_call_summaries(text))
    return summaries


def _extract_no_progress_loop_signals(call_summaries: list[str], *, limit: int = 4) -> list[str]:
    window = [summary for summary in call_summaries[-32:] if summary]
    if not window:
        return []
    if any(_is_mutation_tool_summary(summary) for summary in window[-12:]):
        return []

    records: dict[str, dict[str, int | str]] = {}
    for index, summary in enumerate(window):
        if not _is_no_progress_loop_tool_summary(summary):
            continue
        key = _recent_evidence_dedupe_key(summary)
        record = records.setdefault(key, {"summary": summary, "count": 0, "last": 0})
        record["count"] = int(record["count"]) + 1
        record["last"] = index

    repeated = [
        record
        for record in records.values()
        if int(record["count"]) >= 3
    ]
    repeated.sort(key=lambda item: (-int(item["count"]), -int(item["last"])))
    signals: list[str] = []
    for record in repeated[: max(1, int(limit or 1))]:
        summary = str(record["summary"])
        count = int(record["count"])
        signals.append(
            f"{summary} repeated {count} times recently. Do not request the same input again; "
            "treat the earlier DS2API_HISTORY/preserved result as current context, then use a materially "
            "different input, Edit/Write when ready, or answer from available evidence."
        )
    return signals


def _is_no_progress_loop_tool_summary(summary: str) -> bool:
    name = _compact_name((summary or "").split("(", 1)[0])
    return name in {"read", "readfile", "fileread", "glob", "grep", "ls", "lsp", "list", "listdir", "listdirectory"}


def _is_mutation_tool_summary(summary: str) -> bool:
    name = _compact_name((summary or "").split("(", 1)[0])
    return name in {"edit", "editfile", "multiedit", "write", "writefile", "applypatch", "patch", "notebookedit"}


def _tool_call_summary(name: str, params: dict[str, str]) -> str:
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
    return f"{name}({', '.join(pieces)})" if pieces else f"{name}()"


def _plain_tool_call_summaries(text: str) -> list[str]:
    if not _PLAIN_TOOL_CALL_HEADER_RE.search(text or ""):
        return []
    summaries: list[str] = []
    for line in (text or "").splitlines():
        match = _PLAIN_TOOL_CALL_RE.match(line)
        if not match:
            continue
        name = match.group("name").strip()
        if not name:
            continue
        summaries.append(_tool_call_summary(name, _parse_plain_tool_call_args(match.group("args") or "")))
    return summaries


def _parse_plain_tool_call_args(raw: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for match in _PLAIN_TOOL_ARG_RE.finditer(raw or ""):
        key = match.group("key").strip()
        value = _strip_plain_arg_value(match.group("value"))
        if key and value:
            params[key] = _one_line(value, 1000)
    return params


def _strip_plain_arg_value(value: str) -> str:
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _extract_tool_result_signals(entries: list[tuple[str, str]]) -> list[str]:
    signals: list[str] = []
    for role, text in entries:
        if str(role or "").strip().lower() not in {"tool", "user"}:
            continue
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if _RESULT_SIGNAL_RE.search(line):
                signal = line
                if _looks_like_tool_result_error_header(line):
                    detail = _first_tool_result_detail_after(lines, index)
                    if detail:
                        signal = f"{line} {detail}"
                signals.append(signal)
                break
    return signals


def _extract_recent_read_result_excerpts(
    entries: list[tuple[str, str]],
    *,
    limit: int,
    max_excerpt_chars: int,
) -> list[tuple[str, str]]:
    pending_read_paths: list[str] = []
    captured: list[tuple[str, str]] = []

    for role, text in entries:
        content = (text or "").strip()
        if not content:
            continue
        label = _history_role_label(role)
        if label == "ASSISTANT":
            pending_read_paths.extend(_read_like_tool_paths_from_assistant_text(content))
            if len(pending_read_paths) > 16:
                pending_read_paths = pending_read_paths[-16:]
            continue
        if label != "TOOL" or not pending_read_paths:
            continue

        path = pending_read_paths.pop(0)
        reusable_content = _strip_tool_result_metadata_for_snapshot(content)
        if not _is_reusable_read_result_content(reusable_content):
            continue
        captured.append((path, _compact_multiline_excerpt(reusable_content, max_chars=max_excerpt_chars)))

    kept: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path, excerpt in reversed(captured):
        key = _recent_read_result_key(path)
        if key in seen:
            continue
        seen.add(key)
        kept.insert(0, (path, excerpt))
        if len(kept) >= max(1, int(limit or 1)):
            break
    return kept


def _read_like_tool_paths_from_assistant_text(text: str) -> list[str]:
    paths: list[str] = []
    for name, params in _iter_invocations(text):
        if _is_read_like_tool_name(name):
            paths.append(_read_path_from_tool_params(params))
    if _PLAIN_TOOL_CALL_HEADER_RE.search(text or ""):
        for line in (text or "").splitlines():
            match = _PLAIN_TOOL_CALL_RE.match(line)
            if not match:
                continue
            name = match.group("name").strip()
            if _is_read_like_tool_name(name):
                paths.append(_read_path_from_tool_params(_parse_plain_tool_call_args(match.group("args") or "")))
    return paths


def _read_path_from_tool_params(params: dict[str, str]) -> str:
    for key in ("file_path", "path", "filename", "file", "target_file"):
        value = params.get(key)
        if value:
            return _one_line(value, 180)
    return "unknown read target"


def _strip_tool_result_metadata_for_snapshot(content: str) -> str:
    lines = (content or "").strip().splitlines()
    while len(lines) >= 2:
        first = lines[0].strip()
        if not (_looks_like_non_error_tool_result_header(first) or _looks_like_ds2api_tool_result_header(first)):
            break
        lines = lines[1:]
    stripped = "\n".join(lines).strip()
    return stripped or (content or "").strip()


def _is_reusable_read_result_content(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    head_lines = [line.strip() for line in text.splitlines() if line.strip()][:6]
    first_line = head_lines[0] if head_lines else ""
    if _looks_like_tool_result_error_header(first_line):
        return False
    for line in head_lines:
        if len(line) <= 500 and _RESULT_SIGNAL_RE.search(line):
            return False
    lowered = "\n".join(head_lines).lower()
    if "the tool call failed" in lowered[:300]:
        return False
    return True


def _compact_multiline_excerpt(value: str, *, max_chars: int) -> str:
    text = (value or "").strip()
    limit = max(120, int(max_chars or 600))
    if len(text) <= limit:
        return text
    omission = "\n...[read result excerpt truncated]...\n"
    body_budget = max(80, limit - len(omission))
    head_budget = max(60, int(body_budget * 0.65))
    tail_budget = max(20, body_budget - head_budget)
    return (text[:head_budget].rstrip() + omission + text[-tail_budget:].lstrip())[:limit]


def _indent_snapshot_block(value: str) -> list[str]:
    lines = (value or "").splitlines() or [""]
    return [f"  {line}" if line else "  " for line in lines]


def _recent_read_result_key(path: str) -> str:
    return re.sub(r"[\\]+", "/", (path or "").strip().lower())


def _read_path_alias_text(path: str) -> str:
    normalized = _recent_read_result_key(path)
    if not normalized or "/" not in normalized:
        return ""
    leaf = normalized.rsplit("/", 1)[-1]
    if not leaf or leaf == normalized:
        return ""
    return f"also matches {leaf}"


def _looks_like_tool_result_error_header(line: str) -> bool:
    lowered = (line or "").lower()
    return "tool result for" in lowered and re.search(r"\bis_error:\s*true\b", lowered) is not None


def _looks_like_non_error_tool_result_header(line: str) -> bool:
    lowered = (line or "").lower()
    return "tool result for" in lowered and re.search(r"\bis_error:\s*false\b", lowered) is not None


def _looks_like_ds2api_tool_result_header(line: str) -> bool:
    return re.match(r"^\s*\[(?:name|tool_call_id)=[^\]]+\]\s*$", line or "", re.IGNORECASE) is not None


def _first_tool_result_detail_after(lines: list[str], index: int) -> str:
    for line in lines[index + 1 :]:
        lowered = line.lower()
        if lowered.startswith("the tool call failed.") or lowered.startswith("use this tool result to continue"):
            continue
        return line
    return ""


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
