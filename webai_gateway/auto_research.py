from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from webai_gateway.tool_bridge_replay import ReplayCase, load_replay_cases, run_replay_case


ACTIONABLE_ERROR_KINDS = {
    "unsafe_local_shell_command",
    "write_after_failed_read_without_discovery",
    "write_after_failed_path_without_discovery",
    "deferred_code_change_without_call",
    "deferred_tool_action_without_call",
    "tool_denial_without_call",
    "unverified_code_change_completion",
}

_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(cookie|authorization|bearer|api[_-]?key|session|token|secret|qwen_session|ds_session_id)"
    r"([\w.-]*\s*[:=]\s*)([^\s,;}\]\"']+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_MAX_CAPTURED_TEXT_CHARS = 2000


@dataclass(frozen=True)
class ReplayReport:
    total: int
    passed: int
    failed: int
    failures: list[dict[str, Any]]


def collect_failure_fixtures(transcript_path: str | Path) -> list[ReplayCase]:
    """Extract actionable ToolBridge replay fixtures from a Claude JSONL transcript."""

    path = Path(transcript_path)
    messages: list[dict[str, Any]] = []
    task_text = ""
    cases: list[ReplayCase] = []
    seen_ids: set[str] = set()

    for row in _iter_jsonl(path):
        message = row.get("message") if isinstance(row.get("message"), dict) else None
        if not message:
            continue
        role = str(message.get("role") or row.get("type") or "")
        content = message.get("content")
        timestamp = str(row.get("timestamp") or "")

        if role == "user" and not _is_tool_result_content(content):
            task_text = _user_task_text(content) or task_text

        if role == "assistant":
            for call in _assistant_tool_calls(content):
                payload = _build_replay_payload(
                    call=call,
                    messages=messages,
                    task_text=task_text,
                    timestamp=timestamp,
                )
                case = ReplayCase(id=payload["id"], path=Path(f"<auto-research:{payload['id']}>"), payload=payload)
                actual = run_replay_case(case)
                error = actual.get("error")
                if error in ACTIONABLE_ERROR_KINDS and case.id not in seen_ids:
                    payload = _with_case_id(payload, str(error), timestamp)
                    payload["expected"] = {
                        "error": error,
                        "tool_calls": actual["tool_calls"],
                        "warning_contains": None,
                    }
                    cases.append(ReplayCase(id=payload["id"], path=Path(f"<auto-research:{payload['id']}>"), payload=payload))
                    seen_ids.add(payload["id"])

        converted = _convert_message(message)
        if converted:
            messages.append(converted)

    return cases


def write_failure_fixtures(cases: list[ReplayCase], output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case in cases:
        path = root / f"{_safe_id(case.id)}.json"
        path.write_text(json.dumps(case.payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def run_replay_report(root: str | Path) -> ReplayReport:
    cases = load_replay_cases(Path(root))
    failures: list[dict[str, Any]] = []
    for case in cases:
        actual = run_replay_case(case)
        expected = case.payload["expected"]
        warning_contains = expected.get("warning_contains")
        matches = actual["error"] == expected["error"] and actual["tool_calls"] == expected["tool_calls"]
        if warning_contains:
            matches = matches and bool(actual["warning"] and warning_contains in actual["warning"])
        else:
            matches = matches and actual["warning"] is None
        if not matches:
            failures.append(
                {
                    "id": case.id,
                    "path": str(case.path),
                    "expected": expected,
                    "actual": actual,
                }
            )
    return ReplayReport(total=len(cases), passed=len(cases) - len(failures), failed=len(failures), failures=failures)


def build_auto_research_status(root: str | Path) -> dict[str, Any]:
    fixture_root = Path(root)
    collect_command = f"python -m webai_gateway.auto_research collect <claude-jsonl> --output {fixture_root}"
    report_command = f"python -m webai_gateway.auto_research report --fixtures {fixture_root}"
    if not fixture_root.exists():
        return {
            "available": False,
            "fixtureDir": str(fixture_root),
            "total": 0,
            "passed": 0,
            "failed": 0,
            "passRate": None,
            "failureKinds": [],
            "taxonomy": {},
            "knownFixtureIds": [],
            "recent": [],
            "failures": [],
            "collectCommand": collect_command,
            "reportCommand": report_command,
            "message": "Replay fixture 目录不存在，请先运行 collect 命令采集失败样本。",
        }
    try:
        cases = load_replay_cases(fixture_root)
        report = run_replay_report(fixture_root)
    except (AssertionError, OSError, ValueError) as exc:
        return {
            "available": False,
            "fixtureDir": str(fixture_root),
            "total": 0,
            "passed": 0,
            "failed": 0,
            "passRate": None,
            "failureKinds": [],
            "taxonomy": {},
            "knownFixtureIds": [],
            "recent": [],
            "failures": [{"message": str(exc)}],
            "collectCommand": collect_command,
            "reportCommand": report_command,
            "message": "Replay fixture 暂不可用，请检查 JSON 格式或回放目录。",
        }

    failure_counts: dict[str, int] = {}
    for case in cases:
        error = case.payload["expected"].get("error") or "allowed_tool_call"
        failure_counts[str(error)] = failure_counts.get(str(error), 0) + 1
    taxonomy = dict(sorted(failure_counts.items()))
    recent = sorted(cases, key=lambda case: _file_mtime(case.path), reverse=True)[:8]
    return {
        "available": True,
        "fixtureDir": str(fixture_root),
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "passRate": round(report.passed / report.total, 4) if report.total else None,
        "failureKinds": [
            {"kind": kind, "count": count}
            for kind, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "taxonomy": taxonomy,
        "knownFixtureIds": sorted(case.id for case in cases),
        "recent": [_case_summary(case) for case in recent],
        "failures": report.failures[:8],
        "collectCommand": collect_command,
        "reportCommand": report_command,
        "message": "Replay 回放全部通过。" if report.failed == 0 else "存在未通过的 replay，请先修复 Gateway 适配层。",
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="WebLLM Gateway auto-research helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect replay fixtures from a Claude JSONL transcript")
    collect.add_argument("transcript", type=Path, nargs="+")
    collect.add_argument("--output", type=Path, default=Path("tests/fixtures/tool_bridge_replays"))

    report = subparsers.add_parser("report", help="Run replay fixtures and print a JSON report")
    report.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/tool_bridge_replays"))

    args = parser.parse_args(argv)
    if args.command == "collect":
        cases_by_id: dict[str, ReplayCase] = {}
        for transcript in args.transcript:
            for case in collect_failure_fixtures(transcript):
                cases_by_id[case.id] = case
        written = write_failure_fixtures(list(cases_by_id.values()), args.output)
        print(json.dumps({"written": [str(path) for path in written]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "report":
        result = run_replay_report(args.fixtures)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        return 1 if result.failed else 0
    return 2


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(_redact_value(item))
    return rows


def _build_replay_payload(
    *,
    call: dict[str, Any],
    messages: list[dict[str, Any]],
    task_text: str,
    timestamp: str,
) -> dict[str, Any]:
    model_text = "```tool_json\n" + json.dumps({"calls": [call]}, ensure_ascii=False, separators=(",", ":")) + "\n```"
    fixture_messages = _fixture_messages(messages, task_text)
    return {
        "id": f"claude_pending_{_timestamp_suffix(timestamp)}",
        "description": "Auto-collected Claude Code ToolBridge failure candidate.",
        "protocol": "anthropic_messages",
        "model": "qwen-coder/qwen-coder-plus",
        "tool_bridge_config": {
            "mode": "strict",
            "activationPolicy": "auto",
            "exposurePolicy": "local-agent",
            "toolProfile": "auto",
            "maxCallsPerTurn": 4,
        },
        "input": {
            "task_text": task_text or "Local agent task",
            "messages": fixture_messages,
            "tools": _tool_defs_for_messages(fixture_messages, call),
            "model_text": model_text,
        },
        "expected": {"error": None, "tool_calls": [], "warning_contains": None},
    }


def _fixture_messages(messages: list[dict[str, Any]], task_text: str) -> list[dict[str, Any]]:
    useful = [message for message in messages if not _is_noise_user_message(message)]
    compact = list(useful[-12:])
    has_task = any(
        message.get("role") == "user" and not _is_tool_result_content(message.get("content"))
        for message in compact
    )
    if task_text and not has_task:
        compact.insert(0, {"role": "user", "content": task_text})
    return compact


def _convert_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = str(message.get("role") or "")
    content = _compact_value(_redact_value(message.get("content")))
    if role in {"user", "assistant", "system", "tool"}:
        out = {"role": role, "content": content}
        if role == "tool" and message.get("tool_call_id"):
            out["tool_call_id"] = str(message["tool_call_id"])
        return out
    return None


def _assistant_tool_calls(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        input_value = block.get("input") if isinstance(block.get("input"), dict) else {}
        calls.append(
            {
                "id": str(block.get("id") or f"call_{len(calls) + 1}"),
                "name": name.strip(),
                "input": _compact_value(_redact_value(input_value)),
            }
        )
    return calls


def _tool_defs_for_messages(messages: list[dict[str, Any]], call: dict[str, Any]) -> list[dict[str, Any]]:
    names = {"Read", "Glob", "Grep", "Edit", "Write", str(call.get("name") or "")}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for existing in _assistant_tool_calls(message.get("content")):
            names.add(str(existing.get("name") or ""))
    ordered = [name for name in ("Read", "Glob", "Grep", "Edit", "Write", "Bash", "Skill", "WebFetch") if name in names]
    return [_tool_def(name) for name in ordered]


def _tool_def(name: str) -> dict[str, Any]:
    properties: dict[str, Any]
    required: list[str]
    if name in {"Read", "Edit", "Write"}:
        properties = {"file_path": {"type": "string"}, "content": {"type": "string"}}
        required = ["file_path"]
    elif name in {"Glob", "Grep"}:
        properties = {"path": {"type": "string"}, "pattern": {"type": "string"}}
        required = ["pattern"]
    elif name == "Bash":
        properties = {"command": {"type": "string"}}
        required = ["command"]
    else:
        properties = {}
        required = []
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _is_tool_result_content(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _user_task_text(content: Any) -> str:
    text = _content_text(content).strip()
    if not text:
        return ""
    command_args = _extract_tag(text, "command-args")
    if command_args:
        return command_args.strip()
    if _is_noise_text(text):
        return ""
    return text


def _is_noise_user_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user" or _is_tool_result_content(message.get("content")):
        return False
    return _is_noise_text(_content_text(message.get("content")))


def _is_noise_text(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    return any(
        marker in stripped
        for marker in (
            "Base directory for this skill:",
            "<EXTREMELY-IMPORTANT>",
            "<command-message>",
            "[Request interrupted by user]",
        )
    )


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text or "", flags=re.DOTALL)
    return match.group(1) if match else ""


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        redacted = _BEARER_RE.sub("Bearer [REDACTED]", value)
        return _SENSITIVE_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", redacted)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in {"cookie", "authorization", "bearer", "api_key", "apikey", "token", "secret"}:
                out[key] = "[REDACTED]"
            else:
                out[key] = _redact_value(item)
        return out
    return value


def _compact_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= _MAX_CAPTURED_TEXT_CHARS:
            return value
        omitted = len(value) - _MAX_CAPTURED_TEXT_CHARS
        return value[:_MAX_CAPTURED_TEXT_CHARS] + f"\n...[truncated {omitted} chars]"
    if isinstance(value, list):
        return [_compact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _compact_value(item) for key, item in value.items()}
    return value


def _timestamp_suffix(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%H%M%S")
    except ValueError:
        return "unknown"


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "auto_research_case"


def _case_id(error: str, timestamp: str) -> str:
    return _safe_id(f"claude_{error}_{_timestamp_suffix(timestamp)}")


def _with_case_id(payload: dict[str, Any], error: str, timestamp: str) -> dict[str, Any]:
    payload = dict(payload)
    payload["id"] = _case_id(error, timestamp)
    return payload


def _case_summary(case: ReplayCase) -> dict[str, Any]:
    expected = case.payload["expected"]
    return {
        "id": case.id,
        "path": case.path.name,
        "error": expected.get("error"),
        "description": str(case.payload.get("description") or ""),
        "updatedAt": datetime.fromtimestamp(_file_mtime(case.path), tz=timezone.utc).isoformat(),
    }


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
