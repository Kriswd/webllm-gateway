from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from webai_gateway.app import create_app
from webai_gateway.anthropic_api import anthropic_body_to_openai
from webai_gateway.config import (
    GatewayConfig,
    ObservationPolicyConfig,
    ProviderRuntimeConfig,
    ServerConfig,
    ToolBridgeConfig,
    UpstreamConfig,
    load_config,
)
from webai_gateway.openai_api import (
    EMPTY_ASSISTANT_RESPONSE_TEXT,
    build_tool_call_sse,
    build_tool_refusal_recovery_payload,
    parse_chat_response,
    tool_bridge_rejected_response_text,
)
from webai_gateway.deepseek_web import DeepSeekWebClient, is_deepseek_web_model, normalize_deepseek_model
from webai_gateway.tool_controller import RetryState, classify_bridge_result
from webai_gateway.prompt_compaction import compact_web_prompt
from webai_gateway.qwen_web import (
    QwenWebClient,
    _collect_qwen_stream_lines,
    normalize_qwen_model,
    parse_qwen_stream_text,
    qwen_messages_to_prompt_and_files,
)
from webai_gateway.qwen_coder import (
    QwenCoderClient,
    _collect_qwen_coder_stream_lines,
    normalize_qwen_coder_model,
    parse_qwen_coder_stream_text,
    qwen_coder_messages_to_prompt_and_files,
)
from webai_gateway.tool_bridge import (
    BridgeError,
    BridgeResult,
    ToolCallDraft,
    build_local_repo_preflight_tool_call,
    build_context,
    build_repair_messages,
    parse_tool_response,
    prefer_local_tools_for_local_agent_task,
    prepare_openai_messages,
    compress_observation,
    sanitize_leaked_tool_protocol_output,
)
from webai_gateway.web_auth import (
    CredentialStore,
    DeepSeekWebAuthService,
    PROVIDERS,
    _read_qwen_credential,
    credential_summary,
    provider_payload,
)


def _config() -> GatewayConfig:
    return GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
    )


def _openai_response(content: str) -> dict[str, Any]:
    return {
        "id": "upstream-chatcmpl",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }


def _client(handler) -> TestClient:
    transport = httpx.MockTransport(handler)
    return TestClient(create_app(config=_config(), http_client=httpx.Client(transport=transport)))


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer local-dev-key"}


def _controller_context_with_tools(
    names: list[str],
    *,
    task_text: str = "Review this local project.",
    recent_tool_call_names: tuple[str, ...] = (),
):
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in names
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    return replace(
        context,
        task_text=task_text,
        has_tool_loop=True,
        recent_tool_call_names=recent_tool_call_names,
    )


def test_tool_controller_classifies_tool_calls_as_tool_call() -> None:
    result = BridgeResult(
        content="",
        tool_calls=[ToolCallDraft(id="call_1", name="Read", input={"file_path": "README.md"})],
    )

    decision = classify_bridge_result(
        result,
        _controller_context_with_tools(["Read"]),
        RetryState(),
    )

    assert decision.state == "TOOL_CALL"


def test_tool_controller_classifies_repairable_error_as_retry() -> None:
    result = BridgeResult(
        content="bad",
        tool_calls=[],
        error=BridgeError("tool_denial_without_call", "denied", repairable=True),
        raw_content="bad",
    )

    decision = classify_bridge_result(
        result,
        _controller_context_with_tools(["Read"]),
        RetryState(),
    )

    assert decision.state == "RETRY"
    assert decision.retry_kind == "repair"


def test_tool_bridge_accepts_ds2api_dsml_tool_calls() -> None:
    context = _controller_context_with_tools(["Read"])

    result = parse_tool_response(
        '<|DSML|tool_calls>\n'
        '  <|DSML|invoke name="Read">\n'
        '    <|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter>\n'
        '  </|DSML|invoke>\n'
        '</|DSML|tool_calls>',
        context,
    )

    assert result.error is None
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id.startswith("call_")
    assert result.tool_calls[0].name == "Read"
    assert result.tool_calls[0].input == {"file_path": "README.md"}


def test_tool_bridge_generates_ds2api_style_unique_ids_for_idless_xml_calls() -> None:
    context = _controller_context_with_tools(["Read"])
    raw = (
        '<|DSML|tool_calls>'
        '<|DSML|invoke name="Read">'
        '<|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter>'
        '</|DSML|invoke>'
        '</|DSML|tool_calls>'
    )

    first = parse_tool_response(raw, context)
    second = parse_tool_response(raw, context)

    assert first.error is None
    assert second.error is None
    assert first.tool_calls[0].id.startswith("call_")
    assert second.tool_calls[0].id.startswith("call_")
    assert first.tool_calls[0].id != second.tool_calls[0].id


def test_tool_bridge_accepts_ds2api_canonical_xml_tool_calls() -> None:
    context = _controller_context_with_tools(["Read"])

    result = parse_tool_response(
        '<tool_calls><invoke name="Read"><parameter name="file_path">README.md</parameter></invoke></tool_calls>',
        context,
    )

    assert result.error is None
    assert [(call.name, call.input) for call in result.tool_calls] == [("Read", {"file_path": "README.md"})]


def test_tool_bridge_accepts_ds2api_full_width_pipe_alias() -> None:
    context = _controller_context_with_tools(["Read"])

    result = parse_tool_response(
        '<｜DSML｜tool_calls><｜DSML｜invoke name="Read">'
        '<｜DSML｜parameter name="file_path">README.md</｜DSML｜parameter>'
        '</｜DSML｜invoke></｜DSML｜tool_calls>',
        context,
    )

    assert result.error is None
    assert [(call.name, call.input) for call in result.tool_calls] == [("Read", {"file_path": "README.md"})]


def test_tool_bridge_ignores_ds2api_tool_xml_inside_markdown_fence() -> None:
    context = _controller_context_with_tools(["Read"])

    result = parse_tool_response(
        'Example only:\n```xml\n'
        '<tool_calls><invoke name="Read"><parameter name="file_path">README.md</parameter></invoke></tool_calls>\n'
        '```\nDo not execute it.',
        context,
    )

    assert result.tool_calls == []
    assert result.error is None
    assert "Example only" in result.content


def test_tool_bridge_preserves_ds2api_cdata_with_code_fence_content() -> None:
    context = _controller_context_with_tools(["Write"])
    content = "before\n```python\nprint('hi')\n```\nafter"

    result = parse_tool_response(
        '<|DSML|tool_calls><|DSML|invoke name="Write">'
        '<|DSML|parameter name="file_path"><![CDATA[notes.md]]></|DSML|parameter>'
        f'<|DSML|parameter name="content"><![CDATA[{content}]]></|DSML|parameter>'
        '</|DSML|invoke></|DSML|tool_calls>',
        context,
    )

    assert result.error is None
    assert result.tool_calls[0].input == {"file_path": "notes.md", "content": content}


def test_tool_bridge_parses_ds2api_nested_xml_array_parameters() -> None:
    context = _controller_context_with_tools(["MultiEdit"])

    result = parse_tool_response(
        '<|DSML|tool_calls><|DSML|invoke name="MultiEdit">'
        '<|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter>'
        '<|DSML|parameter name="edits">'
        '<item><old_string><![CDATA[foo]]></old_string><new_string><![CDATA[bar]]></new_string></item>'
        '<item><old_string><![CDATA[baz]]></old_string><new_string><![CDATA[qux]]></new_string></item>'
        '</|DSML|parameter>'
        '</|DSML|invoke></|DSML|tool_calls>',
        context,
    )

    assert result.error is None
    assert result.tool_calls[0].input == {
        "file_path": "README.md",
        "edits": [
            {"old_string": "foo", "new_string": "bar"},
            {"old_string": "baz", "new_string": "qux"},
        ],
    }


@pytest.mark.parametrize(
    ("open_tag", "invoke_tag", "param_tag", "param_close", "invoke_close", "close_tag"),
    [
        ("<DSML|DSML|tool_calls>", "<DSML|DSML|invoke name=\"Read\">", "<DSML|DSML|parameter name=\"file_path\">", "</DSML|DSML|parameter>", "</DSML|DSML|invoke>", "</DSML|DSML|tool_calls>"),
        ("<<|DSML|tool_calls>", "<<|DSML|invoke name=\"Read\">", "<<|DSML|parameter name=\"file_path\">", "</|DSML|parameter>", "</|DSML|invoke>", "</|DSML|tool_calls>"),
        ("<|DSML tool_calls>", "<|DSML invoke name=\"Read\">", "<|DSML parameter name=\"file_path\">", "</|DSML parameter>", "</|DSML invoke>", "</|DSML tool_calls>"),
        ("<DSMLtool_calls>", "<DSMLinvoke name=\"Read\">", "<DSMLparameter name=\"file_path\">", "</DSMLparameter>", "</DSMLinvoke>", "</DSMLtool_calls>"),
    ],
)
def test_tool_bridge_tolerates_ds2api_dsml_tag_noise(
    open_tag: str,
    invoke_tag: str,
    param_tag: str,
    param_close: str,
    invoke_close: str,
    close_tag: str,
) -> None:
    context = _controller_context_with_tools(["Read"])

    result = parse_tool_response(
        f"{open_tag}{invoke_tag}{param_tag}README.md{param_close}{invoke_close}{close_tag}",
        context,
    )

    assert result.error is None
    assert [(call.name, call.input) for call in result.tool_calls] == [("Read", {"file_path": "README.md"})]


def test_tool_bridge_repairs_ds2api_missing_opening_wrapper() -> None:
    context = _controller_context_with_tools(["Read"])

    result = parse_tool_response(
        '<|DSML|invoke name="Read">'
        '<|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter>'
        '</|DSML|invoke></|DSML|tool_calls>',
        context,
    )

    assert result.error is None
    assert result.content == ""
    assert [(call.name, call.input) for call in result.tool_calls] == [("Read", {"file_path": "README.md"})]


def test_tool_bridge_ignores_ds2api_tool_xml_inside_unclosed_markdown_fence() -> None:
    context = _controller_context_with_tools(["Read"])

    result = parse_tool_response(
        'Example only:\n```xml\n'
        '<tool_calls><invoke name="Read"><parameter name="file_path">README.md</parameter></invoke></tool_calls>\n',
        context,
    )

    assert result.tool_calls == []
    assert result.error is None
    assert "Example only" in result.content


def test_sanitize_leaked_tool_protocol_output_matches_ds2api_tool_wrapper_cleanup() -> None:
    text = (
        "before\n"
        "<|DSML|tool_calls>\n"
        '  <|DSML|invoke name="Bash">\n'
        '    <|DSML|parameter name="command"><![CDATA[pwd]]></|DSML|parameter>\n'
        "  </|DSML|invoke>\n"
        "</|DSML|tool_calls>\n"
        "after"
    )

    assert sanitize_leaked_tool_protocol_output(text) == "before\n\nafter"


def test_sanitize_leaked_tool_protocol_output_matches_ds2api_agent_xml_cleanup() -> None:
    text = "Done.<attempt_completion><result>Some final answer</result></attempt_completion>"

    assert sanitize_leaked_tool_protocol_output(text) == "Done.Some final answer"


def test_tool_controller_stops_after_repair_budget() -> None:
    result = BridgeResult(
        content="bad",
        tool_calls=[],
        error=BridgeError("malformed_json", "bad json", repairable=True),
        raw_content="bad",
    )

    decision = classify_bridge_result(
        result,
        _controller_context_with_tools(["Read"]),
        RetryState(repair_attempts=1),
    )

    assert decision.state == "FINAL"
    assert decision.reason == "retry_budget_exhausted"


def test_tool_controller_asks_user_for_clarification_final_after_repair_budget() -> None:
    text = "Goal: Review code. Status: No changes found via git diff. Next: Confirm target branch or files for review."
    result = BridgeResult(
        content=text,
        tool_calls=[],
        error=BridgeError("unproven_final_answer_without_tool_call", "needs clarification", repairable=True),
        raw_content=text,
    )

    decision = classify_bridge_result(
        result,
        _controller_context_with_tools(
            ["AskUserQuestion", "Bash", "Glob", "Read"],
            task_text="Review this local project.",
            recent_tool_call_names=("Bash", "Glob", "Read"),
        ),
        RetryState(repair_attempts=1),
    )

    assert decision.state == "ASK_USER"
    assert decision.tool_calls is not None
    assert decision.tool_calls[0].name == "AskUserQuestion"
    assert "target branch or files" in json.dumps(decision.tool_calls[0].input)


def test_tool_bridge_routes_scope_selection_plain_text_to_ask_user_after_repair_budget() -> None:
    text = (
        "[CAVEMAN] Git clean. No changes. Review target unclear. Pick action: "
        "1. Review `rewrite_content.py` (AI rewrite logic) "
        "2. Review `sync_to_feishu.py` (Feishu sync logic) "
        "3. Review `process_answer.py` (Interview manager wrapper) "
        '4. Scan all root `.py` for quality issues Specify file or "scan all".'
    )
    context = _controller_context_with_tools(
        ["AskUserQuestion", "Bash", "Glob", "Read"],
        task_text="Simplify: Code Review and Cleanup",
        recent_tool_call_names=("Read", "Glob", "Bash", "Skill", "Glob"),
    )

    result = parse_tool_response(text, context)
    decision = classify_bridge_result(result, context, RetryState(repair_attempts=1))

    assert result.error is not None
    assert result.error.kind in {"premature_clarification_without_tool_call", "unproven_final_answer_without_tool_call"}
    assert decision.state == "ASK_USER"
    assert decision.tool_calls is not None
    assert decision.tool_calls[0].name == "AskUserQuestion"


def test_tool_controller_rejects_final_without_read_evidence_for_local_review() -> None:
    result = BridgeResult(
        content="Review complete. No issues found.",
        tool_calls=[],
        raw_content="Review complete. No issues found.",
    )

    decision = classify_bridge_result(
        result,
        _controller_context_with_tools(["Read"], task_text="Review this local project."),
        RetryState(),
    )

    assert decision.state == "RETRY"
    assert decision.reason == "insufficient_final_evidence"


def test_tool_controller_all_profile_passes_final_without_read_evidence_like_ds2api() -> None:
    result = BridgeResult(
        content="Review complete. No issues found.",
        tool_calls=[],
        raw_content="Review complete. No issues found.",
    )
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read tool",
                    "parameters": {"type": "object"},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
    )
    context = replace(context, task_text="Review this local project.", has_tool_loop=True)

    decision = classify_bridge_result(result, context, RetryState())

    assert decision.state == "FINAL"
    assert decision.reason == ""


def test_tool_controller_allows_final_after_read_evidence_for_readonly_review() -> None:
    result = BridgeResult(
        content="Review summary: no critical issues were found in inspected files.",
        tool_calls=[],
        raw_content="Review summary: no critical issues were found in inspected files.",
    )

    decision = classify_bridge_result(
        result,
        _controller_context_with_tools(
            ["Glob", "Read", "Grep"],
            task_text="Review this local project.",
            recent_tool_call_names=("Glob", "Read", "Grep"),
        ),
        RetryState(),
    )

    assert decision.state == "FINAL"


def test_tool_controller_retries_status_only_final_after_tool_result() -> None:
    result = BridgeResult(
        content="Bash completed with no output. Project dir exist.",
        tool_calls=[],
        raw_content="Bash completed with no output. Project dir exist.",
    )

    decision = classify_bridge_result(
        result,
        _controller_context_with_tools(
            ["Bash", "Glob", "Read"],
            task_text="Review this local project and give a detailed improvement plan.",
            recent_tool_call_names=("Bash", "Glob", "Read"),
        ),
        RetryState(),
    )

    assert decision.state == "RETRY"
    assert decision.reason == "status_only_final_without_task_answer"


def test_tool_controller_retries_no_task_final_after_tool_result() -> None:
    result = BridgeResult(
        content="Caveman mode active. No task given. Wait for input.",
        tool_calls=[],
        raw_content="Caveman mode active. No task given. Wait for input.",
    )

    decision = classify_bridge_result(
        result,
        _controller_context_with_tools(
            ["Glob", "Read"],
            task_text="Review this local project and give a detailed improvement plan.",
            recent_tool_call_names=("Glob",),
        ),
        RetryState(),
    )

    assert decision.state == "RETRY"
    assert decision.reason == "status_only_final_without_task_answer"


def test_tool_controller_retries_ds2api_history_summary_final_in_all_profile() -> None:
    text = (
        "根据提供的 `DS2API_HISTORY.txt` 上下文，当前的工作状态如下："
        "用户之前正在探索如何为 Claude 创建 Skills，助手读取了相关文档。"
        "当前状态是文件内容已经压缩返回，下一步可以继续阅读或总结。"
    )
    result = BridgeResult(content=text, tool_calls=[], raw_content=text)
    context = _controller_context_with_tools(
        ["Glob", "Read", "Grep", "Skill"],
        task_text="审查当前项目的代码，看看有什么需要改进的",
        recent_tool_call_names=("Glob", "Read", "Skill"),
    )
    context = replace(context, options=ToolBridgeConfig(exposure_policy="all", tool_profile="all"))

    decision = classify_bridge_result(result, context, RetryState())

    assert decision.state == "RETRY"
    assert decision.reason == "history_summary_final_without_task_answer"


def test_tool_controller_retries_off_task_environment_config_final_after_tool_loop() -> None:
    text = (
        'CAVEMAN MODE ACTIVE. Statusline badge not configured. Add to `~/.claude/settings.json`: '
        '```json\n"statusLine": {"type": "command", "command": "powershell -File caveman-statusline.ps1"}\n```'
    )
    result = BridgeResult(content=text, tool_calls=[], raw_content=text)

    decision = classify_bridge_result(
        result,
        _controller_context_with_tools(
            ["AskUserQuestion", "Glob", "Read", "Skill"],
            task_text="继续",
            recent_tool_call_names=("Glob", "Skill", "Skill"),
        ),
        RetryState(),
    )

    assert decision.state == "RETRY"
    assert decision.reason == "off_task_environment_configuration_final"


def test_tool_controller_retries_repairable_shell_without_gateway_ask_user() -> None:
    result = BridgeResult(
        content="",
        tool_calls=[],
        error=BridgeError("unsafe_local_shell_command", "needs shell confirmation", repairable=True),
        raw_content='```tool_json\n{"calls":[{"name":"Bash","input":{"command":"pip install -r requirements.txt"}}]}\n```',
    )

    decision = classify_bridge_result(
        result,
        _controller_context_with_tools(["AskUserQuestion", "Read", "Bash"]),
        RetryState(),
    )

    assert decision.state == "RETRY"
    assert decision.reason == "unsafe_local_shell_command"
    assert decision.tool_calls is None


def test_tool_bridge_normalizes_malformed_ask_user_question_schema() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "AskUserQuestion",
                    "parameters": {
                        "type": "object",
                        "required": ["questions"],
                        "properties": {"questions": {"type": "array"}},
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"AskUserQuestion","input":{"questions":[{"type":"string","content":"Allow git commands?"}]}}]}\n```',
        context,
    )

    assert result.error is None
    assert result.tool_calls[0].name == "AskUserQuestion"
    assert result.tool_calls[0].input["questions"][0]["question"] == "Allow git commands?"
    assert "header" not in result.tool_calls[0].input["questions"][0]
    assert "options" not in result.tool_calls[0].input["questions"][0]


def test_tool_bridge_normalizes_ask_user_question_description_only_schema() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "AskUserQuestion",
                    "parameters": {
                        "type": "object",
                        "required": ["questions"],
                        "properties": {"questions": {"type": "array"}},
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"AskUserQuestion","input":{"questions":[{"type":"string","description":"Need permission to run git commands. Allow?"}]}}]}\n```',
        context,
    )

    assert result.error is None
    assert result.tool_calls[0].input["questions"][0]["question"] == "Need permission to run git commands. Allow?"
    assert "header" not in result.tool_calls[0].input["questions"][0]
    assert "options" not in result.tool_calls[0].input["questions"][0]


def test_tool_bridge_does_not_synthesize_ask_user_question_options() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "AskUserQuestion",
                    "parameters": {
                        "type": "object",
                        "required": ["questions"],
                        "properties": {"questions": {"type": "array"}},
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_1","name":"AskUserQuestion","input":{"questions":[{"question":"您希望我对列出的 Skills 文件执行什么操作？"}]}}]}\n'
        "```",
        context,
    )

    assert result.error is None
    first_question = result.tool_calls[0].input["questions"][0]
    assert first_question["question"] == "您希望我对列出的 Skills 文件执行什么操作？"
    assert "options" not in first_question


def _credential_store(tmp_path: Path) -> CredentialStore:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    return store


def _qwen_coder_credential_store(tmp_path: Path) -> CredentialStore:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen-coder",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    return store


def _not_found_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404, request=request)))


def test_gitignore_protects_local_credentials_and_runtime_state() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "```" not in gitignore
    for pattern in (
        "config.json",
        "credentials/",
        ".webai-gateway/",
        ".codex-logs/",
        ".learnings/",
        ".pytest_cache/",
        "webui/node_modules/",
    ):
        assert pattern in gitignore


def _sse_events(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in (body or "").split("\n\n"):
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if not data_lines:
            continue
        raw = "\n".join(data_lines)
        item = json.loads(raw)
        item["_event"] = event_name
        events.append(item)
    return events


def _openai_sse_payloads(body: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for block in (body or "").split("\n\n"):
        data_lines = [
            line[len("data:") :].strip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        raw = "\n".join(data_lines)
        if raw == "[DONE]":
            continue
        payloads.append(json.loads(raw))
    return payloads


def test_models_returns_configured_model_when_upstream_models_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    client = _client(handler)

    response = client.get("/v1/models", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "web-model"
    model_ids = {item["id"] for item in body["data"]}
    assert "deepseek-v4-pro" in model_ids
    assert "deepseek-v4-pro[1m]" not in model_ids
    assert "deepseek-web/deepseek-chat" not in model_ids
    assert "deepseek" not in model_ids
    assert "gpt-instant" in model_ids
    assert "gemini-3-pro" in model_ids
    assert "qwen-web/qwen3.6-max-preview" in model_ids
    assert "qwen-web/qwen3.6-plus" in model_ids
    assert "qwen-web/qwen3-max" in model_ids
    assert "qwen-web/qwen3.6-max" not in model_ids
    qwen_preview = next(item for item in body["data"] if item["id"] == "qwen-web/qwen3.6-max-preview")
    assert qwen_preview["capabilities"]["tool_bridge"] is True
    assert qwen_preview["capabilities"]["supports_native_tools"] is False
    assert "seed" in model_ids
    assert "sora-2" in model_ids


def test_deepseek_web_catalog_prefers_v4_pro_models() -> None:
    deepseek = PROVIDERS["deepseek-web"]

    assert deepseek.models == ("deepseek-v4-pro",)


def test_deepseek_web_provider_payload_exposes_verified_ds2api_model(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "deepseek-web",
        {
            "cookie": "ds_session_id=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )

    deepseek = next(item for item in provider_payload(store)["providers"] if item["id"] == "deepseek-web")

    assert "deepseek-v4-pro" in deepseek["models"]
    assert "deepseek-v4-pro[1m]" not in deepseek["models"]
    assert deepseek["availableModels"] == ["deepseek-v4-pro"]
    assert deepseek["modelCount"] == 1
    assert deepseek["advertiseModels"] is True
    assert "ds2api" in deepseek["availabilityMessage"]
    assert "bearer-secret" not in json.dumps(deepseek, ensure_ascii=False)


def test_deepseek_web_routes_official_v4_pro_alias() -> None:
    assert is_deepseek_web_model("deepseek-v4-pro[1m]") is True
    assert normalize_deepseek_model("deepseek-v4-pro[1m]") == "deepseek-v4-pro"
    assert normalize_deepseek_model("deepseek-web/deepseek-v4-pro[1m]") == "deepseek-v4-pro"


def test_deepseek_web_client_forwards_to_ds2api_with_saved_bearer() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "ds2api-test",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
            },
            request=request,
        )

    deepseek = DeepSeekWebClient(
        {"bearer": "bearer-secret", "cookie": "ds_session_id=session-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ds2api_base_url="http://ds2api.test/v1",
    )

    response = deepseek.chat_completions(
        {
            "model": "deepseek-v4-pro[1m]",
            "messages": [{"role": "user", "content": "只回?OK"}],
            "stream": True,
            "_webai_native_web_search": True,
        }
    )

    assert seen["path"] == "/v1/chat/completions"
    assert seen["authorization"] == "Bearer bearer-secret"
    assert seen["body"]["model"] == "deepseek-v4-pro"
    assert seen["body"]["stream"] is False
    assert "_webai_native_web_search" not in seen["body"]
    assert response["choices"][0]["message"]["content"] == "OK"


def test_requires_bearer_api_key() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    response = client.get("/v1/models")

    assert response.status_code == 401


def test_chat_rejects_invalid_json_with_400() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    response = client.post(
        "/v1/chat/completions",
        headers={**_headers(), "Content-Type": "application/json"},
        content="{bad json",
    )

    assert response.status_code == 400
    assert "Request body must be valid JSON" in response.json()["detail"]


def test_forwards_normal_non_streaming_chat_without_tool_prompt() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("plain reply"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "web-model", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "plain reply"
    assert seen["path"] == "/v1/chat/completions"
    assert "tools" not in seen["body"]
    assert seen["body"]["messages"][0]["role"] == "system"
    assert "WebAI Gateway response language policy" in seen["body"]["messages"][0]["content"]
    assert "默认使用简体中文" in seen["body"]["messages"][0]["content"]
    assert seen["body"]["messages"][1:] == [{"role": "user", "content": "hello"}]


def test_preserves_webai2api_catalog_model_when_forwarding() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("webai2api reply"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "gpt-instant", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert seen["body"]["model"] == "gpt-instant"


def test_injects_tools_and_returns_openai_tool_calls() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = (
            '<|DSML|tool_calls><|DSML|invoke name="read_file">'
            '<|DSML|parameter name="path"><![CDATA[README.md]]></|DSML|parameter>'
            '</|DSML|invoke></|DSML|tool_calls>'
        )
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a local file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "tools" not in seen["body"]
    assert "tool_choice" not in seen["body"]
    system_messages = [m["content"] for m in seen["body"]["messages"] if m["role"] == "system"]
    assert system_messages
    assert "WebAI Gateway response language policy" in system_messages[0]
    assert "默认使用简体中文" in system_messages[0]
    assert "<|DSML|tool_calls>" in system_messages[0]
    assert '"name": "read_file"' in system_messages[0]
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "read_file"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}


def test_auto_tool_profile_preserves_explicit_required_tool_choice() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = (
            '<|DSML|tool_calls><|DSML|invoke name="read_file">'
            '<|DSML|parameter name="path"><![CDATA[README.md]]></|DSML|parameter>'
            '</|DSML|invoke></|DSML|tool_calls>'
        )
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a local file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "tools" not in seen["body"]
    assert "tool_choice" not in seen["body"]
    prompt = "\n".join(str(m["content"]) for m in seen["body"]["messages"] if m["role"] == "system")
    assert "<|DSML|tool_calls>" in prompt
    assert '"name": "read_file"' in prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "read_file"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}


def test_strict_tool_bridge_accepts_calls_array_and_strips_content() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = '```tool_json\n{"calls":[{"id":"call_read","name":"read_file","input":{"path":"README.md"}}]}\n```'
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a local file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "tools" not in seen["body"]
    assert "tool_choice" not in seen["body"]
    prompt = "\n".join(str(m["content"]) for m in seen["body"]["messages"] if m["role"] == "system")
    assert "<|DSML|tool_calls>" in prompt
    assert '<|DSML|invoke name="tool_name">' in prompt
    assert "no natural language outside it" in prompt
    assert "Never omit the opening <|DSML|tool_calls> tag" in prompt
    assert "Wrong 1 - mixed text after DSML" in prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["id"] == "call_read"
    assert tool_call["function"]["name"] == "read_file"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}


def test_prompt_tool_bridge_hides_runtime_tools_for_any_client() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = '```tool_json\n{"name":"fetch_url","args":{"url":"https://github.com/Einsia/OpenChronicle"}}\n```'
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "inspect a GitHub project"}],
            "tool_choice": "required",
            "tools": [
                {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert '"name": "fetch_url"' in prompt
    assert '"name": "terminal"' not in prompt
    assert '"name": "write_file"' not in prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "fetch_url"


def test_strict_tool_prompt_guides_followup_and_machine_readable_urls() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("plain reply"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "inspect a GitHub project"}],
            "tool_choice": "required",
            "tools": [
                {"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert "After a Tool result message" in prompt
    assert "machine-readable endpoints" in prompt
    assert "https://api.github.com/repos/<owner>/<repo>" in prompt
    assert "raw.githubusercontent.com" in prompt


def test_strict_tool_prompt_forbids_ask_user_for_tool_permission() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("plain reply"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "修改项目文件并运行必要命令"}],
            "tools": [
                {"type": "function", "function": {"name": "AskUserQuestion", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Bash", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Write", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert "Do not call AskUserQuestion to request permission" in prompt
    assert "Do not call AskUserQuestion for optional next-step or scope selection" in prompt
    assert "Request the listed tool directly" in prompt


def test_strict_tool_bridge_repairs_malformed_tool_json_once() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, json=_openai_response('```tool_json\n{"calls":[{"name":"read_file","input":\n```'), request=request)
        content = '```tool_json\n{"calls":[{"name":"read_file","input":{"path":"README.md"}}]}\n```'
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    repair_text = "\n".join(str(m.get("content", "")) for m in requests[1]["messages"])
    assert "Previous tool call was invalid" in repair_text
    assert "<|DSML|tool_calls>" in repair_text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read_file"


def test_strict_tool_bridge_repairs_allowed_tool_denial_text() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_openai_response(
                    "Tool Glob does not exists.Tool Bash does not exist.Tool Read does not exist."
                    "I cannot directly access the filesystem or execute tools in this environment."
                ),
                request=request,
            )
        return httpx.Response(
            200,
            json=_openai_response('```tool_json\n{"calls":[{"id":"toolu_read","name":"Read","input":{"file_path":"README.md"}}]}\n```'),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages?beta=true",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "/init"}],
            "tools": [
                {"name": "Glob", "input_schema": {"type": "object"}},
                {"name": "Bash", "input_schema": {"type": "object"}},
                {"name": "Read", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    repair_text = "\n".join(str(m.get("content", "")) for m in requests[1]["messages"])
    assert "The listed tools are available through the downstream client" in repair_text
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [{"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {"file_path": "README.md"}}]


def test_openai_tool_controller_exposes_bash_like_ds2api_for_local_agent_task() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Bash","input":{"command":"cd E:/ProjectX/mindcraft && git log --oneline -5"}}]}'
                "\n```"
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "Review local project code and make a detailed improvement plan."}],
            "tools": [
                {"type": "function", "function": {"name": "AskUserQuestion", "parameters": {"type": "object"}}},
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Run shell commands",
                        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                    },
                },
                {"type": "function", "function": {"name": "Glob", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 1
    prompt = "\n".join(str(message.get("content", "")) for message in requests[0]["messages"] if message.get("role") == "system")
    assert '"name": "Bash"' in prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "Bash"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "cd E:/ProjectX/mindcraft && git log --oneline -5"}
    assert "x-webai-tool-bridge-error" not in response.headers


def test_tool_bridge_accepts_mutating_bash_call_like_ds2api_when_declared() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "Implement the requested local project fix."}],
    )

    result = parse_tool_response(
        (
            '<|DSML|tool_calls><|DSML|invoke name="Bash">'
            '<|DSML|parameter name="command"><![CDATA[pip install pytest pytest-cov && pytest]]></|DSML|parameter>'
            '</|DSML|invoke></|DSML|tool_calls>'
        ),
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id.startswith("call_")
    assert result.tool_calls[0].name == "Bash"
    assert result.tool_calls[0].input == {"command": "pip install pytest pytest-cov && pytest"}


def test_openai_tool_controller_recovers_repeated_off_task_scope_question() -> None:
    requests: list[dict[str, Any]] = []
    off_task_question = (
        '```tool_json\n'
        '{"calls":[{"id":"call_question","name":"AskUserQuestion","input":{"questions":[{"header":"Scope",'
        '"question":"May I migrate project modules into src and rewrite module boundaries?",'
        '"options":[{"label":"Allow","description":"Migrate modules"},{"label":"Skip","description":"Only provide the plan"}]}]}}]}'
        "\n```"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        if len(requests) <= 2:
            return httpx.Response(200, json=_openai_response(off_task_question), request=request)
        return httpx.Response(
            200,
            json=_openai_response("Detailed improvement plan: keep the current structure, add focused tests, and stage changes by risk."),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {"name": "Read", "arguments": "{\"file_path\":\"README.md\"}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_read", "content": "Project overview."},
            ],
            "tools": [
                {"type": "function", "function": {"name": "AskUserQuestion", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Glob", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Edit", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Write", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 3
    recovery_prompt = "\n".join(str(message.get("content", "")) for message in requests[2]["messages"])
    assert "off_task_scope_escalation_question" in recovery_prompt
    assert "Do not call AskUserQuestion" in recovery_prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "Detailed improvement plan" in choice["message"]["content"]
    assert "x-webai-tool-bridge-error" not in response.headers


def test_openai_tool_controller_recovers_optional_scope_question() -> None:
    requests: list[dict[str, Any]] = []
    optional_question = (
        '```tool_json\n'
        '{"calls":[{"id":"call_question","name":"AskUserQuestion","input":{"questions":[{"header":"Focus",'
        '"question":"Improvement plan focus?",'
        '"options":[{"label":"Architecture/Code quality","description":"Refactor, patterns, testing"},'
        '{"label":"All of above","description":"Comprehensive roadmap"}]}]}}]}'
        "\n```"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        if len(requests) == 1:
            return httpx.Response(200, json=_openai_response(optional_question), request=request)
        return httpx.Response(
            200,
            json=_openai_response("Detailed improvement plan: cover architecture, quality, reliability, and tests."),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {"name": "Read", "arguments": "{\"file_path\":\"README.md\"}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_read", "content": "Project overview."},
            ],
            "tools": [
                {"type": "function", "function": {"name": "AskUserQuestion", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Glob", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    repair_prompt = "\n".join(str(message.get("content", "")) for message in requests[1]["messages"])
    assert "optional_scope_question_without_need" in repair_prompt
    assert "Do not call AskUserQuestion" in repair_prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "Detailed improvement plan" in choice["message"]["content"]
    assert "x-webai-tool-bridge-error" not in response.headers


def test_openai_tool_controller_recovers_repeated_ask_user_question() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_openai_response(
                    '```tool_json\n'
                    '{"calls":[{"id":"call_2","name":"AskUserQuestion","input":{"questions":[{"header":"Secret management",'
                    '"question":"How does this project handle API key and database password secrets?",'
                    '"options":[{"label":"Environment variables","description":"Use injected env vars"}]}]}}]}'
                    "\n```"
                ),
                request=request,
            )
        return httpx.Response(
            200,
            json=_openai_response("Implementation summary: use the existing environment-variable answer and continue with the plan."),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "AskUserQuestion",
                                "arguments": (
                                    '{"questions":[{"header":"Secret management",'
                                    '"question":"How does this project handle API keys and database passwords?",'
                                    '"options":[{"label":"Environment variables","description":"Use injected env vars"}]}]}'
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "User answered: use environment variables without .env files.",
                },
            ],
            "tools": [
                {"type": "function", "function": {"name": "AskUserQuestion", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Glob", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in requests[1]["messages"])
    assert "repeat_same_ask_user_without_progress" in retry_prompt
    assert "Do not call AskUserQuestion again" in retry_prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "environment-variable answer" in choice["message"]["content"]
    assert "x-webai-tool-bridge-error" not in response.headers


def test_openai_tool_controller_retries_final_without_review_evidence() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        if len(requests) == 1:
            return httpx.Response(200, json=_openai_response("Review complete. No issues found."), request=request)
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Glob","input":{"path":".","pattern":"*.py"}}]}\n```'
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "Review this local project."}],
            "tools": [{"type": "function", "function": {"name": "Glob", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in requests[1]["messages"])
    assert "insufficient_final_evidence" in retry_prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "Glob"


def test_openai_tool_controller_recovers_repeated_discovery_after_repair_retry() -> None:
    requests: list[dict[str, Any]] = []
    repeated_glob = (
        '```tool_json\n'
        '{"calls":[{"id":"call_repeat","name":"Glob","input":{"path":"E:/ProjectX/mindcraft","pattern":"*.py"}}]}'
        "\n```"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        if len(requests) <= 2:
            return httpx.Response(200, json=_openai_response(repeated_glob), request=request)
        return httpx.Response(
            200,
            json=_openai_response("Review summary: use the earlier Glob result and focus on the listed Python files."),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_glob",
                            "type": "function",
                            "function": {
                                "name": "Glob",
                                "arguments": "{\"path\":\"E:/ProjectX/mindcraft\",\"pattern\":\"*.py\"}",
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_glob", "content": "rewrite_content.py\nsync_to_feishu.py"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "Glob", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 3
    recovery_prompt = "\n".join(str(message.get("content", "")) for message in requests[2]["messages"])
    assert "repeat_discovery_call_without_progress" in recovery_prompt
    assert "Do not repeat" in recovery_prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "earlier Glob result" in choice["message"]["content"]
    assert "x-webai-tool-bridge-error" not in response.headers


def test_openai_tool_controller_retry_budget_exhaustion_does_not_leak_dsml() -> None:
    requests: list[dict[str, Any]] = []
    repeated_glob = (
        "<|DSML|tool_calls>\n"
        '  <|DSML|invoke name="Glob">\n'
        '    <|DSML|parameter name="path"><![CDATA[E:/ProjectX/mindcraft]]></|DSML|parameter>\n'
        '    <|DSML|parameter name="pattern"><![CDATA[*.py]]></|DSML|parameter>\n'
        "  </|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=_openai_response(repeated_glob), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_glob",
                            "type": "function",
                            "function": {
                                "name": "Glob",
                                "arguments": "{\"path\":\"E:/ProjectX/mindcraft\",\"pattern\":\"*.py\"}",
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_glob", "content": "rewrite_content.py\nsync_to_feishu.py"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "Glob", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 3
    assert response.headers["x-webai-tool-bridge-error"] == "repeat_discovery_call_without_progress"
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert "<|DSML|tool_calls>" not in choice["message"]["content"]
    assert "repeat_discovery_call_without_progress" in choice["message"]["content"]
    assert "Glob" in choice["message"]["content"]


def test_anthropic_tool_controller_retry_budget_exhaustion_does_not_leak_dsml() -> None:
    requests: list[dict[str, Any]] = []
    repeated_glob = (
        "<|DSML|tool_calls>\n"
        '  <|DSML|invoke name="Glob">\n'
        '    <|DSML|parameter name="path"><![CDATA[E:/ProjectX/mindcraft]]></|DSML|parameter>\n'
        '    <|DSML|parameter name="pattern"><![CDATA[*.py]]></|DSML|parameter>\n'
        "  </|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=_openai_response(repeated_glob), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages?beta=true",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_glob",
                            "name": "Glob",
                            "input": {"path": "E:/ProjectX/mindcraft", "pattern": "*.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_glob",
                            "content": "rewrite_content.py\nsync_to_feishu.py",
                        }
                    ],
                },
            ],
            "tools": [
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(requests) == 3
    assert response.headers["x-webai-tool-bridge-error"] == "repeat_discovery_call_without_progress"
    body = response.json()
    assert body["stop_reason"] == "end_turn"
    assert body["content"][0]["type"] == "text"
    assert "<|DSML|tool_calls>" not in body["content"][0]["text"]
    assert "repeat_discovery_call_without_progress" in body["content"][0]["text"]
    assert "Glob" in body["content"][0]["text"]


def test_openai_tool_controller_retries_status_only_final_after_tool_result() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_openai_response("Bash completed with no output. Project dir exist."),
                request=request,
            )
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n{"calls":[{"id":"call_read_again","name":"Read","input":{"file_path":"README.md"}}]}\n```'
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "Review this local project and give a detailed improvement plan."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_bash",
                            "type": "function",
                            "function": {"name": "Bash", "arguments": "{\"command\":\"ls -la\"}"},
                        },
                        {
                            "id": "call_glob",
                            "type": "function",
                            "function": {"name": "Glob", "arguments": "{\"pattern\":\"**/*.py\"}"},
                        },
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {"name": "Read", "arguments": "{\"file_path\":\"README.md\"}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_bash", "content": "total 12"},
                {"role": "tool", "tool_call_id": "call_glob", "content": "README.md\napp.py"},
                {"role": "tool", "tool_call_id": "call_read", "content": "project readme"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "Bash", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Glob", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in requests[1]["messages"])
    assert "status_only_final_without_task_answer" in retry_prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "Read"


def test_openai_tool_controller_retries_no_task_final_after_tool_result() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_openai_response("Caveman mode active. No task given. Wait for input."),
                request=request,
            )
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n{"calls":[{"id":"call_read_again","name":"Read","input":{"file_path":"README.md"}}]}\n```'
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "Review this local project and give a detailed improvement plan."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_glob",
                            "type": "function",
                            "function": {"name": "Glob", "arguments": "{\"pattern\":\"**/*.py\"}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_glob",
                    "content": "rewrite_content.py\ncreate_feishu_view.py",
                },
            ],
            "tools": [
                {"type": "function", "function": {"name": "Glob", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in requests[1]["messages"])
    assert "status_only_final_without_task_answer" in retry_prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "Read"


def test_semantic_final_judge_shadow_records_metadata_without_enforcing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{"file_path":"README.md"}}]}\n```'
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="all", semantic_final_judge="shadow"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "Read local README.md"}],
            "tools": [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    diagnostics = client.get("/api/admin/request-diagnostics").json()["events"]
    event = diagnostics[-1]
    assert event["semanticFinalJudgeMode"] == "shadow"
    assert event["semanticFinalJudgeVerdict"] == "allow"
    assert isinstance(event["semanticFinalJudgeConfidence"], float)


def test_parse_chat_response_flags_no_bridge_local_tool_denial_when_context_has_tools() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read files",
                    "parameters": {"type": "object"},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )

    parsed, result = parse_chat_response(
        _openai_response("I cannot directly access the project files or codebase to perform a detailed review."),
        bridge=False,
        allowed_tools=set(),
        model="qwen-coder/qwen-coder-plus",
        bridge_context=context,
        return_bridge_result=True,
    )

    assert parsed["choices"][0]["message"]["content"]
    assert result.error is not None
    assert result.error.kind == "tool_denial_without_call"
    assert result.error.repairable is True


def test_strict_tool_bridge_repairs_chinese_permission_denial_text() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_openai_response(
                    "I cannot directly update the project files or run Git commands. "
                    "The environment prevents direct filesystem access and shell execution."
                ),
                request=request,
            )
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"toolu_write","name":"Write",'
                '"input":{"file_path":"SECURITY.md","content":"# Security hardening\\n"}}]}'
                "\n```"
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages?beta=true",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "更新项目并提?"}],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Write", "description": "Write files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    repair_text = "\n".join(str(m.get("content", "")) for m in requests[1]["messages"])
    assert "The listed tools are available through the downstream client" in repair_text
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_write",
            "name": "Write",
            "input": {"file_path": "SECURITY.md", "content": "# Security hardening\n"},
        }
    ]


def test_strict_tool_bridge_repairs_simulated_tool_environment_denial() -> None:
    requests: list[dict[str, Any]] = []
    denial = (
        "# Understanding the Current Situation\n"
        "The prompt appears to be simulating a specialized security hardening environment called Claude Code. "
        "These indicate that the simulated environment is trying to use specialized tools like using-superpowers "
        "that aren't available here. Since we don't have access to the simulated tool environment, "
        "I can still help with step-by-step implementation manually."
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, json=_openai_response(denial), request=request)
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"toolu_write","name":"Write",'
                '"input":{"file_path":"SECURITY.md","content":"# Security hardening\\n"}}]}'
                "\n```"
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages?beta=true",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "继续落地安全加固"}],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Write", "description": "Write files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    repair_text = "\n".join(str(m.get("content", "")) for m in requests[1]["messages"])
    assert "The listed tools are available through the downstream client" in repair_text
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_write",
            "name": "Write",
            "input": {"file_path": "SECURITY.md", "content": "# Security hardening\n"},
        }
    ]


def test_strict_tool_bridge_reports_repair_failure_without_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_response('```tool_json\n{"calls":[{"name":"missing_tool","input":{}}]}\n```'), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    assert response.headers["x-webai-tool-bridge-error"] == "unknown_tool"
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert "unknown_tool" in choice["message"]["content"]
    assert "missing_tool" in choice["message"]["content"]
    assert "read_file" in choice["message"]["content"]
    assert choice["message"]["webai_tool_bridge"]["error"] == "unknown_tool"


def test_tool_bridge_rejection_text_distinguishes_policy_denial_for_allowed_tool() -> None:
    result = BridgeResult(
        content="",
        tool_calls=[],
        raw_content=(
            "```tool_json\n"
            '{"calls":[{"id":"call_1","name":"AskUserQuestion","input":{"questions":[{"question":"允许重构目录结构吗？"}]}}]}'
            "\n```"
        ),
        error=BridgeError(
            "off_task_scope_escalation_question",
            "The model asked the user to authorize broad restructuring that the current task did not request.",
            repairable=True,
        ),
    )

    text = tool_bridge_rejected_response_text(result, {"AskUserQuestion", "Read"})

    assert "已注册工具" in text
    assert "工具使用策略" in text
    assert "模型请求的工具：AskUserQuestion" in text
    assert "当前允许工具：AskUserQuestion, Read" in text
    assert "未注册工具" not in text
    assert "不会绕过任务边界策略" in text


def test_tool_bridge_rejection_text_keeps_unregistered_tool_guidance_for_unknown_tool() -> None:
    result = BridgeResult(
        content="",
        tool_calls=[],
        raw_content='```tool_json\n{"calls":[{"id":"call_1","name":"Task","input":{}}]}\n```',
        error=BridgeError("unknown_tool", "未知工具：Task", repairable=True),
    )

    text = tool_bridge_rejected_response_text(result, {"Read"})

    assert "未允许、未注册或格式无效" in text
    assert "模型请求的工具：Task" in text
    assert "Gateway 不会执行未注册工具" in text


def test_tool_bridge_policy_denial_does_not_retry_upstream() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git checkout -b security/hotfix-secrets-rotation"}}]}\n```'
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "审查代码并制定改进计?"}],
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(requests) == 1
    assert response.headers["x-webai-tool-bridge-error"] == "unsafe_review_shell_command"
    body = response.json()
    assert body["stop_reason"] == "end_turn"
    assert body["content"][0]["type"] == "text"
    assert "unsafe_review_shell_command" in body["content"][0]["text"]


def test_openai_upstream_tool_bridge_runs_unknown_tool_recovery_after_repair() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        if len(requests) <= 2:
            return httpx.Response(
                200,
                json=_openai_response(
                    '```tool_json\n{"calls":[{"id":"call_1","name":"Task","input":{"description":"inspect"}}]}\n```'
                ),
                request=request,
            )
        return httpx.Response(200, json=_openai_response("Recovered with a final answer."), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "inspect repository"}],
            "tools": [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 3
    recovery_prompt = "\n".join(str(message.get("content", "")) for message in requests[2]["messages"])
    assert "不要再请求未列出的工具" in recovery_prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Recovered with a final answer."
    assert "tool_calls" not in choice["message"]


def test_converts_tool_messages_for_web_upstream() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("continued"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "file text"},
            ],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    assert all(m["role"] != "tool" for m in seen["body"]["messages"])
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "Tool result for read_file" in joined
    assert "call_1" in joined
    assert "file text" in joined


def test_converts_failed_tool_result_as_error_observation() -> None:
    seen: dict[str, Any] = {}
    failed_result = json.dumps(
        {
            "url": "https://raw.githubusercontent.com/Einsia/OpenChronicle/main/README.md",
            "error": "network_error",
            "message": "Request failed after 3 attempts",
        },
        ensure_ascii=False,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("continued"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "fetch_url", "arguments": "{\"url\":\"https://raw.githubusercontent.com/Einsia/OpenChronicle/main/README.md\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "name": "fetch_url", "content": failed_result},
            ],
            "tools": [{"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "Tool result for fetch_url" in joined
    assert "is_error: true" in joined
    assert "is_error: false" not in joined
    assert "Request failed after 3 attempts" in joined
    assert "different allowed tool or input" in joined


def test_converts_tool_result_with_observation_compression() -> None:
    seen: dict[str, Any] = {}
    long_result = "A" * 7000

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("continued"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": long_result},
            ],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "Tool result was too long and has been compressed" in joined
    assert "Original length" in joined
    assert len(joined) < 7600


def test_compress_observation_filters_dependency_path_lists() -> None:
    raw = "\n".join(
        [
            "webui\\node_modules\\.pnpm\\@vue+shared@3.5.25\\node_modules\\@vue\\shared\\README.md",
            "webui\\node_modules\\.pnpm\\nanoid@3.3.11\\node_modules\\nanoid\\README.md",
            "webui\\dist\\assets\\index.js",
            "README.md",
            "AGENTS.md",
            "requirements.txt",
        ]
    )

    compressed = compress_observation(raw, ToolBridgeConfig(observation_max_chars=1000))

    assert "dependency/build paths were omitted" in compressed
    assert "node_modules" not in compressed
    assert ".pnpm" not in compressed
    assert "webui\\dist" not in compressed
    assert "README.md" in compressed
    assert "AGENTS.md" in compressed
    assert "requirements.txt" in compressed


def test_compress_observation_reports_when_only_dependency_paths_match() -> None:
    raw = "\n".join(
        [
            "webui\\node_modules\\.pnpm\\@vue+shared@3.5.25\\node_modules\\@vue\\shared\\README.md",
            "webui\\node_modules\\.pnpm\\nanoid@3.3.11\\node_modules\\nanoid\\README.md",
        ]
    )

    compressed = compress_observation(raw, ToolBridgeConfig(observation_max_chars=1000))

    assert "Only dependency/build/cache paths were returned" in compressed
    assert "request a narrower pattern" in compressed
    assert "node_modules" not in compressed


def test_compress_observation_uses_configured_excluded_path_parts() -> None:
    raw = "\n".join(
        [
            "third_party\\sdk\\README.md",
            "vendor_cache\\generated\\notes.md",
            "README.md",
            "docs\\usage.md",
        ]
    )
    config = ToolBridgeConfig(
        observation_max_chars=1000,
        observation_policy=ObservationPolicyConfig(excluded_path_parts=("third_party", "vendor_cache")),
    )

    compressed = compress_observation(raw, config)

    assert "dependency/build paths were omitted" in compressed
    assert "third_party" not in compressed
    assert "vendor_cache" not in compressed
    assert "README.md" in compressed
    assert "docs\\usage.md" in compressed


def test_compress_observation_uses_configured_excluded_path_globs() -> None:
    raw = "\n".join(
        [
            "src\\generated\\client.md",
            "packages\\app\\src\\generated\\schema.md",
            "src\\main.py",
            "tests\\test_gateway.py",
        ]
    )
    config = ToolBridgeConfig(
        observation_max_chars=1000,
        observation_policy=ObservationPolicyConfig(excluded_path_globs=("**/generated/**",)),
    )

    compressed = compress_observation(raw, config)

    assert "dependency/build paths were omitted" in compressed
    assert "generated" not in compressed
    assert "src\\main.py" in compressed
    assert "tests\\test_gateway.py" in compressed


def test_tool_bridge_rewrites_repository_wide_glob_to_shallow_pattern() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all", max_calls_per_turn=4),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Glob","input":{"pattern":"**/*.md"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Glob", {"pattern": "*.md"})
    ]


def test_tool_bridge_rewrites_repository_wide_all_files_glob_to_directory_list() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "LS",
                    "description": "List a directory.",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "Update repository E:\\ProjectX\\mindcraft\\MediaCrawler from origin/main"}],
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Glob","input":{"pattern":"**/*"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "LS", {"path": "."})
    ]


def test_tool_bridge_uses_command_args_instead_of_superpowers_skill_body_for_task_anchor() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    command_args = "Implement the improvement plan in this local codebase."
    skill_body = (
        "Base directory for this skill: C:\\Users\\woody\\.codex\\plugins\\cache\\superpowers\\using-superpowers\n\n"
        "<EXTREMELY-IMPORTANT>\n"
        "Do not treat the skill body as the user's repository task.\n"
        "</EXTREMELY-IMPORTANT>\n\n"
        "# Using Skills\n"
        + ("large skill instruction block\n" * 300)
        + f"\nARGUMENTS: {command_args}"
    )

    refined = prefer_local_tools_for_local_agent_task(
        context,
        [
            {
                "role": "user",
                "content": (
                    "<command-message>using-superpowers</command-message>\n"
                    "<command-name>/using-superpowers</command-name>\n"
                    f"<command-args>{command_args}</command-args>"
                ),
            },
            {"role": "user", "content": [{"type": "text", "text": skill_body}]},
        ],
    )

    assert refined.task_text == command_args
    assert len(refined.task_text) < 100
    assert "EXTREMELY-IMPORTANT" not in refined.task_text


def test_tool_bridge_normalizes_glob_path_with_embedded_pattern() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"list_py_files","name":"Glob","input":{"path":"E:\\\\ProjectX\\\\mindcraft\\\\**\\\\*.py","max_depth":5}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("list_py_files", "Glob", {"path": "E:\\ProjectX\\mindcraft", "pattern": "**/*.py"})
    ]


def test_tool_bridge_maps_unregistered_ls_to_allowed_glob() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"LS","input":{"path":"."}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Glob", {"path": ".", "pattern": "*"})
    ]


def test_tool_bridge_maps_unregistered_list_to_allowed_glob() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"List","input":{"path":"."}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Glob", {"path": ".", "pattern": "*"})
    ]


def test_tool_bridge_maps_read_directory_path_to_glob_listing() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Glob", {"path": "E:/ProjectX/mindcraft", "pattern": "*"})
    ]


def test_tool_bridge_keeps_read_for_common_extensionless_file_names() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft/README"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Read", {"file_path": "E:/ProjectX/mindcraft/README"})
    ]


def test_write_after_failed_read_error_lists_only_available_discovery_tools() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "pattern": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Write",
                    "description": "Write files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["file_path", "content"],
                    },
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "/using-superpowers 按计划落地改?"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "tests/test_integration.py"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "is_error": True,
                        "content": "File does not exist.",
                    }
                ],
            },
        ],
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_2","name":"Write","input":{"file_path":"tests/test_integration.py","content":"# guessed"}}]}\n```',
        context,
    )

    assert result.error is not None
    assert result.error.kind == "write_after_failed_read_without_discovery"
    assert "Glob or Read" in result.error.message
    assert "LS" not in result.error.message


def test_tool_refusal_recovery_prompt_lists_only_allowed_discovery_tools() -> None:
    bridge_result = BridgeResult(
        content="",
        tool_calls=[],
        error=BridgeError(
            "write_after_failed_read_without_discovery",
            "The model tried to create a missing path. Use Glob/Grep/LS or Read an existing project file.",
            repairable=True,
        ),
        raw_content='```tool_json\n{"calls":[{"id":"call_1","name":"Write","input":{"file_path":"tests/test_integration.py"}}]}\n```',
    )

    recovered = build_tool_refusal_recovery_payload(
        {"messages": [{"role": "user", "content": "继续落地改进"}]},
        bridge_result,
        allowed_tools={"Glob", "Read"},
    )
    prompt = recovered["messages"][-1]["content"]

    assert "Use Glob or Read" in prompt
    assert "LS" not in prompt


def test_tool_bridge_rewrites_absolute_repository_wide_glob_pattern_to_shallow_pattern() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Glob","input":{"pattern":"E:/ProjectX/mindcraft/**/*"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Glob", {"path": "E:/ProjectX/mindcraft", "pattern": "*"})
    ]


def test_tool_bridge_maps_file_search_alias_to_glob() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"file_search","input":{"directory":"E:\\\\ProjectX\\\\mindcraft","pattern":"*.py,*.md,*.json"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Glob", {"path": "E:\\ProjectX\\mindcraft", "pattern": "**/*.{py,md,json}"})
    ]


def test_tool_bridge_normalizes_glob_patterns_array_to_pattern() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_1","name":"Glob","input":{"patterns":["docs/*.md","docs/*.txt"]}}]}'
        "\n```",
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Glob", {"pattern": "docs/{*.md,*.txt}"})
    ]


def test_tool_bridge_extracts_multiline_tool_summary_calls() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Agent",
                    "description": "Launch a subagent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["prompt", "description"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all", max_calls_per_turn=4),
    )

    result = parse_tool_response(
        '''
I see the issue - when launching Agent tools, I need to provide both prompt and description parameters.

Assistant requested tool calls:
- Agent({
"prompt": "搜索代码库中与制定完善的改进计划并落地相关的现有实现模式?",
"description": "搜索现有改进计划模板和文档模?"
})
- Agent({
"prompt": "分析代码库中与计划执行相关的核心组件?",
"description": "分析计划执行相关的核心组?"
})
''',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        (
            "toolu_summary_1",
            "Agent",
            {
                "prompt": "搜索代码库中与制定完善的改进计划并落地相关的现有实现模式?",
                "description": "搜索现有改进计划模板和文档模?",
            },
        ),
        (
            "toolu_summary_2",
            "Agent",
            {
                "prompt": "分析代码库中与计划执行相关的核心组件?",
                "description": "分析计划执行相关的核心组?",
            },
        ),
    ]


def test_tool_bridge_synthesizes_agent_description_from_prompt() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Agent",
                    "description": "Launch a subagent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["prompt", "description"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all", max_calls_per_turn=4),
    )

    result = parse_tool_response(
        '''
Assistant requested tool calls:
- Agent({
"prompt": "Analyze plan execution components.\\n1. Find workflow engine modules.\\n2. Identify validation mechanisms."
})
''',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        (
            "toolu_summary_1",
            "Agent",
            {
                "prompt": "Analyze plan execution components.\n1. Find workflow engine modules.\n2. Identify validation mechanisms.",
                "description": "Analyze plan execution components.",
            },
        )
    ]


def test_tool_bridge_rejects_echoed_tool_history_summary_without_tool_json() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "List files.",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read files.",
                    "parameters": {"type": "object"},
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_calls_per_turn=4),
    )

    result = parse_tool_response(
        '''
Assistant requested tool calls:
- Glob({"pattern":"*.py","path":"E:/ProjectX/mindcraft"})
User: Tool result for Glob (call id: toolu_call_3, is_error: false):
["E:/ProjectX/mindcraft/rewrite_content.py"]
Use this tool result to continue the task.
Assistant: Assistant requested tool calls:
- Read({"file_path":"E:/ProjectX/mindcraft/rewrite_content.py"})
''',
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "echoed_tool_history_without_tool_json"
    assert result.error.repairable is True


def test_direct_provider_sanitizes_expensive_glob_without_retry(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class BroadGlobClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Glob","input":{"pattern":"**/*.md"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=BroadGlobClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "/init"}],
            "tools": [
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 1
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_call_1", "name": "Glob", "input": {"pattern": "*.md"}}
    ]


def test_anthropic_messages_returns_tool_use_block() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = '```tool_json\n{"calls":[{"id":"toolu_read","name":"read_file","input":{"path":"README.md"}}]}\n```'
        return httpx.Response(200, json=_openai_response(content), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "system": "You are Claude Code.",
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [{"name": "read_file", "description": "Read a local file", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert "tools" not in seen["body"]
    assert "tool_choice" not in seen["body"]
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [{"type": "tool_use", "id": "toolu_read", "name": "read_file", "input": {"path": "README.md"}}]


def test_anthropic_converter_recovers_omitted_tool_use_from_registry() -> None:
    openai_body = anthropic_body_to_openai(
        {
            "model": "web-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_read",
                            "content": (
                                "File unchanged since last read. The content from the earlier Read tool_result "
                                "in this conversation is still current - refer to that instead of re-reading."
                            ),
                        }
                    ],
                }
            ],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
        tool_call_registry={
            "toolu_read": {
                "id": "toolu_read",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": "{\"file_path\":\"E:/ProjectX/mindcraft/rewrite_content.py\"}",
                },
            }
        },
    )

    assert openai_body["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "toolu_read",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": "{\"file_path\":\"E:/ProjectX/mindcraft/rewrite_content.py\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_read",
            "name": "Read",
            "content": (
                "File unchanged since last read. The content from the earlier Read tool_result "
                "in this conversation is still current - refer to that instead of re-reading."
            ),
        },
    ]


def test_anthropic_messages_uses_registry_to_repair_repeated_read_when_tool_use_omitted() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        if len(requests) == 1:
            content = (
                "```tool_json\n"
                "{\"calls\":[{\"id\":\"toolu_read\",\"name\":\"Read\",\"input\":{\"file_path\":\"E:/ProjectX/mindcraft/rewrite_content.py\"}}]}"
                "\n```"
            )
        elif len(requests) == 2:
            content = (
                "```tool_json\n"
                "{\"calls\":[{\"id\":\"toolu_repeat\",\"name\":\"Read\",\"input\":{\"file_path\":\"E:/ProjectX/mindcraft/rewrite_content.py\"}}]}"
                "\n```"
            )
        else:
            content = "详细改进计划：基于已有读取结果继续审查，不再重复读取 unchanged 文件?"
        return httpx.Response(200, json=_openai_response(content), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="local-agent", tool_profile="auto"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    first = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "Review local project code and make a detailed improvement plan."}],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert first.status_code == 200
    assert first.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_read",
            "name": "Read",
            "input": {"file_path": "E:/ProjectX/mindcraft/rewrite_content.py"},
        }
    ]

    second = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_read",
                            "content": (
                                "File unchanged since last read. The content from the earlier Read tool_result "
                                "in this conversation is still current - refer to that instead of re-reading."
                            ),
                        }
                    ],
                },
            ],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert second.status_code == 200
    assert len(requests) == 3
    assert "repeat_unchanged_read_without_progress" in json.dumps(requests[2]["messages"], ensure_ascii=False)
    assert second.json()["content"] == [
        {"type": "text", "text": "详细改进计划：基于已有读取结果继续审查，不再重复读取 unchanged 文件?"}
    ]


def test_anthropic_messages_registry_tracks_toolu_alias_for_idless_ds2api_call() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        if len(requests) == 1:
            content = (
                "<|DSML|tool_calls>"
                '<|DSML|invoke name="Bash">'
                '<|DSML|parameter name="command"><![CDATA[git -C "E:/ProjectX/mindcraft" status]]></|DSML|parameter>'
                "</|DSML|invoke>"
                "</|DSML|tool_calls>"
            )
        else:
            content = "done"
        return httpx.Response(200, json=_openai_response(content), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    first = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "Check repository status."}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert first.status_code == 200
    tool_use = first.json()["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["id"].startswith("toolu_call_")

    second = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "Check repository status."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "is_error": True,
                            "content": "fatal: not a git repository",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert second.status_code == 200
    joined = "\n".join(str(message["content"]) for message in requests[1]["messages"])
    assert "Assistant requested tool calls" in joined
    assert "Tool result for Bash" in joined
    assert 'git -C "E:/ProjectX/mindcraft" status' in joined


def test_anthropic_messages_converts_tool_result_to_web_observation() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("done"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_read", "name": "read_file", "input": {"path": "README.md"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_read", "content": "file text"}]},
            ],
            "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "Tool result for read_file" in joined
    assert "toolu_read" in joined
    assert "file text" in joined
    assert response.json()["content"] == [{"type": "text", "text": "done"}]


def test_anthropic_messages_preserves_tool_result_error_state() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("done"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_fetch", "name": "fetch_url", "input": {"url": "https://example.com"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_fetch", "is_error": True, "content": "No data returned"}]},
            ],
            "tools": [{"name": "fetch_url", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "Tool result for fetch_url" in joined
    assert "is_error: true" in joined
    assert "different allowed tool or input" in joined


def test_anthropic_tool_result_summarizes_non_text_blocks_without_raw_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("done"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_screen", "name": "screenshot", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_screen",
                            "content": [
                                {"type": "text", "text": "captured"},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
                            ],
                        }
                    ],
                },
            ],
            "tools": [{"name": "screenshot", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    joined = "\n".join(str(m["content"]) for m in seen["body"]["messages"])
    assert "captured" in joined
    assert "[image: media_type=image/png, source=base64, data_length=8]" in joined
    assert "aGVsbG8=" not in joined
    assert "{'type':" not in joined


def test_anthropic_messages_streams_text_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_response("plain reply"), request=request)

    client = _client(handler)

    with client.stream(
        "POST",
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
            "stream": True,
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = _sse_events(body)
    assert [event["_event"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2]["delta"] == {"type": "text_delta", "text": "plain reply"}
    assert events[4]["delta"]["stop_reason"] == "end_turn"


def test_anthropic_messages_accepts_x_api_key_and_passes_common_fields() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("ok"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "local-dev-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "system": [{"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}],
            "max_tokens": 128,
            "stop_sequences": ["</stop>"],
            "top_p": 0.9,
            "metadata": {"user_id": "local-user"},
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    )

    assert response.status_code == 200
    assert seen["body"]["stop"] == ["</stop>"]
    assert seen["body"]["top_p"] == 0.9
    assert seen["body"]["metadata"] == {"user_id": "local-user"}
    assert "reasoning" not in seen["body"]
    assert "cache_control" not in json.dumps(seen["body"], ensure_ascii=False)


def test_anthropic_count_tokens_returns_estimate_for_claude_code() -> None:
    client = _client(lambda request: httpx.Response(500, request=request))

    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "local-dev-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "system": "You are Claude Code.",
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "README.md"}}],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file text"}]},
            ],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["input_tokens"] > 0


def test_tool_bridge_full_exposure_allows_runtime_and_mcp_tools() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n{"calls":[{"id":"toolu_bash","name":"Bash","input":{"command":"pwd"}},{"id":"toolu_mcp","name":"mcp__repo__search","input":{"query":"gateway"}}]}\n```'
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="all", max_calls_per_turn=4),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "run command and search mcp"}],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Edit", "description": "Edit files", "input_schema": {"type": "object"}},
                {"name": "mcp__repo__search", "description": "Search repository", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert '"name": "Bash"' in prompt
    assert '"name": "Edit"' in prompt
    assert '"name": "mcp__repo__search"' in prompt
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert [block["name"] for block in body["content"]] == ["Bash", "mcp__repo__search"]


def test_tool_bridge_rewrites_cli_tool_name_to_bash_when_bash_allowed() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"gh","input":{"command":"repo view Kriswd/web-gateway"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "gh repo view Kriswd/web-gateway"})
    ]


def test_tool_bridge_records_parse_phases_for_successful_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read files",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"read_file","input":{"path":"README.md"}}]}\n```',
        context,
    )

    phases = [(phase.name, phase.status, phase.detail) for phase in result.phases]
    assert ("extract", "ok", "candidate_count=1") in phases
    assert ("parse/normalize", "ok", "call_count=1") in phases
    assert ("validate", "ok", "call_count=1") in phases


def test_tool_bridge_rewrites_terminal_tool_name_to_bash_without_prefixing_wrapper() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"terminal","input":{"command":"git status --short"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "git status --short"})
    ]


def test_tool_bridge_normalizes_windows_paths_in_bash_command() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"cd E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler && git fetch origin && git status"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "cd E:/ProjectX/mindcraft/MediaCrawler && git fetch origin && git status"})
    ]


def test_tool_bridge_normalizes_windows_paths_in_exec_command() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Exec",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "Update repository E:\\ProjectX\\mindcraft\\MediaCrawler from origin/main"}],
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Exec","input":{"command":"cd E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler && git pull origin main"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Exec", {"command": "cd E:/ProjectX/mindcraft/MediaCrawler && git pull origin main"})
    ]


def test_tool_bridge_rejects_incomplete_git_clone_command() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git clone"}}]}\n```',
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "incomplete_shell_command"
    assert result.error.repairable is True


@pytest.mark.parametrize(
    ("command", "expected_kind", "expected_repairable"),
    [
        (
            "python-dotenv>=1.0.0 && pytest>=7.0.0 && pytest-asyncio>=0.21.0",
            "unsafe_local_shell_command",
            True,
        ),
        ("git clone <repository-url>", "unsafe_local_shell_command", True),
        ("pytest.fail('expected failure') && pytest.fail('second failure')", "unsafe_local_shell_command", True),
        ("python test_improvements.py && python sync_to_feishu.py", "unsafe_local_shell_command", True),
    ],
)
def test_tool_bridge_rejects_real_web_model_bad_shell_artifacts(
    command: str,
    expected_kind: str,
    expected_repairable: bool,
) -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "Implement the improvement plan in this local codebase."}],
    )
    payload = json.dumps(
        {"calls": [{"id": "call_1", "name": "Bash", "input": {"command": command}}]},
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == expected_kind
    assert result.error.repairable is expected_repairable


def test_tool_bridge_marks_package_requirement_bash_artifact_repairable() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "Implement the improvement plan in this local codebase."}],
    )

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_1","name":"Bash",'
        '"input":{"command":"python-dotenv>=1.0.0 && pytest>=7.0.0"}}]}'
        "\n```",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_local_shell_command"
    assert result.error.repairable is True


def test_tool_bridge_trims_trailing_shell_operator() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git status --short &&"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "git status --short"})
    ]


def test_tool_bridge_unescapes_html_shell_operators() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git status --short &amp;&amp; git log --oneline -n 5"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "git status --short && git log --oneline -n 5"})
    ]


def test_tool_bridge_collapses_duplicate_shell_operator() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git status --short && && git log --oneline -n 5"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "git status --short && git log --oneline -n 5"})
    ]


def test_tool_bridge_rejects_destructive_git_for_readonly_review_task() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "审查下当前项目的代码，看看有什么需要改进的?"}],
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git -C E:/ProjectX/mindcraft/MediaCrawler reset --hard origin/main && git -C E:/ProjectX/mindcraft/MediaCrawler clean -fdx"}}]}\n```',
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_review_shell_command"
    assert result.error.repairable is False


def test_tool_bridge_rejects_full_test_suite_for_readonly_review_task() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "review this codebase and suggest improvements"}],
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"pytest tests/"}}]}\n```',
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_review_shell_command"
    assert result.error.repairable is False


@pytest.mark.parametrize(
    "command",
    [
        "pip install pylint mypy ruff && ruff check api/ store/ config/ database/",
        "python -m venv .venv && pip install -r requirements.txt",
        "ruff check api/ config/ --fix",
    ],
)
def test_tool_bridge_rejects_slow_or_mutating_setup_for_readonly_review_task(command: str) -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "你审查下当前项目的代码，看看有什么需要改进的?"}],
    )
    payload = json.dumps(
        {"calls": [{"id": "call_1", "name": "Bash", "input": {"command": command}}]},
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_review_shell_command"
    assert result.error.repairable is False


@pytest.mark.parametrize(
    "command",
    [
        (
            'git grep -i "password\\|api_key\\|secret\\|token" -- "*.py" "*.md" "*.yaml" && '
            'git grep -i "input\\|request" -- "*.py" | grep -v "validate" && '
            'git add .gitignore && git rm -r --cached . && git add . && '
            'git commit -m "feat: 添加 .gitignore 并整理项目结?'
        ),
        (
            'git grep -i "input\\|request" -- "*.py" | grep -v "validate" && '
            "git filter-repo --force --invert-paths --path-glob '*.md' --path-glob '*.yaml'"
        ),
        (
            'git grep -l "API_KEY\\|SECRET\\|TOKEN" | xargs sed -i "s/=[^ ]*//g" && '
            "pip install pytest pytest-cov && "
            "git checkout -b security-fix/immediate-mitigation"
        ),
        'git grep -l "API_KEY\\|SECRET\\|TOKEN" | xargs sed -i "s/=[^ ]*//g"',
    ],
)
def test_tool_bridge_rejects_mutating_shell_pipeline_for_chinese_review_plan_task(command: str) -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "审查代码并制定改进计?"}],
    )
    payload = json.dumps(
        {"calls": [{"id": "call_1", "name": "Bash", "input": {"command": command}}]},
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_review_shell_command"
    assert result.error.repairable is False


def test_tool_bridge_allows_readonly_git_grep_for_chinese_review_plan_task() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "审查代码并制定改进计?"}],
    )
    command = 'git grep -i "password\\|api_key\\|secret\\|token" -- "*.py" "*.md" "*.yaml"'
    payload = json.dumps(
        {"calls": [{"id": "call_1", "name": "Bash", "input": {"command": command}}]},
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": command})
    ]


@pytest.mark.parametrize(
    "command",
    [
        "pip install pytest pytest-cov && pytest --cov=src",
        "python -m pytest --collect-only && python scripts/fix_windows_paths.py",
        "git checkout -b fix/test-collection && pytest --collect-only",
    ],
)
def test_tool_bridge_rejects_high_risk_shell_for_review_plan_landing_task(command: str) -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "审查代码并制定完整改进计划并落地"}],
    )
    payload = json.dumps(
        {"calls": [{"id": "call_1", "name": "Bash", "input": {"command": command}}]},
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_local_shell_command"
    assert result.error.repairable is True


def test_tool_bridge_rejects_shell_setup_for_code_edit_task_without_setup_request() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "description": "Edit files",
                    "parameters": {"type": "object"},
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "修复这个 Bug 并落?"}],
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"pip install pytest pytest-cov && pytest"}}]}\n```',
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_local_shell_command"
    assert result.error.repairable is True


def test_tool_bridge_uses_original_human_prompt_after_tool_result_for_review_policy() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "审查下代码并且制定改进计?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Bash",
                        "input": {"command": "git status --short"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "On branch security/fix-secrets-leak\nnothing to commit, working tree clean",
                    }
                ],
            },
        ],
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_2","name":"Bash","input":{"command":"git checkout -b security/hotfix-secrets-rotation"}}]}\n```',
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_review_shell_command"
    assert result.error.repairable is False


@pytest.mark.parametrize(
    "command",
    [
        "git checkout -b security/hotfix-secrets-rotation",
        "pip install pytest pytest-cov && pytest --cov=src --cov-report=term-missing",
        "python -c \"print('FEISHU_APP_ID=')\" > .env.example",
        "git secrets --register-aws",
        "python utils/rotate_secrets.py --all",
        "git add . && git commit -m \"feat: security hardening\"",
        "python skills/security-reviewer/audit.py --full-scan",
    ],
)
def test_tool_bridge_claude_code_review_policy_blocks_mutating_command_matrix(command: str) -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "审查下代码并且制定改进计?"}],
    )
    payload = json.dumps(
        {"calls": [{"id": "call_1", "name": "Bash", "input": {"command": command}}]},
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_review_shell_command"
    assert result.error.repairable is False


def test_tool_bridge_review_policy_uses_original_task_for_continue_turn() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "审查下代码并且制定改进计?"},
            {"role": "assistant", "content": "我先看一下项目结构?"},
            {"role": "user", "content": "继续"},
        ],
    )
    payload = json.dumps(
        {
            "calls": [
                {
                    "id": "call_1",
                    "name": "Bash",
                    "input": {
                        "command": "pip install pytest pytest-cov && git rm skills/feishu-syncer/test-url*.js"
                    },
                }
            ]
        },
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_review_shell_command"
    assert result.error.repairable is False


@pytest.mark.parametrize(
    "command",
    [
        "pytest --cov=. && pytest --cov=src --cov-report=html",
        "pip install -r requirements.txt --upgrade-strategy eager && pip install -e .",
        "git checkout -b fix/test-collection && pytest --collect-only",
        "pytest --create-minimal-suite --output tests/minimal_suite",
    ],
)
def test_tool_bridge_blocks_high_risk_shell_when_task_anchor_is_missing(command: str) -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    payload = json.dumps(
        {"calls": [{"id": "call_1", "name": "Bash", "input": {"command": command}}]},
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_shell_command_requires_explicit_task"
    assert result.error.repairable is False


def test_tool_bridge_allows_readonly_shell_when_task_anchor_is_missing() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git -C E:/ProjectX/mindcraft status --short"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "git -C E:/ProjectX/mindcraft status --short"})
    ]


def test_tool_bridge_allows_broad_test_when_user_explicitly_requests_tests() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "请运行测试并验证当前项目"}],
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"pytest --collect-only"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "pytest --collect-only"})
    ]


def test_tool_bridge_review_plan_text_is_not_repaired_into_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Write",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "审查代码并制定改进计?"}],
    )

    result = parse_tool_response(
        "改进计划：\n1. 修复配置加载问题。\n2. 添加基础测试。\n3. 后续再落地实现?",
        context,
    )

    assert result.tool_calls == []
    assert result.error is None
    assert "改进计划" in result.content


@pytest.mark.parametrize(
    "command",
    [
        "git -C E:/ProjectX/mindcraft status --short",
        "git -C E:/ProjectX/mindcraft log --oneline -n 5",
        'git grep -i "secret" -- "*.py"',
        "ls -la E:/ProjectX/mindcraft",
    ],
)
def test_tool_bridge_claude_code_review_policy_allows_readonly_command_matrix(command: str) -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "审查下代码并且制定改进计?"}],
    )
    payload = json.dumps(
        {"calls": [{"id": "call_1", "name": "Bash", "input": {"command": command}}]},
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": command})
    ]


def test_tool_bridge_rewrites_windows_dir_syntax_for_bash() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "你审查下当前项目的代码，看看有什么需要改进的?"}],
    )
    payload = json.dumps(
        {
            "calls": [
                {
                    "id": "call_1",
                    "name": "Bash",
                    "input": {"command": 'dir "E:/ProjectX/mindcraft/MediaCrawler" /s'},
                }
            ]
        },
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "find E:/ProjectX/mindcraft/MediaCrawler -maxdepth 3 -type f"})
    ]


def test_tool_bridge_rewrites_windows_dir_syntax_to_glob_when_available() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    payload = json.dumps(
        {
            "calls": [
                {
                    "id": "call_1",
                    "name": "Bash",
                    "input": {"command": 'dir "E:/ProjectX/mindcraft/MediaCrawler" /s'},
                }
            ]
        },
        ensure_ascii=False,
    )

    result = parse_tool_response(f"```tool_json\n{payload}\n```", context)

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Glob", {"path": "E:/ProjectX/mindcraft/MediaCrawler", "pattern": "*"})
    ]


def test_tool_bridge_allows_readonly_git_status_for_review_task() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "审查下当前项目的代码，看看有什么需要改进的?"}],
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git -C E:/ProjectX/mindcraft/MediaCrawler status --short && git -C E:/ProjectX/mindcraft/MediaCrawler log --oneline -n 5"}}]}\n```',
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        (
            "call_1",
            "Bash",
            {
                "command": "git -C E:/ProjectX/mindcraft/MediaCrawler status --short && git -C E:/ProjectX/mindcraft/MediaCrawler log --oneline -n 5"
            },
        )
    ]


def test_tool_bridge_does_not_rewrite_cli_name_without_bash() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read files",
                    "parameters": {"type": "object"},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"gh","input":{"command":"repo view Kriswd/web-gateway"}}]}\n```',
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unknown_tool"


def test_tool_bridge_adds_loop_guard_for_repeated_shell_housekeeping() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    messages = [
        {"role": "user", "content": "Update the local repository from origin/main."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": json.dumps({"command": "git fetch origin && git reset --hard origin/main"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "HEAD is now at abc123 latest"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": json.dumps({"command": "git status --short"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": ""},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": json.dumps({"command": "git rev-parse HEAD && git rev-parse origin/main"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_3", "content": "abc123\nabc123"},
    ]

    converted = prepare_openai_messages(messages, context)

    assert converted[-1]["role"] == "user"
    assert "Tool loop guard" in converted[-1]["content"]
    assert "Bash" in converted[-1]["content"]
    assert "stop requesting equivalent tools" in converted[-1]["content"]


def test_tool_bridge_rejects_repeated_shell_housekeeping_without_progress_in_agent_profile() -> None:
    messages = [
        {"role": "user", "content": "Fix the local repository test failure and verify the result."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": json.dumps({"command": "cd E:/ProjectX/mindcraft && git status --short"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": ""},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": json.dumps({"command": "cd E:/ProjectX/mindcraft && git diff HEAD"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": ""},
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {"name": "Edit", "description": "Edit files", "parameters": {"type": "object"}},
            },
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        "<|DSML|tool_calls>\n"
        '  <|DSML|invoke name="Bash">\n'
        '    <|DSML|parameter name="command"><![CDATA[cd E:/ProjectX/mindcraft && git status --short]]></|DSML|parameter>\n'
        "  </|DSML|invoke>\n"
        "</|DSML|tool_calls>",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "repeat_shell_housekeeping_without_progress"
    assert result.error.repairable is True


def test_tool_bridge_normalizes_cmd_cd_drive_switch_for_bash() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "Review this local Windows repository."}],
    )

    result = parse_tool_response(
        "<|DSML|tool_calls>"
        '<|DSML|invoke name="Bash">'
        '<|DSML|parameter name="command"><![CDATA[cd /d E:\\ProjectX\\mindcraft && git status]]></|DSML|parameter>'
        "</|DSML|invoke>"
        "</|DSML|tool_calls>",
        context,
    )

    assert result.error is None
    assert result.tool_calls[0].input["command"] == "cd E:/ProjectX/mindcraft && git status"


def test_tool_bridge_all_exposure_does_not_truncate_or_invent_toolsearch() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json=_openai_response('```tool_json\n{"calls":[{"id":"toolu_39","name":"Tool39","input":{"value":39}}]}\n```'),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="all", max_tools_in_prompt=32),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    tools = [{"name": f"Tool{i}", "description": f"Tool {i}", "input_schema": {"type": "object"}} for i in range(40)]

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={"model": "web-model", "messages": [{"role": "user", "content": "use the last tool"}], "tools": tools, "max_tokens": 1024},
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert '"name": "Tool39"' in prompt
    assert "ToolSearch" not in prompt
    assert response.json()["content"] == [{"type": "tool_use", "id": "toolu_39", "name": "Tool39", "input": {"value": 39}}]


def test_tool_bridge_all_exposure_compacts_large_tool_prompt_without_hiding_tools() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json=_openai_response('```tool_json\n{"calls":[{"id":"toolu_79","name":"Tool79","input":{"value":79}}]}\n```'),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", tool_profile="all", tool_prompt_max_chars=6000),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    tools = [
        {
            "name": f"Tool{i}",
            "description": "long description " * 120,
            "input_schema": {"type": "object", "properties": {f"field_{j}": {"type": "string", "description": "x" * 80} for j in range(8)}},
        }
        for i in range(80)
    ]

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={"model": "web-model", "messages": [{"role": "user", "content": "use the last tool"}], "tools": tools, "max_tokens": 1024},
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert len(prompt) <= 7000
    assert "Tool prompt manifest was compacted" in prompt
    assert "Tool79" in prompt
    assert "ToolSearch" not in prompt
    assert response.json()["content"] == [{"type": "tool_use", "id": "toolu_79", "name": "Tool79", "input": {"value": 79}}]


def test_tool_bridge_flags_runaway_plain_text_without_tool_call() -> None:
    context = build_context(
        [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = replace(context, task_text="Update the local project in E:\\ProjectX\\mindcraft", has_tool_loop=True)
    raw = "This is verbose plain assistant output without a tool call.\n" * 180

    result = parse_tool_response(raw, context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "runaway_plain_text_without_tool_call"
    assert result.error.repairable is True


def test_tool_bridge_rejects_post_write_shell_verification_in_final_text() -> None:
    context = build_context(
        [
            {"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "Write", "parameters": {"type": "object"}}},
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )
    context = replace(context, has_tool_loop=True, recent_tool_call_names=("Skill", "Write"))
    raw = (
        "Mindcraft config manager has been created in `config/manager.py`.\n\n"
        "Verify the configuration file:\n"
        "```bash\n"
        "ls -l E:/ProjectX/mindcraft/config/manager.py\n"
        "```\n\n"
        "Test configuration loading:\n"
        "```bash\n"
        "python -c \"from config.manager import get_config; print(get_config('wechat.timeout'))\"\n"
        "```"
    )

    result = parse_tool_response(raw, context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unexecuted_verification_command_after_write"
    assert result.error.repairable is True


def test_openai_tool_bridge_repairs_runaway_plain_text_without_tool_call() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=_openai_response("This is verbose plain assistant output without a tool call.\n" * 180),
                request=request,
            )
        return httpx.Response(
            200,
            json=_openai_response('```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{"file_path":"README.md"}}]}\n```'),
            request=request,
        )

    client = _client(handler)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [
                {"role": "user", "content": "Update the local project in E:\\ProjectX\\mindcraft"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_prior",
                            "type": "function",
                            "function": {"name": "Read", "arguments": "{\"file_path\":\"README.md\"}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_prior", "name": "Read", "content": "prior observation"},
            ],
            "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    assert len(requests) == 2
    repair_prompt = "\n".join(str(message.get("content", "")) for message in requests[1]["messages"])
    assert "runaway_plain_text_without_tool_call" in repair_prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "Read"


def test_anthropic_tool_bridge_parses_claude_code_tool_summary_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = (
            "Searched for 15 patterns (ctrl+o to expand)\n\n"
            "Assistant requested tool calls:\n"
            '- Read({"file_path":"E:\\\\ProjectX\\\\webai-gateway\\\\README.md"})\n'
            '- Read({"file_path":"E:\\\\ProjectX\\\\webai-gateway\\\\requirements.txt"})'
        )
        return httpx.Response(200, json=_openai_response(content), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all", max_readonly_calls_per_turn=8),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "/init"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert [block["name"] for block in body["content"]] == ["Read", "Read"]
    assert body["content"][0]["id"].startswith("toolu_")
    assert body["content"][0]["input"] == {"file_path": "E:\\ProjectX\\webai-gateway\\README.md"}
    assert body["content"][1]["input"] == {"file_path": "E:\\ProjectX\\webai-gateway\\requirements.txt"}


def test_anthropic_tool_bridge_allows_code_review_batch_read_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        calls = "\n".join(
            f'- Read({{"file_path":"E:\\\\ProjectX\\\\mindcraft\\\\file{i}.py"}})' for i in range(9)
        )
        content = f"Assistant requested tool calls:\n{calls}"
        return httpx.Response(200, json=_openai_response(content), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="local-agent"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "审查当前项目代码并制定改进计?"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert [block["name"] for block in body["content"]] == ["Read"] * 9


def test_anthropic_tool_bridge_normalizes_provider_search_markup_to_allowed_search_tool() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        content = (
            "<search>\n"
            "<query>智谱GLM-5.1网页版体验地址 2026</query>\n"
            "<current_date>2026-04-17</current_date>\n"
            "</search>"
        )
        return httpx.Response(200, json=_openai_response(content), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "联网搜索下，5.1都发布了"}],
            "tools": [
                {
                    "name": "WebSearch",
                    "description": "Search the web",
                    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert "provider-native" in prompt
    assert "<search>" in prompt
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_search_1",
            "name": "WebSearch",
            "input": {"query": "智谱GLM-5.1网页版体验地址 2026"},
        }
    ]


def test_anthropic_tool_bridge_returns_text_when_provider_search_markup_has_no_search_tool() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = "<search><query>智谱GLM-5.1网页版体验地址 2026</query></search>"
        return httpx.Response(200, json=_openai_response(content), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "联网搜索下，5.1都发布了"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "end_turn"
    assert body["content"][0]["type"] == "text"
    assert "模型请求联网搜索" in body["content"][0]["text"]
    assert "当前允许工具中没有可用的搜索工具" in body["content"][0]["text"]
    assert "智谱GLM-5.1网页版体验地址 2026" in body["content"][0]["text"]
    assert "<search>" not in body["content"][0]["text"]


def test_anthropic_tool_use_ids_are_toolu_prefixed_for_claude_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_openai_response('```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{"file_path":"README.md"}}]}\n```'),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "read"}],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"][0]["type"] == "tool_use"
    assert body["content"][0]["id"].startswith("toolu_")
    assert body["content"][0]["id"] != "call_1"


def test_tool_bridge_safe_exposure_hides_claude_code_write_tools() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_openai_response("plain reply"), request=request)

    client = _client(handler)

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "inspect files"}],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
                {"name": "Edit", "description": "Edit files", "input_schema": {"type": "object"}},
                {"name": "Write", "description": "Write files", "input_schema": {"type": "object"}},
                {"name": "MultiEdit", "description": "Edit many files", "input_schema": {"type": "object"}},
                {"name": "NotebookEdit", "description": "Edit notebooks", "input_schema": {"type": "object"}},
                {"name": "TodoWrite", "description": "Write todos", "input_schema": {"type": "object"}},
                {"name": "mcp__repo__search", "description": "Search repository", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["body"]["messages"] if message.get("role") == "system")
    assert '"name": "Read"' in prompt
    assert '"name": "Glob"' in prompt
    assert '"name": "mcp__repo__search"' in prompt
    for blocked_name in ("Edit", "Write", "MultiEdit", "NotebookEdit", "TodoWrite"):
        assert f'"name": "{blocked_name}"' not in prompt


def test_tool_bridge_local_agent_exposure_keeps_minimal_coding_tools() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in [
                "Read",
                "Glob",
                "Grep",
                "Bash",
                "Edit",
                "Write",
                "MultiEdit",
                "WebFetch",
                "WebSearch",
                "Agent",
                "Skill",
                "TodoWrite",
                "NotebookEdit",
                "mcp__repo__search",
            ]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", tool_profile="agent", max_tools_in_prompt=32),
    )

    names = {tool.name for tool in context.tools}

    assert {"Read", "Glob", "Grep", "Bash", "Edit", "Write", "MultiEdit", "WebFetch", "WebSearch", "Skill", "mcp__repo__search"} <= names
    assert {"Agent", "TodoWrite", "NotebookEdit"}.isdisjoint(names)


def test_tool_bridge_local_agent_hides_bash_until_shell_is_explicitly_requested() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Skill"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", tool_profile="agent", max_tools_in_prompt=32),
    )

    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "审查当前项目代码并制定完整改进计?"}],
    )

    assert {tool.name for tool in context.tools} == {"Read", "Glob", "Grep", "Edit", "Write", "Skill"}


def test_tool_bridge_local_agent_chinese_implementation_task_hides_bash() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Skill"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", tool_profile="auto", max_tools_in_prompt=32),
    )

    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "制定一个完整的改进计划并落地"}],
    )

    assert {tool.name for tool in context.tools} == {"Read", "Glob", "Grep", "Edit", "Write", "Skill"}


def test_tool_bridge_local_agent_chinese_test_task_keeps_bash_without_write_tools() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Skill"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", tool_profile="auto", max_tools_in_prompt=32),
    )

    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "运行测试并验证当前项目是否可?"}],
    )

    assert {tool.name for tool in context.tools} == {"Read", "Glob", "Grep", "Bash", "Skill"}


def test_tool_bridge_local_agent_tool_loop_hides_bash_when_task_text_is_weak() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Skill"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )

    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {
                "role": "user",
                "content": (
                    "<command-name>/using-superpowers</command-name>\n"
                    "<command-args>do it</command-args>"
                ),
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "tests/test_config.py"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "is_error": True,
                        "content": "File does not exist. Note: your current working directory is E:\\ProjectX\\mindcraft.",
                    }
                ],
            },
        ],
    )

    names = {tool.name for tool in context.tools}
    assert "Bash" not in names
    assert "Write" not in names
    assert {"Read", "Glob", "Grep", "Edit", "Skill"} <= names


def test_tool_bridge_local_agent_exposes_write_after_repo_discovery() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Skill"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1", "name": "Glob", "input": {"path": ".", "pattern": "*.py"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "fetch_wechat.py\nrewrite_content.py"}],
        },
    ]

    context = prefer_local_tools_for_local_agent_task(context, messages)

    names = {tool.name for tool in context.tools}
    assert "Bash" not in names
    assert {"Read", "Glob", "Grep", "Edit", "Write", "Skill"} <= names


def test_tool_bridge_local_agent_keeps_write_after_successful_write_result() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write", "Skill"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Write",
                    "input": {
                        "file_path": "E:/ProjectX/mindcraft/config/manager.py",
                        "content": "class ConfigManager:\n    pass\n",
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "File created successfully at: E:/ProjectX/mindcraft/config/manager.py",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "def analyze_style():\n    return 'ok'\n",
                }
            ],
        },
    ]

    context = prefer_local_tools_for_local_agent_task(context, messages)

    names = {tool.name for tool in context.tools}
    assert "Write" in names
    assert {"Read", "Glob", "Grep", "Edit", "Skill"} <= names


def test_tool_bridge_local_agent_keeps_bash_for_explicit_shell_task() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Skill"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", tool_profile="agent", max_tools_in_prompt=32),
    )

    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "运行测试并验证当前项目是否可?"}],
    )

    assert "Bash" in {tool.name for tool in context.tools}


def test_tool_prompt_warns_after_missing_read_result_not_to_repeat_same_path() -> None:
    messages = [
        {"role": "user", "content": "Implement the plan in the local codebase."},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "tests/test_config.py"}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "is_error": True,
                    "content": "File does not exist. Note: your current working directory is E:\\ProjectX\\mindcraft.",
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    prepared = prepare_openai_messages(messages, context)
    prompt = "\n".join(str(message.get("content", "")) for message in prepared)

    assert "Do not repeat the same Read input" in prompt
    assert "tests/test_config.py" in prompt
    assert "Glob/Grep" in prompt


def test_tool_prompt_warns_after_unchanged_read_result_not_to_repeat_same_path() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/rewrite_content.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": (
                        "File unchanged since last read. The content from the earlier Read tool_result "
                        "in this conversation is still current - refer to that instead of re-reading."
                    ),
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    prepared = prepare_openai_messages(messages, context)
    prompt = "\n".join(str(message.get("content", "")) for message in prepared)

    assert "Tool result guard: the last Read result reported unchanged content" in prompt
    assert "E:/ProjectX/mindcraft/rewrite_content.py" in prompt
    assert "Do not repeat the same Read input" in prompt


def test_tool_bridge_rejects_repeat_read_after_unchanged_result() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/rewrite_content.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": (
                        "File unchanged since last read. The content from the earlier Read tool_result "
                        "in this conversation is still current - refer to that instead of re-reading."
                    ),
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft/rewrite_content.py"}}]}\n'
        "```",
        context,
    )

    assert result.error is not None
    assert result.error.kind == "repeat_unchanged_read_without_progress"
    assert result.tool_calls == []


def test_tool_bridge_all_profile_passes_repeated_unchanged_read_like_ds2api() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all", max_tools_in_prompt=32),
    )
    context = replace(
        context,
        has_tool_loop=True,
        recent_unchanged_read_paths=("E:/ProjectX/mindcraft/get_feishu_fields.py",),
        recent_unchanged_read_summaries=(
            'Read input: {"file_path":"E:\\ProjectX\\mindcraft\\get_feishu_fields.py"}',
        ),
    )

    result = parse_tool_response(
        "<|DSML|tool_calls>"
        '<|DSML|invoke name="Read">'
        '<|DSML|parameter name="file_path"><![CDATA[E:\\ProjectX\\mindcraft\\get_feishu_fields.py]]></|DSML|parameter>'
        "</|DSML|invoke>"
        "</|DSML|tool_calls>",
        context,
    )

    assert result.error is None
    assert [(call.name, call.input) for call in result.tool_calls] == [
        ("Read", {"file_path": "E:\\ProjectX\\mindcraft\\get_feishu_fields.py"})
    ]


def test_tool_bridge_resolved_all_profile_passes_repeated_unchanged_read_like_ds2api() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", max_tools_in_prompt=32),
    )
    context = replace(
        context,
        has_tool_loop=True,
        recent_unchanged_read_paths=("E:/ProjectX/mindcraft/get_feishu_fields.py",),
        recent_unchanged_read_summaries=(
            'Read input: {"file_path":"E:\\ProjectX\\mindcraft\\get_feishu_fields.py"}',
        ),
    )

    result = parse_tool_response(
        "<|DSML|tool_calls>"
        '<|DSML|invoke name="Read">'
        '<|DSML|parameter name="file_path"><![CDATA[E:\\ProjectX\\mindcraft\\get_feishu_fields.py]]></|DSML|parameter>'
        "</|DSML|invoke>"
        "</|DSML|tool_calls>",
        context,
    )

    assert result.error is None
    assert [(call.name, call.input) for call in result.tool_calls] == [
        ("Read", {"file_path": "E:\\ProjectX\\mindcraft\\get_feishu_fields.py"})
    ]


def test_tool_bridge_drops_repeated_unchanged_read_when_batch_has_progress() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/rewrite_content.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": (
                        "File unchanged since last read. The content from the earlier Read tool_result "
                        "in this conversation is still current - refer to that instead of re-reading."
                    ),
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":['
        '{"id":"call_repeat","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft/rewrite_content.py"}},'
        '{"id":"call_glob","name":"Glob","input":{"path":"E:/ProjectX/mindcraft","pattern":"*.py"}},'
        '{"id":"call_read","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft/sync_to_feishu.py"}}'
        "]}\n"
        "```",
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_glob", "Glob", {"path": "E:/ProjectX/mindcraft", "pattern": "*.py"}),
        ("call_read", "Read", {"file_path": "E:/ProjectX/mindcraft/sync_to_feishu.py"}),
    ]


def test_tool_bridge_rejects_repeated_read_without_progress() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "def analyze_style():\n    return 'ok'\n",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "def analyze_style():\n    return 'ok'\n",
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_3","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"}}]}\n'
        "```",
        context,
    )

    assert result.error is not None
    assert result.error.kind == "repeat_read_call_without_progress"
    assert result.tool_calls == []


def test_tool_bridge_drops_repeated_read_when_batch_has_progress() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "def analyze_style():\n    return 'ok'\n",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "def analyze_style():\n    return 'ok'\n",
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":['
        '{"id":"call_repeat","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"}},'
        '{"id":"call_read","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft/skills/story-interviewer/interview_tools.py"}}'
        "]}\n"
        "```",
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_read", "Read", {"file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/interview_tools.py"}),
    ]


def test_tool_bridge_rejects_repeated_read_batch_without_progress() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"},
                },
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/interview_tools.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "analyze_style content"},
                {"type": "tool_result", "tool_use_id": "call_2", "content": "interview_tools content"},
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        "<|DSML|tool_calls>\n"
        '  <|DSML|invoke name="Read">\n'
        '    <|DSML|parameter name="file_path"><![CDATA[E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py]]></|DSML|parameter>\n'
        "  </|DSML|invoke>\n"
        '  <|DSML|invoke name="Read">\n'
        '    <|DSML|parameter name="file_path"><![CDATA[E:/ProjectX/mindcraft/skills/story-interviewer/interview_tools.py]]></|DSML|parameter>\n'
        "  </|DSML|invoke>\n"
        "</|DSML|tool_calls>",
        context,
    )

    assert result.error is not None
    assert result.error.kind == "repeat_read_call_without_progress"
    assert result.tool_calls == []


def test_tool_bridge_allows_repeated_read_after_mutation() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "old content"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "Edit",
                    "input": {
                        "file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py",
                        "old_string": "old",
                        "new_string": "new",
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_2", "content": "Updated 1 file."}]},
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_3","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"}}]}\n'
        "```",
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_3", "Read", {"file_path": "E:/ProjectX/mindcraft/skills/story-interviewer/analyze_style.py"}),
    ]


def test_tool_bridge_rejects_repeated_discovery_without_progress() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Glob",
                    "input": {"path": "E:/ProjectX/mindcraft", "pattern": "**/*.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "rewrite_content.py\nsync_to_feishu.py",
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"Glob","input":{"path":"E:/ProjectX/mindcraft","pattern":"**/*.py"}}]}\n'
        "```",
        context,
    )

    assert result.error is not None
    assert result.error.kind == "repeat_discovery_call_without_progress"
    assert result.tool_calls == []


def test_tool_bridge_drops_repeated_discovery_when_batch_has_progress() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Glob",
                    "input": {"path": "E:/ProjectX/mindcraft", "pattern": "**/*.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "rewrite_content.py\nsync_to_feishu.py",
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":['
        '{"id":"call_repeat","name":"Glob","input":{"path":"E:/ProjectX/mindcraft","pattern":"**/*.py"}},'
        '{"id":"call_read","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft/sync_to_feishu.py"}}'
        "]}\n"
        "```",
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_read", "Read", {"file_path": "E:/ProjectX/mindcraft/sync_to_feishu.py"}),
    ]


def test_tool_bridge_allows_repeated_discovery_after_mutation() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Glob",
                    "input": {"path": "E:/ProjectX/mindcraft", "pattern": "**/*.py"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "rewrite_content.py"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "Edit",
                    "input": {"file_path": "E:/ProjectX/mindcraft/rewrite_content.py", "old_string": "a", "new_string": "b"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_2", "content": "Updated 1 file."}]},
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_3","name":"Glob","input":{"path":"E:/ProjectX/mindcraft","pattern":"**/*.py"}}]}\n'
        "```",
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_3", "Glob", {"path": "E:/ProjectX/mindcraft", "pattern": "**/*.py"}),
    ]


def test_tool_bridge_rejects_repeated_same_skill_without_progress() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make an improvement plan."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Skill",
                    "input": {"skill": "using-superpowers"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Launching skill: using-superpowers"}],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Skill", "Read", "Glob", "Grep"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"Skill","input":{"skill":"using-superpowers"}}]}\n'
        "```",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "repeat_same_skill_without_progress"
    assert result.error.repairable is True


def test_tool_bridge_rejects_repeated_skill_after_injected_skill_result_without_assistant_history() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make an improvement plan."},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_call_1", "content": "Launching skill: update-config"}],
        },
        {
            "role": "user",
            "content": (
                "# Update Config Skill\n\n"
                "Modify Claude Code configuration by updating settings.json files.\n\n"
                "## When Hooks Are Required (Not Memory)"
            ),
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Skill", "Read", "Glob", "Grep"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"Skill","input":{"skill":"update-config","args":"add statusLine command"}}]}\n'
        "```",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "repeat_same_skill_without_progress"


def test_tool_bridge_rejects_repeated_skill_after_injected_skill_text_only() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make an improvement plan."},
        {
            "role": "user",
            "content": (
                "# Update Config Skill\n\n"
                "Modify Claude Code configuration by updating settings.json files.\n\n"
                "## When Hooks Are Required (Not Memory)"
            ),
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Skill", "Read", "Glob", "Grep"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"Skill","input":{"skill":"update-config","args":"add statusLine command"}}]}\n'
        "```",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "repeat_same_skill_without_progress"


def test_tool_bridge_rejects_repeated_simplify_skill_after_colon_titled_skill_text_only() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make improvements."},
        {
            "role": "user",
            "content": (
                "# Simplify: Code Review and Cleanup\n\n"
                "Review all changed files for reuse, quality, and efficiency. Fix any issues found.\n\n"
                "## Phase 1: Identify Changes\n\n"
                "Run `git diff` to see what changed.\n\n"
                "## Phase 2: Launch Review Agents"
            ),
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Skill", "Read", "Glob", "Grep"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"Skill","input":{"skill":"simplify"}}]}\n'
        "```",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "repeat_same_skill_without_progress"


def test_tool_bridge_rejects_repeated_skill_name_alias_without_progress() -> None:
    messages = [
        {"role": "user", "content": "Review changed files and fix any quality issues."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Skill",
                    "input": {"skill_name": "simplify"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Launching skill: simplify"}],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Skill", "Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_2","name":"Skill","input":{"skill":"simplify"}}]}\n```',
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "repeat_same_skill_without_progress"
    assert result.error.repairable is True


def test_tool_bridge_resolved_all_profile_passes_repeated_skill_like_ds2api() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Skill", "Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", max_tools_in_prompt=32),
    )
    context = replace(
        context,
        has_tool_loop=True,
        recent_skill_names=("simplify",),
        recent_tool_call_summaries=('Skill input: {"skill":"simplify"}',),
    )

    result = parse_tool_response(
        "<|DSML|tool_calls>"
        '<|DSML|invoke name="Skill">'
        '<|DSML|parameter name="skill"><![CDATA[simplify]]></|DSML|parameter>'
        "</|DSML|invoke>"
        "</|DSML|tool_calls>",
        context,
    )

    assert result.error is None
    assert [(call.name, call.input) for call in result.tool_calls] == [
        ("Skill", {"skill": "simplify"})
    ]


def test_tool_bridge_adds_guard_after_injected_skill_text_without_assistant_history() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make an improvement plan."},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_call_1", "content": "Launching skill: update-config"}],
        },
        {
            "role": "user",
            "content": (
                "# Update Config Skill\n\n"
                "Modify Claude Code configuration by updating settings.json files.\n\n"
                "## When Hooks Are Required (Not Memory)"
            ),
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Skill", "Read", "Glob", "Grep"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    prepared = prepare_openai_messages(messages, context)

    assert prepared[-1]["role"] == "user"
    assert "Skill progress guard" in prepared[-1]["content"]
    assert "Do not request the same Skill again" in prepared[-1]["content"]
    assert "Skill input: {\"skill\":\"update-config\"}" in prepared[-1]["content"]


def test_tool_bridge_adds_guard_after_injected_skill_text_only() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make an improvement plan."},
        {
            "role": "user",
            "content": (
                "# Update Config Skill\n\n"
                "Modify Claude Code configuration by updating settings.json files.\n\n"
                "## When Hooks Are Required (Not Memory)"
            ),
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Skill", "Read", "Glob", "Grep"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    prepared = prepare_openai_messages(messages, context)

    assert prepared[-1]["role"] == "user"
    assert "Skill progress guard" in prepared[-1]["content"]
    assert "Do not request the same Skill again" in prepared[-1]["content"]
    assert "Skill input: {\"skill\":\"update-config\"}" in prepared[-1]["content"]


def test_tool_bridge_adds_guard_after_skill_result() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make an improvement plan."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Skill",
                    "input": {"skill": "using-superpowers"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Launching skill: using-superpowers"}],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Skill", "Read", "Glob", "Grep"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    prepared = prepare_openai_messages(messages, context)

    assert prepared[-1]["role"] == "user"
    assert "Skill progress guard" in prepared[-1]["content"]
    assert "Do not request the same Skill again" in prepared[-1]["content"]
    assert "Skill input: {\"skill\":\"using-superpowers\"}" in prepared[-1]["content"]


def test_repeat_skill_repair_prompt_forbids_same_skill() -> None:
    error = BridgeError(
        "repeat_same_skill_without_progress",
        "The model repeated the same Skill call.",
        repairable=True,
    )

    repaired = build_repair_messages(
        [{"role": "user", "content": "Review local project code and make an improvement plan."}],
        '```tool_json\n{"calls":[{"id":"call_2","name":"Skill","input":{"skill":"using-superpowers"}}]}\n```',
        error,
    )

    repair_prompt = repaired[-1]["content"]
    assert "Do not request the same Skill again" in repair_prompt
    assert "different allowed tool" in repair_prompt
    assert "substantive final answer" in repair_prompt


def test_tool_bridge_rejects_off_task_environment_configuration_question() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/process_answer.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "project file content"}],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["AskUserQuestion", "Read", "Glob", "Grep", "Skill"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"AskUserQuestion","input":{"questions":[{"header":"Configure status line",'
        '"question":"Do you want me to configure the caveman statusLine badge in settings.json?",'
        '"options":[{"label":"Yes","description":"Add statusLine config"},{"label":"No","description":"Skip"}]}]}}]}\n'
        "```",
        context,
    )

    assert result.error is not None
    assert result.error.kind == "off_task_environment_configuration_question"
    assert result.tool_calls == []


def test_tool_bridge_rejects_off_task_claude_path_question() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/CLAUDE.md"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "project rules"}]},
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["AskUserQuestion", "Read", "Glob", "Grep", "Skill"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"AskUserQuestion","input":{"questions":[{"header":"Policy",'
        '"question":"Should I update .claude/settings.json for this task?",'
        '"options":[{"label":"Yes","description":"Update agent settings"},{"label":"No","description":"Skip"}]}]}}]}\n'
        "```",
        context,
    )

    assert result.error is not None
    assert result.error.kind == "off_task_environment_configuration_question"
    assert result.tool_calls == []


def test_tool_bridge_rejects_off_task_scope_escalation_question() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/CLAUDE.md"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "project rules"}]},
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["AskUserQuestion", "Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    scope_question_payload = json.dumps(
        {
            "calls": [
                {
                    "id": "call_2",
                    "name": "AskUserQuestion",
                    "input": {
                        "questions": [
                            {
                                "header": "重构",
                                "question": "允许重构目录结构吗？将脚本迁移到 src 并重写模块边界？",
                                "options": [
                                    {"label": "允许", "description": "执行目录迁移"},
                                    {"label": "暂缓", "description": "只输出改进计划"},
                                ],
                            }
                        ]
                    },
                }
            ]
        },
        ensure_ascii=False,
    )
    result = parse_tool_response(f"```tool_json\n{scope_question_payload}\n```", context)

    assert result.error is not None
    assert result.error.kind == "off_task_scope_escalation_question"
    assert result.tool_calls == []


def test_tool_bridge_allows_scope_question_when_user_requested_restructure() -> None:
    messages = [
        {"role": "user", "content": "请重构项目目录结构，必要时先问我迁移范围。"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/CLAUDE.md"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "project rules"}]},
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["AskUserQuestion", "Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    scope_question_payload = json.dumps(
        {
            "calls": [
                {
                    "id": "call_2",
                    "name": "AskUserQuestion",
                    "input": {
                        "questions": [
                            {
                                "header": "重构",
                                "question": "允许重构目录结构吗？将脚本迁移到 src 并重写模块边界？",
                                "options": [
                                    {"label": "允许", "description": "执行目录迁移"},
                                    {"label": "暂缓", "description": "只输出改进计划"},
                                ],
                            }
                        ]
                    },
                }
            ]
        },
        ensure_ascii=False,
    )
    result = parse_tool_response(f"```tool_json\n{scope_question_payload}\n```", context)

    assert result.error is None
    assert [(call.id, call.name) for call in result.tool_calls] == [("call_2", "AskUserQuestion")]


def test_tool_bridge_rejects_optional_scope_question_when_review_can_continue() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/README.md"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "project overview"}]},
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["AskUserQuestion", "Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"AskUserQuestion","input":{"questions":[{"header":"Focus",'
        '"question":"Improvement plan focus?",'
        '"options":[{"label":"Architecture/Code quality","description":"Refactor, patterns, testing"},'
        '{"label":"All of above","description":"Comprehensive roadmap"}]}]}}]}\n'
        "```",
        context,
    )

    assert result.error is not None
    assert result.error.kind == "optional_scope_question_without_need"
    assert result.tool_calls == []


def test_tool_bridge_rejects_second_ask_user_question_after_progress() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_question",
                    "name": "AskUserQuestion",
                    "input": {
                        "questions": [
                            {
                                "header": "Focus",
                                "question": "Improvement plan focus?",
                                "options": [{"label": "All of above", "description": "Comprehensive roadmap"}],
                            }
                        ]
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_question",
                    "content": "User answered: All of above.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_read",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/README.md"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_read", "content": "project overview"}]},
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["AskUserQuestion", "Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"AskUserQuestion","input":{"questions":[{"header":"Area",'
        '"question":"Which area to review first?",'
        '"options":[{"label":"Core scripts","description":"Content pipeline"},'
        '{"label":"Full scan","description":"Everything"}]}]}}]}\n'
        "```",
        context,
    )

    assert result.error is not None
    assert result.error.kind == "ask_user_question_budget_exceeded"
    assert result.tool_calls == []


def test_tool_bridge_rejects_repeated_ask_user_question_without_progress() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "AskUserQuestion",
                    "input": {
                        "questions": [
                            {
                                "header": "Secret management",
                                "question": "How does this project handle API keys and database passwords?",
                                "options": [
                                    {"label": "Environment variables", "description": "Use injected env vars"}
                                ],
                            }
                        ]
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "User answered: use environment variables without .env files.",
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["AskUserQuestion", "Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"AskUserQuestion","input":{"questions":[{"header":"Secret management",'
        '"question":"How does this project handle API key and database password secrets?",'
        '"options":[{"label":"Environment variables","description":"Use injected env vars"}]}]}}]}\n'
        "```",
        context,
    )

    assert result.error is not None
    assert result.error.kind == "repeat_same_ask_user_without_progress"
    assert result.tool_calls == []


def test_tool_bridge_all_profile_passes_repeated_ask_user_question_like_ds2api() -> None:
    messages = [
        {"role": "user", "content": "Review local project code and make a detailed improvement plan."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "AskUserQuestion",
                    "input": {
                        "questions": [
                            {
                                "header": "Secret management",
                                "question": "How does this project handle API keys?",
                                "options": [{"label": "Environment variables", "description": "Use env vars"}],
                            }
                        ]
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Use environment variables."}],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["AskUserQuestion", "Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"AskUserQuestion","input":{"questions":[{"header":"Secret management",'
        '"question":"How does this project handle API key and database password secrets?",'
        '"options":[{"label":"Environment variables","description":"Use injected env vars"}]}]}}]}\n'
        "```",
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "AskUserQuestion"


def test_tool_bridge_rejects_ask_user_question_claiming_successful_bash_was_blocked() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers 落地改进"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Bash",
                    "input": {
                        "command": "cd E:/ProjectX/mindcraft && git diff HEAD",
                        "description": "Get git diff to identify changed files",
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "(Bash completed with no output)",
                    "is_error": False,
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["AskUserQuestion", "Bash", "Glob", "Read"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    blocked_question_payload = json.dumps(
        {
            "calls": [
                {
                    "id": "call_2",
                    "name": "AskUserQuestion",
                    "input": {
                        "questions": [
                            {
                                "header": "获取 Diff",
                                "question": "Bash 命令被阻止。请手动运行 `git diff HEAD` 并将输出粘贴给我，或者授权我执行 Bash 命令以继续代码审查？",
                                "options": [
                                    {"label": "粘贴 git diff 输出", "description": "用户手动提供变更内容"},
                                    {"label": "授权 Bash", "description": "允许代理执行 git 命令"},
                                ],
                            }
                        ]
                    },
                }
            ]
        },
        ensure_ascii=False,
    )
    result = parse_tool_response(f"```tool_json\n{blocked_question_payload}\n```", context)

    assert result.error is not None
    assert result.error.kind == "successful_tool_misread_as_blocked_question"
    assert result.tool_calls == []


def test_tool_bridge_rejects_ask_user_question_claiming_recent_successful_bash_was_blocked() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers review, fix, and implement improvements in the current project"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_bash",
                    "name": "Bash",
                    "input": {
                        "command": "cd E:/ProjectX/mindcraft && git diff HEAD",
                        "description": "Get full git diff for review",
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_bash",
                    "content": "(Bash completed with no output)",
                    "is_error": False,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_scope",
                    "name": "AskUserQuestion",
                    "input": {
                        "questions": [
                            {
                                "header": "Scope",
                                "question": "Review scope: entire project or only top-level scripts?",
                                "options": [
                                    {"label": "Entire project", "description": "Review everything"},
                                    {"label": "Top-level only", "description": "Skip subdirectories"},
                                ],
                            }
                        ]
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_scope",
                    "content": (
                        'User has answered your questions: "Review scope: entire project or only top-level scripts?"'
                        '="Entire project". You can now continue with the user\'s answers in mind.'
                    ),
                    "is_error": False,
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["AskUserQuestion", "Bash", "Glob", "Read", "Skill"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_3","name":"AskUserQuestion","input":{"questions":[{"header":"Bash Permission",'
        '"question":"Need git diff to review changes. Grant Bash permission for `git diff`?",'
        '"options":[{"label":"Grant Bash","description":"Allow running git diff via Bash"},'
        '{"label":"Specify Files","description":"Manually list files to review instead"}]}]}}]}\n'
        "```",
        context,
    )

    assert result.error is not None
    assert result.error.kind == "successful_tool_misread_as_blocked_question"
    assert result.tool_calls == []


def test_tool_bridge_rejects_write_to_same_path_after_missing_read_without_discovery() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/mindcraft_system.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "is_error": True,
                    "content": "File does not exist. Note: your current working directory is E:\\ProjectX\\mindcraft.",
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"Write","input":{"file_path":"mindcraft_system.py","content":"# guessed module"}}]}'
        "\n```",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "write_after_failed_read_without_discovery"
    assert result.error.repairable is True


def test_tool_bridge_rejects_write_near_missing_read_without_discovery() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/tests/test_integration.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "is_error": True,
                    "content": "File does not exist. Note: your current working directory is E:\\ProjectX\\mindcraft.",
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"Write","input":{"file_path":"tests/integration/test_full_workflow.py","content":"# guessed integration test"}}]}'
        "\n```",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "write_after_failed_read_without_discovery"
    assert result.error.repairable is True


def test_tool_bridge_keeps_failed_read_guard_after_unrelated_write_without_discovery() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"file_path": "E:/ProjectX/mindcraft/config/settings.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "is_error": True,
                    "content": "File does not exist. Note: your current working directory is E:\\ProjectX\\mindcraft.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "Write",
                    "input": {
                        "file_path": "E:/ProjectX/mindcraft/tests/integration/test_full_workflow.py",
                        "content": "# unrelated guessed test",
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "File created successfully at: E:/ProjectX/mindcraft/tests/integration/test_full_workflow.py",
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_3","name":"Write","input":{"file_path":"config/settings.py","content":"# guessed config"}}]}'
        "\n```",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "write_after_failed_read_without_discovery"
    assert result.error.repairable is True


def test_tool_bridge_rejects_write_under_failed_glob_directory_without_discovery() -> None:
    messages = [
        {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Glob",
                    "input": {"path": "E:/ProjectX/mindcraft/tests", "pattern": "*"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "is_error": True,
                    "content": "Directory does not exist: E:/ProjectX/mindcraft/tests. Note: your current working directory is E:\\ProjectX\\mindcraft.",
                }
            ],
        },
    ]
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(context, messages)

    prepared = prepare_openai_messages(messages, context)
    prompt = "\n".join(str(message.get("content", "")) for message in prepared)
    assert "last file-discovery call failed" in prompt
    assert "do not create that same guessed path or its children with Write" in prompt

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"Write","input":{"file_path":"tests/integration_test.py","content":"# guessed test"}}]}'
        "\n```",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "write_after_failed_path_without_discovery"
    assert result.error.repairable is True


def _profile_tool_defs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
        }
        for name in ["Read", "Glob", "Grep", "Bash", "Edit", "Write"]
    ]


def test_tool_bridge_none_tool_profile_hides_all_tools() -> None:
    context = build_context(
        _profile_tool_defs(),
        ToolBridgeConfig(tool_profile="none", exposure_policy="all", max_tools_in_prompt=32),
    )

    assert context.enabled is False
    assert context.tools == []


def test_tool_bridge_read_only_tool_profile_keeps_only_read_tools() -> None:
    context = build_context(
        _profile_tool_defs(),
        ToolBridgeConfig(tool_profile="read-only", exposure_policy="all", max_tools_in_prompt=32),
    )

    assert {tool.name for tool in context.tools} == {"Read", "Glob", "Grep"}


def test_auto_tool_profile_filters_write_tools_but_keeps_guarded_readonly_shell() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object"}},
            }
            for name in ["Bash", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", max_tools_in_prompt=32),
    )

    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "Review current codebase and report findings only."}],
    )

    assert context.enabled is True
    assert {tool.name for tool in context.tools} == {"Bash"}


def test_tool_bridge_agent_tool_profile_hides_bash_until_shell_is_explicit() -> None:
    context = build_context(
        _profile_tool_defs(),
        ToolBridgeConfig(tool_profile="agent", exposure_policy="all", max_tools_in_prompt=32),
    )

    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "Review the repo and implement the requested change."}],
    )

    assert {tool.name for tool in context.tools} == {"Read", "Glob", "Grep", "Edit", "Write"}


def test_tool_bridge_all_tool_profile_keeps_runtime_tools() -> None:
    context = build_context(
        _profile_tool_defs(),
        ToolBridgeConfig(tool_profile="all", exposure_policy="safe", max_tools_in_prompt=32),
    )

    assert {tool.name for tool in context.tools} == {"Read", "Glob", "Grep", "Bash", "Edit", "Write"}


def test_anthropic_messages_streams_tool_use_events_for_qwen_web(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            qwen_client_factory=_FakeQwenToolClient,
            http_client=_not_found_client(),
        )
    )

    with client.stream(
        "POST",
        "/v1/messages?beta=true",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [{"name": "list_local_files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
            "stream": True,
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    events = _sse_events(body)
    assert events[1]["content_block"]["type"] == "tool_use"
    assert events[1]["content_block"]["id"]
    assert events[1]["content_block"]["name"] == "list_local_files"
    assert events[1]["content_block"]["input"] == {}
    assert events[2]["delta"] == {"type": "input_json_delta", "partial_json": '{"path":"LOCAL_WORKSPACE"}'}
    assert events[4]["delta"]["stop_reason"] == "tool_use"


def test_anthropic_messages_streams_batch_readonly_tool_events_for_qwen_web(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            qwen_client_factory=_FakeQwenBatchToolClient,
            http_client=_not_found_client(),
        )
    )

    with client.stream(
        "POST",
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "/init"}],
            "tools": [
                {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}},
                        "required": ["pattern"],
                    },
                }
            ],
            "max_tokens": 1024,
            "stream": True,
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    events = _sse_events(body)
    tool_starts = [
        event
        for event in events
        if event["_event"] == "content_block_start" and event["content_block"]["type"] == "tool_use"
    ]
    assert len(tool_starts) == 6
    assert {event["content_block"]["name"] for event in tool_starts} == {"Glob"}
    assert not any(
        event.get("delta", {}).get("type") == "text_delta" and "\"calls\"" in event["delta"].get("text", "")
        for event in events
    )
    assert events[-2]["delta"]["stop_reason"] == "tool_use"


def test_streaming_tool_json_becomes_openai_tool_call_chunk() -> None:
    content = '```tool_json\n{"name":"read_file","args":{"path":"README.md"}}\n```'
    event = json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {event}\n\ndata: [DONE]\n\n".encode("utf-8"),
            request=request,
        )

    client = _client(handler)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "stream": True,
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        },
    ) as response:
        chunks = list(response.iter_text())

    assert response.status_code == 200
    joined = "".join(chunks)
    assert "tool_json" not in joined
    assert '"tool_calls"' in joined
    assert '"finish_reason":"tool_calls"' in joined.replace(" ", "")


def test_tool_bridge_parses_legacy_function_calls_markup() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "skills_list",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '<function_calls>\n<invoke name="skills_list">\n</invoke>\n</function_calls>',
        context,
    )

    assert result.content == ""
    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "skills_list"
    assert result.tool_calls[0].input == {}


def test_tool_bridge_parses_legacy_function_calls_parameters() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "skills_search",
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '<function_calls><invoke name="skills_search"><parameter name="query">github trending</parameter></invoke></function_calls>',
        context,
    )

    assert result.error is None
    assert result.tool_calls[0].name == "skills_search"
    assert result.tool_calls[0].input == {"query": "github trending"}


def test_tool_bridge_parses_xml_tool_call_function_equals_markup() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "skills_list",
                    "description": "List installed skills",
                    "parameters": {"type": "object"},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        "<tool_call>\n<function=skills_list>\n</function>\n</tool_call>",
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "toolu_xml_function_1"
    assert result.tool_calls[0].name == "skills_list"
    assert result.tool_calls[0].input == {}


def test_tool_bridge_parses_bare_xml_function_equals_markup() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "skills_list",
                    "description": "List installed skills",
                    "parameters": {"type": "object"},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        "<function=skills_list>\n</function>",
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "toolu_xml_function_1"
    assert result.tool_calls[0].name == "skills_list"
    assert result.tool_calls[0].input == {}


def test_tool_bridge_treats_plain_json_object_as_text_when_not_tool_shaped() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read files",
                    "parameters": {"type": "object"},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    raw = '{"status":"allowed","reason":"Skill definition is safe and available."}'
    result = parse_tool_response(raw, context)

    assert result.error is None
    assert result.tool_calls == []
    assert result.content == raw


def test_tool_bridge_repairs_skill_name_alias_to_required_skill_parameter() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Skill",
                    "description": "Load a skill",
                    "parameters": {
                        "type": "object",
                        "properties": {"skill": {"type": "string"}},
                        "required": ["skill"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Skill","input":{"skill_name":"using-superpowers"}}]}\n```',
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "Skill"
    assert result.tool_calls[0].input == {"skill": "using-superpowers"}


def test_tool_bridge_repairs_snake_case_alias_to_required_camel_case_parameter() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "TaskUpdate",
                    "description": "Update a task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "taskId": {"type": "string"},
                            "status": {"type": "string"},
                        },
                        "required": ["taskId", "status"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '<|DSML|tool_calls><|DSML|invoke name="TaskUpdate">'
        '<|DSML|parameter name="task_id"><![CDATA[1]]></|DSML|parameter>'
        '<|DSML|parameter name="status"><![CDATA[in_progress]]></|DSML|parameter>'
        "</|DSML|invoke></|DSML|tool_calls>",
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "TaskUpdate"
    assert result.tool_calls[0].input == {"taskId": "1", "status": "in_progress"}


def test_tool_bridge_repairs_description_alias_to_required_subject_parameter() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "TaskCreate",
                    "description": "Create a task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "prompt": {"type": "string"},
                        },
                        "required": ["subject", "prompt"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '<|DSML|tool_calls><|DSML|invoke name="TaskCreate">'
        '<|DSML|parameter name="description"><![CDATA[代码审查 - Python]]></|DSML|parameter>'
        '<|DSML|parameter name="prompt"><![CDATA[审查项目代码并列出问题。]]></|DSML|parameter>'
        "</|DSML|invoke></|DSML|tool_calls>",
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "TaskCreate"
    assert result.tool_calls[0].input == {"subject": "代码审查 - Python", "prompt": "审查项目代码并列出问题。"}


def test_tool_bridge_maps_explore_alias_to_agent_with_focus_as_prompt() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Agent",
                    "description": "Launch an explorer agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["prompt", "description"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '<execute>\n{"name":"Explore","arguments":{"focus":"Project structure and planning workflow."}}\n</execute>',
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "Agent"
    assert result.tool_calls[0].input == {
        "prompt": "Project structure and planning workflow.",
        "description": "Project structure and planning workflow.",
    }


def test_tool_bridge_maps_readfile_alias_to_read_file_path_schema() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"ReadFile","input":{"path":"E:/ProjectX/mindcraft/CLAUDE.md"}}]}\n```',
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "Read"
    assert result.tool_calls[0].input == {"file_path": "E:/ProjectX/mindcraft/CLAUDE.md"}


def test_tool_bridge_repairs_read_path_alias_to_required_file_path() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{"path":"E:/ProjectX/mindcraft/CLAUDE.md"}}]}\n```',
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "Read"
    assert result.tool_calls[0].input == {"file_path": "E:/ProjectX/mindcraft/CLAUDE.md"}


def test_tool_bridge_parses_xml_tool_call_function_equals_parameters() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "skills_search",
                    "description": "Search skills",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        '<tool_call><function=skills_search>{"query":"github trending"}</function></tool_call>',
        context,
    )

    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "skills_search"
    assert result.tool_calls[0].input == {"query": "github trending"}


def test_tool_bridge_parses_tool_code_function_call_markup() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "skills_list",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        "为了准确回答，我将先检查当前已安装?skill 列表。\n\n<tool_code>\nskills_list()\n</tool_code>",
        context,
    )

    assert result.content == ""
    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "skills_list"
    assert result.tool_calls[0].input == {}


def test_tool_bridge_parses_bare_allowed_function_call_line() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "skills_search",
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response(
        "我来搜索一下。\nskills_search(query='github trending')",
        context,
    )

    assert result.content == ""
    assert result.error is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "skills_search"
    assert result.tool_calls[0].input == {"query": "github trending"}


def test_tool_bridge_does_not_parse_unknown_bare_function_call_line() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "skills_list",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )

    result = parse_tool_response("example_helper()", context)

    assert result.content == "example_helper()"
    assert result.tool_calls == []
    assert result.error is None


def test_openai_text_sse_emits_content_before_final_stop_chunk() -> None:
    body = build_tool_call_sse(
        "GitHub skills are available.",
        allowed_tools=set(),
        model="web-model",
    )

    payloads = _openai_sse_payloads(body)

    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    content_chunks = [
        payload
        for payload in payloads
        if payload["choices"][0]["delta"].get("content")
    ]
    assert content_chunks
    assert all(payload["choices"][0]["finish_reason"] is None for payload in content_chunks)
    assert (
        "".join(payload["choices"][0]["delta"]["content"] for payload in content_chunks)
        == "GitHub skills are available."
    )
    assert payloads[-1]["choices"][0]["delta"] == {}
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_openai_text_sse_rejects_dsml_tool_policy_error_without_leak() -> None:
    body = build_tool_call_sse(
        '<|DSML|tool_calls><|DSML|invoke name="Task">'
        '<|DSML|parameter name="description"><![CDATA[inspect]]></|DSML|parameter>'
        "</|DSML|invoke></|DSML|tool_calls>",
        allowed_tools={"Read"},
        model="web-model",
    )

    payloads = _openai_sse_payloads(body)
    text = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
    )

    assert "<|DSML|tool_calls>" not in text
    assert "unknown_tool" in text
    assert "Task" in text
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_startup_bat_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "start_webai_gateway.bat").exists()


def test_root_serves_webai2api_native_spa_when_available(tmp_path: Path) -> None:
    native_dist = tmp_path / "webui" / "dist"
    native_dist.mkdir(parents=True)
    (native_dist / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><head><title>WebAI2API</title></head><body><div id="app"></div></body></html>',
        encoding="utf-8",
    )
    client = TestClient(create_app(config=_config(), native_ui_dir=native_dist, http_client=_not_found_client()))

    response = client.get("/")

    assert response.status_code == 200
    assert "WebAI2API" in response.text
    assert "WebAI Gateway 鎺у埗鍙?" not in response.text


def test_assets_serve_webai2api_native_files(tmp_path: Path) -> None:
    native_dist = tmp_path / "webui" / "dist"
    assets_dir = native_dist / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "index.js").write_text("console.log('native webai2api');", encoding="utf-8")
    client = TestClient(create_app(config=_config(), native_ui_dir=native_dist, http_client=_not_found_client()))

    response = client.get("/assets/index.js")

    assert response.status_code == 200
    assert "native webai2api" in response.text
    assert response.headers["content-type"].startswith("text/javascript") or response.headers["content-type"].startswith("application/javascript")


def test_admin_routes_proxy_to_webai2api_sidecar_root(tmp_path: Path) -> None:
    native_dist = tmp_path / "webui" / "dist"
    native_dist.mkdir(parents=True)
    (native_dist / "index.html").write_text("<html>native</html>", encoding="utf-8")
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"status": "running"}, request=request)

    client = TestClient(
        create_app(
            config=_config(),
            native_ui_dir=native_dist,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    response = client.get("/admin/status?fresh=1", headers={"Authorization": "Bearer webai2api-token"})

    assert response.status_code == 200
    assert response.json() == {"status": "running"}
    assert seen["method"] == "GET"
    assert seen["url"] == "http://upstream.test/admin/status?fresh=1"
    assert seen["authorization"] == "Bearer webai2api-token"


def test_admin_proxy_reports_sidecar_unavailable_without_crashing(tmp_path: Path) -> None:
    native_dist = tmp_path / "webui" / "dist"
    native_dist.mkdir(parents=True)
    (native_dist / "index.html").write_text("<html>native</html>", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sidecar refused connection", request=request)

    client = TestClient(
        create_app(
            config=_config(),
            native_ui_dir=native_dist,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    response = client.get("/admin/status")

    assert response.status_code == 502
    body = response.json()
    assert body["detail"]["code"] == "webai2api_sidecar_unavailable"
    assert body["detail"]["upstream"] == "http://upstream.test/admin/status"


def test_v1_routes_remain_gateway_owned_when_admin_proxy_exists(tmp_path: Path) -> None:
    native_dist = tmp_path / "webui" / "dist"
    native_dist.mkdir(parents=True)
    (native_dist / "index.html").write_text("<html>native</html>", encoding="utf-8")
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"object": "list", "data": [{"id": "sidecar-model", "object": "model"}]}, request=request)

    client = TestClient(
        create_app(
            config=_config(),
            native_ui_dir=native_dist,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    response = client.get("/v1/models", headers=_headers())

    assert response.status_code == 200
    assert seen_paths == ["/v1/models"]
    model_ids = {item["id"] for item in response.json()["data"]}
    assert "sidecar-model" in model_ids
    assert "gpt-instant" in model_ids


def test_onboarding_returns_gateway_providers_and_models(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "deepseek-web",
        {
            "cookie": "ds_session_id=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "sidecar-model", "object": "model"}]},
            request=request,
        )

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    response = client.get("/api/admin/onboarding")

    assert response.status_code == 200
    body = response.json()
    assert body["gateway"]["baseUrl"] == "/v1"
    assert body["gateway"]["apiKey"] == "local-dev-key"
    assert body["gateway"]["defaultModel"] == "web-model"
    assert body["summary"]["providers"] >= 10
    assert body["summary"]["models"] >= 3
    assert body["summary"]["authorizedDirectProviders"] == 1
    assert body["summary"]["webAI2APIProviders"] >= 1
    assert {item["id"] for item in body["models"]} >= {
        "sidecar-model",
        "gpt-instant",
        "qwen-web/qwen3.6-plus",
    }
    assert "deepseek-v4-pro" in {item["id"] for item in body["models"]}
    assert "deepseek-v4-pro[1m]" not in {item["id"] for item in body["models"]}
    assert "deepseek-web/deepseek-chat" not in {item["id"] for item in body["models"]}
    deepseek = next(item for item in body["providers"] if item["id"] == "deepseek-web")
    assert deepseek["credential"]["authorized"] is True
    assert deepseek["availableModels"] == ["deepseek-v4-pro"]
    assert deepseek["modelCount"] == 1
    assert "ds2api" in deepseek["availabilityMessage"]
    assert "session-secret" not in response.text
    assert "bearer-secret" not in response.text


def test_vendored_webai2api_frontend_has_gateway_bridge_page() -> None:
    root = Path(__file__).resolve().parents[1]
    main_js = root / "webui" / "src" / "main.js"
    app_vue = root / "webui" / "src" / "App.vue"
    bridge_vue = root / "webui" / "src" / "components" / "gateway" / "KrisBridge.vue"

    assert main_js.exists()
    assert app_vue.exists()
    assert bridge_vue.exists()
    main_source = main_js.read_text(encoding="utf-8")
    app_source = app_vue.read_text(encoding="utf-8")
    bridge_source = bridge_vue.read_text(encoding="utf-8")
    assert "path: '/'" in main_source
    assert "KrisBridge.vue" in main_source.split("path: '/'", 1)[1].split("}", 1)[0]
    assert "/dashboard" in main_source
    assert "/gateway/kris-bridge" in main_source
    assert "网页登录向导" in app_source
    assert "高级管理" in app_source
    assert "publicRoutes" in app_source
    assert "管理登录" in app_source
    assert "/api/admin/onboarding" in bridge_source
    assert "/api/admin/web-auth/browser/start" in bridge_source
    assert "/admin/restart" in bridge_source
    assert "打开授权浏览器" in bridge_source
    assert "未授权" in bridge_source
    assert "可用模型" in bridge_source
    assert "http://127.0.0.1:8610/v1" in bridge_source
    assert "Claude Code" in bridge_source


def test_admin_root_serves_management_ui() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    response = client.get("/")

    assert response.status_code == 200
    assert 'lang="zh-CN"' in response.text
    assert "<title>WebAI2API</title>" in response.text
    assert "/assets/index.js" in response.text
    assert "WebAI Gateway 鎺у埗鍙?" not in response.text


def test_admin_static_script_uses_chinese_ui_messages() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "配置已保存并生效" in response.text
    assert "令牌已复制。" in response.text
    assert "请在弹出的浏览器里完成 DeepSeek 登录" in response.text
    assert "打开 WebAI2API 登录管理" in response.text
    assert "WebAI2API 原生界面管理" in response.text


def test_static_management_ui_exposes_observation_policy_controls() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    index_response = client.get("/static/index.html")
    script_response = client.get("/static/app.js")

    assert index_response.status_code == 200
    assert script_response.status_code == 200
    assert "observationPolicyPathParts" in index_response.text
    assert "observationPolicyPathGlobs" in index_response.text
    assert "路径列表压缩" in index_response.text
    assert "excludedPathParts" in script_response.text
    assert "excludedPathGlobs" in script_response.text
    assert "pathListMaxItems" in script_response.text
    assert "providerRuntimeTimeoutSeconds" in index_response.text
    assert "providerRuntimePromptMaxChars" in index_response.text
    assert "providerRuntimeResponseLanguage" in index_response.text
    assert "deepseekDs2apiBaseUrl" in index_response.text
    assert "responseLanguage" in script_response.text
    assert "deepseekDs2apiBaseUrl" in script_response.text
    assert "providerRuntime" in script_response.text


def test_static_management_ui_exposes_request_diagnostics() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    index_response = client.get("/static/index.html")
    script_response = client.get("/static/app.js")

    assert index_response.status_code == 200
    assert script_response.status_code == 200
    assert "requestDiagnosticsList" in index_response.text
    assert "request-diagnostics" in script_response.text
    assert "renderToolBridgeSummary" in script_response.text
    assert "toolBridgeError" in script_response.text
    assert "toolBridgeControllerState" in script_response.text
    assert "semanticFinalJudgeMode" in script_response.text
    assert "providerPromptCompacted" in script_response.text


def test_static_management_ui_exposes_auto_research_dashboard() -> None:
    client = _client(lambda request: httpx.Response(200, json={}, request=request))

    index_response = client.get("/static/index.html")
    script_response = client.get("/static/app.js")

    assert index_response.status_code == 200
    assert script_response.status_code == 200
    assert "autoResearchStatus" in index_response.text
    assert "autoResearchRecent" in index_response.text
    assert "autoResearchCandidates" in index_response.text
    assert "自我改进" in index_response.text
    assert "auto-research/status" in script_response.text
    assert "auto-research/candidates" in script_response.text
    assert "renderAutoResearchStatus" in script_response.text


def test_provider_runtime_timeout_round_trips_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.put(
        "/api/admin/config",
        headers=_headers(),
        json={"providerRuntime": {"requestTimeoutSeconds": 240}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["providerRuntime"]["requestTimeoutSeconds"] == 240
    assert body["providerRuntime"]["promptMaxChars"] == 32000
    assert body["providerRuntime"]["responseLanguage"] == "zh-CN"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["providerRuntime"]["requestTimeoutSeconds"] == 240
    assert saved["providerRuntime"]["promptMaxChars"] == 32000
    assert saved["providerRuntime"]["responseLanguage"] == "zh-CN"
    health = client.get("/health", headers=_headers()).json()
    assert health["config"]["providerRuntime"]["requestTimeoutSeconds"] == 240
    assert health["config"]["providerRuntime"]["promptMaxChars"] == 32000
    assert health["config"]["providerRuntime"]["responseLanguage"] == "zh-CN"


def test_provider_runtime_deepseek_ds2api_base_url_round_trips_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.put(
        "/api/admin/config",
        headers=_headers(),
        json={"providerRuntime": {"deepseekDs2apiBaseUrl": "http://127.0.0.1:9555/v1"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["providerRuntime"]["deepseekDs2apiBaseUrl"] == "http://127.0.0.1:9555/v1"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["providerRuntime"]["deepseekDs2apiBaseUrl"] == "http://127.0.0.1:9555/v1"
    health = client.get("/health", headers=_headers()).json()
    assert health["config"]["providerRuntime"]["deepseekDs2apiBaseUrl"] == "http://127.0.0.1:9555/v1"


def test_health_reports_runtime_source_freshness(tmp_path: Path) -> None:
    client = TestClient(create_app(config=_config(), config_path=tmp_path / "config.json", http_client=_not_found_client()))

    response = client.get("/health", headers=_headers())

    assert response.status_code == 200
    runtime = response.json()["runtime"]
    assert runtime["sourceFresh"] is True
    assert runtime["sourceStale"] is False
    assert runtime["latestSource"]["path"].endswith(".py")
    assert runtime["statusText"] == "运行代码是最新的"


def test_health_flags_runtime_when_source_changed_after_start(tmp_path: Path) -> None:
    app = create_app(config=_config(), config_path=tmp_path / "config.json", http_client=_not_found_client())
    app.state.runtime_started_epoch = 0
    client = TestClient(app)

    response = client.get("/health", headers=_headers())

    assert response.status_code == 200
    runtime = response.json()["runtime"]
    assert runtime["sourceFresh"] is False
    assert runtime["sourceStale"] is True
    assert runtime["statusText"] == "源码已更新，请重启 Gateway 让补丁生效"


def test_provider_runtime_response_language_round_trips_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.put(
        "/api/admin/config",
        headers=_headers(),
        json={"providerRuntime": {"responseLanguage": "en-US"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["providerRuntime"]["responseLanguage"] == "en-US"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["providerRuntime"]["responseLanguage"] == "en-US"
    health = client.get("/health", headers=_headers()).json()
    assert health["config"]["providerRuntime"]["responseLanguage"] == "en-US"


def test_admin_config_returns_manageable_local_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.get("/api/admin/config")

    assert response.status_code == 200
    body = response.json()
    assert body["server"]["apiKey"] == "local-dev-key"
    assert body["upstream"]["baseUrl"] == "http://upstream.test/v1"
    assert body["upstream"]["model"] == "web-model"


def test_admin_auto_research_status_returns_replay_summary(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "replays"
    fixture_dir.mkdir()
    (fixture_dir / "blocked_shell.json").write_text(
        json.dumps(
            {
                "id": "blocked_shell",
                "description": "Dashboard sample.",
                "protocol": "anthropic_messages",
                "model": "qwen-coder/qwen-coder-plus",
                "tool_bridge_config": {
                    "mode": "strict",
                    "activationPolicy": "auto",
                    "exposurePolicy": "local-agent",
                    "toolProfile": "auto",
                },
                "input": {
                    "task_text": "审查项目",
                    "messages": [{"role": "user", "content": "/using-superpowers 审查项目"}],
                    "tools": [
                        {
                            "name": "Bash",
                            "description": "Run shell command.",
                            "input_schema": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                                "required": ["command"],
                            },
                        }
                    ],
                    "model_text": '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git status --short"}}]}\n```',
                },
                "expected": {
                    "error": "unsafe_local_shell_command",
                    "tool_calls": [],
                    "warning_contains": None,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            config=_config(),
            config_path=tmp_path / "config.json",
            http_client=_not_found_client(),
            auto_research_fixture_dir=fixture_dir,
        )
    )

    response = client.get("/api/admin/auto-research/status")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["total"] == 1
    assert body["passed"] == 1
    assert body["failed"] == 0
    assert body["failureKinds"] == [{"kind": "unsafe_local_shell_command", "count": 1}]
    assert body["recent"][0]["id"] == "blocked_shell"


def test_admin_provider_smoke_reports_qwen_tool_loop_without_secrets(tmp_path: Path) -> None:
    class SmokeQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            text = json.dumps(payload, ensure_ascii=False)
            if "tool_result" in text or "tool_call_id" in text or "北京今天晴" in text:
                return _openai_response("北京今天晴，气温 22°C。")
            if "PROVIDER_SMOKE_OK" in text:
                return _openai_response("PROVIDER_SMOKE_OK")
            if "get_weather" in text:
                return _openai_response('get_weather(city="Beijing")')
            return _openai_response("unexpected smoke prompt")

    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "",
            "userAgent": "Chrome Test",
            "metadata": {"sessionToken": "session-token"},
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            qwen_client_factory=SmokeQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post("/api/admin/provider-smoke/qwen")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "qwen"
    assert body["authorized"] is True
    assert body["ok"] is True
    assert body["passed"] == 5
    result_ids = {item["id"] for item in body["results"]}
    assert result_ids == {"models", "openai_text", "openai_tool_use", "anthropic_tool_use", "anthropic_tool_result"}
    tool_use = next(item for item in body["results"] if item["id"] == "anthropic_tool_use")
    assert tool_use["ok"] is True
    assert tool_use["detail"]["name"] == "get_weather"
    assert tool_use["detail"]["input"] == {"city": "Beijing"}
    assert "session-secret" not in response.text


def test_admin_provider_smoke_returns_step_failures_without_secrets(tmp_path: Path) -> None:
    class FailingQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("upstream failed with qwen_session=session-secret")

    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "",
            "userAgent": "Chrome Test",
            "metadata": {"sessionToken": "session-token"},
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            qwen_client_factory=FailingQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post("/api/admin/provider-smoke/qwen")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "qwen"
    assert body["authorized"] is True
    assert body["ok"] is False
    assert body["passed"] == 1
    failed = [item for item in body["results"] if not item["ok"]]
    assert len(failed) == 4
    assert all("message" in item for item in failed)
    assert "session-secret" not in response.text
    assert "qwen_session" not in response.text


def test_admin_provider_smoke_returns_auth_failure_when_provider_not_authorized(tmp_path: Path) -> None:
    client = TestClient(create_app(config=_config(), credential_store=CredentialStore(tmp_path / "credentials"), http_client=_not_found_client()))

    response = client.post("/api/admin/provider-smoke/qwen")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "qwen"
    assert body["authorized"] is False
    assert body["ok"] is False
    assert body["passed"] == 0
    assert body["results"][0]["id"] == "auth"
    assert body["results"][0]["ok"] is False


def test_admin_config_update_persists_and_reloads_auth_token(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"object": "list", "data": [{"id": "new-model", "object": "model"}]}, request=request)

    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.put(
        "/api/admin/config",
        json={
            "server": {"apiKey": "new-local-key"},
            "upstream": {
                "baseUrl": "http://changed-upstream.test/v1",
                "apiKey": "upstream-key",
                "model": "new-model",
                "toolMode": "prompt",
            },
        },
    )

    assert response.status_code == 200
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["server"]["apiKey"] == "new-local-key"
    assert saved["upstream"]["baseUrl"] == "http://changed-upstream.test/v1"
    assert client.get("/v1/models", headers=_headers()).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer new-local-key"}).status_code == 200
    assert seen["path"] == "/v1/models"


def test_tool_bridge_observation_policy_round_trips_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.put(
        "/api/admin/config",
        headers=_headers(),
        json={
            "providerRuntime": {"nativeWebSearchPolicy": "force"},
            "tool_bridge": {
                "activationPolicy": "always",
                "observationPolicy": {
                    "excludedPathParts": ["third_party", "vendor_cache"],
                    "excludedPathGlobs": ["**/generated/**"],
                    "pathListMaxItems": 12,
                }
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["providerRuntime"]["nativeWebSearchPolicy"] == "force"
    assert body["tool_bridge"]["activationPolicy"] == "always"
    policy = body["tool_bridge"]["observationPolicy"]
    assert policy["excludedPathParts"] == ["third_party", "vendor_cache"]
    assert policy["excludedPathGlobs"] == ["**/generated/**"]
    assert policy["pathListMaxItems"] == 12

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["providerRuntime"]["nativeWebSearchPolicy"] == "force"
    assert saved["tool_bridge"]["activationPolicy"] == "always"
    assert saved["tool_bridge"]["observationPolicy"] == policy
    health = client.get("/health", headers=_headers()).json()
    assert health["config"]["providerRuntime"]["nativeWebSearchPolicy"] == "force"
    assert health["config"]["tool_bridge"]["activationPolicy"] == "always"
    assert health["config"]["tool_bridge"]["observationPolicy"] == policy


def test_tool_profile_config_round_trips_public_admin_update(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.put(
        "/api/admin/config",
        headers=_headers(),
        json={
            "tool_bridge": {
                "toolProfile": "agent",
                "readonlyToolNames": ["Read", "Glob", "Grep", "mcp__repo__search"],
                "writeToolNames": ["Edit", "Write"],
                "shellToolNames": ["Bash"],
            }
        },
    )

    assert response.status_code == 200
    admin_body = response.json()["tool_bridge"]
    assert admin_body["toolProfile"] == "agent"
    assert admin_body["readonlyToolNames"] == ["Read", "Glob", "Grep", "mcp__repo__search"]
    assert admin_body["writeToolNames"] == ["Edit", "Write"]
    assert admin_body["shellToolNames"] == ["Bash"]

    saved = json.loads(config_path.read_text(encoding="utf-8"))["tool_bridge"]
    assert saved["toolProfile"] == "agent"
    assert saved["readonlyToolNames"] == ["Read", "Glob", "Grep", "mcp__repo__search"]
    assert saved["writeToolNames"] == ["Edit", "Write"]
    assert saved["shellToolNames"] == ["Bash"]
    loaded = load_config(config_path)
    assert loaded.tool_bridge.tool_profile == "agent"
    assert loaded.tool_bridge.readonly_tool_names == ("Read", "Glob", "Grep", "mcp__repo__search")
    assert loaded.tool_bridge.write_tool_names == ("Edit", "Write")
    assert loaded.tool_bridge.shell_tool_names == ("Bash",)

    public_body = client.get("/health", headers=_headers()).json()["config"]["tool_bridge"]
    assert public_body["toolProfile"] == "agent"
    assert public_body["readonlyToolNames"] == ["Read", "Glob", "Grep", "mcp__repo__search"]
    assert public_body["writeToolNames"] == ["Edit", "Write"]
    assert public_body["shellToolNames"] == ["Bash"]


def test_tool_bridge_config_exposes_semantic_final_judge(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    default_body = client.get("/health", headers=_headers()).json()["config"]["tool_bridge"]
    assert default_body["semanticFinalJudge"] == "off"

    response = client.put(
        "/api/admin/config",
        headers=_headers(),
        json={"tool_bridge": {"semanticFinalJudge": "shadow"}},
    )

    assert response.status_code == 200
    assert response.json()["tool_bridge"]["semanticFinalJudge"] == "shadow"
    saved = json.loads(config_path.read_text(encoding="utf-8"))["tool_bridge"]
    assert saved["semanticFinalJudge"] == "shadow"
    loaded = load_config(config_path)
    assert loaded.tool_bridge.semantic_final_judge == "shadow"
    public_body = client.get("/health", headers=_headers()).json()["config"]["tool_bridge"]
    assert public_body["semanticFinalJudge"] == "shadow"


def test_admin_token_rotation_persists_and_updates_gateway_auth(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    client = TestClient(create_app(config=_config(), config_path=config_path, http_client=_not_found_client()))

    response = client.post("/api/admin/token/rotate")

    assert response.status_code == 200
    token = response.json()["server"]["apiKey"]
    assert token.startswith("wg_")
    assert len(token) > 20
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["server"]["apiKey"] == token
    assert client.get("/v1/models", headers=_headers()).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_web_auth_providers_list_deepseek_without_secrets(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "deepseek-web",
        {
            "cookie": "ds_session_id=session-secret; HWSID=hws-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(create_app(config=_config(), credential_store=store, http_client=_not_found_client()))

    response = client.get("/api/admin/web-auth/providers")

    assert response.status_code == 200
    body = response.json()
    deepseek = next(item for item in body["providers"] if item["id"] == "deepseek-web")
    assert deepseek["name"] == "DeepSeek Web"
    assert deepseek["status"] == "available"
    assert deepseek["credential"]["authorized"] is True
    assert deepseek["credential"]["fields"] == {"cookie": True, "bearer": True, "userAgent": True}
    assert deepseek["availableModels"] == ["deepseek-v4-pro"]
    assert deepseek["modelCount"] == 1
    assert deepseek["advertiseModels"] is True
    assert "ds2api" in deepseek["availabilityMessage"]
    assert "session-secret" not in response.text
    assert "bearer-secret" not in response.text


def test_web_auth_providers_cover_webai2api_supported_sites(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(tmp_path / "credentials"),
            http_client=_not_found_client(),
        )
    )

    response = client.get("/api/admin/web-auth/providers")

    assert response.status_code == 200
    providers = {item["id"]: item for item in response.json()["providers"]}
    assert {
        "lmarena",
        "gemini-biz",
        "nano-banana-free",
        "zai",
        "gemini",
        "zenmux",
        "chatgpt",
        "qwen",
        "qwen-cn",
        "deepseek-web",
        "sora",
        "google-flow",
        "doubao",
    }.issubset(providers)
    assert providers["lmarena"]["capabilities"] == {"text": True, "image": True, "video": False}
    assert providers["gemini-biz"]["capabilities"] == {"text": True, "image": True, "video": True}
    assert providers["nano-banana-free"]["capabilities"] == {"text": False, "image": True, "video": False}
    assert providers["sora"]["capabilities"] == {"text": False, "image": False, "video": True}
    assert providers["deepseek-web"]["route"] == "direct"
    assert providers["qwen"]["route"] == "direct"
    assert providers["qwen"]["toolBridge"] == "strict"
    assert providers["qwen"]["supportsNativeTools"] is False
    assert providers["qwen"]["preferredProtocol"] == "openai"
    assert providers["qwen-cn"]["route"] == "webai2api"
    assert providers["chatgpt"]["route"] == "webai2api"
    assert "chatgpt_text" in providers["chatgpt"]["adapters"]
    assert providers["qwen"]["loginUrl"] == "https://chat.qwen.ai/"
    assert providers["qwen-cn"]["loginUrl"] == "https://www.qianwen.com/"
    assert providers["qwen"]["capabilities"] == {"text": True, "image": False, "video": False}
    assert "qwen_web" in providers["qwen"]["adapters"]
    assert "qwen_cn_web" in providers["qwen-cn"]["adapters"]
    assert "qwen-web/qwen3.6-max-preview" in providers["qwen"]["models"]
    assert "qwen-web/qwen3.6-plus" in providers["qwen"]["models"]
    assert "qwen-web/qwen3-max" in providers["qwen"]["models"]
    assert "qwen-web/qwen3.6-max" not in providers["qwen"]["models"]
    assert "Qwen3.5-Plus" in providers["qwen-cn"]["models"]


def test_qwen_cookie_only_credential_is_not_authorized() -> None:
    summary = credential_summary(
        "qwen",
        {
            "provider": "qwen",
            "cookie": "visitor_id=visitor; device_id=device",
            "bearer": "",
            "userAgent": "Chrome Test",
            "metadata": {"sessionToken": ""},
            "updatedAt": "2026-04-26T00:00:00+00:00",
        },
    )

    assert summary["authorized"] is False
    assert summary["fields"] == {"cookie": True, "bearer": False, "userAgent": True}


def test_qwen_session_token_credential_is_authorized() -> None:
    summary = credential_summary(
        "qwen",
        {
            "provider": "qwen",
            "cookie": "visitor_id=visitor; qwen_session=session-token",
            "bearer": "",
            "userAgent": "Chrome Test",
            "metadata": {"sessionToken": "session-token"},
            "updatedAt": "2026-04-26T00:00:00+00:00",
        },
    )

    assert summary["authorized"] is True


def test_deepseek_cookie_only_credential_is_not_authorized() -> None:
    summary = credential_summary(
        "deepseek-web",
        {
            "provider": "deepseek-web",
            "cookie": "ds_session_id=session-secret; HWSID=hws-secret",
            "bearer": "",
            "userAgent": "Chrome Test",
            "updatedAt": "2026-05-01T00:00:00+00:00",
        },
    )

    assert summary["authorized"] is False
    assert summary["fields"] == {"cookie": True, "bearer": False, "userAgent": True}


def test_deepseek_store_rejects_cookie_only_credential(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")

    with pytest.raises(ValueError, match="DeepSeek.*token"):
        store.save(
            "deepseek-web",
            {
                "cookie": "ds_session_id=session-secret; HWSID=hws-secret",
                "bearer": "",
                "userAgent": "Chrome Test",
            },
        )

    assert store.get("deepseek-web") is None


def test_qwen_store_rejects_visitor_cookie_only_credential(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")

    with pytest.raises(ValueError, match="Qwen"):
        store.save(
            "qwen",
            {
                "cookie": "visitor_id=visitor; device_id=device",
                "bearer": "",
                "userAgent": "Chrome Test",
                "metadata": {"sessionToken": ""},
            },
        )

    assert store.get("qwen") is None


def test_onboarding_marks_existing_qwen_cookie_only_file_unauthorized(tmp_path: Path) -> None:
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()
    (credential_dir / "qwen.json").write_text(
        json.dumps(
            {
                "provider": "qwen",
                "cookie": "visitor_id=visitor; device_id=device",
                "bearer": "",
                "userAgent": "Chrome Test",
                "metadata": {"sessionToken": ""},
                "updatedAt": "2026-04-26T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(credential_dir),
            http_client=_not_found_client(),
        )
    )

    response = client.get("/api/admin/onboarding")

    assert response.status_code == 200
    qwen = next(item for item in response.json()["providers"] if item["id"] == "qwen")
    assert qwen["credential"]["authorized"] is False
    assert qwen["credential"]["fields"]["cookie"] is True
    assert "visitor" not in response.text


class _FakeQwenContext:
    def __init__(self, cookies: list[dict[str, str]]) -> None:
        self._cookies = cookies

    async def cookies(self, urls):
        return self._cookies


class _FakeQwenPage:
    async def evaluate(self, script: str) -> str:
        return "Chrome Test"


def test_read_qwen_credential_ignores_visitor_cookies() -> None:
    credential = asyncio.run(
        _read_qwen_credential(
            _FakeQwenContext(
                [
                    {"name": "visitor_id", "value": "visitor"},
                    {"name": "device_id", "value": "device"},
                ]
            ),
            _FakeQwenPage(),
        )
    )

    assert credential is None


def test_read_qwen_credential_accepts_bearer_login_state() -> None:
    credential = asyncio.run(
        _read_qwen_credential(
            _FakeQwenContext([{"name": "visitor_id", "value": "visitor"}]),
            _FakeQwenPage(),
            bearer_seen="bearer-token",
        )
    )

    assert credential is not None
    assert credential["bearer"] == "bearer-token"
    assert credential["metadata"]["sessionToken"] == "bearer-token"


def test_read_qwen_coder_credential_uses_coder_origin() -> None:
    context = _FakeQwenContext([{"name": "qwen_session", "value": "coder-session"}])

    credential = asyncio.run(
        _read_qwen_credential(
            context,
            _FakeQwenPage(),
            origins=("https://coder.qwen.ai", "https://qwen.ai"),
        )
    )

    assert credential is not None
    assert credential["metadata"]["sessionToken"] == "coder-session"


def test_qwen_coder_auth_service_captures_browser_login(monkeypatch: pytest.MonkeyPatch) -> None:
    progress: list[str] = []
    seen: dict[str, Any] = {}

    class FakePage:
        def on(self, event: str, handler: Any) -> None:
            seen["event"] = event

        async def goto(self, url: str) -> None:
            seen["goto"] = url

        async def evaluate(self, script: str) -> str:
            return "Chrome Test"

    class FakeContext:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        async def cookies(self, urls: Any) -> list[dict[str, str]]:
            seen["cookie_urls"] = urls
            return [{"name": "qwen_session", "value": "coder-session"}]

    class FakeBrowser:
        def __init__(self) -> None:
            self.contexts = [FakeContext()]

        async def new_context(self) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            seen["closed"] = True

    class FakeChromium:
        async def connect_over_cdp(self, cdp_url: str) -> FakeBrowser:
            seen["cdp_url"] = cdp_url
            return FakeBrowser()

    class FakePlaywrightManager:
        async def __aenter__(self) -> Any:
            return types.SimpleNamespace(chromium=FakeChromium())

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

    fake_async_api = types.SimpleNamespace(async_playwright=lambda: FakePlaywrightManager())
    monkeypatch.setitem(sys.modules, "playwright", types.SimpleNamespace(async_api=fake_async_api))
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_async_api)

    credential = asyncio.run(
        DeepSeekWebAuthService().capture(
            "qwen-coder",
            "http://127.0.0.1:9222",
            progress=progress.append,
            timeout_seconds=3,
        )
    )

    assert seen["goto"] == "https://coder.qwen.ai/"
    assert "https://coder.qwen.ai" in seen["cookie_urls"]
    assert credential["metadata"]["sessionToken"] == "coder-session"
    assert credential["userAgent"] == "Chrome Test"
    assert seen["closed"] is True
    assert any("Qwen Coder" in item for item in progress)


class _FakeAuthService:
    async def capture(self, provider_id: str, cdp_url: str, progress):
        progress("已连接授权浏览器，正在等待登录状?")
        return {
            "cookie": "ds_session_id=session-secret; HWSID=hws-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        }


class _FakeVisitorQwenAuthService:
    async def capture(self, provider_id: str, cdp_url: str, progress):
        progress("等待 Qwen 登录?")
        return {
            "cookie": "visitor_id=visitor; device_id=device",
            "bearer": "",
            "userAgent": "Chrome Test",
            "metadata": {"sessionToken": ""},
        }


def test_qwen_auth_job_does_not_succeed_with_visitor_cookies(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            web_auth_service=_FakeVisitorQwenAuthService(),
            run_auth_jobs_inline=True,
            http_client=_not_found_client(),
        )
    )

    response = client.post("/api/admin/web-auth/jobs", json={"provider": "qwen", "cdpUrl": "http://127.0.0.1:9222"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "Qwen" in body["message"]
    assert store.get("qwen") is None


class _FakeBrowserLauncher:
    def start(self, provider_id: str, cdp_url: str) -> dict[str, Any]:
        return {
            "provider": provider_id,
            "cdpUrl": cdp_url,
            "loginUrl": "https://chat.deepseek.com",
            "started": True,
            "message": "授权浏览器已启动",
        }


def test_web_auth_job_captures_and_persists_credentials(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            web_auth_service=_FakeAuthService(),
            run_auth_jobs_inline=True,
            http_client=_not_found_client(),
        )
    )

    response = client.post("/api/admin/web-auth/jobs", json={"provider": "deepseek-web", "cdpUrl": "http://127.0.0.1:9222"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["credential"]["authorized"] is True
    assert "bearer-secret" not in response.text
    assert store.get("deepseek-web")["bearer"] == "bearer-secret"


def test_web_auth_browser_start_returns_beginner_friendly_action() -> None:
    client = TestClient(create_app(config=_config(), browser_launcher=_FakeBrowserLauncher(), http_client=_not_found_client()))

    response = client.post("/api/admin/web-auth/browser/start", json={"provider": "deepseek-web"})

    assert response.status_code == 200
    body = response.json()
    assert body["started"] is True
    assert body["loginUrl"] == "https://chat.deepseek.com"
    assert "授权浏览器已启动" in body["message"]


class _FakeDeepSeekClient:
    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.credential["bearer"] == "bearer-secret"
        return {
            "id": "deepseek-web-test",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "来自网页登录模型"},
                }
            ],
        }


class _FakeQwenClient:
    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.credential["cookie"] == "qwen_session=session-secret"
        return {
            "id": "qwen-web-test",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "来自 Qwen 网页模型"},
                }
            ],
        }


class _FakeQwenCoderClient:
    captured_payload: dict[str, Any] = {}

    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        _FakeQwenCoderClient.captured_payload = payload
        assert self.credential["cookie"] == "qwen_session=session-secret"
        return {
            "id": "qwen-coder-test",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "来自 Qwen Coder 网页模型"},
                }
            ],
        }


class _FakeQwenToolClient:
    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "qwen-web/qwen3.6-plus"
        assert "tools" not in payload
        assert "tool_choice" not in payload
        return _openai_response('```tool_json\n{"name":"list_local_files","args":{"path":"LOCAL_WORKSPACE"}}\n```')


class _FakeQwenBatchToolClient:
    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert payload["model"] == "qwen-web/qwen3.6-plus"
        assert "tools" not in payload
        assert "tool_choice" not in payload
        calls = [
            {"id": "call_1", "name": "Glob", "input": {"pattern": "*.md"}},
            {"id": "call_2", "name": "Glob", "input": {"pattern": "pyproject.toml"}},
            {"id": "call_3", "name": "Glob", "input": {"pattern": "requirements*.txt"}},
            {"id": "call_4", "name": "Glob", "input": {"pattern": ".cursorrules"}},
            {"id": "call_5", "name": "Glob", "input": {"pattern": ".cursor/rules/**"}},
            {"id": "call_6", "name": "Glob", "input": {"pattern": ".github/copilot-instructions.md"}},
        ]
        return _openai_response(json.dumps({"calls": calls}, separators=(",", ":")))


class _FakeQwenMultimodalClient:
    captured_payload: dict[str, Any] = {}

    def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
        self.credential = credential

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        _FakeQwenMultimodalClient.captured_payload = payload
        return _openai_response("我看到了图片")


def test_deepseek_web_chat_uses_saved_credentials(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "deepseek-web",
        {
            "cookie": "ds_session_id=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            deepseek_client_factory=_FakeDeepSeekClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "deepseek-web/deepseek-chat", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "来自网页登录模型"


def test_deepseek_official_v4_pro_id_uses_saved_web_credentials(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingDeepSeekClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            seen["credential"] = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return {
                "id": "deepseek-web-test",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "来自 V4 Pro 网页模型"},
                    }
                ],
            }

    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "deepseek-web",
        {
            "cookie": "ds_session_id=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            deepseek_client_factory=CapturingDeepSeekClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "deepseek-v4-pro[1m]", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    assert seen["credential"]["bearer"] == "bearer-secret"
    assert seen["payload"]["model"] == "deepseek-v4-pro[1m]"
    assert response.json()["choices"][0]["message"]["content"] == "来自 V4 Pro 网页模型"


def test_deepseek_openai_tool_calls_are_forwarded_to_ds2api(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class ToolDeepSeekClient:
        def __init__(self, credential: dict[str, Any], **_: Any) -> None:
            seen["credential"] = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return {
                "id": "deepseek-tool-test",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_weather_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"},
                                }
                            ],
                        },
                    }
                ],
            }

    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "deepseek-web",
        {
            "cookie": "ds_session_id=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            deepseek_client_factory=ToolDeepSeekClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "请调?get_weather"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查询天气",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
    )

    assert response.status_code == 200
    assert "tools" in seen["payload"]
    assert "tool_choice" in seen["payload"]
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_deepseek_anthropic_tool_use_round_trip_uses_native_tool_calls(tmp_path: Path) -> None:
    class ToolDeepSeekClient:
        def __init__(self, credential: dict[str, Any], **_: Any) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": "deepseek-tool-test",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_weather_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"},
                                }
                            ],
                        },
                    }
                ],
            }

    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "deepseek-web",
        {
            "cookie": "ds_session_id=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            deepseek_client_factory=ToolDeepSeekClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers=_headers(),
        json={
            "model": "deepseek-v4-pro",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "请调?get_weather"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "查询天气",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        },
    )

    assert response.status_code == 200
    block = response.json()["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "get_weather"
    assert block["input"] == {"city": "Beijing"}


def test_qwen_web_client_uses_qwen_chat_api_and_parses_stream() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/chats/new":
            return httpx.Response(200, json={"data": {"id": "chat-1"}}, request=request)
        if request.url.path == "/api/v2/chat/completions":
            seen["body"] = json.loads(request.content.decode("utf-8"))
            seen["query"] = dict(request.url.params)
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"你好"}}]}\n\ndata: {"output":{"text":"，Qwen"}}\n\ndata: [DONE]\n',
                request=request,
            )
        return httpx.Response(404, request=request)

    client = QwenWebClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = client.chat_completions(
        {"model": "qwen-web/qwen3.6-plus", "messages": [{"role": "user", "content": "你好"}]}
    )

    assert seen["query"]["chat_id"] == "chat-1"
    assert seen["body"]["model"] == "qwen3.6-plus"
    assert seen["body"]["messages"][0]["feature_config"]["thinking_enabled"] is False
    assert response["choices"][0]["message"]["content"] == "你好，Qwen"


def test_qwen_web_client_records_task_state_snapshot_diagnostics() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/chats/new":
            return httpx.Response(200, json={"data": {"id": "chat-1"}}, request=request)
        if request.url.path == "/api/v2/chat/completions":
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    task_update = (
        '<|DSML|tool_calls><|DSML|invoke name="TaskUpdate">'
        '<|DSML|parameter name="task_id"><![CDATA[4]]></|DSML|parameter>'
        '<|DSML|parameter name="status"><![CDATA[in_progress]]></|DSML|parameter>'
        '<|DSML|parameter name="description"><![CDATA[Extract Feishu client]]></|DSML|parameter>'
        "</|DSML|invoke></|DSML|tool_calls>"
    )
    client = QwenWebClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        prompt_max_chars=2200,
    )

    response = client.chat_completions(
        {
            "model": "qwen-web/qwen3.6-plus",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "system bootstrap\n"
                        + ("skill listing entry\n" * 600)
                        + "\nYou are using WebAI Gateway's strict tool bridge.\n"
                        + "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
                    ),
                },
                {"role": "assistant", "content": "Assistant requested tool calls:\n" + task_update},
                {"role": "tool", "content": "Task 4 updated"},
                {"role": "user", "content": "Continue all four tasks."},
            ],
        }
    )

    sent_prompt = seen["body"]["messages"][0]["content"]
    assert response["choices"][0]["message"]["content"] == "ok"
    assert "# WebAI Gateway preserved task state" in sent_prompt
    assert client.last_diagnostic["prompt_task_state_preserved"] is True
    assert client.last_diagnostic["prompt_task_count"] == 1
    assert client.last_diagnostic["prompt_recent_tool_call_count"] >= 1


def test_qwen_web_client_records_layered_compaction_diagnostics() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/chats/new":
            return httpx.Response(200, json={"data": {"id": "chat-1"}}, request=request)
        if request.url.path == "/api/v2/chat/completions":
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "system bootstrap\n"
                + ("tool schema detail\n" * 700)
                + "You are using WebAI Gateway's strict tool bridge.\n"
                + "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
            ),
        }
    ]
    for index in range(80):
        messages.extend(
            [
                {"role": "user", "content": f"old request {index}\n" + ("old observation\n" * 100)},
                {"role": "assistant", "content": f"Assistant requested tool calls:\nRead(path=old_{index}.py)"},
                {"role": "tool", "content": f"old_{index}.py content\n" + ("legacy content\n" * 100)},
            ]
        )
    messages.append({"role": "user", "content": "Continue. LATEST_DIAGNOSTIC_SENTINEL"})
    client = QwenWebClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        prompt_max_chars=48000,
    )

    response = client.chat_completions({"model": "qwen-web/qwen3.6-plus", "messages": messages})

    sent_prompt = seen["body"]["messages"][0]["content"]
    assert response["choices"][0]["message"]["content"] == "ok"
    assert len(sent_prompt) <= 36000
    assert "LATEST_DIAGNOSTIC_SENTINEL" in sent_prompt
    assert client.last_diagnostic["prompt_compaction_strategy"] == "ds2api_layered_history"
    assert client.last_diagnostic["prompt_history_entry_count"] > client.last_diagnostic["prompt_latest_entry_count"]
    assert client.last_diagnostic["prompt_latest_entry_count"] >= 3


def test_qwen_web_client_raises_on_qwen_success_false_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/chats/new":
            return httpx.Response(200, json={"data": {"id": "chat-1"}}, request=request)
        if request.url.path == "/api/v2/chat/completions":
            return httpx.Response(
                200,
                json={"success": False, "data": {"code": "Not_Found", "details": "Model not found"}},
                request=request,
            )
        return httpx.Response(404, request=request)

    client = QwenWebClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError, match="Not_Found.*Model not found"):
        client.chat_completions(
            {"model": "qwen-web/qwen3.6-max", "messages": [{"role": "user", "content": "hello"}]}
        )


def test_qwen_web_helpers_normalize_and_parse_common_stream_shapes() -> None:
    text = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            'data: {"data":{"content":"B"}}',
            '{"answer":"C"}',
            "data: [DONE]",
        ]
    )

    assert normalize_qwen_model("qwen-web/qwen3.5-plus") == "qwen3.5-plus"
    assert parse_qwen_stream_text(text) == "ABC"


def test_qwen_messages_compacts_oversized_prompt_for_web_provider() -> None:
    huge_system = "system bootstrap\n" + ("skill listing entry\n" * 4000)
    tool_protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools: Read, Glob, Write.\n"
        "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
    )
    prompt, files = qwen_messages_to_prompt_and_files(
        [
            {"role": "system", "content": huge_system + "\n\n" + tool_protocol},
            {"role": "user", "content": "Please analyze this codebase and create a CLAUDE.md file."},
        ],
        max_prompt_chars=1200,
    )

    assert files == []
    assert len(prompt) <= 1200
    assert "Prompt content was compacted" in prompt
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert "Please analyze this codebase" in prompt
    assert prompt.count("skill listing entry") < 10


def test_qwen_messages_prepend_stateless_web_api_guard() -> None:
    prompt, files = qwen_messages_to_prompt_and_files(
        [{"role": "user", "content": "只回复 OK"}],
        max_prompt_chars=4000,
    )

    assert files == []
    assert prompt.startswith("System: You are serving a stateless WebAI Gateway API request.")
    assert "不要引用网页端旧会话" in prompt
    assert "User: 只回复 OK" in prompt


def test_qwen_messages_compaction_preserves_tool_bridge_protocol_with_large_later_history() -> None:
    huge_system_prefix = "system bootstrap\n" + ("skill listing entry\n" * 500)
    tool_protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools: Read, Glob, Write.\n"
        "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
    )
    large_later_history = ("dependency path .venv/Lib/site-packages/pkg/module.py\n" * 300) + "LATEST_SENTINEL"

    prompt, files = qwen_messages_to_prompt_and_files(
        [
            {"role": "system", "content": huge_system_prefix + "\n\n" + tool_protocol},
            {"role": "user", "content": large_later_history},
        ],
        max_prompt_chars=1600,
    )

    assert files == []
    assert len(prompt) <= 1600
    assert "Prompt content was compacted" in prompt
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert "Required tool-call format" in prompt
    assert "<|DSML|tool_calls>" in prompt
    assert "LATEST_SENTINEL" in prompt


def test_qwen_messages_compaction_uses_ds2api_history_continuation() -> None:
    huge_system_prefix = "system bootstrap\n" + ("skill listing entry\n" * 500)
    tool_protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools: Read, Glob, Write.\n"
        "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
    )

    prompt, files = qwen_messages_to_prompt_and_files(
        [
            {"role": "system", "content": huge_system_prefix + "\n\n" + tool_protocol},
            {"role": "assistant", "content": "Assistant requested tool calls: Read({\"file_path\":\"README.md\"})"},
            {"role": "tool", "content": "README content"},
            {"role": "user", "content": "Continue the implementation. LATEST_SENTINEL"},
        ],
        max_prompt_chars=1800,
    )

    assert files == []
    assert len(prompt) <= 1800
    assert "# DS2API_HISTORY.txt" in prompt
    assert "Prior conversation history and tool progress." in prompt
    assert "Continue from the latest state in the provided DS2API_HISTORY.txt context" in prompt
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert "<|DSML|tool_calls>" in prompt
    assert "LATEST_SENTINEL" in prompt
    assert "=== CURRENT USER REQUEST (highest priority) ===" in prompt
    assert "Do not summarize DS2API_HISTORY.txt" in prompt
    assert prompt.rfind("LATEST_SENTINEL") > prompt.rfind("Continue from the latest state")


def test_qwen_messages_compaction_does_not_promote_tool_history_without_task_updates() -> None:
    huge_system_prefix = "system bootstrap\n" + ("skill listing entry\n" * 500)
    tool_protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools: Skill, Glob, AskUserQuestion.\n"
        "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
    )

    prompt, files = qwen_messages_to_prompt_and_files(
        [
            {"role": "system", "content": huge_system_prefix + "\n\n" + tool_protocol},
            {
                "role": "assistant",
                "content": (
                    'Assistant requested tool calls:\n<|DSML|tool_calls><|DSML|invoke name="Skill">'
                    '<|DSML|parameter name="skill"><![CDATA[caveman]]></|DSML|parameter>'
                    "</|DSML|invoke></|DSML|tool_calls>"
                ),
            },
            {"role": "tool", "content": "Caveman mode disabled"},
            {"role": "user", "content": "现在处理一个全新的问题。LATEST_NEW_TASK_SENTINEL"},
        ],
        max_prompt_chars=1800,
    )

    assert files == []
    assert "# WebAI Gateway preserved task state" not in prompt
    assert "Recent tool calls:" not in prompt
    assert "=== CURRENT USER REQUEST (highest priority) ===" in prompt
    assert prompt.rfind("LATEST_NEW_TASK_SENTINEL") > prompt.rfind("Continue from the latest state")


def test_qwen_messages_current_request_skips_skill_control_text() -> None:
    huge_system_prefix = "system bootstrap\n" + ("skill listing entry\n" * 500)
    tool_protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools: Skill, Glob, Read.\n"
        "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
    )

    prompt, files = qwen_messages_to_prompt_and_files(
        [
            {"role": "system", "content": huge_system_prefix + "\n\n" + tool_protocol},
            {"role": "user", "content": "/using-superpowers 审查当前项目的代码，看看有什么需要改进的"},
            {
                "role": "assistant",
                "content": (
                    'Assistant requested tool calls:\n<|DSML|tool_calls><|DSML|invoke name="Skill">'
                    '<|DSML|parameter name="skill"><![CDATA[using-superpowers]]></|DSML|parameter>'
                    "</|DSML|invoke></|DSML|tool_calls>"
                ),
            },
            {
                "role": "tool",
                "content": (
                    "Loaded using-superpowers. Do not request the same Skill again. "
                    "Use the loaded skill instructions already in the conversation, then continue the original user task."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Do not request the same Skill again. Use the loaded skill instructions already in the conversation, "
                    "then continue the original user task."
                ),
            },
        ],
        max_prompt_chars=2200,
    )

    assert files == []
    marker = "=== CURRENT USER REQUEST (highest priority) ==="
    assert marker in prompt
    current_block = prompt[prompt.rfind(marker) :]
    assert "审查当前项目的代码，看看有什么需要改进的" in current_block
    assert "/using-superpowers" not in current_block
    assert "Do not request the same Skill again" not in current_block


def test_qwen_messages_compaction_preserves_task_state_snapshot() -> None:
    huge_system_prefix = "system bootstrap\n" + ("skill listing entry\n" * 600)
    tool_protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools: TaskUpdate, Read, Write.\n"
        "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
    )
    task_update_done = (
        '<|DSML|tool_calls><|DSML|invoke name="TaskUpdate">'
        '<|DSML|parameter name="task_id"><![CDATA[1]]></|DSML|parameter>'
        '<|DSML|parameter name="status"><![CDATA[completed]]></|DSML|parameter>'
        '<|DSML|parameter name="description"><![CDATA[Create config module]]></|DSML|parameter>'
        "</|DSML|invoke></|DSML|tool_calls>"
    )
    task_update_pending = (
        '<|DSML|tool_calls><|DSML|invoke name="TaskUpdate">'
        '<|DSML|parameter name="task_id"><![CDATA[4]]></|DSML|parameter>'
        '<|DSML|parameter name="status"><![CDATA[in_progress]]></|DSML|parameter>'
        '<|DSML|parameter name="description"><![CDATA[Extract Feishu client]]></|DSML|parameter>'
        "</|DSML|invoke></|DSML|tool_calls>"
    )
    noisy_middle = "old observation\n" * 600

    prompt, files = qwen_messages_to_prompt_and_files(
        [
            {"role": "system", "content": huge_system_prefix + "\n\n" + tool_protocol},
            {"role": "assistant", "content": "Assistant requested tool calls:\n" + task_update_done},
            {"role": "tool", "content": "Task 1 updated"},
            {"role": "user", "content": noisy_middle},
            {"role": "assistant", "content": "Assistant requested tool calls:\n" + task_update_pending},
            {"role": "tool", "content": "Read result for E:/ProjectX/mindcraft/scripts/feishu_client.py: file does not exist"},
            {"role": "user", "content": "Continue all four tasks. LATEST_SENTINEL"},
        ],
        max_prompt_chars=2200,
    )

    assert files == []
    assert len(prompt) <= 2200
    assert "# WebAI Gateway preserved task state" in prompt
    assert "Task 1: completed - Create config module" in prompt
    assert "Task 4: in_progress - Extract Feishu client" in prompt
    assert "feishu_client.py" in prompt
    assert "file does not exist" in prompt
    assert "LATEST_SENTINEL" in prompt


def test_qwen_messages_task_state_snapshot_preserves_taskupdate_todo_descriptions() -> None:
    task_update = (
        '<|DSML|tool_calls><|DSML|invoke name="TaskUpdate">'
        '<|DSML|parameter name="todos">'
        "<item>"
        "<content><![CDATA[创建 config.py 配置模块]]></content>"
        "<status><![CDATA[completed]]></status>"
        "<activeForm><![CDATA[正在创建 config.py 配置模块]]></activeForm>"
        "</item>"
        "<item>"
        "<content><![CDATA[重构 sync_to_feishu.py 使用 Config 和 FeishuClient]]></content>"
        "<status><![CDATA[in_progress]]></status>"
        "<activeForm><![CDATA[正在重构 sync_to_feishu.py]]></activeForm>"
        "</item>"
        "</|DSML|parameter>"
        "</|DSML|invoke></|DSML|tool_calls>"
    )

    prompt, files = qwen_messages_to_prompt_and_files(
        [
            {
                "role": "system",
                "content": (
                    "system bootstrap\n"
                    + ("tool schema detail\n" * 600)
                    + "You are using WebAI Gateway's strict tool bridge.\n"
                    + "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
                ),
            },
            {"role": "assistant", "content": "Assistant requested tool calls:\n" + task_update},
            {"role": "tool", "content": "Todos updated"},
            {"role": "user", "content": "任务 1-2 完成了什么？LATEST_SENTINEL"},
        ],
        max_prompt_chars=2200,
    )

    assert files == []
    assert "# WebAI Gateway preserved task state" in prompt
    assert "Task 1: completed - 创建 config.py 配置模块" in prompt
    assert "Task 2: in_progress - 重构 sync_to_feishu.py 使用 Config 和 FeishuClient" in prompt
    assert "no description" not in prompt
    assert "LATEST_SENTINEL" in prompt


def test_qwen_messages_layered_compaction_keeps_latest_turns_without_filling_provider_budget() -> None:
    tool_protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools: TaskUpdate, Read, Grep, Write.\n"
        "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
    )
    task_update = (
        '<|DSML|tool_calls><|DSML|invoke name="TaskUpdate">'
        '<|DSML|parameter name="task_id"><![CDATA[6]]></|DSML|parameter>'
        '<|DSML|parameter name="status"><![CDATA[in_progress]]></|DSML|parameter>'
        '<|DSML|parameter name="description"><![CDATA[Refactor Feishu sync]]></|DSML|parameter>'
        "</|DSML|invoke></|DSML|tool_calls>"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": ("system bootstrap\n" + ("tool schema detail\n" * 700) + tool_protocol)},
        {"role": "assistant", "content": "Assistant requested tool calls:\n" + task_update},
        {"role": "tool", "content": "Task 6 updated"},
    ]
    for index in range(90):
        messages.extend(
            [
                {"role": "user", "content": f"old request {index}\nANCIENT_OBSERVATION_SENTINEL\n" + ("old observation\n" * 120)},
                {"role": "assistant", "content": f"Assistant requested tool calls:\nRead(path=old_{index}.py)"},
                {"role": "tool", "content": f"old_{index}.py content\n" + ("legacy content\n" * 120)},
            ]
        )
    messages.extend(
        [
            {"role": "assistant", "content": "Assistant requested tool calls:\nRead(path=sync_to_feishu.py)"},
            {"role": "tool", "content": "LATEST_TOOL_RESULT_SENTINEL\nsync_to_feishu.py imports config and FeishuClient"},
            {"role": "user", "content": "Continue from the current refactor state. LATEST_USER_TURN_SENTINEL"},
        ]
    )

    prompt, files = qwen_messages_to_prompt_and_files(messages, max_prompt_chars=48000)

    assert files == []
    assert len(prompt) <= 36000
    assert "# WebAI Gateway preserved task state" in prompt
    assert "# DS2API_HISTORY.txt" in prompt
    assert "[Layered history compaction]" in prompt
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert "<|DSML|tool_calls>" in prompt
    assert "Task 6: in_progress - Refactor Feishu sync" in prompt
    assert "LATEST_TOOL_RESULT_SENTINEL" in prompt
    assert "LATEST_USER_TURN_SENTINEL" in prompt
    assert "ANCIENT_OBSERVATION_SENTINEL" not in prompt


def test_web_prompt_compaction_preserves_required_tool_format_when_tool_manifest_is_large() -> None:
    protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools:\n"
        + "\n".join(f'{{"name":"Tool{i}","description":"{"x" * 20}"}}' for i in range(200))
        + "\nRequired tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
    )
    raw = "system bootstrap\n" + ("skill listing entry\n" * 200) + protocol + ("\nold observation" * 300) + "\nLATEST_SENTINEL"

    prompt = compact_web_prompt(raw, max_chars=1800)

    assert len(prompt) <= 1800
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert "Required tool-call format" in prompt
    assert "<|DSML|tool_calls>" in prompt
    assert "LATEST_SENTINEL" in prompt


def test_qwen_web_parser_prefers_answer_phase_over_think_phase() -> None:
    text = "\n\n".join(
        [
            'data: {"choices":[{"delta":{"content":"thinking","phase":"think"}}]}',
            'data: {"choices":[{"delta":{"content":"OK","phase":"answer"}}]}',
            "data: [DONE]",
        ]
    )

    assert parse_qwen_stream_text(text) == "OK"


def test_qwen_web_stream_returns_complete_tool_json_before_done() -> None:
    consumed = 0
    tool_text = '```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{}}]}\n```'

    def lines():
        nonlocal consumed
        consumed += 1
        yield "data: " + json.dumps({"choices": [{"delta": {"content": tool_text}}]})
        consumed += 1
        raise AssertionError("stream should stop once a complete tool_json block is available")

    content = _collect_qwen_stream_lines(lines(), deadline_seconds=30)

    assert consumed == 1
    assert '"name":"Read"' in content


def test_qwen_web_stream_returns_complete_dsml_tool_block_before_done() -> None:
    consumed = 0
    tool_text = (
        '<|DSML|tool_calls><|DSML|invoke name="Read">'
        '<|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter>'
        '</|DSML|invoke></|DSML|tool_calls>'
    )

    def lines():
        nonlocal consumed
        consumed += 1
        yield "data: " + json.dumps({"choices": [{"delta": {"content": tool_text}}]})
        consumed += 1
        raise AssertionError("stream should stop once a complete DSML tool block is available")

    content = _collect_qwen_stream_lines(lines(), deadline_seconds=30)

    assert consumed == 1
    assert "<|DSML|tool_calls>" in content


def test_qwen_web_stream_returns_runaway_tool_bridge_text_before_timeout() -> None:
    consumed = 0

    def lines():
        nonlocal consumed
        for _ in range(10):
            consumed += 1
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "plain text without a tool call " * 4, "phase": "answer"}}]})
        raise AssertionError("stream should stop before the hard provider timeout")

    content = _collect_qwen_stream_lines(
        lines(),
        deadline_seconds=300,
        max_output_chars_without_tool_json=180,
    )

    assert consumed < 10
    assert len(content) >= 180
    assert "tool_json" not in content


def test_qwen_web_stream_has_wall_clock_deadline_for_heartbeat_streams() -> None:
    ticks = iter([0.0, 31.0])

    with pytest.raises(TimeoutError, match="Qwen Web request exceeded 30s"):
        _collect_qwen_stream_lines(
            ['data: {"choices":[{"delta":{"content":"still thinking","phase":"think"}}]}'],
            deadline_seconds=30,
            monotonic=lambda: next(ticks),
        )


def test_qwen_web_stream_checks_deadline_for_non_json_heartbeats() -> None:
    ticks = iter([0.0, 31.0])

    with pytest.raises(TimeoutError, match="Qwen Web request exceeded 30s"):
        _collect_qwen_stream_lines(
            [": keep-alive"],
            deadline_seconds=30,
            monotonic=lambda: next(ticks),
        )


def test_qwen_web_chat_uses_saved_credentials(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            qwen_client_factory=_FakeQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "qwen-web/qwen3.6-plus", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "来自 Qwen 网页模型"


def test_anthropic_messages_routes_qwen_coder_direct_provider(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=_FakeQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "来自 Qwen Coder 网页模型"
    assert _FakeQwenCoderClient.captured_payload["model"] == "qwen-coder/qwen-coder-plus"


def _repo_update_after_tool_loop_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": r"E:\ProjectX\mindcraft\MediaCrawler update this original GitHub repository",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_probe",
                    "name": "Bash",
                    "input": {"command": "git -C E:/ProjectX/mindcraft/MediaCrawler remote -v"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_probe",
                    "content": "origin https://github.com/NanmiCoder/MediaCrawler.git (fetch)",
                }
            ],
        },
        {
            "role": "user",
            "content": r"Continue update E:\ProjectX\mindcraft\MediaCrawler repository",
        },
    ]


def test_qwen_coder_keeps_tool_bridge_for_windows_path_update_task(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"cd E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler && git status --short"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=CapturingQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {
                    "role": "user",
                    "content": r"E:\ProjectX\mindcraft\MediaCrawler inspect this local repository",
                }
            ],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "web-search", "description": "Search the web", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    payload = seen["payload"]
    assert payload.get("_webai_native_web_search") is False
    assert "tools" not in payload
    prompt = "\n".join(str(message.get("content", "")) for message in payload["messages"])
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert '"name": "Bash"' in prompt
    assert '"name": "web-search"' not in prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Bash",
            "input": {"command": "cd E:/ProjectX/mindcraft/MediaCrawler && git status --short"},
        }
    ]


def test_qwen_coder_local_repo_update_preflights_before_model_guess(tmp_path: Path) -> None:
    class UnexpectedQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("local repository preflight should not call the web model")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=UnexpectedQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {
                    "role": "user",
                    "content": r"E:\ProjectX\mindcraft\MediaCrawler update this original GitHub repository",
                }
            ],
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_web_preflight_1",
            "name": "Bash",
            "input": {
                "command": "git -C E:/ProjectX/mindcraft/MediaCrawler remote -v && git -C E:/ProjectX/mindcraft/MediaCrawler status --short"
            },
        }
    ]
    events_response = client.get("/api/admin/tool-bridge-events")
    assert events_response.status_code == 200
    events = events_response.json()["events"]
    assert events[-1]["kind"] == "local_repo_preflight"
    assert events[-1]["model"] == "qwen-coder/qwen-coder-plus"
    assert events[-1]["tool"] == "Bash"
    assert events[-1]["commandPreview"] == (
        "git -C E:/ProjectX/mindcraft/MediaCrawler remote -v && "
        "git -C E:/ProjectX/mindcraft/MediaCrawler status --short"
    )


def test_local_repo_preflight_ignores_file_paths_from_context_noise() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {
                "role": "user",
                "content": (
                    "你审查下代码，看看有什么改进的地方\n"
                    "Loaded C:\\Users\\woody\\.claude\\plugins\\cache\\caveman\\hooks\\caveman-statusline.ps1 "
                    "and recently updated status text."
                ),
            }
        ],
    )

    assert build_local_repo_preflight_tool_call(context) is None


def test_qwen_coder_accepts_bash_string_input_after_tool_loop(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class StringInputQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Bash",'
                '"input":"git -C \\"E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler\\" pull origin main"}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=StringInputQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": _repo_update_after_tool_loop_messages(),
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 1
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Bash",
            "input": {"command": 'git -C "E:/ProjectX/mindcraft/MediaCrawler" pull origin main'},
        }
    ]


def test_qwen_coder_tool_bridge_rejection_is_recorded_with_safe_details(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class RepeatingUnknownToolQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Task",'
                '"input":{"description":"inspect","token":"session-secret"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=RepeatingUnknownToolQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "inspect repository"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 3
    assert response.json()["content"][0]["type"] == "text"
    events_response = client.get("/api/admin/tool-bridge/events")
    assert events_response.status_code == 200
    events = events_response.json()["events"]
    rejection = [event for event in events if event["kind"] == "tool_bridge_rejection"][-1]
    assert rejection["model"] == "qwen-coder/qwen-coder-plus"
    assert rejection["errorKind"] == "unknown_tool"
    assert rejection["allowedTools"] == ["Read"]
    assert "未知工具：Task" in rejection["errorMessage"]
    assert "Task" in rejection["rawPreview"]
    assert "session-secret" not in rejection["rawPreview"]
    assert "[redacted]" in rejection["rawPreview"]


def test_qwen_coder_recovers_prose_completion_claim_with_bad_bash_artifact(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class ProseCompletionWithBadBashClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 2:
                return _openai_response(
                    '```tool_json\n'
                    '{"calls":[{"id":"toolu_read","name":"Read","input":{"file_path":"README.md"}}]}'
                    "\n```"
                )
            return _openai_response(
                "All improvement work is complete.\n\n"
                "Assistant requested tool calls:\n"
                '- Bash({"command":"python-dotenv>=1.0.0 && pytest>=7.0.0"})\n\n'
                "Superbrain mode completed successfully."
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=ProseCompletionWithBadBashClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
            json={
                "model": "qwen-coder/qwen-coder-plus",
                "messages": [{"role": "user", "content": "Implement the improvement plan in this local codebase."}],
                "tools": [
                    {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                    {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                ],
                "max_tokens": 1024,
            },
        )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    repair_text = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "insufficient_final_evidence" in repair_text
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {"file_path": "README.md"}}
    ]


def test_qwen_coder_recovers_repeated_hidden_shell_to_structured_edit(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class HiddenShellThenStructuredEditClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) <= 2:
                return _openai_response(
                    '```tool_json\n'
                    '{"calls":[{"id":"call_1","name":"bash",'
                    '"input":{"command":"ls -la E:/ProjectX/mindcraft/"}}]}'
                    "\n```"
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_2","name":"Edit",'
                '"input":{"file_path":"E:/ProjectX/mindcraft/README.md","old_string":"TODO","new_string":"Done"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=HiddenShellThenStructuredEditClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "真正落地修改 Mindcraft 项目代码"}],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
                {"name": "Grep", "description": "Search files", "input_schema": {"type": "object"}},
                {"name": "Edit", "description": "Edit files", "input_schema": {"type": "object"}},
                {"name": "Write", "description": "Write files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 3
    assert "x-webai-tool-bridge-error" not in response.headers
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_2",
            "name": "Edit",
            "input": {
                "file_path": "E:/ProjectX/mindcraft/README.md",
                "old_string": "TODO",
                "new_string": "Done",
            },
        }
    ]


def test_qwen_coder_repairs_empty_code_change_promise_to_write_tool(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class PromiseThenWriteClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    "Need to actually modify the Mindcraft project code as requested. "
                    "Will make real changes to the files now."
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Write",'
                '"input":{"file_path":"E:/ProjectX/mindcraft/config/settings.py","content":"CONFIG = {}\\n"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=PromiseThenWriteClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "真正落地修改 Mindcraft 项目代码"}],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Write", "description": "Write files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "deferred_code_change_without_call" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Write",
            "input": {"file_path": "E:/ProjectX/mindcraft/config/settings.py", "content": "CONFIG = {}\n"},
        }
    ]


def test_qwen_coder_repairs_hidden_shell_for_code_edit_task_to_structured_edit(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class HiddenShellThenEditClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    '```tool_json\n'
                    '{"calls":[{"id":"call_1","name":"Bash",'
                    '"input":{"command":"pip install pytest pytest-cov && pytest"}}]}'
                    "\n```"
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_2","name":"Edit",'
                '"input":{"file_path":"config.py","old_string":"DEBUG = True","new_string":"DEBUG = False"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=HiddenShellThenEditClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "修复这个 Bug 并落?"}],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
                {"name": "Grep", "description": "Search files", "input_schema": {"type": "object"}},
                {"name": "Edit", "description": "Edit files", "input_schema": {"type": "object"}},
                {"name": "Write", "description": "Write files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    first_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[0]["messages"])
    assert '"name": "Bash"' not in first_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_2",
            "name": "Edit",
            "input": {"file_path": "config.py", "old_string": "DEBUG = True", "new_string": "DEBUG = False"},
        }
    ]


def test_tool_bridge_rejection_records_compiler_phases_without_secrets(tmp_path: Path) -> None:
    class UnknownToolQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                "```tool_json\n"
                '{"calls":[{"id":"call_1","name":"MissingTool",'
                '"input":{"token":"session-secret","api_key":"super-secret-key"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=UnknownToolQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "修复代码"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    events_response = client.get("/api/admin/tool-bridge/events")
    assert events_response.status_code == 200
    events = events_response.json()["events"]
    rejection = [event for event in events if event["kind"] == "tool_bridge_rejection"][-1]
    assert rejection["errorKind"] == "unknown_tool"
    assert rejection["phases"]
    assert {"name": "extract", "status": "ok", "detail": "candidate_count=1"} in rejection["phases"]
    assert {"name": "parse/normalize", "status": "error", "detail": "unknown_tool"} in rejection["phases"]
    assert {"name": "validate", "status": "error", "detail": "unknown_tool"} in rejection["phases"]
    assert "taskPreview" not in rejection
    assert rejection["hasTaskText"] is True
    assert isinstance(rejection["taskChars"], int)
    serialized = json.dumps(rejection["phases"], ensure_ascii=False)
    assert "session-secret" not in serialized
    assert "super-secret-key" not in serialized
    assert "api_key" not in serialized.lower()


@pytest.mark.parametrize(
    "command",
    [
        "git checkout -b security/hotfix-secrets-rotation",
        "pip install pytest pytest-cov && pytest --cov=src --cov-report=term-missing",
        "python -c \"print('FEISHU_APP_ID=')\" > .env.example",
        "git secrets --register-aws",
        "python utils/rotate_secrets.py --all",
        "git add . && git commit -m \"feat: security hardening\"",
        "python skills/security-reviewer/audit.py --full-scan",
    ],
)
def test_qwen_coder_blocks_claude_code_review_mutation_without_retry(tmp_path: Path, command: str) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class ReviewMutationQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            return _openai_response(
                '```tool_json\n'
                f'{{"calls":[{{"id":"call_1","name":"Bash","input":{{"command":{json.dumps(command, ensure_ascii=False)}}}}}]}}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=ReviewMutationQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "审查下代码并且制定改进计?"}],
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 1
    assert response.headers["x-webai-tool-bridge-error"] == "unsafe_review_shell_command"
    body = response.json()
    assert body["content"][0]["type"] == "text"
    assert "unsafe_review_shell_command" in body["content"][0]["text"]


@pytest.mark.parametrize(
    "command",
    [
        "git -C E:/ProjectX/mindcraft status --short",
        "git -C E:/ProjectX/mindcraft log --oneline -n 5",
        'git grep -i "secret" -- "*.py"',
        "ls -la E:/ProjectX/mindcraft",
    ],
)
def test_qwen_coder_allows_claude_code_review_readonly_command_matrix(tmp_path: Path, command: str) -> None:
    class ReviewReadonlyQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                '```tool_json\n'
                f'{{"calls":[{{"id":"call_1","name":"Bash","input":{{"command":{json.dumps(command, ensure_ascii=False)}}}}}]}}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=ReviewReadonlyQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "审查下代码并且制定改进计?"}],
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert "x-webai-tool-bridge-error" not in response.headers
    body = response.json()
    assert body["content"][0]["type"] == "tool_use"
    assert body["content"][0]["name"] == "Bash"
    assert body["content"][0]["input"]["command"] == command


def test_qwen_coder_rewrites_windows_dir_shell_to_glob_tool(tmp_path: Path) -> None:
    class WindowsDirQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential
            self.last_diagnostic = {}

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Bash","input":{"command":"dir \\"E:/ProjectX/mindcraft/MediaCrawler\\" /s"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=WindowsDirQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "审查这个本地项目的代码结?"}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
                },
                {
                    "name": "Glob",
                    "description": "Find files by pattern.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                },
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Glob",
            "input": {"path": "E:/ProjectX/mindcraft/MediaCrawler", "pattern": "*"},
        }
    ]


def test_qwen_coder_review_policy_keeps_original_prompt_after_tool_result(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class ReviewAfterToolResultQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_2","name":"Bash","input":{"command":"pip install pytest pytest-cov && pytest --cov=src"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=ReviewAfterToolResultQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "审查下代码并且制定改进计?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Bash",
                            "input": {"command": "git status --short"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "On branch security/fix-secrets-leak\nnothing to commit, working tree clean",
                        }
                    ],
                },
            ],
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 1
    assert response.headers["x-webai-tool-bridge-error"] == "unsafe_review_shell_command"
    body = response.json()
    assert body["content"][0]["type"] == "text"
    assert "unsafe_review_shell_command" in body["content"][0]["text"]


def test_qwen_coder_review_policy_keeps_original_prompt_after_continue(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class ReviewContinueQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Bash","input":{"command":"pip install pytest pytest-cov && git rm skills/feishu-syncer/test-url*.js"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=ReviewContinueQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "审查下代码并且制定改进计?"},
                {"role": "assistant", "content": "我先看一下项目结构?"},
                {"role": "user", "content": "继续"},
            ],
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 1
    assert response.headers["x-webai-tool-bridge-error"] == "unsafe_review_shell_command"
    body = response.json()
    assert body["content"][0]["type"] == "text"
    assert "unsafe_review_shell_command" in body["content"][0]["text"]


def test_qwen_coder_blocks_high_risk_shell_when_continue_has_no_task_anchor(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class UnanchoredContinueQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git checkout -b fix/test-collection && pytest --collect-only"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=UnanchoredContinueQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "继续"}],
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 1
    assert response.headers["x-webai-tool-bridge-error"] == "unsafe_shell_command_requires_explicit_task"
    body = response.json()
    assert body["content"][0]["type"] == "text"
    assert "unsafe_shell_command_requires_explicit_task" in body["content"][0]["text"]


def test_qwen_coder_recovers_unavailable_skill_loader_tool_from_loaded_skill_context(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class SkillLoaderThenAnswerQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) <= 2:
                return _openai_response(
                    "Using using-superpowers to structure the plan.\n"
                    "```tool_json\n"
                    '{"calls":[{"id":"call_1","name":"Skill","input":{"skill_name":"using-superpowers"}}]}'
                    "\n```"
                )
            return _openai_response("I will continue from the already loaded using-superpowers instructions.")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=SkillLoaderThenAnswerQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use the using-superpowers skill.\n"
                        "Base directory for this skill: C:\\Users\\woody\\.claude\\skills\\using-superpowers\n"
                        "The skill body is already loaded in this conversation."
                    ),
                }
            ],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 3
    final_retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[2]["messages"])
    assert "The requested Skill/SlashCommand loader content is already present" in final_retry_prompt
    assert "Do not call Skill" in final_retry_prompt
    assert response.json()["content"] == [
        {"type": "text", "text": "I will continue from the already loaded using-superpowers instructions."}
    ]


def test_qwen_coder_maps_agent_alias_to_task_tool(tmp_path: Path) -> None:
    class AgentAliasQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Agent",'
                '"input":{"prompt":"Inspect plan execution patterns.","description":"Inspect planning patterns"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=AgentAliasQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "make a plan with subagents"}],
            "tools": [
                {
                    "name": "Task",
                    "description": "Launch a subagent",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["prompt", "description"],
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Task",
            "input": {"prompt": "Inspect plan execution patterns.", "description": "Inspect planning patterns"},
        }
    ]


def test_qwen_coder_maps_readfile_alias_to_read_tool_use(tmp_path: Path) -> None:
    class ReadFileAliasQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"ReadFile",'
                '"input":{"path":"E:/ProjectX/mindcraft/CLAUDE.md"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=ReadFileAliasQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "Read the project instructions"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Read",
            "input": {"file_path": "E:/ProjectX/mindcraft/CLAUDE.md"},
        }
    ]


def test_qwen_coder_repairs_agent_missing_description_before_tool_use(tmp_path: Path) -> None:
    class MissingAgentDescriptionClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Agent",'
                '"input":{"prompt":"Analyze plan execution components.\\nFind validation hooks."}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=MissingAgentDescriptionClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "make a plan"}],
            "tools": [
                {
                    "name": "Agent",
                    "description": "Launch a subagent",
                    "input_schema": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}, "description": {"type": "string"}},
                        "required": ["prompt", "description"],
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Agent",
            "input": {
                "prompt": "Analyze plan execution components.\nFind validation hooks.",
                "description": "Analyze plan execution components.",
            },
        }
    ]


def test_qwen_coder_repairs_glob_patterns_array_before_tool_use(tmp_path: Path) -> None:
    class GlobPatternsClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Glob",'
                '"input":{"patterns":["docs/*.md","docs/*.txt"]}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=GlobPatternsClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "find docs"}],
            "tools": [
                {
                    "name": "Glob",
                    "description": "Find files",
                    "input_schema": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                        "required": ["pattern"],
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_call_1", "name": "Glob", "input": {"pattern": "docs/{*.md,*.txt}"}}
    ]


def test_qwen_coder_repairs_incomplete_git_clone_before_tool_use(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class RepairingQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git clone"}}]}\n```'
                )
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_2","name":"Bash","input":{"command":"git -C E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler remote -v && git -C E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler status --short"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=RepairingQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": _repo_update_after_tool_loop_messages(),
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "git clone is missing the repository URL" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_2",
            "name": "Bash",
            "input": {
                "command": "git -C E:/ProjectX/mindcraft/MediaCrawler remote -v && git -C E:/ProjectX/mindcraft/MediaCrawler status --short"
            },
        }
    ]


def test_qwen_coder_repairs_premature_repo_clarification_to_local_probe(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class ClarifyingQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    "请确?MediaCrawler 是指哪个具体?GitHub 仓库？也请说明您想更新哪些具体内容?"
                )
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git -C E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler remote -v && git -C E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler status --short"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=ClarifyingQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": _repo_update_after_tool_loop_messages(),
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "asked the user to provide repository details" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Bash",
            "input": {
                "command": "git -C E:/ProjectX/mindcraft/MediaCrawler remote -v && git -C E:/ProjectX/mindcraft/MediaCrawler status --short"
            },
        }
    ]


def test_qwen_coder_repairs_clone_to_requested_local_path_to_probe(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class CloneFirstQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git clone https://github.com/NanmiCoder/MediaCrawler.git E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler"}}]}\n```'
                )
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_2","name":"Bash","input":{"command":"git -C E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler remote -v && git -C E:\\\\ProjectX\\\\mindcraft\\\\MediaCrawler status --short"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=CloneFirstQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": _repo_update_after_tool_loop_messages(),
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "git clone targets the local path named by the task" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_2",
            "name": "Bash",
            "input": {
                "command": "git -C E:/ProjectX/mindcraft/MediaCrawler remote -v && git -C E:/ProjectX/mindcraft/MediaCrawler status --short"
            },
        }
    ]


def test_qwen_coder_repairs_deferred_shell_execution_without_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class DeferredShellQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    'I will execute these commands:\n'
                    'git -C "E:/ProjectX/mindcraft/MediaCrawler" fetch origin\n'
                    'git -C "E:/ProjectX/mindcraft/MediaCrawler" reset --hard origin/main'
                )
            raise AssertionError("unwrapped shell commands should be converted without asking the web model to repair")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=DeferredShellQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": _repo_update_after_tool_loop_messages()
            + [{"role": "user", "content": "Local changes can be discarded; execute the update."}],
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 1
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_web_shell_1",
            "name": "Bash",
            "input": {
                "command": 'git -C "E:/ProjectX/mindcraft/MediaCrawler" fetch origin && git -C "E:/ProjectX/mindcraft/MediaCrawler" reset --hard origin/main'
            },
        }
    ]


def test_qwen_coder_repairs_unwrapped_shell_command_block_without_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class UnwrappedShellQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    "I understand you'd like to discard the local changes and update the repository. "
                    "Since you've confirmed it's okay to delete the local modifications, I'll use a simpler approach:\n\n"
                    "# Discard all local changes and reset to match the remote\n"
                    'git -C "E://ProjectX//mindcraft//MediaCrawler" reset --hard origin/main\n\n'
                    "# Ensure we have the latest updates\n"
                    'git -C "E://ProjectX//mindcraft//MediaCrawler" pull origin main\n\n'
                    "This will completely discard your local changes and pull the 65 new commits."
                )
            raise AssertionError("unwrapped shell commands should be converted without asking the web model to repair")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=UnwrappedShellQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": _repo_update_after_tool_loop_messages()
            + [{"role": "user", "content": "好的，本地的修改可以删掉"}],
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 1
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_web_shell_1",
            "name": "Bash",
            "input": {
                "command": 'git -C "E:/ProjectX/mindcraft/MediaCrawler" reset --hard origin/main && git -C "E:/ProjectX/mindcraft/MediaCrawler" pull origin main'
            },
        }
    ]


def test_tool_bridge_converts_fenced_bash_command_to_shell_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review local project code."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_preflight",
                        "name": "Bash",
                        "input": {"command": "git -C E:/ProjectX/mindcraft status --short"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_preflight", "content": ""}],
            },
        ],
    )

    result = parse_tool_response("```bash git -C E:/ProjectX/mindcraft diff --name-only ```", context)

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_web_shell_1", "Bash", {"command": "git -C E:/ProjectX/mindcraft diff --name-only"})
    ]


def test_qwen_coder_converts_fenced_bash_command_without_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class FencedShellQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            return _openai_response("```bash git -C E:/ProjectX/mindcraft diff --name-only ```")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=FencedShellQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "Review local project code."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_preflight",
                            "name": "Bash",
                            "input": {"command": "git -C E:/ProjectX/mindcraft status --short"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_preflight", "content": ""}],
                },
            ],
            "tools": [{"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 1
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_web_shell_1",
            "name": "Bash",
            "input": {"command": "git -C E:/ProjectX/mindcraft diff --name-only"},
        }
    ]


def test_tool_bridge_accepts_hidden_readonly_git_shell_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "List files",
                    "parameters": {"type": "object"},
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [{"role": "user", "content": "Review local project code and inspect current changes."}],
    )

    assert "Bash" not in {tool.name for tool in context.tools}

    result = parse_tool_response(
        '```tool_json\n'
        '{"calls":[{"id":"call_1","name":"Bash","input":{"command":"git -C E:/ProjectX/mindcraft diff --name-only HEAD"}}]}'
        "\n```",
        context,
    )

    assert result.error is None
    assert [(call.id, call.name, call.input) for call in result.tool_calls] == [
        ("call_1", "Bash", {"command": "git -C E:/ProjectX/mindcraft diff --name-only HEAD"})
    ]


def test_tool_bridge_rejects_fenced_bash_command_when_shell_is_hidden() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Glob",
                    "description": "List files",
                    "parameters": {"type": "object"},
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review local project code."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_preflight",
                        "name": "Bash",
                        "input": {"command": "git -C E:/ProjectX/mindcraft status --short"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_preflight", "content": ""}],
            },
            {"role": "user", "content": "继续"},
        ],
    )

    result = parse_tool_response("```bash cd E:/ProjectX/mindcraft && ls -la ```", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_local_shell_command"
    assert result.error.repairable is True


def test_qwen_coder_repairs_fenced_bash_command_when_shell_is_hidden(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class HiddenFencedShellQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("```bash cd E:/ProjectX/mindcraft && ls -la ```")
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Glob",'
                '"input":{"path":"E:/ProjectX/mindcraft","pattern":"*"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=HiddenFencedShellQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "Review local project code."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_preflight",
                            "name": "Bash",
                            "input": {"command": "git -C E:/ProjectX/mindcraft status --short"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_preflight", "content": ""}],
                },
                {"role": "user", "content": "继续"},
            ],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "List files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "unsafe_local_shell_command" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Glob",
            "input": {"path": "E:/ProjectX/mindcraft", "pattern": "*"},
        }
    ]


def test_tool_bridge_rejects_prose_shell_command_intent_when_shell_is_hidden() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            },
            {"type": "function", "function": {"name": "Read", "description": "Read files", "parameters": {"type": "object"}}},
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review local project code."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_preflight",
                        "name": "Bash",
                        "input": {"command": "git -C E:/ProjectX/mindcraft status --short"},
                    }
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_preflight", "content": ""}]},
            {"role": "user", "content": "继续"},
        ],
    )

    result = parse_tool_response(
        "Reviewing mindcraft project code for improvements. "
        "Need to examine current changes and identify optimization opportunities. "
        "Running git diff to see pending changes.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unsafe_local_shell_command"
    assert result.error.repairable is True


def test_qwen_coder_repairs_prose_shell_command_intent_when_shell_is_hidden(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class ProseShellIntentQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    "Reviewing mindcraft project code for improvements. "
                    "Need to examine current changes and identify optimization opportunities. "
                    "Running git diff to see pending changes."
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Read",'
                '"input":{"file_path":"E:/ProjectX/mindcraft/requirements.txt"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=ProseShellIntentQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "Review local project code."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_preflight",
                            "name": "Bash",
                            "input": {"command": "git -C E:/ProjectX/mindcraft status --short"},
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_preflight", "content": ""}]},
                {"role": "user", "content": "继续"},
            ],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "unsafe_local_shell_command" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Read",
            "input": {"file_path": "E:/ProjectX/mindcraft/requirements.txt"},
        }
    ]


def test_qwen_coder_repairs_code_patch_snippet_without_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class CodePatchSnippetQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    "Bug in auth middleware. Token expiry check use `<` not `<=`. Fix:\n"
                    "```python\n"
                    "# In auth.py\n"
                    "if token.expiry <= datetime.now():\n"
                    "    raise AuthError(\"Token expired\")\n"
                    "```\n"
                    "Change to:\n"
                    "```python\n"
                    "if token.expiry < datetime.now():\n"
                    "    raise AuthError(\"Token expired\")\n"
                    "```"
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Edit",'
                '"input":{"file_path":"E:/ProjectX/mindcraft/auth.py",'
                '"old_string":"if token.expiry <= datetime.now():\\n    raise AuthError(\\"Token expired\\")",'
                '"new_string":"if token.expiry < datetime.now():\\n    raise AuthError(\\"Token expired\\")"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=CodePatchSnippetQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "Review local project code and implement required fixes."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_read_auth",
                            "name": "Read",
                            "input": {"file_path": "E:/ProjectX/mindcraft/auth.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_read_auth",
                            "content": "if token.expiry <= datetime.now():\n    raise AuthError(\"Token expired\")",
                        }
                    ],
                },
                {"role": "user", "content": "继续"},
            ],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Edit", "description": "Edit files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "incomplete_fix_stub_without_tool_call" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Edit",
            "input": {
                "file_path": "E:/ProjectX/mindcraft/auth.py",
                "old_string": 'if token.expiry <= datetime.now():\n    raise AuthError("Token expired")',
                "new_string": 'if token.expiry < datetime.now():\n    raise AuthError("Token expired")',
            },
        }
    ]


def test_qwen_coder_repairs_local_file_inspection_intent_without_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class LocalInspectionIntentQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    "Bug in git status check. Commands fail silently. "
                    "Files exist but git ignore setup prevent tracking. "
                    "Need direct file review. Review files in E:\\ProjectX\\mindcraft. "
                    "Check structure and content for improvement areas."
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Glob",'
                '"input":{"path":"E:/ProjectX/mindcraft","pattern":"*.py"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=LocalInspectionIntentQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "Review local project code and make an improvement plan."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_status",
                            "name": "Bash",
                            "input": {"command": "git -C E:/ProjectX/mindcraft status --short"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_status",
                            "content": "",
                        }
                    ],
                },
                {"role": "user", "content": "继续"},
            ],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "List files", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "deferred_tool_action_without_call" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Glob",
            "input": {"path": "E:/ProjectX/mindcraft", "pattern": "*.py"},
        }
    ]


def test_qwen_coder_repairs_codebase_overview_intent_without_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class CodebaseOverviewIntentQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    "Review mindcraft project code and create improvement plan. "
                    "Need current codebase overview first. "
                    "Use different file path for project structure analysis."
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Glob",'
                '"input":{"path":"E:/ProjectX/mindcraft","pattern":"*"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=CodebaseOverviewIntentQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "Review mindcraft project code and create improvement plan."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_read",
                            "name": "Read",
                            "input": {"file_path": "E:/ProjectX/mindcraft/README.md"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_read",
                            "content": "README exists",
                        }
                    ],
                },
                {"role": "user", "content": "继续"},
            ],
            "tools": [
                {"name": "Glob", "description": "List files", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "deferred_tool_action_without_call" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Glob",
            "input": {"path": "E:/ProjectX/mindcraft", "pattern": "*"},
        }
    ]


def test_qwen_coder_repairs_unproven_final_answer_without_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class UnprovenFinalQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    "Need a better project map before proceeding. Continue with another file path."
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Glob",'
                '"input":{"path":"E:/ProjectX/mindcraft","pattern":"*"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=UnprovenFinalQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "Review mindcraft project code and create improvement plan."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_read",
                            "name": "Read",
                            "input": {"file_path": "E:/ProjectX/mindcraft/README.md"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_read",
                            "content": "README exists",
                        }
                    ],
                },
                {"role": "user", "content": "继续"},
            ],
            "tools": [
                {"name": "Glob", "description": "List files", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "unproven_final_answer_without_tool_call" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Glob",
            "input": {"path": "E:/ProjectX/mindcraft", "pattern": "*"},
        }
    ]


def test_qwen_coder_repairs_deferred_code_change_without_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class DeferredImplementationQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    "# Continuing Security Hardening Implementation\n"
                    "I'm continuing with the implementation plan. "
                    "Let's implement the environment variable management system first."
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Write",'
                '"input":{"file_path":"E:/ProjectX/mindcraft/config.py","content":"CONFIG = {}\\n"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=DeferredImplementationQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "继续落地改代码，实现安全加固"}],
            "tools": [
                {
                    "name": "Write",
                    "description": "Write a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["file_path", "content"],
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "deferred_code_change_without_call" in retry_prompt
    assert "request a tool" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Write",
            "input": {"file_path": "E:/ProjectX/mindcraft/config.py", "content": "CONFIG = {}\n"},
        }
    ]


def test_qwen_coder_repairs_echoed_tool_history_without_tool_json(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class EchoedHistoryQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    "Assistant requested tool calls:\n"
                    '- Glob({"pattern":"*.py","path":"E:/ProjectX/mindcraft"})\n'
                    "User: Tool result for Glob (call id: toolu_call_3, is_error: false):\n"
                    '["E:/ProjectX/mindcraft/rewrite_content.py"]\n'
                    "Use this tool result to continue the task.\n"
                    "Assistant: Assistant requested tool calls:\n"
                    '- Read({"file_path":"E:/ProjectX/mindcraft/rewrite_content.py"})'
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_1","name":"Read",'
                '"input":{"file_path":"E:/ProjectX/mindcraft/rewrite_content.py"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=EchoedHistoryQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "Review local project code and continue the plan."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_call_3",
                            "name": "Glob",
                            "input": {"pattern": "*.py", "path": "E:/ProjectX/mindcraft"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_call_3",
                            "content": '["E:/ProjectX/mindcraft/rewrite_content.py"]',
                        }
                    ],
                },
            ],
            "tools": [
                {"name": "Glob", "description": "List files", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "echoed_tool_history_without_tool_json" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_1",
            "name": "Read",
            "input": {"file_path": "E:/ProjectX/mindcraft/rewrite_content.py"},
        }
    ]


def test_tool_bridge_flags_chinese_deferred_code_change_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Write",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["file_path", "content"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="agent"),
    )

    result = parse_tool_response("I will implement the environment variable management module and update the config file.", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "deferred_code_change_without_call"
    assert result.error.repairable is True


def test_tool_bridge_flags_empty_code_change_promise_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Write",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["file_path", "content"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )

    result = parse_tool_response(
        "Need to actually modify the Mindcraft project code as requested. "
        "Will make real changes to the files now.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "deferred_code_change_without_call"
    assert result.error.repairable is True


def test_tool_bridge_flags_manual_code_after_tool_restriction_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read files",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "description": "Edit files",
                    "parameters": {"type": "object"},
                },
            },
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )

    result = parse_tool_response(
        "Due to tool restrictions, I provided the complete code implementations that you need to manually create in your project.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "tool_denial_without_call"
    assert result.error.repairable is True


def test_tool_bridge_flags_chinese_empty_code_change_promise_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Write",
                    "description": "Write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["file_path", "content"],
                    },
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="local-agent"),
    )

    result = parse_tool_response("Now I will implement the improvement plan by modifying files.", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "deferred_code_change_without_call"
    assert result.error.repairable is True


def test_tool_bridge_rejects_named_tool_intent_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review the current local project code and make an improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Glob", "input": {"path": ".", "pattern": "*.py"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "rewrite_content.py"}]},
        ],
    )

    result = parse_tool_response(
        "Need inspect the project Python files. Next step run Glob tool to list all .py files.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "deferred_named_tool_action_without_call"
    assert result.error.repairable is True


def test_tool_bridge_allows_final_reference_to_prior_tool_result() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review the current local project code and make an improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Glob", "input": {"path": ".", "pattern": "*.py"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "rewrite_content.py"}]},
        ],
    )

    result = parse_tool_response(
        "Review summary: use the earlier Glob result and focus on the listed Python files.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is None


def test_tool_bridge_rejects_local_file_inspection_intent_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review the current local project code and make an improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Glob", "input": {"path": ".", "pattern": "*.py"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "rewrite_content.py"}]},
        ],
    )

    result = parse_tool_response(
        "Bug in git status check. Commands fail silently. "
        "Files exist but git ignore setup prevent tracking. "
        "Need direct file review. Review files in E:\\ProjectX\\mindcraft. "
        "Check structure and content for improvement areas.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "deferred_tool_action_without_call"
    assert result.error.repairable is True


def test_tool_bridge_rejects_codebase_overview_intent_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review mindcraft project code and create improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "README.md"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "README exists"}]},
        ],
    )

    result = parse_tool_response(
        "Review mindcraft project code and create improvement plan. "
        "Need current codebase overview first. "
        "Use different file path for project structure analysis.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "deferred_tool_action_without_call"
    assert result.error.repairable is True


def test_tool_bridge_rejects_unproven_final_answer_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review mindcraft project code and create improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "README.md"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "README exists"}]},
        ],
    )

    result = parse_tool_response(
        "Need a better project map before proceeding. Continue with another file path.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unproven_final_answer_without_tool_call"
    assert result.error.repairable is True


def test_tool_bridge_all_profile_passes_unproven_plain_text_like_ds2api() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review mindcraft project code and create improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "README.md"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "README exists"}]},
        ],
    )

    result = parse_tool_response(
        "Need a better project map before proceeding. Continue with another file path.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is None
    assert result.content == "Need a better project map before proceeding. Continue with another file path."


def test_tool_bridge_rejects_chinese_unproven_review_next_step_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review mindcraft project code and create improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "README.md"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "README exists"}]},
        ],
    )

    result = parse_tool_response(
        "\u5ba1\u67e5mindcraft\u9879\u76ee\u4ee3\u7801\u53d1\u73b0\u5b89\u5168\u6f0f\u6d1e"
        "\u548c\u8d28\u91cf\u95ee\u9898\u3002\u9700\u7acb\u5373\u4fee\u590dSQL\u6ce8\u5165\u3001"
        "\u786c\u7f16\u7801\u5bc6\u94a5\u7b49\u98ce\u9669\u3002\u4e0b\u4e00\u6b65\u5b9e\u65bd"
        "\u53c2\u6570\u5316\u67e5\u8be2\u548c\u5bc6\u94a5\u7ba1\u7406\u3002",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unproven_final_answer_without_tool_call"
    assert result.error.repairable is True


def test_tool_bridge_rejects_chinese_generic_review_completion_next_steps_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review mindcraft project code and create improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "README.md"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "README exists"}]},
        ],
    )

    result = parse_tool_response(
        "Mindcraft项目代码审查完成。当前工作树E:\\ProjectX\\mindcraft非Git仓库，无远程同步?"
        "项目含内容抓?改写/同步功能，需补充Git初始化、测试覆盖至80%、安全扫描及文档完善?"
        "建议按TDD流程逐步添加单元/集成/E2E测试，建立CI流水线，规范PR流程?",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unproven_final_answer_without_tool_call"
    assert result.error.repairable is True


def test_tool_bridge_rejects_chinese_short_review_next_step_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "继续执行"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "README.md"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "README exists"}]},
        ],
    )

    result = parse_tool_response(
        "审查Mindcraft项目代码质量。当前工作树非Git仓库，需初始化版本控制?"
        "下一步建立测试框架达80%覆盖，补全安全扫描与文档?",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unproven_final_answer_without_tool_call"
    assert result.error.repairable is True


def test_tool_bridge_treats_chinese_continue_execute_as_continuation() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read tool",
                    "parameters": {"type": "object"},
                },
            }
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "落地 Mindcraft 项目安全改进。"},
            {"role": "user", "content": "继续执行"},
        ],
    )

    assert context.task_text == "落地 Mindcraft 项目安全改进。\n继续执行"


def test_tool_bridge_allows_complete_review_summary_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review mindcraft project code and create improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "README.md"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "README exists"}]},
        ],
    )

    result = parse_tool_response(
        "Review summary: no critical issues were found in the inspected README. "
        "Recommendation: keep the current structure and document any future auth changes in SECURITY.md.",
        context,
    )

    assert result.error is None
    assert result.tool_calls == []
    assert "Review summary" in result.content


def test_tool_bridge_rejects_chinese_local_file_inspection_intent_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "审查 mindcraft 项目代码并制定改进计划?"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Glob", "input": {"path": ".", "pattern": "*.py"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "rewrite_content.py"}]},
        ],
    )

    result = parse_tool_response(
        "审查mindcraft项目代码并制定改进计划。需直接查看文件内容识别问题。下一步检查项目结构和核心文件?",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "deferred_tool_action_without_call"
    assert result.error.repairable is True


def test_tool_bridge_rejects_inline_fix_step_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review the current local project code and make an improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "auth.py"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "if token.expiry < now: pass"}]},
        ],
    )

    result = parse_tool_response(
        "Bug in auth middleware. Token expiry check use `<` not `<=`. "
        "Fix: change to `<=` ensure expired tokens rejected. [next step].",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "incomplete_fix_stub_without_tool_call"
    assert result.error.repairable is True


def test_tool_bridge_rejects_incomplete_fix_stub_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review the current local project code and make an improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "auth.py"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "def expired(token): return token.exp < now"}]},
        ],
    )

    result = parse_tool_response("Bug in auth middleware. Token expiry check uses `<` not `<=`. Fix:", context)

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "incomplete_fix_stub_without_tool_call"
    assert result.error.repairable is True


def test_tool_bridge_rejects_code_patch_snippet_without_tool_call() -> None:
    context = build_context(
        [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ["Read", "Glob", "Grep", "Edit", "Write"]
        ],
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "Review the current local project code and make an improvement plan."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "Read", "input": {"file_path": "auth.py"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "def expired(token): return token.exp < now"}]},
        ],
    )

    result = parse_tool_response(
        "Bug in auth middleware. Token expiry check use `<` not `<=`. Fix:\n"
        "```python\n"
        "# In auth.py\n"
        "if token.expiry <= datetime.now():\n"
        "    raise AuthError(\"Token expired\")\n"
        "```\n"
        "Change to:\n"
        "```python\n"
        "if token.expiry < datetime.now():\n"
        "    raise AuthError(\"Token expired\")\n"
        "```",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "incomplete_fix_stub_without_tool_call"
    assert result.error.repairable is True


def test_tool_bridge_rejects_isolated_write_completion_claim_after_tool_result() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object"},
            },
        }
        for name in ["Read", "Glob", "Grep", "Edit", "Write"]
    ]
    context = build_context(
        tools,
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "/using-superpowers 你实际没有改代码啊，真正落地一?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "arguments": json.dumps(
                                {
                                    "file_path": "E:/ProjectX/mindcraft/security.py",
                                    "content": "def validate_url(url):\n    return True\n",
                                }
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "File created successfully at: E:/ProjectX/mindcraft/security.py"},
        ],
    )

    result = parse_tool_response(
        "Project code changed. Security module added at E:/ProjectX/mindcraft/security.py.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unverified_code_change_completion"
    assert result.error.repairable is True


def test_tool_bridge_infers_write_history_from_created_file_result_without_assistant_call() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object"},
            },
        }
        for name in ["Read", "Glob", "Grep", "Edit", "Write"]
    ]
    context = build_context(
        tools,
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_call_1",
                        "content": "File created successfully at: E:/ProjectX/mindcraft/config/security.py",
                    }
                ],
            },
        ],
    )

    assert "Write" in context.recent_tool_call_names

    result = parse_tool_response(
        "Mindcraft project review completed. Code structure, security, and tests are improved. Project ready.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unverified_code_change_completion"
    assert result.error.repairable is True


def test_tool_bridge_rejects_markdown_fenced_tool_call_artifact_without_tool_json() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object"},
            },
        }
        for name in ["Read", "Glob", "Grep", "Edit", "Write"]
    ]
    context = build_context(
        tools,
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Glob",
                        "input": {"path": "E:/ProjectX/mindcraft", "pattern": "*"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "README.md\nrewrite_content.py\nconfig/settings.py",
                    }
                ],
            },
        ],
    )

    result = parse_tool_response(
        '```bash\nGlob({"path":"E:/ProjectX/mindcraft","pattern":"*"})\n```\n'
        "I've created the Mindcraft project. Your next step is to install dependencies.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "tool_call_markdown_without_tool_json"
    assert result.error.repairable is True


def test_tool_bridge_rejects_completion_claim_without_mutating_tool_history() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object"},
            },
        }
        for name in ["Read", "Glob", "Grep", "Edit", "Write"]
    ]
    context = build_context(
        tools,
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Glob",
                        "input": {"path": "E:/ProjectX/mindcraft", "pattern": "*"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "README.md\nrewrite_content.py\nconfig/settings.py",
                    }
                ],
            },
        ],
    )

    result = parse_tool_response(
        "I've created the Mindcraft project with all core components, configurations, and tests. "
        "Your next step is to install the dependencies and configure API keys.",
        context,
    )

    assert result.tool_calls == []
    assert result.error is not None
    assert result.error.kind == "unverified_code_change_completion"
    assert result.error.repairable is True


def test_qwen_coder_recovers_isolated_write_completion_after_tool_result(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class IsolatedWriteCompletionQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("Project code changed. Security module added at E:/ProjectX/mindcraft/security.py.")
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_2","name":"Read","input":{"file_path":"E:/ProjectX/mindcraft/fetch_wechat.py"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=IsolatedWriteCompletionQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "/using-superpowers 你实际没有改代码啊，真正落地一?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Write",
                            "input": {
                                "file_path": "E:/ProjectX/mindcraft/security.py",
                                "content": "def validate_url(url):\n    return True\n",
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "File created successfully at: E:/ProjectX/mindcraft/security.py",
                        }
                    ],
                },
            ],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Grep", "description": "Search files", "input_schema": {"type": "object"}},
                {"name": "Edit", "description": "Edit files", "input_schema": {"type": "object"}},
                {"name": "Write", "description": "Write files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "unverified_code_change_completion" in retry_prompt
    assert "Read/Grep" in retry_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_2",
            "name": "Read",
            "input": {"file_path": "E:/ProjectX/mindcraft/fetch_wechat.py"},
        }
    ]


def test_qwen_coder_recovers_write_after_failed_read_to_discovery_tool(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []
    bad_write = (
        '```tool_json\n'
        '{"calls":[{"id":"call_2","name":"Write","input":{"file_path":"mindcraft_system.py","content":"# guessed module"}}]}'
        "\n```"
    )

    class MissingReadThenWriteQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) in {1, 2}:
                return _openai_response(bad_write)
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_3","name":"Glob","input":{"path":".","pattern":"*.py"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=MissingReadThenWriteQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Read",
                            "input": {"file_path": "E:/ProjectX/mindcraft/mindcraft_system.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "is_error": True,
                            "content": "File does not exist. Note: your current working directory is E:\\ProjectX\\mindcraft.",
                        }
                    ],
                },
            ],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
                {"name": "Grep", "description": "Search files", "input_schema": {"type": "object"}},
                {"name": "Edit", "description": "Edit files", "input_schema": {"type": "object"}},
                {"name": "Write", "description": "Write files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 3
    repair_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    recovery_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[2]["messages"])
    assert "write_after_failed_read_without_discovery" in repair_prompt
    assert "write_after_failed_read_without_discovery" in recovery_prompt
    assert "do not Write the same missing path" in recovery_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_3",
            "name": "Glob",
            "input": {"path": ".", "pattern": "*.py"},
        }
    ]


def test_qwen_coder_recovers_write_under_failed_glob_directory_to_root_discovery(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class FailedGlobDirectoryThenWriteQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) in {1, 2}:
                return _openai_response(
                    '```tool_json\n'
                    '{"calls":[{"id":"call_2","name":"Write",'
                    '"input":{"file_path":"tests/integration_test.py","content":"# guessed test"}}]}'
                    "\n```"
                )
            return _openai_response(
                '```tool_json\n'
                '{"calls":[{"id":"call_3","name":"Glob","input":{"path":".","pattern":"*.py"}}]}'
                "\n```"
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=FailedGlobDirectoryThenWriteQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [
                {"role": "user", "content": "/using-superpowers implement the local codebase plan"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Glob",
                            "input": {"path": "E:/ProjectX/mindcraft/tests", "pattern": "*"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "is_error": True,
                            "content": "Directory does not exist: E:/ProjectX/mindcraft/tests. Note: your current working directory is E:\\ProjectX\\mindcraft.",
                        }
                    ],
                },
            ],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
                {"name": "Grep", "description": "Search files", "input_schema": {"type": "object"}},
                {"name": "Edit", "description": "Edit files", "input_schema": {"type": "object"}},
                {"name": "Write", "description": "Write files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 3
    recovery_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[2]["messages"])
    assert "write_after_failed_path_without_discovery" in recovery_prompt
    assert "do not Write under the missing path" in recovery_prompt
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_call_3",
            "name": "Glob",
            "input": {"path": ".", "pattern": "*.py"},
        }
    ]


def test_tool_bridge_allows_completion_after_read_and_edit_tool_history() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object"},
            },
        }
        for name in ["Read", "Grep", "Edit", "Write"]
    ]
    context = build_context(
        tools,
        ToolBridgeConfig(exposure_policy="local-agent", max_tools_in_prompt=32),
    )
    context = prefer_local_tools_for_local_agent_task(
        context,
        [
            {"role": "user", "content": "真正落地一个代码改?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": '{"file_path":"fetch_wechat.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "def main():\n    pass\n"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "Edit", "arguments": '{"file_path":"fetch_wechat.py","old_string":"pass","new_string":"return"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "The file fetch_wechat.py has been updated successfully."},
        ],
    )

    result = parse_tool_response(
        "已修?fetch_wechat.py，并基于读取到的现有入口完成接入?",
        context,
    )

    assert result.error is None
    assert result.content


def test_qwen_coder_provider_does_not_claim_gateway_mcp_support() -> None:
    provider = PROVIDERS["qwen-coder"]

    assert provider.supports_native_tools is False
    assert provider.capabilities.get("mcp") is not True
    assert provider.capabilities.get("artifacts") is not True


def test_qwen_coder_legacy_plus_alias_maps_to_web_model() -> None:
    assert normalize_qwen_coder_model("qwen-coder/qwen-coder-plus") == "qwen3-coder-plus"
    assert normalize_qwen_coder_model("qwen-coder/qwen3-coder-plus") == "qwen3-coder-plus"


def test_qwen_coder_client_sends_web_model_alias_for_legacy_plus() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-test"}}, request=request)
        if request.url.path.endswith("/api/v2/chat/completions"):
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    qwen = QwenCoderClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = qwen.chat_completions(
        {
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert seen["body"]["model"] == "qwen3-coder-plus"
    assert seen["body"]["messages"][0]["models"] == ["qwen3-coder-plus"]


def test_qwen_coder_client_does_not_enable_mcp_by_default() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-test"}}, request=request)
        if request.url.path.endswith("/api/v2/chat/completions"):
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    qwen = QwenCoderClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = qwen.chat_completions(
        {
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    feature_config = seen["body"]["messages"][0]["feature_config"]
    assert feature_config.get("mcp_enabled") is not True
    assert feature_config.get("artifacts_enabled") is not True
    assert feature_config.get("thinking_enabled") is False
    assert "output_schema" not in feature_config


def test_qwen_coder_stream_ignores_metadata_only_title_events() -> None:
    text = "\n\n".join(
        [
            'data: {"choices":[{"delta":{"content":"{\\"title\\":\\"Review code and plan improvements\\"}","phase":"answer"}}]}',
            "data: [DONE]",
        ]
    )

    assert parse_qwen_coder_stream_text(text) == ""


def test_qwen_coder_stream_checks_deadline_for_non_json_heartbeats() -> None:
    ticks = iter([0.0, 31.0])

    with pytest.raises(TimeoutError, match="Qwen Coder Web request exceeded 30s"):
        _collect_qwen_coder_stream_lines(
            [": keep-alive"],
            deadline_seconds=30,
            monotonic=lambda: next(ticks),
        )


def test_qwen_coder_stream_returns_complete_dsml_tool_block_before_done() -> None:
    consumed = 0
    tool_text = (
        '<|DSML|tool_calls><|DSML|invoke name="Read">'
        '<|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter>'
        '</|DSML|invoke></|DSML|tool_calls>'
    )

    def lines():
        nonlocal consumed
        consumed += 1
        yield "data: " + json.dumps({"choices": [{"delta": {"content": tool_text, "phase": "answer"}}]})
        consumed += 1
        raise AssertionError("stream should stop once a complete DSML tool block is available")

    content = _collect_qwen_coder_stream_lines(lines(), deadline_seconds=30)

    assert consumed == 1
    assert "<|DSML|tool_calls>" in content


def test_qwen_coder_stream_returns_runaway_tool_bridge_text_before_timeout() -> None:
    consumed = 0

    def lines():
        nonlocal consumed
        for _ in range(10):
            consumed += 1
            yield "data: " + json.dumps({"choices": [{"delta": {"content": "plain text without a tool call " * 4, "phase": "answer"}}]})
        raise AssertionError("stream should stop before the hard provider timeout")

    content = _collect_qwen_coder_stream_lines(
        lines(),
        deadline_seconds=300,
        max_output_chars_without_tool_json=180,
    )

    assert consumed < 10
    assert len(content) >= 180
    assert "tool_json" not in content


def test_qwen_coder_client_keeps_coding_prompt_budget_above_generic_default() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-test"}}, request=request)
        if request.url.path.endswith("/api/v2/chat/completions"):
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    qwen = QwenCoderClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        prompt_max_chars=12000,
    )

    response = qwen.chat_completions(
        {
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "x" * 16000}],
        }
    )

    sent_prompt = seen["body"]["messages"][0]["content"]
    assert response["choices"][0]["message"]["content"] == "ok"
    assert len(sent_prompt) > 12000
    assert "Prompt content was compacted" not in sent_prompt
    assert qwen.last_diagnostic["prompt_max_chars"] == 32000
    assert qwen.last_diagnostic["prompt_compacted"] is False


def test_qwen_coder_messages_compaction_preserves_tool_bridge_protocol_with_large_later_history() -> None:
    huge_system_prefix = "system bootstrap\n" + ("skill listing entry\n" * 500)
    tool_protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools: Read, Glob, Write.\n"
        "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
    )
    large_later_history = ("dependency path .venv/Lib/site-packages/pkg/module.py\n" * 300) + "LATEST_SENTINEL"

    prompt, files = qwen_coder_messages_to_prompt_and_files(
        [
            {"role": "system", "content": huge_system_prefix + "\n\n" + tool_protocol},
            {"role": "user", "content": large_later_history},
        ],
        max_prompt_chars=1600,
    )

    assert files == []
    assert len(prompt) <= 1600
    assert "Prompt content was compacted" in prompt
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert "Required tool-call format" in prompt
    assert "<|DSML|tool_calls>" in prompt
    assert "LATEST_SENTINEL" in prompt


def test_qwen_coder_messages_prepend_stateless_web_api_guard() -> None:
    prompt, files = qwen_coder_messages_to_prompt_and_files(
        [{"role": "user", "content": "只回复 OK"}],
        max_prompt_chars=4000,
    )

    assert files == []
    assert prompt.startswith("System: You are serving a stateless WebAI Gateway API request.")
    assert "不要引用网页端旧会话" in prompt
    assert "User: 只回复 OK" in prompt


def test_qwen_coder_messages_compaction_uses_ds2api_history_continuation() -> None:
    huge_system_prefix = "system bootstrap\n" + ("skill listing entry\n" * 500)
    tool_protocol = (
        "You are using WebAI Gateway's strict tool bridge.\n"
        "Available tools: Read, Glob, Edit.\n"
        "Required tool-call format:\n<|DSML|tool_calls>\n</|DSML|tool_calls>"
    )

    prompt, files = qwen_coder_messages_to_prompt_and_files(
        [
            {"role": "system", "content": huge_system_prefix + "\n\n" + tool_protocol},
            {"role": "assistant", "content": "Assistant requested tool calls: Read({\"file_path\":\"README.md\"})"},
            {"role": "tool", "content": "README content"},
            {"role": "user", "content": "Continue the implementation. LATEST_SENTINEL"},
        ],
        max_prompt_chars=1800,
    )

    assert files == []
    assert len(prompt) <= 1800
    assert "# DS2API_HISTORY.txt" in prompt
    assert "Prior conversation history and tool progress." in prompt
    assert "Continue from the latest state in the provided DS2API_HISTORY.txt context" in prompt
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert "<|DSML|tool_calls>" in prompt
    assert "LATEST_SENTINEL" in prompt


def test_qwen_coder_client_retries_metadata_only_phase_response() -> None:
    attempts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-test"}}, request=request)
        if request.url.path.endswith("/api/v2/chat/completions"):
            attempts.append(json.loads(request.content.decode("utf-8")))
            if len(attempts) == 1:
                content = b'data: {"choices":[{"delta":{"content":"{\\"title\\":\\"Review code and plan improvements\\"}","phase":"answer"}}]}\n\ndata: [DONE]\n\n'
            else:
                content = b'data: {"choices":[{"delta":{"content":"final answer","phase":"answer"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=content, request=request, headers={"content-type": "text/event-stream"})
        return httpx.Response(404, request=request)

    qwen = QwenCoderClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = qwen.chat_completions(
        {
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "review code"}],
        }
    )

    assert response["choices"][0]["message"]["content"] == "final answer"
    assert len(attempts) == 2
    assert "Do not output JSON metadata" in attempts[1]["messages"][0]["content"]
    assert qwen.last_diagnostic["metadata_only_response"] is True
    assert qwen.last_diagnostic["metadata_retry_count"] == 1


def test_qwen_web_chat_passes_configured_provider_timeout(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingQwenClient:
        def __init__(
            self,
            credential: dict[str, Any],
            http_client: httpx.Client | None = None,
            request_timeout_seconds: float | None = None,
            prompt_max_chars: int | None = None,
        ) -> None:
            seen["timeout"] = request_timeout_seconds
            seen["prompt_max_chars"] = prompt_max_chars

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("来自 Qwen 网页模型")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(request_timeout_seconds=240, prompt_max_chars=30000),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=CapturingQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "qwen-web/qwen3.6-plus", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    assert seen["timeout"] == 240
    assert seen["prompt_max_chars"] == 30000


def test_qwen_coder_chat_passes_configured_provider_runtime(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingQwenCoderClient:
        def __init__(
            self,
            credential: dict[str, Any],
            http_client: httpx.Client | None = None,
            request_timeout_seconds: float | None = None,
            prompt_max_chars: int | None = None,
        ) -> None:
            seen["timeout"] = request_timeout_seconds
            seen["prompt_max_chars"] = prompt_max_chars

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("来自 Qwen Coder 网页模型")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(request_timeout_seconds=240, prompt_max_chars=30000),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=CapturingQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "qwen-coder/qwen-coder-plus", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    assert seen["timeout"] == 240
    assert seen["prompt_max_chars"] == 30000


def test_qwen_web_timeout_returns_gateway_timeout(tmp_path: Path) -> None:
    class TimeoutQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential
            self.last_diagnostic = {
                "prompt_chars": 12000,
                "prompt_max_chars": 12000,
                "prompt_compacted": True,
                "stream_events": 3,
                "output_chars": 0,
                "think_chars": 512,
            }

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise TimeoutError("Qwen Web request exceeded 35s")

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=TimeoutQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={"model": "qwen-web/qwen3.6-plus", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1024},
    )

    assert response.status_code == 504
    assert response.headers["x-should-retry"] == "false"
    detail = response.json()["detail"]
    assert "Qwen Web 响应超时" in detail
    assert "prompt_chars=12000" in detail
    assert "prompt_compacted=True" in detail
    assert "stream_events=3" in detail
    assert "session-secret" not in detail
    assert "bearer-secret" not in detail
    diagnostics = client.get("/api/admin/request-diagnostics").json()["events"]
    error_event = diagnostics[-1]
    assert error_event["kind"] == "completion_error"
    assert error_event["endpoint"] == "/v1/chat/completions"
    assert error_event["route"] == "qwen-web"
    assert error_event["statusCode"] == 504
    assert error_event["errorKind"] == "timeout"
    assert "Qwen Web request exceeded 35s" in error_event["errorPreview"]
    assert error_event["providerPromptChars"] == 12000
    assert error_event["providerPromptCompacted"] is True
    assert error_event["providerStreamEvents"] == 3
    assert "session-secret" not in json.dumps(error_event)
    assert "bearer-secret" not in json.dumps(error_event)


def test_qwen_web_repair_retry_timeout_returns_gateway_timeout(tmp_path: Path) -> None:
    attempts = {"count": 0}

    class RepairTimeoutQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential
            self.last_diagnostic = {
                "prompt_chars": 32000,
                "prompt_max_chars": 32000,
                "prompt_compacted": True,
                "stream_events": 9,
            }

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return _openai_response('```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":\n```')
            raise TimeoutError("Qwen Web repair retry exceeded 300s")

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=RepairTimeoutQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}],
        },
    )

    assert attempts["count"] == 2
    assert response.status_code == 504
    assert response.headers["x-should-retry"] == "false"
    detail = response.json()["detail"]
    assert "Qwen Web" in detail
    assert "repair retry exceeded" in detail
    assert "prompt_compacted=True" in detail
    assert "session-secret" not in detail
    assert "bearer-secret" not in detail


def test_qwen_coder_repair_retry_timeout_returns_gateway_timeout(tmp_path: Path) -> None:
    attempts = {"count": 0}

    class RepairTimeoutQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential
            self.last_diagnostic = {
                "prompt_chars": 32000,
                "prompt_max_chars": 32000,
                "prompt_compacted": True,
                "stream_events": 7,
            }

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return _openai_response('```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":\n```')
            raise TimeoutError("Qwen Coder repair retry exceeded 300s")

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=RepairTimeoutQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "read README"}],
            "tools": [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}],
        },
    )

    assert attempts["count"] == 2
    assert response.status_code == 504
    assert response.headers["x-should-retry"] == "false"
    detail = response.json()["detail"]
    assert "Qwen Coder" in detail
    assert "repair retry exceeded" in detail
    assert "prompt_compacted=True" in detail
    assert "session-secret" not in detail
    assert "bearer-secret" not in detail


def test_qwen_stream_timeout_carries_sanitized_diagnostics() -> None:
    ticks = iter([0.0, 0.0, 181.0])

    with pytest.raises(TimeoutError) as raised:
        _collect_qwen_stream_lines(
            ['data: {"choices":[{"delta":{"content":"thinking","phase":"think"}}]}'],
            deadline_seconds=180,
            monotonic=lambda: next(ticks),
        )

    diagnostic = getattr(raised.value, "diagnostic", {})
    assert diagnostic["stream_events"] == 1
    assert diagnostic["think_chars"] == len("thinking")
    assert diagnostic["output_chars"] == 0


def test_qwen_web_bridge_hides_runtime_tools_from_web_prompt(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response(
                '```tool_json\n{"name":"fetch_url","args":{"url":"https://github.com/Einsia/OpenChronicle"}}\n```'
            )

    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="always"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=store,
            qwen_client_factory=CapturingQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "What is OpenChronicle?"}],
            "tools": [
                {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    payload = seen["payload"]
    assert "tools" not in payload
    prompt = "\n".join(str(message.get("content", "")) for message in payload["messages"] if message.get("role") == "system")
    assert '"name": "fetch_url"' in prompt
    assert '"name": "terminal"' not in prompt
    assert '"name": "write_file"' not in prompt
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "fetch_url"


def test_qwen_web_auto_activation_routes_plain_web_search_to_native_provider(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response("Qwen native web search returned a final answer.")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=CapturingQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "Search the web for the official GLM-5.1 experience URL."}],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    payload = seen["payload"]
    assert payload["_webai_native_web_search"] is True
    assert "tools" not in payload
    assert "tool_choice" not in payload
    prompt = "\n".join(str(message.get("content", "")) for message in payload["messages"])
    assert "WebAI Gateway's strict tool bridge" not in prompt
    assert "Provider" in prompt
    body = response.json()
    assert body["stop_reason"] == "end_turn"
    assert body["content"] == [{"type": "text", "text": "Qwen native web search returned a final answer."}]


def test_qwen_web_native_search_retries_placeholder_response_for_final_answer(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class PlaceholderQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("I will search for the official GLM-5.1 page.")
            return _openai_response("https://chatglm.cn/\nSource: official ChatGLM entry.")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=PlaceholderQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "Search the web for the official GLM-5.1 page."}],
            "tools": [{"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    assert seen_payloads[0]["_webai_native_web_search"] is True
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert seen_payloads[1]["_webai_native_web_search"] is True
    assert "https://chatglm.cn/" in response.json()["content"][0]["text"]


def test_qwen_web_continue_inherits_recent_native_search_context(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class ContinuationQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response("Continuation: verify chatglm.cn and bigmodel.cn first.")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=ContinuationQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [
                {"role": "user", "content": "Search the web for the official GLM-5.1 page."},
                {"role": "assistant", "content": [{"type": "text", "text": "https://chatglm.cn/\nSource: official entry."}]},
                {"role": "user", "content": "继续"},
            ],
            "tools": [{"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    payload = seen["payload"]
    assert payload["_webai_native_web_search"] is True
    assert "tools" not in payload
    prompt = "\n".join(str(message.get("content", "")) for message in payload["messages"])
    assert "WebAI Gateway's strict tool bridge" not in prompt
    assert "Provider" in prompt
    assert response.json()["content"] == [{"type": "text", "text": "Continuation: verify chatglm.cn and bigmodel.cn first."}]


def test_qwen_web_native_search_off_keeps_web_tool_bridge_available(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class ToolQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response('```tool_json\n{"calls":[{"id":"toolu_web","name":"WebSearch","input":{"query":"GLM-5.1 官方体验网页"}}]}\n```')

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="off"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=ToolQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "联网搜索?GLM5.1 官方体验网页是哪?"}],
            "tools": [{"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["payload"]["messages"])
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert seen["payload"].get("_webai_native_web_search") is False
    assert response.json()["content"] == [{"type": "tool_use", "id": "toolu_web", "name": "WebSearch", "input": {"query": "GLM-5.1 官方体验网页"}}]


def test_qwen_web_auto_activation_keeps_tool_bridge_for_local_agent_task(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class CapturingQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return _openai_response('```tool_json\n{"calls":[{"id":"toolu_read","name":"Read","input":{"file_path":"README.md"}}]}\n```')

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=CapturingQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "/init"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    prompt = "\n".join(str(message.get("content", "")) for message in seen["payload"]["messages"])
    assert "WebAI Gateway's strict tool bridge" in prompt
    assert seen["payload"].get("_webai_native_web_search") is not True
    assert response.json()["content"] == [{"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {"file_path": "README.md"}}]


def test_qwen_web_tool_bridge_extracts_embedded_json_tool_call_after_prose(tmp_path: Path) -> None:
    class EmbeddedJsonQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                'Using using-superpowers to structure the implementation plan.\n'
                '{\n'
                '  "calls": [\n'
                '    {\n'
                '      "id": "call_1",\n'
                '      "name": "Skill",\n'
                '      "input": {"skill_name": "using-superpowers"}\n'
                '    }\n'
                '  ]\n'
                '}'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=EmbeddedJsonQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "/using-superpowers"}],
            "tools": [
                {
                    "name": "Skill",
                    "description": "Activate an available skill",
                    "input_schema": {
                        "type": "object",
                        "properties": {"skill_name": {"type": "string"}},
                        "required": ["skill_name"],
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_call_1", "name": "Skill", "input": {"skill_name": "using-superpowers"}}
    ]


def test_qwen_web_rejects_repeated_skill_without_progress(tmp_path: Path) -> None:
    class RepeatedSkillQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_2","name":"Skill","input":{"skill":"simplify"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=RepeatedSkillQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [
                {"role": "user", "content": "继续完成当前代码修复?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Skill",
                            "input": {"skill": "simplify"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "Launching skill: simplify",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": "# Simplify: Code Review and Cleanup\n\nReview all changed files for reuse and quality.",
                },
            ],
            "tools": [
                {
                    "name": "Skill",
                    "description": "Activate an available skill",
                    "input_schema": {"type": "object", "properties": {"skill": {"type": "string"}}},
                },
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Grep", "description": "Search files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    content = response.json()["content"]
    assert content[0]["type"] == "text"
    assert "repeat_same_skill_without_progress" in content[0]["text"]

def test_qwen_web_retries_off_task_environment_config_final_to_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class StatusLineThenReadQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    'CAVEMAN MODE ACTIVE. Statusline badge not configured. Add to `~/.claude/settings.json`: '
                    '```json\n"statusLine": {"type": "command", "command": "powershell -File caveman-statusline.ps1"}\n```'
                )
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_read","name":"Read","input":{"file_path":"README.md"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=StatusLineThenReadQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [
                {"role": "user", "content": "Review the local project and continue the current fix."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_glob", "name": "Glob", "input": {"pattern": "**/*.py"}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_glob", "content": "app.py"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_skill",
                            "name": "Skill",
                            "input": {"skill": "using-superpowers"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call_skill", "content": "Skill loaded"}],
                },
                {"role": "user", "content": "continue"},
            ],
            "tools": [
                {"name": "AskUserQuestion", "description": "Ask user", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "List files", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Skill", "description": "Activate skill", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "off_task_environment_configuration_final" in retry_prompt
    assert "statusLine" in retry_prompt
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_call_read", "name": "Read", "input": {"file_path": "README.md"}}
    ]


def test_qwen_web_tool_bridge_extracts_multiline_agent_summary_calls(tmp_path: Path) -> None:
    class MultilineAgentSummaryQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                '''
I see the issue - when launching Agent tools, I need to provide both prompt and description parameters.

Assistant requested tool calls:
- Agent({
"prompt": "搜索代码库中与制定完善的改进计划并落地相关的现有实现模式?",
"description": "搜索现有改进计划模板和文档模?"
})
- Agent({
"prompt": "分析代码库中与计划执行相关的核心组件?",
"description": "分析计划执行相关的核心组?"
})
'''
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="always", exposure_policy="all", max_calls_per_turn=4),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=MultilineAgentSummaryQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "/using-superpowers 制定一个完善的改进计划并落?"}],
            "tools": [
                {
                    "name": "Agent",
                    "description": "Launch a subagent",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["prompt", "description"],
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert response.json()["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_summary_1",
            "name": "Agent",
            "input": {
                "prompt": "搜索代码库中与制定完善的改进计划并落地相关的现有实现模式?",
                "description": "搜索现有改进计划模板和文档模?",
            },
        },
        {
            "type": "tool_use",
            "id": "toolu_summary_2",
            "name": "Agent",
            "input": {
                "prompt": "分析代码库中与计划执行相关的核心组件?",
                "description": "分析计划执行相关的核心组?",
            },
        },
    ]


def test_qwen_web_tool_bridge_repairs_embedded_unknown_task_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class UnknownTaskThenAllowedToolQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response(
                    '?{"calls":[{"id":"call_1","name":"Task","input":{"description":"搜索智谱清言/GLM5支持",'
                    '"prompt":"在当前代码库中搜索是否包含智谱清言","subagent_type":"general-purpose"}}]}'
                )
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_2","name":"Read","input":{"file_path":"webai_gateway/web_auth.py"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=UnknownTaskThenAllowedToolQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "原始 web2api 项目支持智谱清言的网页和 GLM5 模型吗？"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "未知工具：Task" in retry_prompt
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_call_2", "name": "Read", "input": {"file_path": "webai_gateway/web_auth.py"}}
    ]


def test_qwen_web_tool_bridge_recovers_when_unknown_tool_repair_repeats(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class RepeatedUnknownTaskThenAnswerQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) <= 2:
                return _openai_response(
                    '{"calls":[{"id":"call_1","name":"Task","input":{"description":"搜索智谱清言/GLM5支持",'
                    '"prompt":"检查是否支持智谱清言","subagent_type":"general-purpose"}}]}'
                )
            return _openai_response("结论：当前只发现 zAI/zai_is_text ?glm-4.6 支持，未发现 chatglm.cn ?GLM-5 网页模型实现?")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=RepeatedUnknownTaskThenAnswerQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "原始 web2api 项目支持智谱清言的网页和 GLM5 模型吗？"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 3
    final_retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[2]["messages"])
    assert "不要再请求未列出的工具" in final_retry_prompt
    assert "Task" in final_retry_prompt
    assert response.json()["content"] == [
        {"type": "text", "text": "结论：当前只发现 zAI/zai_is_text ?glm-4.6 支持，未发现 chatglm.cn ?GLM-5 网页模型实现?"}
    ]


def test_qwen_web_retries_incomplete_prelude_for_direct_answer(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class PreludeThenAnswerQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("我来设计一个支?GLM-5 网页授权的实现计划?")
            return _openai_response("完整计划：第一步梳?provider 能力，第二步实现授权捕获，第三步补充配置和测试?")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=PreludeThenAnswerQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "设计 GLM-5 网页授权计划"}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "只包含准备动作或开头" in retry_prompt
    assert response.json()["content"] == [
        {"type": "text", "text": "完整计划：第一步梳?provider 能力，第二步实现授权捕获，第三步补充配置和测试?"}
    ]


def test_qwen_web_retries_decorated_skill_check_prelude_for_direct_answer(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class SkillCheckPreludeThenAnswerQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("🤖 让我检查一下当前已安装?skill 列表，确认这两个 GitHub 相关?skill 是否已经可用?")
            return _openai_response("这两?GitHub 相关 skill 已经可用，可以继续使用?")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=SkillCheckPreludeThenAnswerQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "已经安装了还是还没安装啊，现在就是可用状态吗?"}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    assert response.json()["content"] == [
        {"type": "text", "text": "这两?GitHub 相关 skill 已经可用，可以继续使用?"}
    ]


def test_qwen_web_incomplete_retry_empty_response_returns_diagnostic_text(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class SkillCheckPreludeThenEmptyQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("🤖 让我检查一下当前已安装?skill 列表，确认这两个 GitHub 相关?skill 是否已经可用?")
            return {}

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=SkillCheckPreludeThenEmptyQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "已经安装了还是还没安装啊，现在就是可用状态吗?"}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    assert response.json()["content"] == [{"type": "text", "text": EMPTY_ASSISTANT_RESPONSE_TEXT}]


def test_parse_chat_response_normalizes_malformed_empty_upstream_response() -> None:
    parsed, bridge_result = parse_chat_response(
        {},
        bridge=False,
        allowed_tools=set(),
        model="web-model",
        return_bridge_result=True,
    )

    assert parsed["choices"][0]["message"]["content"] == EMPTY_ASSISTANT_RESPONSE_TEXT
    assert bridge_result.content == EMPTY_ASSISTANT_RESPONSE_TEXT


def test_parse_chat_response_ignores_spontaneous_tool_json_when_context_has_no_tools() -> None:
    bridge_context = build_context(
        [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        ToolBridgeConfig(activation_policy="auto", exposure_policy="local-agent"),
    )
    bridge_context = prefer_local_tools_for_local_agent_task(
        bridge_context,
        [{"role": "user", "content": "CUA 是否需?API Key?"}],
    )

    parsed, bridge_result = parse_chat_response(
        _openai_response('```tool_json\n{"calls":[{"id":"call_1","name":"capabilities_resolve","input":{"capability_name":"cua"}}]}\n```'),
        bridge=True,
        allowed_tools=set(),
        model="qwen-web/qwen3.6-max-preview",
        bridge_context=bridge_context,
        return_bridge_result=True,
    )

    message = parsed["choices"][0]["message"]
    assert bridge_context.enabled is False
    assert bridge_result.error is None
    assert "webai_tool_bridge" not in message
    assert "capabilities_resolve" in message["content"]


def test_qwen_web_retries_incomplete_prelude_to_tool_call_when_bridge_active(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class PreludeThenToolQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("我来设计一个支?GLM-5 网页授权的实现计划?")
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Read","input":{"file_path":"webai_gateway/web_auth.py"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=PreludeThenToolQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "当前项目支持 GLM-5 网页授权，设计计?"}],
            "tools": [{"name": "Read", "description": "Read files", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "如果确实需要工具" in retry_prompt
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_call_1", "name": "Read", "input": {"file_path": "webai_gateway/web_auth.py"}}
    ]


def test_qwen_web_tool_bridge_repairs_deferred_research_without_tool_call(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []

    class DeferredResearchQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return _openai_response("我来设计一个实现计划。首先让我研究现有的网页授权实现模式?")
            return _openai_response('```tool_json\n{"calls":[{"id":"toolu_read","name":"Read","input":{"file_path":"webai_gateway/web_auth.py"}}]}\n```')

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=DeferredResearchQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "当前这个项目也需要支?GLM5 的网页授权，你设计一个计?"}],
            "tools": [
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
                {"name": "Glob", "description": "Find files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 2
    retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[1]["messages"])
    assert "没有发起工具调用" in retry_prompt
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_read", "name": "Read", "input": {"file_path": "webai_gateway/web_auth.py"}}
    ]


def test_qwen_web_tool_bridge_recovers_when_permission_denial_repair_repeats(tmp_path: Path) -> None:
    seen_payloads: list[dict[str, Any]] = []
    denial = (
        "由于我无法直接访问您的文件系统或执行命令，我将为您提供手动更新GitHub项目的详细步骤?"
        "这是最安全可靠的方式，避免意外数据丢失?"
    )

    class RepeatedPermissionDenialThenToolQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen_payloads.append(payload)
            if len(seen_payloads) <= 2:
                return _openai_response(denial)
            return _openai_response(
                '```tool_json\n{"calls":[{"id":"toolu_bash","name":"Bash","input":{"command":"git status --short"}}]}\n```'
            )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=RepeatedPermissionDenialThenToolQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "更新项目并推送到 GitHub"}],
            "tools": [
                {"name": "Bash", "description": "Run shell commands", "input_schema": {"type": "object"}},
                {"name": "Read", "description": "Read files", "input_schema": {"type": "object"}},
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert len(seen_payloads) == 3
    final_retry_prompt = "\n".join(str(message.get("content", "")) for message in seen_payloads[2]["messages"])
    assert "manual steps" in final_retry_prompt
    assert "Bash" in final_retry_prompt
    assert response.json()["content"] == [
        {"type": "tool_use", "id": "toolu_bash", "name": "Bash", "input": {"command": "git status --short"}}
    ]


def test_qwen_web_streaming_empty_direct_response_returns_diagnostic_text(tmp_path: Path) -> None:
    class EmptyQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        provider_runtime=ProviderRuntimeConfig(native_web_search_policy="auto"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=EmptyQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "stream": True,
            "messages": [{"role": "user", "content": "联网搜索一下最新模型发布信?"}],
            "tools": [{"name": "WebSearch", "description": "Search the web", "input_schema": {"type": "object"}}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    assert "上游模型返回了空响应" in response.text


def test_qwen_web_streaming_text_uses_standard_openai_sse_chunks(tmp_path: Path) -> None:
    class TextQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("这两个 GitHub 相关 skill 已经可用。")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=TextQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "stream": True,
            "messages": [{"role": "user", "content": "已经安装了还是还没安装啊，现在就是可用状态吗?"}],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    payloads = _openai_sse_payloads(response.text)
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert payloads[1]["choices"][0]["delta"] == {"content": "这两个 GitHub 相关 skill 已经可用。"}
    assert payloads[1]["choices"][0]["finish_reason"] is None
    assert payloads[-1]["choices"][0]["delta"] == {}
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_request_diagnostics_records_qwen_stream_content(tmp_path: Path) -> None:
    class TextQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("这两个 GitHub 相关 skill 已经可用。")

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=TextQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "stream": True,
            "messages": [{"role": "user", "content": "确认 skill 是否可用"}],
            "max_tokens": 1024,
        },
    )
    diagnostics = client.get("/api/admin/request-diagnostics").json()["events"]

    assert response.status_code == 200
    event = diagnostics[-1]
    assert event["endpoint"] == "/v1/chat/completions"
    assert event["route"] == "qwen-web"
    assert event["stream"] is True
    assert event["responseKind"] == "sse"
    assert event["ssePayloadCount"] >= 3
    assert event["responseContentChars"] == len("这两个 GitHub 相关 skill 已经可用。")
    assert event["finishReason"] == "stop"
    assert "GitHub 相关 skill" in event["responseContentPreview"]


def test_request_diagnostics_records_request_start_with_tool_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_response("plain reply"), request=request)

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="always", exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "Read the local project README."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Read files",
                        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                    },
                }
            ],
            "max_tokens": 1024,
        },
    )
    diagnostics = client.get("/api/admin/request-diagnostics").json()["events"]

    assert response.status_code == 200
    started = next(event for event in diagnostics if event["kind"] == "completion_request_started")
    assert started["endpoint"] == "/v1/chat/completions"
    assert started["route"] == "upstream"
    assert started["bridge"] is True
    assert started["requestToolCount"] == 1
    assert started["toolBridgeAllowedToolCount"] == 1
    assert started["toolBridgeAllowedTools"] == ["Read"]
    assert started["toolBridgeExposurePolicy"] == "all"
    assert diagnostics[-1]["kind"] == "completion_response"


def test_request_diagnostics_records_qwen_coder_provider_diagnostic(tmp_path: Path) -> None:
    class DiagnosticQwenCoderClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential
            self.last_diagnostic = {
                "prompt_chars": 12345,
                "prompt_max_chars": 12000,
                "prompt_compacted": True,
                "prompt_task_state_preserved": True,
                "prompt_task_state_chars": 640,
                "prompt_task_count": 4,
                "prompt_recent_tool_call_count": 3,
                "prompt_compaction_strategy": "ds2api_layered_history",
                "prompt_history_entry_count": 120,
                "prompt_latest_entry_count": 5,
                "message_count": 3,
                "artifacts_enabled": False,
                "mcp_enabled": False,
                "thinking_enabled": False,
                "metadata_only_response": True,
                "metadata_retry_count": 1,
            }

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("final answer")

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_qwen_coder_credential_store(tmp_path),
            qwen_coder_client_factory=DiagnosticQwenCoderClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-coder/qwen-coder-plus",
            "messages": [{"role": "user", "content": "review code"}],
            "max_tokens": 1024,
        },
    )
    diagnostics = client.get("/api/admin/request-diagnostics").json()["events"]

    assert response.status_code == 200
    event = diagnostics[-1]
    assert event["route"] == "qwen-coder"
    assert event["providerPromptChars"] == 12345
    assert event["providerPromptMaxChars"] == 12000
    assert event["providerPromptCompacted"] is True
    assert event["providerPromptTaskStatePreserved"] is True
    assert event["providerPromptTaskStateChars"] == 640
    assert event["providerPromptTaskCount"] == 4
    assert event["providerRecentToolCallCount"] == 3
    assert event["providerPromptCompactionStrategy"] == "ds2api_layered_history"
    assert event["providerPromptHistoryEntryCount"] == 120
    assert event["providerPromptLatestEntryCount"] == 5
    assert event["providerMessageCount"] == 3
    assert event["providerArtifactsEnabled"] is False
    assert event["providerThinkingEnabled"] is False
    assert event["providerMetadataOnlyResponse"] is True
    assert event["providerMetadataRetryCount"] == 1


def test_request_diagnostics_records_tool_bridge_error_for_upstream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"python-dotenv>=1.0.0 && pytest>=7.0.0"}}]}\n```'
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "Implement the improvement plan in this local codebase."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Run shell commands",
                        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                    },
                }
            ],
        },
    )
    diagnostics = client.get("/api/admin/request-diagnostics").json()["events"]

    assert response.status_code == 200
    event = diagnostics[-1]
    assert "toolBridgeError" not in event
    assert event["responseToolCallCount"] == 0


def test_admin_auto_research_candidates_reports_recent_bridge_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_openai_response(
                '```tool_json\n{"calls":[{"id":"call_1","name":"Bash","input":{"command":"python-dotenv>=1.0.0 && pytest>=7.0.0"}}]}\n```'
            ),
            request=request,
        )

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(create_app(config=config, http_client=httpx.Client(transport=httpx.MockTransport(handler))))

    client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "web-model",
            "messages": [{"role": "user", "content": "Implement the improvement plan in this local codebase."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Run shell commands",
                        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                    },
                }
            ],
        },
    )

    candidates = client.get("/api/admin/auto-research/candidates").json()

    assert candidates["total"] == 0


def test_request_diagnostics_redacts_response_preview_secrets(tmp_path: Path) -> None:
    class SecretTextQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("Authorization: Bearer super-secret-token token=session-secret 可以使用?")

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=SecretTextQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "测试诊断脱敏"}],
        },
    )
    diagnostics = client.get("/api/admin/request-diagnostics").json()["events"]

    assert response.status_code == 200
    preview = diagnostics[-1]["responseContentPreview"]
    assert "super-secret-token" not in preview
    assert "session-secret" not in preview
    assert "Authorization: [redacted]" in preview
    assert "token=[redacted]" in preview


def test_anthropic_qwen_web_accepts_image_blocks_as_multimodal_payload(tmp_path: Path) -> None:
    _FakeQwenMultimodalClient.captured_payload = {}
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=_FakeQwenMultimodalClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers={**_headers(), "anthropic-version": "2023-06-01"},
        json={
            "model": "qwen-web/qwen3.6-plus",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请描述这张图"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
                    ],
                }
            ],
            "max_tokens": 1024,
        },
    )

    assert response.status_code == 200
    payload = _FakeQwenMultimodalClient.captured_payload
    assert payload["messages"][0]["role"] == "system"
    assert "WebAI Gateway response language policy" in payload["messages"][0]["content"]
    user_message = next(message for message in payload["messages"] if message.get("role") == "user")
    content = user_message["content"]
    assert content[0] == {"type": "text", "text": "请描述这张图"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
    assert response.json()["content"] == [{"type": "text", "text": "我看到了图片"}]


def test_qwen_web_client_rejects_multimodal_without_upload_support() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-test"}}, request=request)
        if request.url.path.endswith("/api/v2/chat/completions"):
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    qwen = QwenWebClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError, match="Qwen Web .*multimodal"):
        qwen.chat_completions(
            {
                "model": "qwen-web/qwen3.6-plus",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                        ],
                    }
                ],
            }
        )

    assert "body" not in seen


def test_qwen_web_client_enables_auto_search_when_payload_requests_native_web_search() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/v2/chats/new"):
            return httpx.Response(200, json={"data": {"id": "chat-test"}}, request=request)
        if request.url.path.endswith("/api/v2/chat/completions"):
            seen["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"ok","phase":"answer"}}]}\n\ndata: [DONE]\n\n',
                request=request,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404, request=request)

    qwen = QwenWebClient(
        {"cookie": "qwen_session=session-secret", "bearer": "bearer-secret", "userAgent": "Chrome Test"},
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = qwen.chat_completions(
        {
            "model": "qwen-web/qwen3.6-plus",
            "messages": [{"role": "user", "content": "联网搜索 GLM-5.1 官网体验地址"}],
            "_webai_native_web_search": True,
        }
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    feature_config = seen["body"]["messages"][0]["feature_config"]
    assert seen["body"]["search"] is True
    assert feature_config["auto_search"] is True
    assert feature_config["thinking_enabled"] is False


def test_qwen_web_bridge_rejects_filtered_runtime_tool_call(tmp_path: Path) -> None:
    class RuntimeToolQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response('```tool_json\n{"name":"terminal","args":{"command":"gh repo view Einsia/OpenChronicle"}}\n```')

    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="always"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=store,
            qwen_client_factory=RuntimeToolQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "What is OpenChronicle?"}],
            "tools": [
                {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers["x-webai-tool-bridge-error"] == "unknown_tool"
    message = response.json()["choices"][0]["message"]
    assert "tool_calls" not in message
    assert "unknown_tool" in message["content"]
    assert "terminal" in message["content"]
    assert "fetch_url" in message["content"]
    assert message["webai_tool_bridge"]["error"] == "unknown_tool"


def test_qwen_web_streaming_tool_call_returns_openai_tool_chunk(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "bearer-secret",
            "userAgent": "Chrome Test",
        },
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=store,
            qwen_client_factory=_FakeQwenToolClient,
            http_client=_not_found_client(),
        )
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-plus",
            "stream": True,
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [{"type": "function", "function": {"name": "list_local_files", "parameters": {"type": "object"}}}],
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "tool_json" not in body
    assert '"tool_calls"' in body
    assert '"finish_reason":"tool_calls"' in body.replace(" ", "")
    assert "list_local_files" in body


def test_qwen_web_ds2api_dsml_markup_returns_tool_calls(tmp_path: Path) -> None:
    class DsmlQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                '<|DSML|tool_calls><|DSML|invoke name="skills_list"></|DSML|invoke></|DSML|tool_calls>'
            )

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=DsmlQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "检查已安装 skill"}],
            "tools": [{"type": "function", "function": {"name": "skills_list", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == ""
    assert message["tool_calls"][0]["function"]["name"] == "skills_list"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {}


def test_qwen_web_legacy_function_calls_markup_returns_tool_calls(tmp_path: Path) -> None:
    class LegacyFunctionCallQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response('<function_calls><invoke name="skills_list"></invoke></function_calls>')

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=LegacyFunctionCallQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "检查已安装 skill"}],
            "tools": [{"type": "function", "function": {"name": "skills_list", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == ""
    assert message["tool_calls"][0]["function"]["name"] == "skills_list"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {}


def test_qwen_web_tool_code_markup_returns_tool_calls(tmp_path: Path) -> None:
    class ToolCodeQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response(
                "为了准确回答，我将先检查当前已安装?skill 列表。\n\n<tool_code>\nskills_list()\n</tool_code>"
            )

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=ToolCodeQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "检查已安装 skill"}],
            "tools": [{"type": "function", "function": {"name": "skills_list", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == ""
    assert message["tool_calls"][0]["function"]["name"] == "skills_list"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {}


def test_qwen_web_xml_function_equals_markup_returns_tool_calls(tmp_path: Path) -> None:
    class XmlFunctionQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("<tool_call>\n<function=skills_list>\n</function>\n</tool_call>")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=XmlFunctionQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "messages": [{"role": "user", "content": "check installed skills"}],
            "tools": [{"type": "function", "function": {"name": "skills_list", "parameters": {"type": "object"}}}],
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == ""
    assert message["tool_calls"][0]["function"]["name"] == "skills_list"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {}


def test_qwen_web_streaming_legacy_function_calls_markup_returns_tool_chunk(tmp_path: Path) -> None:
    class LegacyFunctionCallQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response('<function_calls><invoke name="skills_list"></invoke></function_calls>')

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=LegacyFunctionCallQwenClient,
            http_client=_not_found_client(),
        )
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "stream": True,
            "messages": [{"role": "user", "content": "检查已安装 skill"}],
            "tools": [{"type": "function", "function": {"name": "skills_list", "parameters": {"type": "object"}}}],
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "function_calls" not in body
    assert '"tool_calls"' in body
    assert "skills_list" in body
    assert '"finish_reason":"tool_calls"' in body.replace(" ", "")


def test_qwen_web_streaming_xml_function_equals_markup_returns_tool_chunk(tmp_path: Path) -> None:
    class XmlFunctionQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("<tool_call>\n<function=skills_list>\n</function>\n</tool_call>")

    config = GatewayConfig(
        server=ServerConfig(api_key="local-dev-key"),
        upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
        tool_bridge=ToolBridgeConfig(activation_policy="auto", exposure_policy="all"),
    )
    client = TestClient(
        create_app(
            config=config,
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=XmlFunctionQwenClient,
            http_client=_not_found_client(),
        )
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "stream": True,
            "messages": [{"role": "user", "content": "check installed skills"}],
            "tools": [{"type": "function", "function": {"name": "skills_list", "parameters": {"type": "object"}}}],
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "<tool_call>" not in body
    assert "function=skills_list" not in body
    assert '"tool_calls"' in body
    assert "skills_list" in body
    assert '"finish_reason":"tool_calls"' in body.replace(" ", "")


def test_qwen_web_allows_safe_generic_tool_use_under_local_agent_policy(tmp_path: Path) -> None:
    class BareFunctionQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response('get_weather(city="Beijing")')

    store = CredentialStore(tmp_path / "credentials")
    store.save(
        "qwen",
        {
            "cookie": "qwen_session=session-secret",
            "bearer": "",
            "userAgent": "Chrome Test",
            "metadata": {"sessionToken": "session-token"},
        },
    )
    client = TestClient(
        create_app(
            config=GatewayConfig(
                server=ServerConfig(api_key="local-dev-key"),
                upstream=UpstreamConfig(base_url="http://upstream.test/v1", model="web-model"),
                tool_bridge=ToolBridgeConfig(exposure_policy="local-agent", activation_policy="auto"),
            ),
            credential_store=store,
            qwen_client_factory=BareFunctionQwenClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/messages",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-plus",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "请调?get_weather，city 必须?Beijing，不要直接回答?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "查询天气",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {"type": "tool_use", "id": "toolu_bare_1", "name": "get_weather", "input": {"city": "Beijing"}}
    ]


def test_qwen_web_streaming_tool_code_markup_returns_tool_chunk(tmp_path: Path) -> None:
    class ToolCodeQwenClient:
        def __init__(self, credential: dict[str, Any], http_client: httpx.Client | None = None) -> None:
            self.credential = credential

        def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _openai_response("<tool_code>\nskills_list()\n</tool_code>")

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=_credential_store(tmp_path),
            qwen_client_factory=ToolCodeQwenClient,
            http_client=_not_found_client(),
        )
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "qwen-web/qwen3.6-max-preview",
            "stream": True,
            "messages": [{"role": "user", "content": "检查已安装 skill"}],
            "tools": [{"type": "function", "function": {"name": "skills_list", "parameters": {"type": "object"}}}],
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "tool_code" not in body
    assert '"tool_calls"' in body
    assert "skills_list" in body
    assert '"finish_reason":"tool_calls"' in body.replace(" ", "")


def test_deepseek_web_chat_requires_browser_login(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(tmp_path / "credentials"),
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "deepseek-web/deepseek-chat", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 401
    assert "DeepSeek" in response.json()["detail"]
    assert "bearer token" in response.json()["detail"]


def test_deepseek_web_chat_rejects_cookie_only_saved_credential(tmp_path: Path) -> None:
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()
    (credential_dir / "deepseek-web.json").write_text(
        json.dumps(
            {
                "provider": "deepseek-web",
                "cookie": "ds_session_id=session-secret; HWSID=hws-secret",
                "bearer": "",
                "userAgent": "Chrome Test",
                "updatedAt": "2026-05-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(credential_dir),
            deepseek_client_factory=_FakeDeepSeekClient,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "deepseek-v4-pro[1m]", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 401
    assert "DeepSeek" in response.json()["detail"]
    assert "授权" in response.json()["detail"]


def test_qwen_web_chat_requires_browser_login(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(tmp_path / "credentials"),
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "qwen-web/qwen3.6-max", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 401
    assert "请先在控制台完成 Qwen 网页登录授权" in response.json()["detail"]


def test_qwen_web_chat_rejects_cookie_only_credentials(tmp_path: Path) -> None:
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir()
    (credential_dir / "qwen.json").write_text(
        json.dumps(
            {
                "provider": "qwen",
                "cookie": "visitor_id=visitor; device_id=device",
                "bearer": "",
                "userAgent": "Chrome Test",
                "metadata": {"sessionToken": ""},
                "updatedAt": "2026-04-26T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Qwen client should not be called with visitor cookies")

    client = TestClient(
        create_app(
            config=_config(),
            credential_store=CredentialStore(credential_dir),
            qwen_client_factory=fail_if_called,
            http_client=_not_found_client(),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "qwen-web/qwen3.6-max", "messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 401
    assert "请先在控制台完成 Qwen 网页登录授权" in response.json()["detail"]
