from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from ds2api_oracle import (
    DS2API_ORACLE_COMMIT,
    DS2API_ORACLE_VERSION,
    assert_oracle_is_latest,
    build_ds2api_runner,
    run_ds2api_runner,
)
from webai_gateway.config import ToolBridgeConfig
from webai_gateway.openai_api import build_tool_call_sse, parse_chat_response
from webai_gateway.tool_bridge import (
    build_context,
    parse_tool_response,
    prepare_openai_messages,
    sanitize_leaked_tool_protocol_output,
    should_bridge_tools,
    to_openai_tool_calls,
)


DS2API_MAIN_COMMIT = DS2API_ORACLE_COMMIT
DS2API_DEV_COMMIT = ""
DS2API_REFERENCE_ROOT = Path(os.environ.get("DS2API_REFERENCE_ROOT", r"E:\ProjectX\_reference\ds2api"))


@pytest.fixture(scope="session")
def ds2api_runner(tmp_path_factory: pytest.TempPathFactory) -> Path:
    assert_oracle_is_latest()
    return build_ds2api_runner(tmp_path_factory.mktemp("ds2api_toolcall_runner"))


def _context(names: list[str]):
    return build_context(
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
        ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
    )


def _openai_tool(name: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": schema or {"type": "object"},
        },
    }


def _tools_for_names(names: list[str]) -> list[dict[str, Any]]:
    return [_openai_tool(name) for name in names]


def _parse(text: str, names: list[str]):
    return parse_tool_response(text, _context(names))


def _single_call(text: str, names: list[str]) -> tuple[str, dict[str, Any]]:
    result = _parse(text, names)
    assert result.error is None
    assert result.content == ""
    assert len(result.tool_calls) == 1
    return result.tool_calls[0].name, result.tool_calls[0].input


def _run_ds2api_runner(
    runner: Path,
    *,
    text: str,
    names: list[str],
    mode: str = "parse",
    thinking: str = "",
    detection_thinking: str = "",
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    content_filter: bool = False,
) -> dict[str, Any]:
    return run_ds2api_runner(
        runner,
        text=text,
        names=names,
        mode=mode,
        thinking=thinking,
        detection_thinking=detection_thinking,
        tools=tools,
        tool_choice=tool_choice,
        content_filter=content_filter,
    )


def _go_test_env(workdir: Path) -> dict[str, str]:
    env = os.environ.copy()
    cache_base = Path(os.environ.get("CODEX_GO_TEST_CACHE", r"D:\CodexCache\go"))
    if not cache_base.drive or not Path(cache_base.drive + "\\").exists():
        cache_base = workdir / ".go-cache"
    (cache_base / "mod").mkdir(parents=True, exist_ok=True)
    (cache_base / "build").mkdir(parents=True, exist_ok=True)
    env.setdefault("GOMODCACHE", str(cache_base / "mod"))
    env.setdefault("GOCACHE", str(cache_base / "build"))
    return env


def _gateway_calls_snapshot(text: str, names: list[str], *, thinking: str = "") -> list[dict[str, Any]]:
    parse_source = thinking if not text.strip() and thinking.strip() else text
    result = parse_tool_response(parse_source, _context(names))
    assert result.error is None, result.error
    return [{"name": call.name, "input": call.input} for call in result.tool_calls]


def _normalize_openai_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        raw_args = fn.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = raw_args
        item = {
            "index": call.get("index", index),
            "type": call.get("type"),
            "function": {
                "name": fn.get("name"),
                "arguments": args,
            },
        }
        normalized.append(item)
    return normalized


def _gateway_openai_snapshot(text: str, tools: list[dict[str, Any]], *, stream: bool = False) -> list[dict[str, Any]]:
    context = build_context(tools, ToolBridgeConfig(exposure_policy="all", tool_profile="all"))
    result = parse_tool_response(text, context)
    assert result.error is None, result.error
    calls = to_openai_tool_calls(result.tool_calls, context)
    if stream:
        calls = [{"index": index, **call} for index, call in enumerate(calls)]
    return _normalize_openai_calls(calls)


DS2API_DIFFERENTIAL_PARSE_CASES: list[dict[str, Any]] = [
    {
        "id": "canonical_wrapper",
        "text": '<tool_calls><invoke name="Bash"><parameter name="command">pwd</parameter><parameter name="description">show cwd</parameter></invoke></tool_calls>',
        "names": ["Bash"],
    },
    {
        "id": "dsml_shell",
        "text": '<|DSML|tool_calls><|DSML|invoke name="Bash"><|DSML|parameter name="command"><![CDATA[pwd]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
        "names": ["Bash"],
    },
    {
        "id": "hyphenated_dsml_shell_heredoc",
        "text": (
            '<dsml-tool-calls>\n'
            '<dsml-invoke name="Bash">\n'
            '<dsml-parameter name="command"><![CDATA[git commit -m "$(cat <<\'EOF\'\n'
            "docs: add architecture updates\n"
            "\n"
            "Co-Authored-By: Claude Opus 4.7 noreply@anthropic.com\n"
            "EOF\n"
            ')"]]></dsml-parameter>\n'
            '<dsml-parameter name="description"><![CDATA[Create commit with architecture doc updates]]></dsml-parameter>\n'
            "</dsml-invoke>\n"
            "</dsml-tool-calls>"
        ),
        "names": ["Bash"],
    },
    {
        "id": "bare_hyphenated_tool_calls_rejected",
        "text": '<tool-calls><invoke name="Bash"><parameter name="command">pwd</parameter></invoke></tool-calls>',
        "names": ["Bash"],
    },
    {
        "id": "dsml_trailing_pipe",
        "text": '<|DSML|tool_calls| <|DSML|invoke name="terminal"><|DSML|parameter name="command"><![CDATA[find "/home" -type d]]></|DSML|parameter><|DSML|parameter name="timeout"><![CDATA[10]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
        "names": ["terminal"],
    },
    {
        "id": "dsml_extra_leading_less_than",
        "text": '<<|DSML|tool_calls><<|DSML|invoke name="Bash"><<|DSML|parameter name="command"><![CDATA[pwd]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
        "names": ["Bash"],
    },
    {
        "id": "dsml_repeated_prefix_noise",
        "text": '<<DSML|DSML|tool_calls><<DSML|DSML|invoke name="Bash"><<DSML|DSML|parameter name="command"><![CDATA[git status]]></DSML|DSML|parameter></DSML|DSML|invoke></DSML|DSML|tool_calls>',
        "names": ["Bash"],
    },
    {
        "id": "mixed_dsml_and_canonical",
        "text": '<|DSML|tool_calls><invoke name="Bash"><|DSML|parameter name="command">pwd</|DSML|parameter></invoke></|DSML|tool_calls>',
        "names": ["Bash"],
    },
    {
        "id": "simple_cdata_inline_markup_as_text",
        "text": '<tool_calls><invoke name="Write"><parameter name="description"><![CDATA[<b>urgent</b>]]></parameter></invoke></tool_calls>',
        "names": ["Write"],
    },
    {
        "id": "unclosed_cdata",
        "text": '<tool_calls><invoke name="Write"><parameter name="content"><![CDATA[hello world</parameter></invoke></tool_calls>',
        "names": ["Write"],
    },
    {
        "id": "multiline_cdata_and_repeated_params",
        "text": '<tool_calls><invoke name="write_file"><parameter name="path">script.sh</parameter><parameter name="content"><![CDATA[#!/bin/bash\necho "hello"\n]]></parameter><parameter name="item">first</parameter><parameter name="item">second</parameter></invoke></tool_calls>',
        "names": ["write_file"],
    },
    {
        "id": "nested_tool_syntax_inside_cdata",
        "text": '<tool_calls><invoke name="Write"><parameter name="content"><![CDATA[# Release notes\n\n```xml\n<tool_calls>\n  <invoke name="demo">\n    <parameter name="value">x</parameter>\n  </invoke>\n</tool_calls>\n```]]></parameter><parameter name="file_path">DS2API-4.0-Release-Notes.md</parameter></invoke></tool_calls>',
        "names": ["Write"],
    },
    {
        "id": "heredoc_cdata_with_fenced_dsml_and_literal_cdata_end",
        "text": (
            '<|DSML|tool_calls><|DSML|invoke name="Bash">'
            '<|DSML|parameter name="command"><![CDATA['
            "cat > docs/project-value.md << 'ENDOFFILE'\n"
            "# DS2API project value\n\n"
            "```xml\n"
            "<|DSML|tool_calls>\n"
            '  <|DSML|invoke name="Bash">\n'
            '    <|DSML|parameter name="command"><![CDATA[grep -E "error|fail" < input.log 2>&1]]></|DSML|parameter>\n'
            "  </|DSML|invoke>\n"
            "</|DSML|tool_calls>\n"
            "```\n\n"
            "Only the literal `]]>` needs special handling.\n\n"
            "ENDOFFILE\n"
            'echo "Done. Lines: $(wc -l < docs/project-value.md)"'
            "]]></|DSML|parameter>"
            '<|DSML|parameter name="description"><![CDATA[Write project value doc]]></|DSML|parameter>'
            "</|DSML|invoke></|DSML|tool_calls>"
        ),
        "names": ["Bash"],
    },
    {
        "id": "compact_fenced_dsml_inside_cdata",
        "text": (
            '<tool_calls><invoke name="Write"><parameter name="content"><![CDATA['
            "```xml\n"
            "<|DSML|tool_calls>\n"
            '  <|DSML|invoke name="Bash">\n'
            '    <|DSML|parameter name="command"><![CDATA[echo compact]]></|DSML|parameter>\n'
            "  </|DSML|invoke>\n"
            "</|DSML|tool_calls>\n"
            "```\n"
            "tail"
            "]]></parameter></invoke></tool_calls>"
        ),
        "names": ["Write"],
    },
    {
        "id": "scalar_json_parameters",
        "text": '<tool_calls><invoke name="configure"><parameter name="count">123</parameter><parameter name="max_tokens"><![CDATA[256]]></parameter><parameter name="enabled">true</parameter></invoke></tool_calls>',
        "names": ["configure"],
    },
    {
        "id": "item_only_body_as_array",
        "text": '<|DSML|tool_calls><|DSML|invoke name="AskUserQuestion"><|DSML|parameter name="questions"><item><question><![CDATA[What would you like to do next?]]></question><header><![CDATA[Next step]]></header><options><item><label><![CDATA[Run tests]]></label><description><![CDATA[Run the test suite]]></description></item><item><label><![CDATA[Other task]]></label><description><![CDATA[Something else entirely]]></description></item></options><multiSelect>false</multiSelect></item></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
        "names": ["AskUserQuestion"],
    },
    {
        "id": "cdata_item_only_body_as_array",
        "text": '<|DSML|tool_calls><|DSML|invoke name="TodoWrite"><|DSML|parameter name="todos"><![CDATA[<br>  <item><br>    <activeForm>Testing EnterWorktree tool</activeForm><br>    <content>Test EnterWorktree tool</content><br>    <status>in_progress</status><br>  </item><br>  <item><br>    <activeForm>Testing TodoWrite tool</activeForm><br>    <content>Test TodoWrite tool</content><br>    <status>completed</status><br>  </item><br>]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
        "names": ["TodoWrite"],
    },
    {
        "id": "single_item_cdata_as_array",
        "text": '<tool_calls><invoke name="TodoWrite"><parameter name="todos"><![CDATA[<item>one</item>]]></parameter></invoke></tool_calls>',
        "names": ["TodoWrite"],
    },
    {
        "id": "loose_json_list_plain",
        "text": '<tool_calls><invoke name="TodoWrite"><parameter name="todos">{"content":"Test TodoWrite tool","status":"completed"}, {"content":"Another task","status":"pending"}</parameter></invoke></tool_calls>',
        "names": ["TodoWrite"],
    },
    {
        "id": "loose_json_list_cdata",
        "text": '<tool_calls><invoke name="TodoWrite"><parameter name="todos"><![CDATA[{"content":"Test TodoWrite tool","status":"completed"}, {"content":"Another task","status":"pending"}]]></parameter></invoke></tool_calls>',
        "names": ["TodoWrite"],
    },
    {
        "id": "preserved_text_parameters",
        "text": '<tool_calls><invoke name="Write"><parameter name="content"><![CDATA[{"content":"Test TodoWrite tool","status":"completed"}, {"content":"Another task","status":"pending"}]]></parameter></invoke></tool_calls>',
        "names": ["Write"],
    },
    {
        "id": "cdata_object_fragment",
        "text": '<tool_calls><invoke name="AskUserQuestion"><parameter name="questions"><![CDATA[<question><![CDATA[Pick one]]></question><options><item><label><![CDATA[A]]></label></item><item><label><![CDATA[B]]></label></item></options>]]></parameter></invoke></tool_calls>',
        "names": ["AskUserQuestion"],
    },
    {
        "id": "raw_command_with_ampersands",
        "text": '<tool_calls><invoke name="execute_command"><parameter name="command">cd /root && git status</parameter></invoke></tool_calls>',
        "names": ["execute_command"],
    },
    {
        "id": "ssh_command_keeps_double_ampersand",
        "text": '<tool_calls><invoke name="execute_command"><parameter name="command">sshpass -p \'xxx\' ssh -o StrictHostKeyChecking=no -p 1111 root@111.111.111.111 \'cd /root && git clone https://github.com/ericc-ch/copilot-api.git\'</parameter><parameter name="cwd"></parameter><parameter name="timeout"></parameter></invoke></tool_calls>',
        "names": ["execute_command"],
    },
    {
        "id": "parameter_named_tool_name_is_not_call_name",
        "text": '<tool_calls><invoke name="execute_command"><parameter name="tool_name">file.txt</parameter><parameter name="command">pwd</parameter></invoke></tool_calls>',
        "names": ["execute_command"],
    },
    {
        "id": "all_empty_parameter_payload_rejected",
        "text": '<tool_calls><invoke name="Bash"><parameter name="command"></parameter><parameter name="description">   </parameter><parameter name="timeout"></parameter></invoke></tool_calls>',
        "names": ["Bash"],
    },
    {
        "id": "zero_arg_tool_call",
        "text": '<tool_calls><invoke name="noop"></invoke></tool_calls>',
        "names": ["noop"],
    },
    {
        "id": "inline_json_tool_object",
        "text": '<tool_calls><invoke name="Bash">{"input":{"command":"pwd","description":"show cwd"}}</invoke></tool_calls>',
        "names": ["Bash"],
    },
    {
        "id": "mismatched_markup_rejected",
        "text": '<tool_calls><invoke name="read_file"><parameter name="path">README.md</function></invoke></tool_calls>',
        "names": ["read_file"],
    },
    {
        "id": "name_inside_params_rejected",
        "text": '<tool_calls><invoke><parameter name="path">README.md</parameter></invoke></tool_calls>',
        "names": ["read_file"],
    },
    {
        "id": "legacy_tools_wrapper_rejected",
        "text": '<tools><tool_call><tool_name>read_file</tool_name><param>{"path":"README.md"}</param></tool_call></tools>',
        "names": ["read_file"],
    },
    {
        "id": "bare_invoke_rejected",
        "text": '<invoke name="read_file"><parameter name="path">README.md</parameter></invoke>',
        "names": ["read_file"],
    },
    {
        "id": "missing_opening_wrapper_repaired",
        "text": 'Before tool call\n<invoke name="read_file"><parameter name="path">README.md</parameter></invoke>\n</tool_calls>\nafter',
        "names": ["read_file"],
    },
    {
        "id": "legacy_canonical_body_rejected",
        "text": '<tool_calls><invoke name="read_file"><tool_name>read_file</tool_name><param>{"path":"README.md"}</param></invoke></tool_calls>',
        "names": ["read_file"],
    },
    {
        "id": "html_entity_arguments",
        "text": '<tool_calls><invoke name="Bash"><parameter name="command">echo a &gt; out.txt</parameter></invoke></tool_calls>',
        "names": ["Bash"],
    },
    {
        "id": "only_non_fenced_tool_call",
        "text": '```xml\n<tool_calls><invoke name="read_file"><parameter name="path">README.md</parameter></invoke></tool_calls>\n```\n<tool_calls><invoke name="search"><parameter name="q">golang</parameter></invoke></tool_calls>',
        "names": ["read_file", "search"],
    },
    {
        "id": "four_backtick_fence",
        "text": '````markdown\n```xml\n<tool_calls><invoke name="read_file"><parameter name="path">README.md</parameter></invoke></tool_calls>\n```\n````\n<tool_calls><invoke name="search"><parameter name="q">outside</parameter></invoke></tool_calls>',
        "names": ["read_file", "search"],
    },
    {
        "id": "dsml_space_separator",
        "text": '<|DSML tool_calls><|DSML invoke name="Read"><|DSML parameter name="file_path"><![CDATA[/tmp/input.txt]]></|DSML parameter></|DSML invoke></|DSML tool_calls>',
        "names": ["Read"],
    },
    {
        "id": "dsml_space_lookalike_rejected",
        "text": '<|DSML tool_calls_extra><|DSML invoke name="Read"><|DSML parameter name="file_path">/tmp/input.txt</|DSML parameter></|DSML invoke></|DSML tool_calls_extra>',
        "names": ["Read"],
    },
    {
        "id": "dsml_collapsed_tags",
        "text": '<DSMLtool_calls><DSMLinvoke name="update_todo_list"><DSMLparameter name="todos"><![CDATA[[x] check parser\n[x] verify]]></DSMLparameter></DSMLinvoke></DSMLtool_calls>',
        "names": ["update_todo_list"],
    },
    {
        "id": "dsml_collapsed_lookalike_rejected",
        "text": '<DSMLtool_calls_extra><DSMLinvoke name="update_todo_list"><DSMLparameter name="todos">x</DSMLparameter></DSMLinvoke></DSMLtool_calls_extra>',
        "names": ["update_todo_list"],
    },
    {
        "id": "prose_mention_before_wrapper",
        "text": 'Summary: support canonical <tool_calls> and DSML <|DSML|tool_calls> wrappers.\n\n<|DSML|tool_calls><|DSML|invoke name="Bash"><|DSML|parameter name="command"><![CDATA[git status]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
        "names": ["Bash"],
    },
    {
        "id": "mixed_case_cdata_ignored_before_wrapper",
        "text": '<![cDaTa[<tool_calls><invoke name="Bash"><parameter name="command">bad</parameter></invoke></tool_calls>]]><tool_calls><invoke name="Read"><parameter name="file_path">README.md</parameter></invoke></tool_calls>',
        "names": ["Bash", "Read"],
    },
]


@pytest.mark.parametrize("case", DS2API_DIFFERENTIAL_PARSE_CASES, ids=[case["id"] for case in DS2API_DIFFERENTIAL_PARSE_CASES])
def test_ds2api_differential_parse_behavior(ds2api_runner: Path, case: dict[str, Any]) -> None:
    reference = _run_ds2api_runner(ds2api_runner, text=case["text"], names=case["names"])
    gateway_calls = _gateway_calls_snapshot(case["text"], case["names"])

    assert gateway_calls == reference["calls"]


def test_ds2api_differential_assistant_uses_thinking_only_when_visible_text_empty(ds2api_runner: Path) -> None:
    thinking = '<tool_calls><invoke name="search"><parameter name="q">go</parameter></invoke></tool_calls>'

    reference = _run_ds2api_runner(ds2api_runner, mode="assistant", text="", thinking=thinking, names=["search"])
    assert _gateway_calls_snapshot("", ["search"], thinking=thinking) == reference["calls"]

    reference = _run_ds2api_runner(
        ds2api_runner,
        mode="assistant",
        text="visible answer",
        thinking=thinking,
        names=["search"],
    )
    assert _gateway_calls_snapshot("visible answer", ["search"], thinking=thinking) == reference["calls"]


def test_ds2api_differential_openai_format_schema_normalization(ds2api_runner: Path) -> None:
    tools = [
        _openai_tool(
            "TaskUpdate",
            {
                "type": "object",
                "properties": {
                    "taskId": {"type": "string"},
                    "payload": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "count": {"type": "number"},
                        },
                    },
                },
            },
        )
    ]
    text = (
        '<tool_calls><invoke name="TaskUpdate">'
        '<parameter name="taskId">1</parameter>'
        '<parameter name="payload"><content><text>hello</text></content>'
        '<tags><item>1</item><item>true</item><item><k>v</k></item></tags>'
        '<count>2</count></parameter>'
        '</invoke></tool_calls>'
    )

    reference = _run_ds2api_runner(ds2api_runner, mode="format", text=text, names=["TaskUpdate"], tools=tools)

    assert _gateway_openai_snapshot(text, tools) == _normalize_openai_calls(reference["openai"])


def test_ds2api_differential_openai_stream_format(ds2api_runner: Path) -> None:
    tools = [_openai_tool("Write", {"type": "object", "properties": {"content": {"type": "string"}}})]
    text = '<tool_calls><invoke name="Write"><parameter name="content"><text>hello</text></parameter></invoke></tool_calls>'

    reference = _run_ds2api_runner(ds2api_runner, mode="stream", text=text, names=["Write"], tools=tools)

    assert _gateway_openai_snapshot(text, tools, stream=True) == _normalize_openai_calls(reference["stream"])


def test_ds2api_reference_commits_are_documented() -> None:
    assert_oracle_is_latest()
    assert DS2API_MAIN_COMMIT == "22a00dc6677c7f022b5ecac92c2c045deacd712a"
    assert DS2API_ORACLE_VERSION == "4.6.0"


def test_ds2api_parity_accepts_hyphenated_dsml_but_rejects_bare_hyphenated_xml() -> None:
    command = "git commit -m \"$(cat <<'EOF'\nfeat: document architecture\nEOF\n)\""
    payload = (
        "<dsml-tool-calls>"
        '<dsml-invoke name="Bash">'
        f'<dsml-parameter name="command"><![CDATA[{command}]]></dsml-parameter>'
        '<dsml-parameter name="description"><![CDATA[Commit architecture docs]]></dsml-parameter>'
        "</dsml-invoke>"
        "</dsml-tool-calls>"
    )

    assert _single_call(payload, ["Bash"]) == (
        "Bash",
        {"command": command, "description": "Commit architecture docs"},
    )

    bare = '<tool-calls><invoke name="Bash"><parameter name="command">pwd</parameter></invoke></tool-calls>'
    result = _parse(bare, ["Bash"])
    assert result.error is None
    assert result.tool_calls == []
    assert result.content == bare


def test_ds2api_parity_preserves_cdata_with_fenced_dsml_and_literal_cdata_end() -> None:
    content = "\n".join(
        [
            "```xml",
            "<|DSML|tool_calls>",
            '  <|DSML|invoke name="Bash">',
            '    <|DSML|parameter name="command"><![CDATA[echo compact]]></|DSML|parameter>',
            "  </|DSML|invoke>",
            "</|DSML|tool_calls>",
            "```",
            "Only literal `]]>` should stay inside content.",
            "tail",
        ]
    )
    payload = (
        '<tool_calls><invoke name="Write">'
        f'<parameter name="content"><![CDATA[{content}]]></parameter>'
        "</invoke></tool_calls>"
    )

    assert _single_call(payload, ["Write"]) == ("Write", {"content": content})


def test_ds2api_assistant_turn_matches_required_and_empty_output_oracle(ds2api_runner: Path) -> None:
    from webai_gateway.assistant_turn import build_assistant_turn

    context = _context(["Read"])
    reference = _run_ds2api_runner(
        ds2api_runner,
        mode="turn",
        text="plain answer",
        names=["Read"],
        tool_choice="required",
    )
    turn = build_assistant_turn(
        raw_text="plain answer",
        visible_text="plain answer",
        thinking="",
        detection_thinking="",
        bridge_context=context,
        tool_choice_required=True,
    )
    assert turn.error_code == reference["turn"]["error"]["code"]
    assert turn.finish_reason == reference["turn"]["finishReason"]
    assert turn.should_fail == reference["turn"]["shouldFail"]

    reference = _run_ds2api_runner(
        ds2api_runner,
        mode="turn",
        text="",
        thinking="reasoning only",
        names=["Read"],
    )
    turn = build_assistant_turn(
        raw_text="",
        visible_text="",
        thinking="reasoning only",
        detection_thinking="",
        bridge_context=context,
    )
    assert turn.error_code == reference["turn"]["error"]["code"]
    assert turn.has_visible_output == reference["turn"]["hasVisibleOutput"]


def test_ds2api_parity_accepts_canonical_and_dsml_xml_wrappers() -> None:
    cases = [
        (
            '<tool_calls><invoke name="read_file"><parameter name="path">README.MD</parameter></invoke></tool_calls>',
            {"path": "README.MD"},
        ),
        (
            '<|DSML|tool_calls><|DSML|invoke name="read_file"><|DSML|parameter name="path">README.MD</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
            {"path": "README.MD"},
        ),
        (
            '<|DSML|tool_calls| <|DSML|invoke name="read_file"><|DSML|parameter name="path">README.MD</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
            {"path": "README.MD"},
        ),
        (
            '<<|DSML|tool_calls><<|DSML|invoke name="read_file"><<|DSML|parameter name="path">README.MD</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
            {"path": "README.MD"},
        ),
        (
            '<DSML|DSML|tool_calls><DSML|DSML|invoke name="read_file"><DSML|DSML|parameter name="path">README.MD</DSML|DSML|parameter></DSML|DSML|invoke></DSML|DSML|tool_calls>',
            {"path": "README.MD"},
        ),
        (
            '<|DSML tool_calls><|DSML invoke name="read_file"><|DSML parameter name="path">README.MD</|DSML parameter></|DSML invoke></|DSML tool_calls>',
            {"path": "README.MD"},
        ),
        (
            '<DSMLtool_calls><DSMLinvoke name="read_file"><DSMLparameter name="path">README.MD</DSMLparameter></DSMLinvoke></DSMLtool_calls>',
            {"path": "README.MD"},
        ),
    ]

    for payload, expected in cases:
        assert _single_call(payload, ["read_file"]) == ("read_file", expected)


def test_ds2api_parity_rejects_dsml_lookalike_tag_names() -> None:
    for payload in [
        '<|DSML tool_calls_extra><|DSML invoke name="Read"><|DSML parameter name="file_path">/tmp/input.txt</|DSML parameter></|DSML invoke></|DSML tool_calls_extra>',
        '<DSMLtool_calls_extra><DSMLinvoke name="update_todo_list"><DSMLparameter name="todos">x</DSMLparameter></DSMLinvoke></DSMLtool_calls_extra>',
    ]:
        result = _parse(payload, ["Read", "update_todo_list"])
        assert result.error is None
        assert result.tool_calls == []
        assert result.content == payload


def test_ds2api_parity_preserves_cdata_and_nested_tool_syntax_as_text() -> None:
    content = '<tool_calls><invoke name="demo"><parameter name="value">x</parameter></invoke></tool_calls>'
    payload = (
        '<|DSML|tool_calls><|DSML|invoke name="write_file">'
        '<|DSML|parameter name="path">notes.md</|DSML|parameter>'
        f'<|DSML|parameter name="content"><![CDATA[{content}]]></|DSML|parameter>'
        '</|DSML|invoke></|DSML|tool_calls>'
    )

    assert _single_call(payload, ["write_file"]) == ("write_file", {"path": "notes.md", "content": content})


def test_ds2api_parity_recovers_unclosed_cdata_inside_valid_wrapper() -> None:
    payload = '<tool_calls><invoke name="Write"><parameter name="content"><![CDATA[hello world</parameter></invoke></tool_calls>'

    assert _single_call(payload, ["Write"]) == ("Write", {"content": "hello world"})


def test_ds2api_parity_parses_scalar_and_structured_parameters() -> None:
    scalar_payload = (
        '<tool_calls><invoke name="configure">'
        '<parameter name="count">123</parameter>'
        '<parameter name="max_tokens"><![CDATA[256]]></parameter>'
        '<parameter name="enabled">true</parameter>'
        '</invoke></tool_calls>'
    )
    assert _single_call(scalar_payload, ["configure"]) == (
        "configure",
        {"count": 123, "max_tokens": 256, "enabled": True},
    )

    question_fragment = (
        "<question><![CDATA[Pick one]]></question>"
        "<options><item><label><![CDATA[A]]></label></item><item><label><![CDATA[B]]></label></item></options>"
    )
    question_payload = (
        '<tool_calls><invoke name="AskUserQuestion">'
        f'<parameter name="questions"><![CDATA[{question_fragment}]]></parameter>'
        '</invoke></tool_calls>'
    )
    assert _single_call(question_payload, ["AskUserQuestion"]) == (
        "AskUserQuestion",
        {"questions": {"question": "Pick one", "options": [{"label": "A"}, {"label": "B"}]}},
    )


def test_ds2api_parity_parses_item_and_loose_json_lists_as_arrays() -> None:
    item_payload = '<tool_calls><invoke name="TodoWrite"><parameter name="todos"><![CDATA[<item>one</item>]]></parameter></invoke></tool_calls>'
    assert _single_call(item_payload, ["TodoWrite"]) == ("TodoWrite", {"todos": ["one"]})

    loose = '{"content":"Test TodoWrite tool","status":"completed"}, {"content":"Another task","status":"pending"}'
    for body in [loose, f"<![CDATA[{loose}]]>"]:
        payload = f'<tool_calls><invoke name="TodoWrite"><parameter name="todos">{body}</parameter></invoke></tool_calls>'
        assert _single_call(payload, ["TodoWrite"]) == (
            "TodoWrite",
            {
                "todos": [
                    {"content": "Test TodoWrite tool", "status": "completed"},
                    {"content": "Another task", "status": "pending"},
                ]
            },
        )

    preserved_payload = f'<tool_calls><invoke name="Write"><parameter name="content"><![CDATA[{loose}]]></parameter></invoke></tool_calls>'
    assert _single_call(preserved_payload, ["Write"]) == ("Write", {"content": loose})


def test_ds2api_parity_preserves_cdata_string_parameter_names() -> None:
    cases = [
        ("file_content", '{"content":"x"}, {"content":"y"}'),
        ("code", "<text>hello</text>"),
        ("pattern", '{"glob":"*.py"}'),
    ]

    for key, raw in cases:
        payload = f'<tool_calls><invoke name="Write"><parameter name="{key}"><![CDATA[{raw}]]></parameter></invoke></tool_calls>'
        assert _single_call(payload, ["Write"]) == ("Write", {key: raw})


def test_ds2api_parity_repairs_loose_json_list_elements() -> None:
    payload = (
        '<tool_calls><invoke name="TodoWrite">'
        '<parameter name="todos"><![CDATA[{content: "x", status: "pending"}, {content: "y", status: "done"}]]></parameter>'
        "</invoke></tool_calls>"
    )

    assert _single_call(payload, ["TodoWrite"]) == (
        "TodoWrite",
        {
            "todos": [
                {"content": "x", "status": "pending"},
                {"content": "y", "status": "done"},
            ]
        },
    )


def test_ds2api_parity_rejects_malformed_or_legacy_xml_body() -> None:
    malformed = '<tool_calls><invoke name="read_file"><parameter name="path">README.md</function></invoke></tool_calls>'
    legacy = '<tool_calls><invoke name="read_file"><tool_name>read_file</tool_name><param>{"path":"README.md"}</param></invoke></tool_calls>'

    for payload in [malformed, legacy]:
        result = _parse(payload, ["read_file"])
        assert result.tool_calls == []
        assert result.content == payload


def test_ds2api_parity_ignores_tool_xml_inside_fenced_code_blocks() -> None:
    payload = (
        "Here is an example:\n"
        "```xml\n"
        '<tool_calls><invoke name="read_file"><parameter name="path">README.md</parameter></invoke></tool_calls>\n'
        "```\n"
        "Do not execute it."
    )
    result = _parse(payload, ["read_file"])

    assert result.error is None
    assert result.tool_calls == []
    assert result.content == payload


def test_ds2api_parity_repairs_missing_opening_wrapper_when_closing_tag_exists() -> None:
    payload = (
        "Before tool call\n"
        '<invoke name="read_file"><parameter name="path">README.md</parameter></invoke>\n'
        "</tool_calls>\n"
        "after"
    )

    assert _single_call(payload, ["read_file"]) == ("read_file", {"path": "README.md"})


def test_ds2api_parity_openai_tool_call_format_uses_unique_call_ids() -> None:
    payload = '<tool_calls><invoke name="search"><parameter name="q">x</parameter></invoke></tool_calls>'

    first = _parse(payload, ["search"]).tool_calls[0]
    second = _parse(payload, ["search"]).tool_calls[0]

    assert first.id.startswith("call_")
    assert second.id.startswith("call_")
    assert first.id != second.id


def test_ds2api_parity_openai_arguments_are_normalized_with_tool_schema() -> None:
    tools = [
        _openai_tool(
            "TaskUpdate",
            {
                "type": "object",
                "properties": {
                    "taskId": {"type": "string"},
                    "payload": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "count": {"type": "number"},
                        },
                    },
                },
            },
        )
    ]
    context = build_context(tools, ToolBridgeConfig(exposure_policy="all"))
    payload = (
        '<tool_calls><invoke name="TaskUpdate">'
        '<parameter name="taskId">1</parameter>'
        '<parameter name="payload"><content><text>hello</text></content>'
        '<tags><item>1</item><item>true</item><item><k>v</k></item></tags>'
        '<count>2</count></parameter>'
        '</invoke></tool_calls>'
    )

    result = parse_tool_response(payload, context)
    assert result.error is None
    args = json.loads(to_openai_tool_calls(result.tool_calls, context)[0]["function"]["arguments"])

    assert args == {
        "taskId": "1",
        "payload": {
            "content": '{"text":"hello"}',
            "tags": ["1", "true", '{"k":"v"}'],
            "count": 2,
        },
    }


def test_ds2api_parity_anthropic_and_camelcase_schemas_normalize_strings() -> None:
    for schema_key in ("input_schema", "inputSchema"):
        context = build_context(
            [
                {
                    "name": "Write",
                    "description": "Write tool",
                    schema_key: {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                    },
                }
            ],
            ToolBridgeConfig(exposure_policy="all"),
        )
        result = parse_tool_response(
            '<tool_calls><invoke name="Write"><parameter name="content"><text>hello</text></parameter></invoke></tool_calls>',
            context,
        )
        args = json.loads(to_openai_tool_calls(result.tool_calls, context)[0]["function"]["arguments"])
        assert args["content"] == '{"text":"hello"}'


def test_ds2api_parity_tool_choice_none_disables_prompt_bridge_even_when_always() -> None:
    tools = [_openai_tool("Read")]

    assert (
        should_bridge_tools(
            tools,
            "strict",
            activation_policy="always",
            tool_choice="none",
            messages=[{"role": "user", "content": "read the repo"}],
        )
        is False
    )


def test_ds2api_parity_forced_tool_choice_filters_visible_tools() -> None:
    context = build_context(
        [_openai_tool("Read"), _openai_tool("Write")],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
        tool_choice={"type": "function", "function": {"name": "Read"}},
    )

    assert context.allowed_names == {"Read"}


def test_ds2api_parity_required_tool_choice_is_injected_into_prompt() -> None:
    context = build_context(
        [_openai_tool("Read")],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
        tool_choice="required",
    )

    prepared = prepare_openai_messages([{"role": "user", "content": "Read README.md"}], context)

    assert "For this response, you MUST call at least one tool from the allowed list." in prepared[0]["content"]


def test_ds2api_parity_forced_tool_choice_is_injected_into_prompt() -> None:
    context = build_context(
        [_openai_tool("Read"), _openai_tool("Write")],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
        tool_choice={"type": "function", "function": {"name": "Read"}},
    )

    prepared = prepare_openai_messages([{"role": "user", "content": "Read README.md"}], context)

    assert "For this response, you MUST call exactly this tool name: Read" in prepared[0]["content"]
    assert "Do not call any other tool." in prepared[0]["content"]


def test_ds2api_parity_allowed_tools_filter_visible_tools() -> None:
    context = build_context(
        [_openai_tool("Read"), _openai_tool("Write")],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
        tool_choice={"type": "auto", "allowed_tools": [{"type": "function", "function": {"name": "Write"}}]},
    )

    assert context.allowed_names == {"Write"}


def test_ds2api_parity_required_tool_choice_flags_plain_text_violation() -> None:
    context = build_context(
        [_openai_tool("Read")],
        ToolBridgeConfig(exposure_policy="all", tool_profile="all"),
        tool_choice="required",
    )

    result = parse_tool_response("I can answer without a tool.", context)

    assert result.error is not None
    assert result.error.kind == "tool_choice_violation"
    assert result.error.repairable is True


def test_ds2api_parity_sse_tool_arguments_are_schema_normalized() -> None:
    context = build_context(
        [
            _openai_tool(
                "Write",
                {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "taskId": {"type": "string"},
                    },
                },
            )
        ],
        ToolBridgeConfig(exposure_policy="all"),
    )
    body = build_tool_call_sse(
        '<tool_calls><invoke name="Write"><parameter name="content"><text>hello</text></parameter><parameter name="taskId">1</parameter></invoke></tool_calls>',
        allowed_tools={"Write"},
        model="deepseek-v4-pro",
        bridge_context=context,
    )
    first = json.loads(body.split("data: ", 1)[1].split("\n\n", 1)[0])
    args = json.loads(first["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"])

    assert args == {"content": '{"text":"hello"}', "taskId": "1"}


def test_ds2api_parity_hidden_thinking_tool_call_is_used_when_visible_text_empty() -> None:
    context = build_context(
        [_openai_tool("search", {"type": "object", "properties": {"q": {"type": "string"}}})],
        ToolBridgeConfig(exposure_policy="all"),
    )
    data = {
        "id": "chatcmpl_hidden",
        "created": 1,
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": '<tool_calls><invoke name="search"><parameter name="q"><text>go</text></parameter></invoke></tool_calls>',
                },
            }
        ],
    }

    parsed = parse_chat_response(
        data,
        bridge=True,
        allowed_tools={"search"},
        model="deepseek-v4-pro",
        bridge_context=context,
    )

    choice = parsed["choices"][0]
    args = json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])
    assert choice["finish_reason"] == "tool_calls"
    assert args == {"q": '{"text":"go"}'}


def test_ds2api_parity_hidden_thinking_is_ignored_when_visible_text_exists() -> None:
    context = build_context(
        [_openai_tool("search", {"type": "object", "properties": {"q": {"type": "string"}}})],
        ToolBridgeConfig(exposure_policy="all"),
    )
    data = {
        "id": "chatcmpl_hidden",
        "created": 1,
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "visible answer",
                    "reasoning_content": '<tool_calls><invoke name="search"><parameter name="q">go</parameter></invoke></tool_calls>',
                },
            }
        ],
    }

    parsed = parse_chat_response(
        data,
        bridge=True,
        allowed_tools={"search"},
        model="deepseek-v4-pro",
        bridge_context=context,
    )

    assert parsed["choices"][0]["message"]["content"] == "visible answer"
    assert "tool_calls" not in parsed["choices"][0]["message"]


def test_ds2api_parity_stream_sse_promotes_tool_call_without_leaking_dsml() -> None:
    body = build_tool_call_sse(
        '<|DSML|tool_calls><|DSML|invoke name="Read"><|DSML|parameter name="file_path"><![CDATA[README.md]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>',
        allowed_tools={"Read"},
        model="qwen-web/qwen3.6-plus",
    )

    assert "<|DSML|tool_calls>" not in body
    assert '"finish_reason":"tool_calls"' in body
    assert '"tool_calls"' in body
    assert '"name":"Read"' in body
    assert json.loads(body.split("data: ", 1)[1].split("\n\n", 1)[0])["choices"][0]["delta"]["tool_calls"][0]["id"].startswith("call_")


def test_ds2api_parity_sanitizes_leaked_output_protocol() -> None:
    cases = [
        ("before\n```json\n```\nafter", "before\n\nafter"),
        (
            '开始\n[{"function":{"arguments":"{\\"command\\":\\"java -version\\"}","name":"exec"},"id":"callb9a321","type":"function"}]< | Tool | >{"content":"openjdk version 21","tool_call_id":"callb9a321"}\n结束',
            "开始\n\n结束",
        ),
        ("A<| end_of_sentence |><| Assistant |>B<| end_of_thinking |>C<| end_of_toolresults |>D", "ABCD"),
        ("A<think>B</think>C<| begin_of_sentence |>D", "ABCD"),
        ("\ue200genui\ue202Ay-D\ue201\n\nRight now in Beijing it is sunny.", "Right now in Beijing it is sunny."),
        ("Answer prefix<think>internal reasoning that never closes", "Answer prefix"),
        (
            'before\n<|DSML|tool_calls><|DSML|invoke name="Bash"><|DSML|parameter name="command">pwd</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>\nafter',
            "before\n\nafter",
        ),
        ("Done.<attempt_completion><result>Some final answer</result></attempt_completion>", "Done.Some final answer"),
        ("Example XML: <result>value</result>", "Example XML: <result>value</result>"),
        ("Done.<attempt_completion><result>Some final answer", "Done.Some final answer"),
        ("Done.Some final answer</result></attempt_completion>", "Done.Some final answer"),
        (
            "Done.<attempt_completion><result>Some final answer\nExample XML: <result>value</result>",
            "Done.Some final answer\nExample XML: <result>value</result>",
        ),
    ]

    for raw, expected in cases:
        assert sanitize_leaked_tool_protocol_output(raw) == expected
