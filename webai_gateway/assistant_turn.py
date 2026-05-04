from __future__ import annotations

from dataclasses import dataclass

from webai_gateway.tool_bridge import BridgeError, BridgeResult, ToolBridgeContext, parse_tool_response


@dataclass(frozen=True)
class AssistantTurn:
    raw_text: str
    visible_text: str
    visible_thinking: str
    detection_thinking: str
    tool_calls: list
    finish_reason: str
    has_tool_calls: bool
    has_visible_text: bool
    has_visible_output: bool
    error_code: str | None
    error_message: str | None
    should_fail: bool
    bridge_result: BridgeResult


def build_assistant_turn(
    *,
    raw_text: str,
    visible_text: str,
    thinking: str,
    detection_thinking: str,
    bridge_context: ToolBridgeContext,
    tool_choice_required: bool = False,
    content_filter: bool = False,
) -> AssistantTurn:
    source = raw_text if raw_text.strip() else (detection_thinking if detection_thinking.strip() else thinking)
    bridge_result = parse_tool_response(source, bridge_context)
    has_tool_calls = bool(bridge_result.tool_calls)
    has_visible_text = bool((visible_text or "").strip())
    has_visible_thinking = bool((thinking or "").strip())
    if source != raw_text and not has_tool_calls:
        bridge_result = _bridge_result_with_content(bridge_result, visible_text)

    if has_tool_calls:
        return AssistantTurn(
            raw_text=raw_text,
            visible_text=visible_text,
            visible_thinking=thinking,
            detection_thinking=detection_thinking,
            tool_calls=bridge_result.tool_calls,
            finish_reason="tool_calls",
            has_tool_calls=True,
            has_visible_text=has_visible_text,
            has_visible_output=True,
            error_code=None,
            error_message=None,
            should_fail=False,
            bridge_result=bridge_result,
        )

    if tool_choice_required:
        error = bridge_result.error or BridgeError(
            "tool_choice_violation",
            "tool_choice requires at least one valid tool call.",
            repairable=False,
        )
        bridge_result = BridgeResult(
            content=visible_text,
            tool_calls=[],
            error=error,
            warning=bridge_result.warning,
            raw_content=bridge_result.raw_content or source,
            phases=bridge_result.phases,
        )
        return AssistantTurn(
            raw_text=raw_text,
            visible_text=visible_text,
            visible_thinking=thinking,
            detection_thinking=detection_thinking,
            tool_calls=[],
            finish_reason="stop",
            has_tool_calls=False,
            has_visible_text=has_visible_text,
            has_visible_output=has_visible_text or has_visible_thinking,
            error_code=error.kind,
            error_message=error.message,
            should_fail=True,
            bridge_result=bridge_result,
        )

    if has_visible_text:
        return AssistantTurn(
            raw_text=raw_text,
            visible_text=visible_text,
            visible_thinking=thinking,
            detection_thinking=detection_thinking,
            tool_calls=[],
            finish_reason="stop",
            has_tool_calls=False,
            has_visible_text=True,
            has_visible_output=True,
            error_code=None,
            error_message=None,
            should_fail=False,
            bridge_result=bridge_result,
        )

    if content_filter:
        error = BridgeError("content_filter", "Upstream content filtered the response and returned no output.")
        bridge_result = BridgeResult(
            content=visible_text,
            tool_calls=[],
            error=error,
            warning=bridge_result.warning,
            raw_content=bridge_result.raw_content or source,
            phases=bridge_result.phases,
        )
        return AssistantTurn(
            raw_text=raw_text,
            visible_text=visible_text,
            visible_thinking=thinking,
            detection_thinking=detection_thinking,
            tool_calls=[],
            finish_reason="content_filter",
            has_tool_calls=False,
            has_visible_text=False,
            has_visible_output=has_visible_thinking,
            error_code=error.kind,
            error_message=error.message,
            should_fail=True,
            bridge_result=bridge_result,
        )

    error = BridgeError(
        "upstream_empty_output",
        (
            "Upstream account hit a rate limit and returned reasoning without visible output."
            if has_visible_thinking
            else "Upstream account hit a rate limit and returned empty output."
        ),
    )
    bridge_result = BridgeResult(
        content=visible_text,
        tool_calls=[],
        error=error,
        warning=bridge_result.warning,
        raw_content=bridge_result.raw_content or source,
        phases=bridge_result.phases,
    )
    return AssistantTurn(
        raw_text=raw_text,
        visible_text=visible_text,
        visible_thinking=thinking,
        detection_thinking=detection_thinking,
        tool_calls=[],
        finish_reason="stop",
        has_tool_calls=False,
        has_visible_text=False,
        has_visible_output=has_visible_thinking,
        error_code=error.kind,
        error_message=error.message,
        should_fail=True,
        bridge_result=bridge_result,
    )


def _bridge_result_with_content(result: BridgeResult, content: str) -> BridgeResult:
    return BridgeResult(
        content=content,
        tool_calls=result.tool_calls,
        error=result.error,
        warning=result.warning,
        raw_content=result.raw_content,
        phases=result.phases,
    )
