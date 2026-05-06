from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from webai_gateway.tool_bridge import BridgeError, BridgeResult, ToolBridgeContext, ToolCallDraft


ControllerState = Literal["TOOL_CALL", "FINAL", "ASK_USER", "RETRY"]


@dataclass(frozen=True)
class RetryState:
    repair_attempts: int = 0
    recovery_attempts: int = 0
    ask_user_attempts: int = 0


@dataclass(frozen=True)
class EvidenceLedger:
    has_discovery: bool = False
    has_file_read: bool = False
    has_search: bool = False
    has_mutation: bool = False
    has_verification: bool = False


@dataclass(frozen=True)
class ControllerDecision:
    state: ControllerState
    retry_kind: str = ""
    reason: str = ""
    bridge_result: BridgeResult | None = None
    tool_calls: list[ToolCallDraft] | None = None
    retry_state: RetryState | None = None


_REPAIRABLE_TOOL_REFUSAL_ERRORS = {
    "tool_denial_without_call",
    "deferred_tool_action_without_call",
    "deferred_code_change_without_call",
    "unverified_code_change_completion",
    "write_after_failed_read_without_discovery",
    "write_after_failed_path_without_discovery",
    "unsafe_local_shell_command",
    "unproven_final_answer_without_tool_call",
    "incomplete_fix_stub_without_tool_call",
}
_READLIKE_TOOL_NAMES = {"read", "readfile", "fileread"}
_DISCOVERY_TOOL_NAMES = {"glob", "grep", "ls", "list", "listdir", "listdirectory", "tree"}
_SEARCH_TOOL_NAMES = {"grep", "search", "find"}
_MUTATION_TOOL_NAMES = {"edit", "editfile", "multiedit", "write", "writefile", "applypatch", "patch"}
_VERIFICATION_TOOL_NAMES = {"bash", "shell", "terminal", "pytest", "test", "run_tests"}
_REVIEW_TASK_RE = re.compile(r"\b(?:review|audit|inspect|analy[sz]e)\b|(?:审查|检查|评审|分析)", re.IGNORECASE)
_MUTATION_TASK_RE = re.compile(
    r"\b(?:implement|fix|repair|modify|update|change|write|edit|refactor|configure|setup|set\s+up|install|deploy)\b|"
    r"(?:实现|修复|修改|落地|更新|编辑|写入|配置|设置|安装|部署)",
    re.IGNORECASE,
)
_VERIFICATION_NOT_RUN_RE = re.compile(
    r"\b(?:not\s+run|not\s+executed|unable\s+to\s+run|could\s+not\s+run)\b|(?:未运行|未执行|没有运行|无法运行)",
    re.IGNORECASE,
)
_STATUS_ONLY_TOOL_FINAL_RE = re.compile(
    r"\b(?:bash|shell|command|tool|read|glob|grep|ls|operation|task)\b.{0,80}"
    r"\b(?:completed|complete|finished|succeeded|done|executed|ran|returned)\b|"
    r"\b(?:completed|complete|finished|succeeded|done)\b.{0,80}"
    r"\b(?:no output|without output|empty output)\b|"
    r"\b(?:project\s+dir|project\s+directory|directory|folder|file|path)\b.{0,60}"
    r"\b(?:exist|exists|existed|found)\b|"
    r"\b(?:no\s+(?:task|request|instruction)s?\s+(?:given|provided|specified)|wait(?:ing)?\s+for\s+(?:your\s+)?input)\b|"
    r"(?:当前无明确任务|无明确任务|没有明确任务|缺乏明确(?:的)?(?:用户)?(?:任务|指令)|请提供(?:明确|具体)(?:的)?(?:指令|任务|需求)|等待(?:用户)?(?:指令|输入))|"
    r"\bcaveman\s+mode\s+active\b|"
    r"(?:command|tool|task|operation).{0,40}(?:no output|empty output)",
    re.IGNORECASE,
)
_EXPLICIT_NO_TASK_FINAL_RE = re.compile(
    r"(?:上下文缺失|无明确任务|没有明确任务|请提供具体需求|请提供具体指令|等待(?:用户)?(?:指令|输入)|new\s+instruction)",
    re.IGNORECASE,
)
_HISTORY_SUMMARY_FINAL_RE = re.compile(
    r"(?:according\s+to|based\s+on|根据|基于).{0,40}DS2API_HISTORY\.txt.{0,80}"
    r"(?:current\s+(?:work\s+)?state|work\s+status|当前(?:的)?(?:工作)?状态|工作状态)|"
    r"DS2API_HISTORY\.txt.{0,120}(?:current\s+(?:work\s+)?state|work\s+status|当前(?:的)?(?:工作)?状态|工作状态)|"
    r"DS2API_HISTORY\.txt.{0,240}(?:tool\s+loop\s+guard|skill\s+progress\s+guard|工具循环保护|技能进度保护|最新的用户请求明确指出|状态分析)",
    re.IGNORECASE | re.DOTALL,
)
_HISTORY_SUMMARY_REQUEST_RE = re.compile(
    r"\b(?:status|state|progress|recap|summar(?:y|ize))\b|(?:状态|进展|总结|回顾|复盘)",
    re.IGNORECASE,
)
_UNKNOWN_PROJECT_STRUCTURE_FINAL_RE = re.compile(
    r"(?:project\s+structure\s+(?:is\s+)?unknown|unknown\s+project\s+structure|项目结构未知)|"
    r"(?:glob\s*(?:result|结果).{0,40}(?:truncated|截断))|"
    r"(?:cannot|can't|unable\s+to|无法).{0,80}(?:review|inspect|audit|analy[sz]e|审查|检查|评审|分析)"
    r".{0,80}(?:without|缺少|没有|需要).{0,80}(?:file|path|directory|repo|project|文件|路径|目录|项目|仓库)|"
    r"(?:(?:please\s+)?(?:provide|specify)|请(?:提供|给出|说明)|需要(?:具体)?(?:提供|说明)?).{0,160}"
    r"(?:project\s+root|root\s+path|directory\s+structure|file\s+list|source\s+files?|language|framework|"
    r"项目根目录|根目录路径|目录结构|文件列表|关键代码|编程语言|框架)",
    re.IGNORECASE | re.DOTALL,
)
_DOC_NEXT_STEP_SUMMARY_FINAL_RE = re.compile(
    r"(?:according\s+to|based\s+on|根据).{0,80}(?:\b[A-Za-z0-9_.-]+\.md\b|document|docs?|文档)"
    r".{0,160}(?:next\s+steps?|下一步).{0,120}(?:recommend|suggest|建议|操作)|"
    r"(?:\b[A-Za-z0-9_.-]+\.md\b|document|docs?|文档).{0,120}(?:listed|列出).{0,80}"
    r"(?:next\s+steps?|下一步).{0,120}(?:recommend|suggest|建议|操作)",
    re.IGNORECASE | re.DOTALL,
)
_OPTIONAL_NEXT_STEP_SELECTION_FINAL_RE = re.compile(
    r"(?:请问|请选择|你希望|您希望|是否需要|需要我|要我).{0,120}"
    r"(?:执行|选择|继续|处理|优先|开始).{0,80}(?:哪一项|哪项|哪一个|哪个|操作|任务|事项|功能)|"
    r"(?:或者|或).{0,40}(?:其他|其它).{0,80}(?:具体任务|任务|需求).{0,40}(?:处理|执行|继续)?|"
    r"(?:which|what|would\s+you\s+like|do\s+you\s+want).{0,160}"
    r"(?:operation|option|task|action|next\s+step)",
    re.IGNORECASE | re.DOTALL,
)
_TASK_REQUESTS_NEXT_STEP_SELECTION_RE = re.compile(
    r"\b(?:next\s+steps?|options?|choose|select)\b|(?:下一步|选项|选择|让我选|让用户选|问我)",
    re.IGNORECASE,
)
_OFF_TASK_ENV_CONFIG_FINAL_RE = re.compile(
    r"\bcaveman\s+mode\s+active\b.{0,200}\b(?:statusline|status\s+line|settings\.json|badge|plugins?|hooks?)\b|"
    r"\b(?:statusline|status\s+line|badge)\b.{0,160}\b(?:settings\.json|configured|configure|command)\b|"
    r"\b(?:settings\.json)\b.{0,160}\b(?:statusline|status\s+line|agent\s+settings|plugin\s+settings|hook\s+config|caveman|badge)\b|"
    r"\b(?:configure|add|update)\b.{0,120}\b(?:statusline|status\s+line|badge|agent\s+settings|plugin\s+settings|hook\s+config)\b|"
    r"(?:~?[\\/]\.claude|\.claude[\\/]).{0,200}\b(?:settings\.json|plugins?|hooks?|statusline|status\s+line)\b",
    re.IGNORECASE | re.DOTALL,
)
_SUBSTANTIVE_TASK_ANSWER_RE = re.compile(
    r"\b(?:finding|findings|issue|issues|risk|risks|recommendation|recommendations|"
    r"improvement|improvements|plan|summary|review|audit|analysis|next\s+steps)\b|"
    r"(?:问题|风险|建议|改进|计划|总结|审查|评审|分析|结论|下一步)",
    re.IGNORECASE,
)
_SHELL_COMMAND_RE = re.compile(r'"command"\s*:\s*"(?P<command>[^"]+)"')
_CLARIFICATION_FINAL_RE = re.compile(
    r"\b(?:confirm|clarify|specify|provide|which|what)\b.{0,120}\b(?:target|branch|files?|path|scope|repo(?:sitory)?)\b|"
    r"\b(?:target\s+(?:branch|files?)|awaiting\s+(?:new\s+)?instructions?)\b",
    re.IGNORECASE | re.DOTALL,
)


def classify_bridge_result(
    result: BridgeResult,
    context: ToolBridgeContext,
    retry_state: RetryState | None = None,
    *,
    max_repair_attempts: int = 1,
) -> ControllerDecision:
    state = retry_state or RetryState()
    if result.tool_calls:
        return ControllerDecision("TOOL_CALL", bridge_result=result, retry_state=state)

    if result.error is not None:
        if _should_ask_user(result, context, state):
            call = _ask_user_tool_call(result)
            return ControllerDecision(
                "ASK_USER",
                reason=result.error.kind,
                bridge_result=result,
                tool_calls=[call],
                retry_state=state,
            )
        if result.error.repairable:
            if state.repair_attempts >= max_repair_attempts:
                return ControllerDecision(
                    "FINAL",
                    reason="retry_budget_exhausted",
                    bridge_result=result,
                    retry_state=state,
                )
            return ControllerDecision(
                "RETRY",
                retry_kind="repair",
                reason=result.error.kind,
                bridge_result=result,
                retry_state=RetryState(
                    repair_attempts=state.repair_attempts + 1,
                    recovery_attempts=state.recovery_attempts,
                    ask_user_attempts=state.ask_user_attempts,
                ),
            )
        return ControllerDecision("FINAL", reason="non_repairable_error", bridge_result=result, retry_state=state)

    if _is_history_summary_final_without_task_answer(context, result.content):
        if state.repair_attempts >= max_repair_attempts:
            return ControllerDecision("FINAL", reason="retry_budget_exhausted", bridge_result=result, retry_state=state)
        return ControllerDecision(
            "RETRY",
            retry_kind="history_summary_final_without_task_answer",
            reason="history_summary_final_without_task_answer",
            bridge_result=result,
            retry_state=RetryState(
                repair_attempts=state.repair_attempts + 1,
                recovery_attempts=state.recovery_attempts,
                ask_user_attempts=state.ask_user_attempts,
            ),
        )

    if _is_unknown_project_structure_final_without_task_answer(context, result.content):
        if state.repair_attempts >= max_repair_attempts:
            return ControllerDecision("FINAL", reason="retry_budget_exhausted", bridge_result=result, retry_state=state)
        return ControllerDecision(
            "RETRY",
            retry_kind="unknown_project_structure_final_without_task_answer",
            reason="unknown_project_structure_final_without_task_answer",
            bridge_result=result,
            retry_state=RetryState(
                repair_attempts=state.repair_attempts + 1,
                recovery_attempts=state.recovery_attempts,
                ask_user_attempts=state.ask_user_attempts,
            ),
        )

    if _is_review_next_step_menu_final_without_task_answer(context, result.content):
        if state.repair_attempts >= max_repair_attempts:
            return ControllerDecision("FINAL", reason="retry_budget_exhausted", bridge_result=result, retry_state=state)
        return ControllerDecision(
            "RETRY",
            retry_kind="review_next_step_menu_final_without_task_answer",
            reason="review_next_step_menu_final_without_task_answer",
            bridge_result=result,
            retry_state=RetryState(
                repair_attempts=state.repair_attempts + 1,
                recovery_attempts=state.recovery_attempts,
                ask_user_attempts=state.ask_user_attempts,
            ),
        )

    if _is_off_task_environment_configuration_final(context, result.content):
        if state.repair_attempts >= max_repair_attempts:
            return ControllerDecision("FINAL", reason="retry_budget_exhausted", bridge_result=result, retry_state=state)
        return ControllerDecision(
            "RETRY",
            retry_kind="off_task_environment_configuration_final",
            reason="off_task_environment_configuration_final",
            bridge_result=result,
            retry_state=RetryState(
                repair_attempts=state.repair_attempts + 1,
                recovery_attempts=state.recovery_attempts,
                ask_user_attempts=state.ask_user_attempts,
            ),
        )

    if _is_status_only_final_without_task_answer(context, result.content):
        if state.repair_attempts >= max_repair_attempts:
            return ControllerDecision("FINAL", reason="retry_budget_exhausted", bridge_result=result, retry_state=state)
        return ControllerDecision(
            "RETRY",
            retry_kind="status_only_final_without_task_answer",
            reason="status_only_final_without_task_answer",
            bridge_result=result,
            retry_state=RetryState(
                repair_attempts=state.repair_attempts + 1,
                recovery_attempts=state.recovery_attempts,
                ask_user_attempts=state.ask_user_attempts,
            ),
        )

    if _allows_ds2api_style_controller_passthrough(context):
        return ControllerDecision("FINAL", bridge_result=result, retry_state=state)

    if _requires_final_evidence(context, result.content) and not _has_required_final_evidence(context, result.content):
        repair_limit = _final_evidence_repair_limit(context, max_repair_attempts)
        if state.repair_attempts >= repair_limit:
            return ControllerDecision("FINAL", reason="retry_budget_exhausted", bridge_result=result, retry_state=state)
        return ControllerDecision(
            "RETRY",
            retry_kind="insufficient_final_evidence",
            reason="insufficient_final_evidence",
            bridge_result=result,
            retry_state=RetryState(
                repair_attempts=state.repair_attempts + 1,
                recovery_attempts=state.recovery_attempts,
                ask_user_attempts=state.ask_user_attempts,
            ),
        )
    return ControllerDecision("FINAL", bridge_result=result, retry_state=state)


def _final_evidence_repair_limit(context: ToolBridgeContext, default_limit: int) -> int:
    if not context.allowed_names:
        return default_limit
    task = context.task_text or ""
    if _MUTATION_TASK_RE.search(task) and not context.has_tool_loop:
        return max(default_limit, 2)
    return default_limit


def _allows_ds2api_style_controller_passthrough(context: ToolBridgeContext) -> bool:
    profile = str(getattr(getattr(context, "options", None), "tool_profile", "") or "").strip().lower().replace("_", "-")
    return profile == "all"


def build_ask_user_question_input(question: str) -> dict[str, object]:
    return {
        "questions": [
            {
                "header": "确认本地命令",
                "question": question,
                "options": [
                    {
                        "label": "允许本次命令",
                        "description": "允许下游客户端按自己的权限系统处理这次命令。",
                    },
                    {
                        "label": "改用只读检查",
                        "description": "要求模型继续使用 Read、Glob、Grep 等只读工具。",
                    },
                    {
                        "label": "取消",
                        "description": "拒绝这次本地命令请求。",
                    },
                ],
                "multiSelect": False,
            }
        ]
    }


def build_evidence_ledger(context: ToolBridgeContext) -> EvidenceLedger:
    names = {_compact_tool_name(name) for name in context.recent_tool_call_names}
    return EvidenceLedger(
        has_discovery=bool(names & _DISCOVERY_TOOL_NAMES),
        has_file_read=bool(names & _READLIKE_TOOL_NAMES),
        has_search=bool(names & _SEARCH_TOOL_NAMES),
        has_mutation=bool(names & _MUTATION_TOOL_NAMES),
        has_verification=bool(names & _VERIFICATION_TOOL_NAMES),
    )


def decision_to_openai_chat_response(decision: ControllerDecision, *, model: str) -> dict[str, object] | None:
    if decision.state != "ASK_USER" or not decision.tool_calls:
        return None
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.input, ensure_ascii=False),
                            },
                        }
                        for call in decision.tool_calls
                    ],
                },
            }
        ],
    }


def _should_ask_user(result: BridgeResult, context: ToolBridgeContext, retry_state: RetryState) -> bool:
    if retry_state.ask_user_attempts > 0:
        return False
    error = result.error
    if error is None:
        return False
    if error.kind in {
        "premature_clarification_without_tool_call",
        "unproven_final_answer_without_tool_call",
    }:
        text = str(result.raw_content or result.content or error.message or "")
        return retry_state.repair_attempts > 0 and _has_tool(context, "AskUserQuestion") and _looks_like_clarification_final(text)
    if error.kind not in {
        "unsafe_shell_command_requires_explicit_task",
        "unsafe_review_shell_command",
    }:
        return False
    if error.repairable:
        return False
    return _has_tool(context, "AskUserQuestion")


def _ask_user_tool_call(result: BridgeResult) -> ToolCallDraft:
    if result.error and result.error.kind in {
        "premature_clarification_without_tool_call",
        "unproven_final_answer_without_tool_call",
    }:
        question = _clarification_question_from_text(str(result.raw_content or result.content or result.error.message or ""))
        return ToolCallDraft(
            id="toolu_ask_user_clarification",
            name="AskUserQuestion",
            input=build_clarification_question_input(question),
        )
    command = _shell_command_from_raw(result.raw_content)
    question = (
        "上游模型请求了需要用户明确确认的本地命令。"
        + (f" 命令：{command}" if command else "")
    )
    return ToolCallDraft(
        id="toolu_ask_user_permission",
        name="AskUserQuestion",
        input=build_ask_user_question_input(question),
    )


def build_clarification_question_input(question: str) -> dict[str, object]:
    return {
        "questions": [
            {
                "header": "Clarify",
                "question": question,
                "options": [
                    {
                        "label": "Use current scope",
                        "description": "Continue using the current branch, files, and gathered evidence.",
                    },
                    {
                        "label": "Specify target",
                        "description": "Provide the branch, files, path, or scope to inspect next.",
                    },
                    {
                        "label": "Stop",
                        "description": "Stop the current task without further tool calls.",
                    },
                ],
                "multiSelect": False,
            }
        ]
    }


def _looks_like_clarification_final(text: str) -> bool:
    return bool(_CLARIFICATION_FINAL_RE.search(text or ""))


def _clarification_question_from_text(text: str) -> str:
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return "Please confirm the target branch, files, path, or scope before continuing."
    if len(raw) > 240:
        raw = raw[:237].rstrip() + "..."
    if raw.endswith("?"):
        return raw
    return f"{raw} Please confirm the target branch, files, path, or scope."


def _requires_final_evidence(context: ToolBridgeContext, text: str) -> bool:
    if not context.allowed_names:
        return False
    task = context.task_text or ""
    return bool(_REVIEW_TASK_RE.search(task) or _MUTATION_TASK_RE.search(task))


def _has_required_final_evidence(context: ToolBridgeContext, text: str) -> bool:
    ledger = build_evidence_ledger(context)
    task = context.task_text or ""
    if _MUTATION_TASK_RE.search(task):
        if not ledger.has_mutation:
            return False
        return ledger.has_verification or bool(_VERIFICATION_NOT_RUN_RE.search(text or ""))
    if _REVIEW_TASK_RE.search(task):
        return ledger.has_discovery and (ledger.has_file_read or ledger.has_search)
    return True


def _is_status_only_final_without_task_answer(context: ToolBridgeContext, text: str) -> bool:
    if not context.has_tool_loop or not context.allowed_names:
        return False
    if not _requires_final_evidence(context, text):
        return False
    raw = " ".join((text or "").split())
    if not raw or len(raw) > 320:
        return False
    if _EXPLICIT_NO_TASK_FINAL_RE.search(raw):
        return True
    if _SUBSTANTIVE_TASK_ANSWER_RE.search(raw):
        return False
    return bool(_STATUS_ONLY_TOOL_FINAL_RE.search(raw))


def _is_history_summary_final_without_task_answer(context: ToolBridgeContext, text: str) -> bool:
    if not context.has_tool_loop or not context.allowed_names:
        return False
    task = context.task_text or ""
    if _HISTORY_SUMMARY_REQUEST_RE.search(task) or "DS2API_HISTORY" in task:
        return False
    if not (_REVIEW_TASK_RE.search(task) or _MUTATION_TASK_RE.search(task)):
        return False
    raw = " ".join((text or "").split())
    if not raw:
        return False
    return bool(_HISTORY_SUMMARY_FINAL_RE.search(raw[:1200]))


def _is_unknown_project_structure_final_without_task_answer(context: ToolBridgeContext, text: str) -> bool:
    if not context.has_tool_loop or not context.allowed_names:
        return False
    task = context.task_text or ""
    if not (_REVIEW_TASK_RE.search(task) or _MUTATION_TASK_RE.search(task)):
        return False
    raw = " ".join((text or "").split())
    if not raw or len(raw) > 1600:
        return False
    if not _UNKNOWN_PROJECT_STRUCTURE_FINAL_RE.search(raw[:1600]):
        return False
    allowed = {_compact_tool_name(tool.name) for tool in context.tools}
    if not (allowed & (_DISCOVERY_TOOL_NAMES | _READLIKE_TOOL_NAMES | {"bash", "shell"})):
        return False
    ledger = build_evidence_ledger(context)
    return ledger.has_discovery or "glob" in raw.lower() or "目录" in raw or "directory" in raw.lower()


def _is_review_next_step_menu_final_without_task_answer(context: ToolBridgeContext, text: str) -> bool:
    if not context.has_tool_loop or not context.allowed_names:
        return False
    task = context.task_text or ""
    if _TASK_REQUESTS_NEXT_STEP_SELECTION_RE.search(task):
        return False
    if not _REVIEW_TASK_RE.search(task):
        return False
    ledger = build_evidence_ledger(context)
    if not (ledger.has_discovery or ledger.has_file_read or ledger.has_search):
        return False
    raw = " ".join((text or "").split())
    if not raw or len(raw) > 2200:
        return False
    if _DOC_NEXT_STEP_SUMMARY_FINAL_RE.search(raw[:2200]) and _OPTIONAL_NEXT_STEP_SELECTION_FINAL_RE.search(raw[:2200]):
        return True
    return bool(_OPTIONAL_NEXT_STEP_SELECTION_FINAL_RE.search(raw[-800:]) and not _looks_like_substantive_review_final(raw))


def _looks_like_substantive_review_final(text: str) -> bool:
    raw = text or ""
    has_finding_language = bool(
        re.search(
            r"\b(?:finding|findings|issue|issues|risk|risks|bug|bugs|defect|defects)\b|"
            r"(?:发现|问题|风险|缺陷|漏洞|隐患)",
            raw,
            re.IGNORECASE,
        )
    )
    has_code_reference = bool(re.search(r"`[^`]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|cs|md)`|[\w./\\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|cs)", raw))
    return has_finding_language and has_code_reference


def _is_off_task_environment_configuration_final(context: ToolBridgeContext, text: str) -> bool:
    if not context.has_tool_loop or not context.allowed_names:
        return False
    if _OFF_TASK_ENV_CONFIG_FINAL_RE.search(context.task_text or ""):
        return False
    raw = " ".join((text or "").split())
    if not raw:
        return False
    return bool(_OFF_TASK_ENV_CONFIG_FINAL_RE.search(raw))


def _has_tool(context: ToolBridgeContext, name: str) -> bool:
    expected = _compact_tool_name(name)
    return any(_compact_tool_name(tool.name) == expected for tool in context.tools)


def _compact_tool_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())


def _shell_command_from_raw(raw: str) -> str:
    match = _SHELL_COMMAND_RE.search(raw or "")
    if not match:
        return ""
    try:
        return bytes(match.group("command"), "utf-8").decode("unicode_escape")
    except Exception:
        return match.group("command")
