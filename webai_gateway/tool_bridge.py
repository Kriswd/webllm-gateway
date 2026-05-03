from __future__ import annotations

import json
import ast
import fnmatch
import html
import re
import shlex
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from webai_gateway.config import ToolBridgeConfig


_FENCED_TOOL_RE = re.compile(r"```tool_json\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_RUNAWAY_PLAIN_TEXT_MIN_CHARS = 8000
_FENCED_CODE_BLOCK_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_+.-]*)[ \t]*(?:\r?\n)?(?P<body>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_XML_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_XML_TOOL_CALLS_OPEN_RE = re.compile(
    r"<+\s*(?!/)[^<>]{0,80}tool_calls(?=\s|[|｜>])[^<>]*>",
    re.IGNORECASE | re.DOTALL,
)
_XML_TOOL_CALLS_CLOSE_RE = re.compile(
    r"<+\s*/\s*[^<>]{0,80}tool_calls(?=\s|[|｜>])[^<>]*>",
    re.IGNORECASE | re.DOTALL,
)
_XML_INVOKE_OPEN_RE = re.compile(
    r"<+\s*(?!/)[^<>]{0,80}invoke(?=\s|[|｜>])[^<>]*>",
    re.IGNORECASE | re.DOTALL,
)
_XML_ANY_TAG_RE = re.compile(r"<+(?P<closing>/?)\s*(?P<body>[^<>]*?)>", re.DOTALL)
_XML_CHILD_START_RE = re.compile(r"<(?P<tag>[A-Za-z_][\w.:-]*)\b(?P<attrs>[^>]*)>", re.DOTALL)
_CDATA_RE = re.compile(r"<!\[CDATA\[(?P<body>.*?)\]\]>", re.DOTALL)
_XML_FUNCTION_EQUALS_RE = re.compile(
    r"<function\s*=\s*(?P<name>[A-Za-z_][\w.:-]*)\s*>(?P<body>.*?)</function>",
    re.IGNORECASE | re.DOTALL,
)
_CDATA_BR_SEPARATOR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_LEGACY_FUNCTION_CALLS_RE = re.compile(r"<function_calls\b[^>]*>(?P<body>.*?)</function_calls>", re.IGNORECASE | re.DOTALL)
_TOOL_CODE_RE = re.compile(r"<tool_code\b[^>]*>(?P<body>.*?)</tool_code>", re.IGNORECASE | re.DOTALL)
_LEGACY_INVOKE_RE = re.compile(r"<invoke\b(?P<attrs>[^>/]*)(?:/>\s*|>(?P<body>.*?)</invoke>)", re.IGNORECASE | re.DOTALL)
_LEGACY_PARAM_RE = re.compile(
    r"<(?P<tag>parameter|param|arg)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
_LEGACY_ATTR_RE = re.compile(r"(?P<name>[A-Za-z_][\w.:-]*)\s*=\s*(?P<quote>[\"'])(?P<value>.*?)(?P=quote)", re.DOTALL)
_PROVIDER_SEARCH_RE = re.compile(r"<search\b[^>]*>(?P<body>.*?)</search>", re.IGNORECASE | re.DOTALL)
_PROVIDER_SEARCH_QUERY_RE = re.compile(r"<query\b[^>]*>(?P<query>.*?)</query>", re.IGNORECASE | re.DOTALL)
_EMPTY_JSON_FENCE_RE = re.compile(r"```json\s*```", re.IGNORECASE | re.DOTALL)
_LEAKED_TOOL_CALL_ARRAY_RE = re.compile(
    r'\[\{\s*"function"\s*:\s*\{[\s\S]*?\}\s*,\s*"id"\s*:\s*"call[^"]*"\s*,\s*"type"\s*:\s*"function"\s*}\]',
    re.IGNORECASE,
)
_LEAKED_TOOL_RESULT_BLOB_RE = re.compile(
    r'<\s*[|｜]\s*tool\s*[|｜]\s*>\s*\{[\s\S]*?"tool_call_id"\s*:\s*"call[^"]*"\s*}',
    re.IGNORECASE,
)
_LEAKED_THINK_TAG_RE = re.compile(r"</?\s*think\s*>", re.IGNORECASE)
_LEAKED_META_MARKER_RE = re.compile(
    r"<[|｜]\s*(?:assistant|tool|begin[_▁]of[_▁]sentence|end[_▁]of[_▁]sentence|"
    r"end[_▁]of[_▁]thinking|end[_▁]of[_▁]toolresults|end[_▁]of[_▁]instructions)\s*[|｜]>",
    re.IGNORECASE,
)
_LEAKED_AGENT_XML_BLOCK_RE = re.compile(
    r"<(?P<tag>attempt_completion|ask_followup_question|new_task)\b[^>]*>(?P<body>.*?)</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
_LEAKED_AGENT_WRAPPER_TAG_RE = re.compile(r"</?(?:attempt_completion|ask_followup_question|new_task)\b[^>]*>", re.IGNORECASE | re.DOTALL)
_LEAKED_AGENT_RESULT_TAG_RE = re.compile(r"</?result>", re.IGNORECASE)
_LEAKED_AGENT_WRAPPER_PLUS_RESULT_OPEN_RE = re.compile(
    r"<(?:attempt_completion|ask_followup_question|new_task)\b[^>]*>\s*<result>",
    re.IGNORECASE | re.DOTALL,
)
_LEAKED_AGENT_RESULT_PLUS_WRAPPER_CLOSE_RE = re.compile(
    r"</result>\s*</(?:attempt_completion|ask_followup_question|new_task)\b[^>]*>",
    re.IGNORECASE | re.DOTALL,
)
_COMMAND_ARGS_RE = re.compile(r"<command-args>(?P<body>.*?)</command-args>", re.IGNORECASE | re.DOTALL)
_SKILL_ARGUMENTS_RE = re.compile(r"(?ims)^\s*ARGUMENTS:\s*(?P<body>.*?)\s*$")
_SKILL_LAUNCH_TOOL_RESULT_RE = re.compile(r"(?im)\bLaunching\s+skill:\s*(?P<skill>[A-Za-z0-9_.:@/-]+)")
_SKILL_DOC_HEADING_RE = re.compile(r"(?im)^\s*#\s+[^\n#]{1,120}\bSkill\b")
_SKILL_DOC_COLON_HEADING_RE = re.compile(
    r"(?im)^\s*#\s+(?P<skill>[A-Za-z][A-Za-z0-9_.:@/-]{1,80})\s*:\s*[^\n#]{1,160}$"
)
_SKILL_FRONTMATTER_RE = re.compile(
    r"^\s*---\s*\n"
    r"(?=[\s\S]{0,1000}^\s*name:\s*[A-Za-z0-9_.:@/-]+\s*$)"
    r"(?=[\s\S]{0,1000}^\s*description:\s*Use\s+when\b)"
    r"[\s\S]{0,1600}?^\s*---\s*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
_SKILL_DOC_MARKER_RE = re.compile(
    r"(?im)(^\s*##\s+(When|Instructions|Procedure|Workflow|Examples)\b|"
    r"^\s*##\s+Phase\b|Base directory for this skill:|Skill instructions|ARGUMENTS:)"
)
_TOOL_BLOCKED_CLAIM_RE = re.compile(
    r"(?:"
    r"\b(?:bash|shell|terminal|tool|command)\b.{0,80}\b(?:blocked|denied|not allowed|permission|unauthorized)\b|"
    r"\b(?:blocked|denied|not allowed|permission|unauthorized)\b.{0,80}\b(?:bash|shell|terminal|tool|command)\b|"
    r"(?:bash|shell|terminal|tool|command|命令|工具).{0,80}(?:被阻止|被拒绝|未授权|没有权限|无权限|权限)|"
    r"(?:授权|允许).{0,80}(?:bash|shell|terminal|命令)"
    r")",
    re.IGNORECASE,
)
_SUCCESSFUL_TOOL_RESULT_RE = re.compile(
    r"(?:completed\s+with\s+no\s+output|exit\s*code\s*[:=]?\s*0|success(?:ful|fully)?|执行完成|成功|完成)",
    re.IGNORECASE,
)
_TOOL_SUMMARY_HEADER_RE = re.compile(r"assistant\s+requested\s+tool\s+calls\s*:", re.IGNORECASE)
_TOOL_SUMMARY_CALL_RE = re.compile(r"^\s*(?:[-*•]\s*)?(?P<name>[A-Za-z_][\w.:-]*)\s*\((?P<input>\{.*\})\)\s*$")
_TOOL_SUMMARY_CALL_START_RE = re.compile(r"(?m)^\s*(?:[-*•]\s*)?(?P<name>[A-Za-z_][\w.:-]*)\s*\(")
_CONVERTED_TOOL_RESULT_RE = re.compile(
    r"(?is)^\s*Tool result for (?P<name>.+?) \(call id: (?P<id>.*?), is_error: (?P<is_error>true|false)\):\n(?P<body>.*)$"
)
_TOOL_HISTORY_ECHO_RE = re.compile(
    r"assistant\s+requested\s+tool\s+calls\s*:.*?"
    r"(?:"
    r"(?:^|\n)\s*user\s*:\s*tool\s+result\b"
    r"|\btool\s+result\s+for\b"
    r"|\buse\s+this\s+tool\s+result\s+to\s+continue\s+the\s+task\b"
    r"|(?:^|\n)\s*assistant\s*:\s*assistant\s+requested\s+tool\s+calls\s*:"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_PRIOR_TOOL_RESULT_REFERENCE_RE = re.compile(
    r"\b(?:earlier|previous|prior|existing|available|already(?:\s+available)?)\b"
    r".{0,100}\b(?:tool\s+)?(?:result|results|evidence|output|outputs|response|responses)\b|"
    r"\b(?:earlier|previous|prior)\b.{0,100}\b(?:result|results|evidence|output|outputs|response|responses)\b|"
    r"\b(?:use|using|based\s+on|from|with)\b.{0,80}\b(?:earlier|previous|prior)\b"
    r".{0,100}\b(?:result|results|evidence|output|outputs|response|responses)\b|"
    r"(?:\u4e4b\u524d|\u524d\u9762|\u4e0a\u4e00\u6b21|\u5df2\u6709|\u73b0\u6709|\u5df2\u7ecf)"
    r".{0,100}(?:\u5de5\u5177\u7ed3\u679c|\u7ed3\u679c|\u8f93\u51fa|\u8bc1\u636e|\u56de\u590d)",
    re.IGNORECASE | re.DOTALL,
)
_BARE_FUNCTION_CALL_LINE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)*)\s*\((?P<args>.*)\)\s*$"
)
_TOOL_RESULT_CLAIM_RE = re.compile(r"(<tool_result\b|工具返回|已读取文件|already read|tool result)", re.IGNORECASE)
_MISSING_FILE_TOOL_RESULT_RE = re.compile(
    r"(file\s+does\s+not\s+exist|directory\s+does\s+not\s+exist|no\s+such\s+(?:file|directory)|"
    r"cannot\s+find\s+the\s+file|path\s+does\s+not\s+exist|not\s+found|"
    r"文件不存在|没有这个文件|找不到文件|路径不存在)",
    re.IGNORECASE,
)
_UNCHANGED_READ_TOOL_RESULT_RE = re.compile(
    r"(file\s+unchanged\s+since\s+last\s+read|unchanged\s+since\s+last\s+read|"
    r"still\s+current.{0,120}re-?reading|refer\s+to\s+that\s+instead\s+of\s+re-?reading|"
    r"文件未变化|文件没有变化|内容未变化|无需重新读取)",
    re.IGNORECASE,
)
_CREATED_FILE_TOOL_RESULT_RE = re.compile(
    r"(?:file\s+created\s+successfully\s+at:\s*|\"filePath\"\s*:\s*\")(?P<path>[^\"\r\n]+)",
    re.IGNORECASE,
)
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
    r"(?:the\s+)?(?:filesystem|file\s+system|project\s+files?|codebase|repo(?:sitory)?|tools?|commands?|shell|bash|terminal)|"
    r"no\s+access\s+to\s+(?:the\s+)?(?:filesystem|file\s+system|project\s+files?|codebase|repo(?:sitory)?|tools?|commands?)|"
    r"(?:do\s+not|don't|does\s+not|doesn't)\s+have\s+access\s+to\s+(?:the\s+)?"
    r"(?:simulated\s+|specialized\s+|current\s+)?(?:tool\s+)?environment|"
    r"(?:tools?|capabilities).{0,80}(?:do\s+not|don't|does\s+not|doesn't|aren't|are\s+not).{0,50}"
    r"(?:exist|available|accessible)|"
    r"(?:simulated|specialized).{0,60}(?:tools?|capabilities|environment).{0,80}"
    r"(?:not\s+available|aren't\s+available|don't\s+exist|doesn't\s+exist)|"
    r"(?:tool\s+restrictions?|restricted\s+tools?).{0,160}"
    r"(?:manual(?:ly)?\s+(?:create|apply|copy)|provided\s+the\s+complete\s+code\s+implementations)|"
    r"(?:无法|不能|不可|没有权限|无权限)\s*(?:直接)?\s*(?:访问|使用|执行|运行|操作)?\s*"
    r"(?:(?:您|你)的)?\s*(?:项目|代码库|本地|任何)?\s*(?:文件系统|文件|代码库|项目文件|工具|操作工具|命令|git命令|终端|shell|bash)|"
    r"(?:系统|环境).{0,24}(?:限制|禁止).{0,24}(?:文件系统|工具|操作工具|命令|git|bash))",
    re.IGNORECASE,
)
_DEFERRED_TOOL_ACTION_RE = re.compile(
    r"("
    r"(?:我(?:来|会|将|先|可以)?|让我|首先|先|需要|准备|接下来).{0,40}"
    r"(?:研究|查看|读取|检查|搜索|检索|分析|了解|调查|梳理|打开|访问)"
    r"|(?:i(?:'ll| will)|i’ll|let me|first|next|i need to|i am going to|i'm going to).{0,60}"
    r"(?:inspect|read|search|check|review|analy[sz]e|look up|investigate|open|access|execute|run|update|pull|sync|stash|reset|apply|merge|push|commit)"
    r"|(?:我(?:来|会|将)|让我|现在|接下来).{0,60}(?:执行|运行|更新|拉取|重置|合并|暂存|应用|删除)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_DEFERRED_CODE_CHANGE_ACTION_RE = re.compile(
    r"("
    r"\b(?:let'?s|let us|i(?:'ll| will| am going to)|i'm going to|we(?:'ll| will| are going to)|continuing|continue)\b"
    r".{0,120}\b(?:implement|modify|edit|write|create|add|fix|refactor|patch|apply|update)\b"
    r"|\b(?:need to|must|now|next|start|begin|will)\b.{0,140}\b(?:actually\s+modify|make\s+real\s+changes|real\s+code\s+changes|actual\s+file\s+modifications|modify\s+the\s+.*?code)\b"
    r"|\b(?:will|now)\b.{0,80}\bmake\s+real\s+changes\b"
    r"|\b(?:continuing|continue).{0,100}\bimplementation\b"
    r"|\bimplementation plan\b.{0,160}\b(?:implement|modify|edit|write|create|add|fix|refactor|patch|apply|update)\b"
    r"|(?:需要|真正|现在|下一步|开始).{0,120}(?:落地|修改代码|实际文件修改|执行实际文件修改|真正落地|改代码)"
    r"|(?:我(?:会|将|来|要)|我们(?:会|将|来|要)|让我|现在|接下来|继续|开始|先).{0,120}"
    r"(?:实现|修改|编辑|写入|创建|新增|添加|修复|重构|打补丁|应用|更新|落地|改代码)"
    r"|(?:实现计划|落地计划|改进计划).{0,160}(?:实现|修改|编辑|写入|创建|新增|添加|修复|重构|应用|更新|落地)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_CODE_PATCH_SNIPPET_WITHOUT_TOOL_RE = re.compile(
    r"\b(?:bug|issue|error|problem|fix|patch|change\s+to|replace\s+with|update\s+to|before|after|old\s+code|new\s+code)\b"
    r"|(?:修复|问题|错误|故障|缺陷|改成|改为|修改为|替换为|旧代码|新代码)",
    re.IGNORECASE,
)
_INLINE_CODE_FIX_STEP_RE = re.compile(
    r"\bfix\s*:\s*(?:change|use|replace|update|set|ensure|make)\b"
    r"|\b(?:change\s+to|replace\s+with|update\s+to)\b"
    r"|\[(?:next\s+step|todo|fixme)\]",
    re.IGNORECASE,
)
_LOCAL_FILE_INSPECTION_INTENT_RE = re.compile(
    r"("
    r"\b(?:need(?:s)?(?:\s+to)?|must|next(?:\s+step)?|will|going\s+to|let'?s|start(?:ing)?|begin)\b"
    r".{0,120}\b(?:inspect|review|read|check|list|search|examine|look\s+at|open)\b"
    r".{0,120}\b(?:files?|project|repo(?:sitory)?|structure|content|core\s+files?|source|codebase)\b"
    r"|\bneed(?:s)?\b.{0,100}\b(?:current\s+)?(?:codebase|project|repo(?:sitory)?)\s+overview\b"
    r"|\buse\b.{0,100}\b(?:different\s+)?file\s+path\b.{0,100}\b(?:project\s+structure|structure\s+analysis|analysis|overview)\b"
    r"|\b(?:inspect|review|read|check|list|search|examine)\s+"
    r"(?:the\s+)?(?:files?|project(?:\s+files?)?|repo(?:sitory)?|structure|content|core\s+files?|source\s+files?|codebase)\b"
    r"|(?:需(?:要)?|必须|下一步|接下来|现在).{0,80}(?:查看|检查|读取|审查|搜索|列出|浏览).{0,80}"
    r"(?:文件|项目|目录|结构|内容|核心文件|代码)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_UNPROVEN_FINAL_ACTION_RE = re.compile(
    r"("
    r"\b(?:need(?:s)?|must|should|continue|proceed|before\s+proceeding|next(?:\s+step)?|first|another|different)\b"
    r".{0,140}\b(?:project\s+map|codebase\s+overview|file\s+path|path|structure|analysis|inspect|review|read|check|list|search|fix|change|update|add|create|establish|initialize|init|test|verify|document|harden|cover|proceed)\b"
    r"|\b(?:continue|proceed)\b.{0,100}\b(?:another|different|next|file|path|analysis|review|inspect|check)\b"
    r"|\[(?:next\s+step|todo|fixme)\]"
    r"|(?:需(?:要)?|还需|必须|应该|建议|下一步|接下来|后续|待|继续|先).{0,120}"
    r"(?:查看|检查|读取|搜索|分析|修复|修改|实施|执行|路径|结构|文件|查询|管理|继续|补充|补全|完善|建立|初始化|添加|新增|规范|测试|覆盖|验证|安全扫描|文档|流水线|版本控制|CI)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_ANSWER_MARKER_RE = re.compile(
    r"\b(?:review\s+summary|summary|findings?|conclusion|recommendations?|result|final\s+answer|no\s+(?:critical\s+)?issues?)\b"
    r"|(?:总结|结论|审查结果|最终答案|建议|未发现|没有发现)",
    re.IGNORECASE,
)
_CODE_CHANGE_COMPLETION_CLAIM_RE = re.compile(
    r"\b(?:project\s+code\s+changed|code\s+changed|implementation\s+complete|implemented|completed|done|fixed|updated|added|created|successfully|ready)\b"
    r"|(?:已|已经|完成|成功|落地|修复|修改|新增|添加|创建|接入)",
    re.IGNORECASE,
)
_FIRST_PERSON_CODE_CHANGE_COMPLETION_CLAIM_RE = re.compile(
    r"\b(?:i(?:'ve| have)|we(?:'ve| have))\s+"
    r"(?:created|implemented|updated|modified|fixed|added|completed|set\s+up|built|changed)\b"
    r"|\b(?:the|this)\s+project\s+is\s+set\s+up\b",
    re.IGNORECASE,
)
_STANDALONE_WRITE_COMPLETION_RE = re.compile(
    r"\b(?:module|file|helper|utility)\s+(?:added|created)\b"
    r"|\b(?:added|created)\s+(?:a\s+)?(?:new\s+)?(?:module|file|helper|utility)\b"
    r"|(?:新增|添加|创建).{0,30}(?:文件|模块|工具)",
    re.IGNORECASE,
)
_QUOTED_WINDOWS_PATH_RE = re.compile(r"(?P<quote>[\"'])(?P<path>[A-Za-z]:[\\/][^\"'\r\n]+)(?P=quote)")
_UNQUOTED_WINDOWS_PATH_RE = re.compile(r"(?<![\w/])(?P<path>[A-Za-z]:[\\/][^\s&|;<>]*)")
_WINDOWS_DRIVE_PATH_RE = re.compile(r"(?<![\w/])(?P<path>[A-Za-z]:[\\/][^\s\"'&|;<>]*)")
_SHELL_COMMAND_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;)\s*")
_SHELL_COMMAND_LINE_RE = re.compile(
    r"(?m)^\s*(?P<command>(?:cd|dir|ls|git|gh|bash|cmd|powershell|pwsh|python|python3|uv|npm|pnpm|npx|node|yarn|pytest|pip|pip3|ruff)\b[^\r\n]*)\s*$"
)
_SHELL_COMMAND_INTENT_RE = re.compile(
    r"\b(?:i(?:'ll| will)|let me|this will|execute|run|update|pull|reset|stash|discard|apply|merge|push|commit|local changes|modifications)\b"
    r"|(?:执行|运行|更新|拉取|重置|删除|本地修改)",
    re.IGNORECASE,
)
_PROSE_SHELL_COMMAND_INTENT_RE = re.compile(
    r"\b(?:running|executing|execute|run|will\s+run|need\s+to\s+run|going\s+to\s+run|"
    r"about\s+to\s+run|using)\b.{0,80}\b"
    r"(?:git\s+(?:diff|status|log|show|remote|branch|rev-parse|fetch|pull|add|commit|stash|reset|clean)|"
    r"ls(?:\s|$)|dir(?:\s|$)|pytest|python(?:3)?|pip(?:3)?|npm|pnpm|node|ruff)\b"
    r"|(?:执行|运行|用|使用).{0,40}"
    r"(?:git\s+(?:diff|status|log|show|remote|branch|rev-parse|fetch|pull|add|commit|stash|reset|clean)|"
    r"ls(?:\s|$)|dir(?:\s|$)|pytest|python(?:3)?|pip(?:3)?|npm|pnpm|node|ruff)\b",
    re.IGNORECASE | re.DOTALL,
)
_OFF_TASK_ENV_CONFIG_QUESTION_RE = re.compile(
    r"(?:\b(?:statusline|status\s+line|badge|settings\.json|agent\s+settings|plugin\s+settings|"
    r"environment\s+settings|hook\s+config|configure\s+(?:the\s+)?(?:agent|status|badge|plugin)|"
    r"caveman)\b|\.claude[\\/]|plugins?[\\/][^\s\"']{0,120}hooks?)",
    re.IGNORECASE | re.DOTALL,
)
_OFF_TASK_SCOPE_ESCALATION_QUESTION_RE = re.compile(
    r"(?:\b(?:restructure|reorganize|rename|move|migrate|delete|remove|overwrite|rewrite)\b"
    r".{0,80}\b(?:directory|folder|project|codebase|repository|src|structure|files?|modules?)\b|"
    r"\b(?:directory|folder|project|codebase|repository|src|structure|files?|modules?)\b"
    r".{0,80}\b(?:restructure|reorganize|rename|move|migrate|delete|remove|overwrite|rewrite)\b|"
    r"(?:重构|重组|迁移|移动|搬迁|删除|移除|覆盖|重写).{0,50}"
    r"(?:目录|结构|文件|代码|项目|仓库|模块|src|\.claude)|"
    r"(?:目录|结构|文件|代码|项目|仓库|模块|src|\.claude).{0,50}"
    r"(?:重构|重组|迁移|移动|搬迁|删除|移除|覆盖|重写))",
    re.IGNORECASE | re.DOTALL,
)
_OPTIONAL_SCOPE_SELECTION_QUESTION_RE = re.compile(
    r"(?:"
    r"\b(?:focus|scope|area|which\s+(?:area|part|module|file)|what\s+specific\s+(?:feature|problem)|"
    r"improvement\s+plan\s+focus|review\s+first|start\s+with|priorit(?:y|ize)|brainstorm\s+today)\b|"
    r"\b(?:all\s+of\s+(?:the\s+)?above|full\s+scan|core\s+scripts|architecture/code\s+quality)\b|"
    r"(?:范围|重点|方向|优先|先(?:看|审查|检查)|哪个(?:区域|模块|文件)|全部扫描|核心脚本)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_EXPLICIT_ASK_USER_TASK_RE = re.compile(
    r"\b(?:ask\s+(?:me|the\s+user)|clarify\s+with\s+(?:me|the\s+user)|interview\s+(?:me|the\s+user)|"
    r"ask\s+questions?)\b|(?:问我|向我提问|询问用户|先问|澄清问题|访谈)",
    re.IGNORECASE,
)
_REVIEW_TASK_MARKERS = (
    "review",
    "audit",
    "inspect",
    "analyze",
    "analyse",
    "understand",
    "familiarize",
    "熟悉",
    "审查",
    "检查",
    "评审",
    "分析",
    "看看",
)
_MUTATING_TASK_MARKERS = (
    "fix",
    "repair",
    "change",
    "modify",
    "implement",
    "update",
    "sync",
    "pull",
    "reset",
    "clean",
    "commit",
    "push",
    "install",
    "setup",
    "修复",
    "修改",
    "改掉",
    "实现",
    "落地",
    "更新",
    "同步",
    "拉取",
    "重置",
    "清理",
    "提交",
    "推送",
    "安装",
    "配置",
    "依赖",
)
_TEST_TASK_MARKERS = ("test", "tests", "pytest", "verify", "verification", "测试", "验证", "跑测试")
_REVIEW_BLOCKED_GIT_SUBCOMMANDS = {
    "add",
    "am",
    "apply",
    "bisect",
    "cherry-pick",
    "checkout",
    "clean",
    "clone",
    "commit",
    "filter-branch",
    "filter-repo",
    "init",
    "merge",
    "mv",
    "pull",
    "push",
    "rebase",
    "reset",
    "restore",
    "rm",
    "secrets",
    "stash",
    "switch",
}
_READONLY_GIT_SUBCOMMANDS = {
    "blame",
    "branch",
    "describe",
    "diff",
    "grep",
    "log",
    "ls-files",
    "remote",
    "rev-parse",
    "show",
    "show-ref",
    "status",
    "tag",
}
_REVIEW_BLOCKED_SHELL_COMMANDS = {"del", "move", "mv", "rd", "remove-item", "ren", "rename", "rm", "rmdir"}
_REVIEW_INSTALL_COMMANDS = {"bun", "conda", "pip", "pip3", "pipenv", "poetry", "uv"}
_REVIEW_PACKAGE_MANAGER_COMMANDS = {"npm", "pnpm", "yarn"}
_SHELL_OPTIONS_WITH_VALUES = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
_SHELL_SETUP_TASK_MARKERS = (
    "install",
    "setup",
    "dependency",
    "dependencies",
    "requirements",
    "pip install",
    "npm install",
    "pnpm install",
    "yarn install",
    "安装",
    "配置环境",
    "安装依赖",
    "依赖",
    "环境",
)
_SHELL_DESTRUCTIVE_TASK_MARKERS = (
    "discard",
    "delete local",
    "remove local",
    "reset hard",
    "reset --hard",
    "clean -fd",
    "clean -fdx",
    "force reset",
    "overwrite",
    "丢弃",
    "删掉",
    "删除本地",
    "重置",
    "清理",
    "覆盖",
)
_SHELL_GIT_UPDATE_TASK_MARKERS = ("pull", "update repo", "update repository", "更新仓库", "拉取", "同步仓库")
_CLARIFICATION_REQUEST_RE = re.compile(
    r"(\?|？|请(?:确认|提供|说明|告诉)|需要(?:确认|提供|说明)|which|what|please\s+(?:confirm|provide|specify))",
    re.IGNORECASE,
)
_REPO_CLARIFICATION_RE = re.compile(
    r"(github|git\s*hub|repository|repo|仓库|项目|更新|MediaCrawler|目录|路径)",
    re.IGNORECASE,
)
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
_REPEAT_DISCOVERY_TOOL_NAMES = frozenset(
    {
        "dir",
        "glob",
        "grep",
        "listdir",
        "listdirectory",
        "ls",
        "lsp",
        "tree",
    }
)
_SHELL_LOOP_TOOL_NAMES = frozenset({"bash", "cmd", "powershell", "shell", "terminal"})
_SHELL_READONLY_GIT_HOUSEKEEPING_MARKERS = (
    " branch",
    " diff",
    " log",
    " rev-parse",
    " show",
    " status",
)
_SHELL_FILE_DISCOVERY_HOUSEKEEPING_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:find|fd|ls|dir|get-childitem)\b|(?:^|[;&|]\s*)rg\s+--files\b",
    re.IGNORECASE,
)
_LOCAL_AGENT_TOOL_NAMES = frozenset(
    {
        "askuserquestion",
        "bash",
        "edit",
        "glob",
        "grep",
        "lsp",
        "ls",
        "multiedit",
        "read",
        "skill",
        "webfetch",
        "websearch",
        "write",
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
class ToolChoicePolicy:
    mode: str = "auto"
    forced_name: str = ""
    allowed_names: frozenset[str] = field(default_factory=frozenset)

    def is_none(self) -> bool:
        return self.mode == "none"

    def is_required(self) -> bool:
        return self.mode in {"required", "forced"}

    def allows(self, name: str) -> bool:
        if not self.allowed_names:
            return True
        return name in self.allowed_names


@dataclass(frozen=True)
class BridgeError:
    kind: str
    message: str
    repairable: bool = False


@dataclass(frozen=True)
class BridgePhase:
    name: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class BridgeResult:
    content: str
    tool_calls: list[ToolCallDraft]
    error: BridgeError | None = None
    warning: str | None = None
    raw_content: str = ""
    phases: list[BridgePhase] = field(default_factory=list)
    controller_state: str = ""
    controller_reason: str = ""
    controller_retry_budget: str = ""
    semantic_final_judge_mode: str = ""
    semantic_final_judge_verdict: str = ""
    semantic_final_judge_confidence: float | None = None
    semantic_final_judge_reason: str = ""


@dataclass(frozen=True)
class ToolBridgeContext:
    enabled: bool
    mode: str
    tools: list[ToolSpec]
    options: ToolBridgeConfig
    task_text: str = ""
    has_tool_loop: bool = False
    recent_tool_call_names: tuple[str, ...] = ()
    recent_tool_call_summaries: tuple[str, ...] = ()
    recent_skill_names: tuple[str, ...] = ()
    recent_successful_tool_result_names: tuple[str, ...] = ()
    recent_successful_tool_result_summaries: tuple[str, ...] = ()
    last_tool_result_text: str = ""
    last_tool_result_is_error: bool = False
    last_failed_read_path: str = ""
    last_failed_read_summary: str = ""
    last_failed_discovery_path: str = ""
    last_failed_discovery_summary: str = ""
    recent_failed_read_paths: tuple[str, ...] = ()
    recent_failed_read_summaries: tuple[str, ...] = ()
    recent_failed_discovery_paths: tuple[str, ...] = ()
    recent_failed_discovery_summaries: tuple[str, ...] = ()
    last_unchanged_read_path: str = ""
    last_unchanged_read_summary: str = ""
    recent_unchanged_read_paths: tuple[str, ...] = ()
    recent_unchanged_read_summaries: tuple[str, ...] = ()
    has_repo_discovery_for_write: bool = False
    hidden_tools: tuple[ToolSpec, ...] = ()
    tool_choice_policy: ToolChoicePolicy = field(default_factory=ToolChoicePolicy)

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
    if _tool_choice_is_none(tool_choice):
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
    latest = _effective_human_task_text(messages)
    if _looks_like_meta_capability_question(latest):
        return False
    if _messages_have_tool_loop(_messages_for_current_human_task(messages)):
        return True
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
    latest = _latest_human_user_text(messages)
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
    tool_choice: Any = None,
) -> ToolBridgeContext:
    cfg = options or ToolBridgeConfig()
    bridge_mode = (mode or cfg.mode or "strict").strip().lower()
    raw_specs = normalize_openai_tools(tools, max_tools=0)
    tool_choice_policy = _parse_tool_choice_policy(tool_choice, raw_specs)
    specs = raw_specs
    specs = _filter_prompt_bridge_tools(specs, cfg)
    specs = _filter_tools_by_tool_choice(specs, tool_choice_policy)
    specs = _limit_prompt_bridge_tools(specs, cfg)
    visible_names = {spec.name for spec in specs}
    hidden_tools = tuple(
        spec
        for spec in raw_specs
        if spec.name not in visible_names and _is_shell_execution_tool(spec, spec.name)
    )
    return ToolBridgeContext(
        enabled=bool(specs) and bridge_mode != "off",
        mode=bridge_mode,
        tools=specs,
        options=cfg,
        hidden_tools=hidden_tools,
        tool_choice_policy=tool_choice_policy,
    )


def _filter_tools_by_tool_choice(specs: list[ToolSpec], policy: ToolChoicePolicy) -> list[ToolSpec]:
    if policy.is_none():
        return []
    if not policy.allowed_names:
        return specs
    return [spec for spec in specs if spec.name in policy.allowed_names]


def _parse_tool_choice_policy(tool_choice: Any, specs: list[ToolSpec]) -> ToolChoicePolicy:
    canonical_by_lower = {spec.name.lower(): spec.name for spec in specs if spec.name}
    declared = frozenset(spec.name for spec in specs if spec.name)
    if tool_choice is None:
        return ToolChoicePolicy(mode="auto")
    if isinstance(tool_choice, str):
        mode = tool_choice.strip().lower()
        if mode in {"", "auto"}:
            return ToolChoicePolicy(mode="auto")
        if mode == "none":
            return ToolChoicePolicy(mode="none")
        if mode in {"required", "any"}:
            return ToolChoicePolicy(mode="required", allowed_names=declared)
        return ToolChoicePolicy(mode="auto")
    if not isinstance(tool_choice, dict):
        return ToolChoicePolicy(mode="auto")

    allowed_override = _parse_allowed_tool_choice_names(tool_choice.get("allowed_tools"), canonical_by_lower)
    choice_type = str(tool_choice.get("type") or "").strip().lower()
    if choice_type in {"none"}:
        return ToolChoicePolicy(mode="none")
    if choice_type in {"required", "any"}:
        return ToolChoicePolicy(mode="required", allowed_names=allowed_override or declared)
    forced_name = _parse_forced_tool_choice_name(tool_choice, canonical_by_lower)
    if forced_name and (choice_type in {"", "auto", "function", "tool"} or _has_tool_choice_selector(tool_choice)):
        return ToolChoicePolicy(mode="forced", forced_name=forced_name, allowed_names=frozenset({forced_name}))
    return ToolChoicePolicy(mode="auto", allowed_names=allowed_override)


def _parse_allowed_tool_choice_names(raw: Any, canonical_by_lower: dict[str, str]) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        return frozenset()
    names: list[str] = []
    for item in raw:
        name = _tool_choice_item_name(item)
        canonical = canonical_by_lower.get(name.lower()) if name else None
        if canonical:
            names.append(canonical)
    return frozenset(names)


def _parse_forced_tool_choice_name(value: dict[str, Any], canonical_by_lower: dict[str, str]) -> str:
    name = _tool_choice_item_name(value)
    return canonical_by_lower.get(name.lower(), name) if name else ""


def _has_tool_choice_selector(value: dict[str, Any]) -> bool:
    return bool(_tool_choice_item_name(value))


def _tool_choice_item_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    fn = value.get("function") if isinstance(value.get("function"), dict) else {}
    return str(
        value.get("name")
        or value.get("tool")
        or value.get("tool_name")
        or fn.get("name")
        or ""
    ).strip()


def prefer_local_tools_for_local_agent_task(
    context: ToolBridgeContext,
    messages: Any,
    *,
    tool_choice: Any = None,
) -> ToolBridgeContext:
    latest = _effective_human_task_text(messages)
    task_messages = _messages_for_current_human_task(messages)
    recent_call_drafts = _recent_tool_call_drafts(task_messages) if isinstance(task_messages, list) else []
    has_tool_loop = _messages_have_tool_loop(task_messages) or any(
        _compact_tool_name(call.name) == "skill" for call in recent_call_drafts
    )
    recent_calls = [
        (call.name, _tool_loop_category(call), _tool_loop_summary(call))
        for call in recent_call_drafts
    ]
    recent_tool_call_names = tuple(name for name, _, _ in recent_calls[-8:])
    recent_tool_call_summaries = tuple(summary for _, _, summary in recent_calls[-8:])
    recent_skill_names = tuple(
        skill_name
        for skill_name in (_skill_name_from_input(call.input) for call in recent_call_drafts[-8:] if _compact_tool_name(call.name) == "skill")
        if skill_name
    )
    last_tool_result_text, last_tool_result_is_error = (
        _last_tool_result_record(task_messages) if isinstance(task_messages, list) else ("", False)
    )
    recent_successful_tool_results = (
        _recent_successful_tool_result_records(task_messages) if isinstance(task_messages, list) else []
    )
    recent_successful_tool_result_names = tuple(
        name for name, _, _ in recent_successful_tool_results[-8:]
    )
    recent_successful_tool_result_summaries = tuple(
        summary for _, summary, _ in recent_successful_tool_results[-8:]
    )
    failed_discovery_calls = _unresolved_failed_discovery_calls(task_messages) if isinstance(task_messages, list) else []
    failed_discovery_call = failed_discovery_calls[-1] if failed_discovery_calls else None
    last_failed_discovery_path = _tool_path_from_input(failed_discovery_call.input) if failed_discovery_call else ""
    last_failed_discovery_summary = _tool_loop_summary(failed_discovery_call) if failed_discovery_call else ""
    failed_read_calls = [
        call for call in failed_discovery_calls if _compact_tool_name(call.name) in {"read", "readfile", "fileread"}
    ]
    failed_read_call = failed_read_calls[-1] if failed_read_calls else None
    last_failed_read_path = _tool_path_from_input(failed_read_call.input) if failed_read_call else ""
    last_failed_read_summary = _tool_loop_summary(failed_read_call) if failed_read_call else ""
    recent_failed_read_paths = tuple(
        path for path in (_tool_path_from_input(call.input) for call in failed_read_calls) if path
    )
    recent_failed_read_summaries = tuple(_tool_loop_summary(call) for call in failed_read_calls)
    recent_failed_discovery_paths = tuple(
        path for path in (_tool_path_from_input(call.input) for call in failed_discovery_calls) if path
    )
    recent_failed_discovery_summaries = tuple(_tool_loop_summary(call) for call in failed_discovery_calls)
    unchanged_read_calls = _recent_unchanged_read_calls(task_messages) if isinstance(task_messages, list) else []
    unchanged_read_call = unchanged_read_calls[-1] if unchanged_read_calls else None
    last_unchanged_read_path = _tool_path_from_input(unchanged_read_call.input) if unchanged_read_call else ""
    last_unchanged_read_summary = _tool_loop_summary(unchanged_read_call) if unchanged_read_call else ""
    recent_unchanged_read_paths = tuple(
        path for path in (_tool_path_from_input(call.input) for call in unchanged_read_calls) if path
    )
    recent_unchanged_read_summaries = tuple(_tool_loop_summary(call) for call in unchanged_read_calls)
    has_repo_discovery_for_write = _has_repo_discovery_for_write(task_messages) if isinstance(task_messages, list) else False
    force_tools = _tool_choice_requires_tool(tool_choice)
    force_activation = (context.options.activation_policy or "auto").strip().lower() == "always"
    shell_explicit = _local_agent_task_explicitly_requests_shell(latest)
    policy = (context.options.exposure_policy or "safe").strip().lower()
    local_agent_policy = policy in {"local-agent", "local_agent", "code-agent", "code_agent"}
    hidden_tools = list(context.hidden_tools)
    preserve_local_agent_filter = bool(context.enabled and has_tool_loop and local_agent_policy)
    if not context.enabled or (not _looks_like_local_agent_task(latest) and not preserve_local_agent_filter):
        if (
            context.enabled
            and _tool_profile(context.options) == "auto"
            and not force_activation
            and not force_tools
            and not has_tool_loop
        ):
            return ToolBridgeContext(
                enabled=False,
                mode=context.mode,
                tools=[],
                options=context.options,
                task_text=latest,
                has_tool_loop=has_tool_loop,
                recent_tool_call_names=recent_tool_call_names,
                recent_tool_call_summaries=recent_tool_call_summaries,
                recent_skill_names=recent_skill_names,
                recent_successful_tool_result_names=recent_successful_tool_result_names,
                recent_successful_tool_result_summaries=recent_successful_tool_result_summaries,
                last_tool_result_text=last_tool_result_text,
                last_tool_result_is_error=last_tool_result_is_error,
                last_failed_read_path=last_failed_read_path,
                last_failed_read_summary=last_failed_read_summary,
                last_failed_discovery_path=last_failed_discovery_path,
                last_failed_discovery_summary=last_failed_discovery_summary,
                recent_failed_read_paths=recent_failed_read_paths,
                recent_failed_read_summaries=recent_failed_read_summaries,
                recent_failed_discovery_paths=recent_failed_discovery_paths,
                recent_failed_discovery_summaries=recent_failed_discovery_summaries,
                last_unchanged_read_path=last_unchanged_read_path,
                last_unchanged_read_summary=last_unchanged_read_summary,
                recent_unchanged_read_paths=recent_unchanged_read_paths,
                recent_unchanged_read_summaries=recent_unchanged_read_summaries,
                has_repo_discovery_for_write=has_repo_discovery_for_write,
                hidden_tools=_dedupe_tool_specs(hidden_tools),
                tool_choice_policy=context.tool_choice_policy,
            )
        task_text = latest if context.enabled and has_tool_loop and latest.strip() else context.task_text
        return ToolBridgeContext(
            enabled=context.enabled,
            mode=context.mode,
            tools=context.tools,
            options=context.options,
            task_text=task_text,
            has_tool_loop=has_tool_loop,
            recent_tool_call_names=recent_tool_call_names,
            recent_tool_call_summaries=recent_tool_call_summaries,
            recent_skill_names=recent_skill_names,
            recent_successful_tool_result_names=recent_successful_tool_result_names,
            recent_successful_tool_result_summaries=recent_successful_tool_result_summaries,
            last_tool_result_text=last_tool_result_text,
            last_tool_result_is_error=last_tool_result_is_error,
            last_failed_read_path=last_failed_read_path,
            last_failed_read_summary=last_failed_read_summary,
            last_failed_discovery_path=last_failed_discovery_path,
            last_failed_discovery_summary=last_failed_discovery_summary,
            recent_failed_read_paths=recent_failed_read_paths,
            recent_failed_read_summaries=recent_failed_read_summaries,
            recent_failed_discovery_paths=recent_failed_discovery_paths,
            recent_failed_discovery_summaries=recent_failed_discovery_summaries,
            last_unchanged_read_path=last_unchanged_read_path,
            last_unchanged_read_summary=last_unchanged_read_summary,
            recent_unchanged_read_paths=recent_unchanged_read_paths,
            recent_unchanged_read_summaries=recent_unchanged_read_summaries,
            has_repo_discovery_for_write=has_repo_discovery_for_write,
            hidden_tools=_dedupe_tool_specs(hidden_tools),
            tool_choice_policy=context.tool_choice_policy,
        )
    candidate_tools = [tool for tool in context.tools if _provider_search_tool_score(tool) <= 0]
    candidate_shell_tools = [tool for tool in candidate_tools if _is_shell_execution_tool(tool, tool.name)]
    local_tools = list(candidate_tools)
    profile = _resolved_tool_profile_for_task(context.options, latest)
    ds2api_passthrough = profile == "all"
    if ds2api_passthrough:
        profile = "all"
        local_tools = list(context.tools)
    elif profile == "read-only":
        if not shell_explicit:
            hidden_tools.extend(candidate_shell_tools)
        local_tools = [tool for tool in candidate_tools if _is_profile_readonly_tool(tool, context.options)]
        if shell_explicit:
            existing = {tool.name for tool in local_tools}
            local_tools.extend(
                tool
                for tool in candidate_tools
                if tool.name not in existing and _is_profile_shell_tool(tool, context.options)
            )
    elif profile == "agent":
        local_tools = [tool for tool in candidate_tools if _is_profile_agent_tool(tool, context.options)]
    if (
        not ds2api_passthrough
        and (profile == "agent" or policy in {"local-agent", "local_agent", "code-agent", "code_agent"})
        and not shell_explicit
    ):
        hidden_tools.extend(tool for tool in local_tools if _is_shell_execution_tool(tool, tool.name))
        local_tools = [tool for tool in local_tools if not _is_shell_execution_tool(tool, tool.name)]
    if (
        not ds2api_passthrough
        and policy in {"local-agent", "local_agent", "code-agent", "code_agent"}
        and has_tool_loop
        and (last_failed_read_path or last_failed_discovery_path)
        and not _task_explicitly_requests_standalone_new_file(latest)
        and not has_repo_discovery_for_write
    ):
        local_tools = [tool for tool in local_tools if _compact_tool_name(tool.name) not in {"write", "writefile"}]
    if not local_tools:
        if profile in {"read-only", "agent", "none"}:
            return ToolBridgeContext(
                enabled=False,
                mode=context.mode,
                tools=[],
                options=context.options,
                task_text=latest,
                has_tool_loop=has_tool_loop,
                recent_tool_call_names=recent_tool_call_names,
                recent_tool_call_summaries=recent_tool_call_summaries,
                recent_skill_names=recent_skill_names,
                recent_successful_tool_result_names=recent_successful_tool_result_names,
                recent_successful_tool_result_summaries=recent_successful_tool_result_summaries,
                last_tool_result_text=last_tool_result_text,
                last_tool_result_is_error=last_tool_result_is_error,
                last_failed_read_path=last_failed_read_path,
                last_failed_read_summary=last_failed_read_summary,
                last_failed_discovery_path=last_failed_discovery_path,
                last_failed_discovery_summary=last_failed_discovery_summary,
                recent_failed_read_paths=recent_failed_read_paths,
                recent_failed_read_summaries=recent_failed_read_summaries,
                recent_failed_discovery_paths=recent_failed_discovery_paths,
                recent_failed_discovery_summaries=recent_failed_discovery_summaries,
                last_unchanged_read_path=last_unchanged_read_path,
                last_unchanged_read_summary=last_unchanged_read_summary,
                recent_unchanged_read_paths=recent_unchanged_read_paths,
                recent_unchanged_read_summaries=recent_unchanged_read_summaries,
                has_repo_discovery_for_write=has_repo_discovery_for_write,
                hidden_tools=_dedupe_tool_specs(hidden_tools),
                tool_choice_policy=context.tool_choice_policy,
            )
        return ToolBridgeContext(
            enabled=context.enabled,
            mode=context.mode,
            tools=context.tools,
            options=context.options,
            task_text=latest,
            has_tool_loop=has_tool_loop,
            recent_tool_call_names=recent_tool_call_names,
            recent_tool_call_summaries=recent_tool_call_summaries,
            recent_skill_names=recent_skill_names,
            recent_successful_tool_result_names=recent_successful_tool_result_names,
            recent_successful_tool_result_summaries=recent_successful_tool_result_summaries,
            last_tool_result_text=last_tool_result_text,
            last_tool_result_is_error=last_tool_result_is_error,
            last_failed_read_path=last_failed_read_path,
            last_failed_read_summary=last_failed_read_summary,
            last_failed_discovery_path=last_failed_discovery_path,
            last_failed_discovery_summary=last_failed_discovery_summary,
            recent_failed_read_paths=recent_failed_read_paths,
            recent_failed_read_summaries=recent_failed_read_summaries,
            recent_failed_discovery_paths=recent_failed_discovery_paths,
            recent_failed_discovery_summaries=recent_failed_discovery_summaries,
            last_unchanged_read_path=last_unchanged_read_path,
            last_unchanged_read_summary=last_unchanged_read_summary,
            recent_unchanged_read_paths=recent_unchanged_read_paths,
            recent_unchanged_read_summaries=recent_unchanged_read_summaries,
            has_repo_discovery_for_write=has_repo_discovery_for_write,
            hidden_tools=_dedupe_tool_specs(hidden_tools),
            tool_choice_policy=context.tool_choice_policy,
        )
    return ToolBridgeContext(
        enabled=bool(local_tools) and context.mode != "off",
        mode=context.mode,
        tools=local_tools,
        options=context.options,
        task_text=latest,
        has_tool_loop=has_tool_loop,
        recent_tool_call_names=recent_tool_call_names,
        recent_tool_call_summaries=recent_tool_call_summaries,
        recent_skill_names=recent_skill_names,
        recent_successful_tool_result_names=recent_successful_tool_result_names,
        recent_successful_tool_result_summaries=recent_successful_tool_result_summaries,
        last_tool_result_text=last_tool_result_text,
        last_tool_result_is_error=last_tool_result_is_error,
        last_failed_read_path=last_failed_read_path,
        last_failed_read_summary=last_failed_read_summary,
        last_failed_discovery_path=last_failed_discovery_path,
        last_failed_discovery_summary=last_failed_discovery_summary,
        recent_failed_read_paths=recent_failed_read_paths,
        recent_failed_read_summaries=recent_failed_read_summaries,
        recent_failed_discovery_paths=recent_failed_discovery_paths,
        recent_failed_discovery_summaries=recent_failed_discovery_summaries,
        last_unchanged_read_path=last_unchanged_read_path,
        last_unchanged_read_summary=last_unchanged_read_summary,
        recent_unchanged_read_paths=recent_unchanged_read_paths,
        recent_unchanged_read_summaries=recent_unchanged_read_summaries,
        has_repo_discovery_for_write=has_repo_discovery_for_write,
        hidden_tools=_dedupe_tool_specs(hidden_tools),
        tool_choice_policy=context.tool_choice_policy,
    )


def build_local_repo_preflight_tool_call(context: ToolBridgeContext) -> ToolCallDraft | None:
    if not context.enabled or context.has_tool_loop:
        return None
    match = _update_existing_repo_path_match(context.task_text)
    if not match:
        return None
    tool = _select_shell_execution_tool(context.tools)
    if not tool:
        return None
    repo_path = _windows_path_to_bash_path(match.group("path").strip("\"'"))
    repo_arg = shlex.quote(repo_path) if re.search(r"\s", repo_path) else repo_path
    command = f"git -C {repo_arg} remote -v && git -C {repo_arg} status --short"
    return ToolCallDraft(
        id="call_web_preflight_1",
        name=tool.name,
        input={_shell_command_key(tool): command},
    )


def normalize_openai_tools(tools: Any, *, max_tools: int = 32) -> list[ToolSpec]:
    if not isinstance(tools, list):
        return []
    specs: list[ToolSpec] = []
    for item in tools:
        fn = item.get("function") if isinstance(item, dict) else None
        if isinstance(fn, dict):
            name = str(fn.get("name") or "").strip()
            description = str(fn.get("description") or "")
            schema = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object", "properties": {}}
            read_only = _is_read_only_tool(name, fn)
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            description = str(item.get("description") or "")
            schema = first_dict(
                item.get("parameters"),
                item.get("input_schema"),
                item.get("inputSchema"),
                item.get("schema"),
            ) or {"type": "object", "properties": {}}
            read_only = _is_read_only_name(name)
        else:
            continue
        if not name:
            continue
        specs.append(
            ToolSpec(
                name=name,
                description=description,
                input_schema=schema,
                read_only=read_only,
            )
        )
    if max_tools <= 0 or len(specs) <= max_tools:
        return specs
    return specs[:max_tools]


def first_dict(*values: Any) -> dict[str, Any] | None:
    for value in values:
        if isinstance(value, dict):
            return value
    return None


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
    profile = _tool_profile(options)
    if profile == "all":
        return specs
    if profile == "none":
        return []
    if profile == "read-only":
        return [spec for spec in specs if _is_profile_readonly_tool(spec, options)]
    if profile == "agent":
        return [spec for spec in specs if _is_profile_agent_tool(spec, options)]
    policy = (options.exposure_policy or "safe").strip().lower()
    if policy == "all":
        return specs
    if policy in {"local-agent", "local_agent", "code-agent", "code_agent"}:
        return [spec for spec in specs if _is_local_agent_prompt_tool(spec)]
    if policy == "allowlist":
        allowed = {name.strip().lower() for name in options.allowed_tool_names if name.strip()}
        if allowed:
            return [spec for spec in specs if spec.name.strip().lower() in allowed]
    return [spec for spec in specs if not _is_prompt_bridge_blocked_tool(spec.name)]


def _limit_prompt_bridge_tools(specs: list[ToolSpec], options: ToolBridgeConfig) -> list[ToolSpec]:
    if _tool_profile(options) == "all":
        return specs
    policy = (options.exposure_policy or "safe").strip().lower()
    if policy == "all":
        return specs
    max_tools = int(options.max_tools_in_prompt or 0)
    if max_tools <= 0 or len(specs) <= max_tools:
        return specs
    return specs[:max_tools]


def _tool_profile(options: ToolBridgeConfig) -> str:
    profile = _configured_tool_profile(options)
    if profile == "auto" and (options.exposure_policy or "").strip().lower() == "all":
        return "all"
    return profile


def _configured_tool_profile(options: ToolBridgeConfig) -> str:
    profile = (options.tool_profile or "auto").strip().lower().replace("_", "-")
    if profile in {"readonly", "read-only"}:
        return "read-only"
    if profile in {"all", "none", "agent", "auto"}:
        return profile
    return "auto"


def _resolved_tool_profile_for_task(options: ToolBridgeConfig, task_text: str) -> str:
    profile = _configured_tool_profile(options)
    if profile != "auto":
        return profile
    if (
        (options.activation_policy or "auto").strip().lower() == "always"
        and (options.exposure_policy or "safe").strip().lower() == "all"
    ):
        return "all"
    if not _looks_like_local_agent_task(task_text):
        return _tool_profile(options)
    if _task_explicitly_allows_mutation(task_text):
        return "agent"
    return "read-only"


def _is_profile_readonly_tool(spec: ToolSpec, options: ToolBridgeConfig) -> bool:
    lowered = spec.name.strip().lower()
    compact = _compact_tool_name(spec.name)
    readonly_names = {name.strip().lower() for name in options.readonly_tool_names if name.strip()}
    if lowered in readonly_names or spec.read_only or _is_read_only_name(spec.name):
        return True
    if _is_readonly_skill_tool_name(compact):
        return True
    if (
        _configured_tool_profile(options) == "auto"
        and (options.exposure_policy or "").strip().lower() == "all"
        and _is_shell_execution_tool(spec, spec.name)
    ):
        return True
    if lowered.startswith(("mcp__", "mcp-")):
        return any(marker in compact for marker in ("read", "list", "search", "fetch", "get", "query", "find"))
    return False


def _is_profile_write_tool(spec: ToolSpec, options: ToolBridgeConfig) -> bool:
    compact = _compact_tool_name(spec.name)
    write_names = {_compact_tool_name(name) for name in options.write_tool_names if name.strip()}
    return compact in write_names or compact in {"edit", "write", "multiedit"}


def _is_profile_shell_tool(spec: ToolSpec, options: ToolBridgeConfig) -> bool:
    compact = _compact_tool_name(spec.name)
    shell_names = {_compact_tool_name(name) for name in options.shell_tool_names if name.strip()}
    return compact in shell_names or _is_shell_execution_tool(spec, spec.name)


def _is_readonly_skill_tool_name(compact_name: str) -> bool:
    if "skill" not in compact_name:
        return False
    if compact_name == "skill":
        return True
    return any(marker in compact_name for marker in ("list", "search", "find", "get", "read", "query", "show"))


def _is_profile_agent_tool(spec: ToolSpec, options: ToolBridgeConfig) -> bool:
    return (
        _is_profile_readonly_tool(spec, options)
        or _is_profile_write_tool(spec, options)
        or _is_profile_shell_tool(spec, options)
        or _is_local_agent_prompt_tool(spec)
    )


def _is_prompt_bridge_blocked_tool(name: str) -> bool:
    lowered = (name or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if lowered in _PROMPT_BRIDGE_BLOCKED_TOOL_NAMES or compact in _PROMPT_BRIDGE_BLOCKED_TOOL_NAMES:
        return True
    parts = [part for part in re.split(r"[^a-z0-9]+", lowered) if part]
    if any(part in _PROMPT_BRIDGE_BLOCKED_TOOL_PARTS for part in parts):
        return True
    return any(compact.endswith(part) for part in {"edit", "write", "applypatch"})


def _is_local_agent_prompt_tool(spec: ToolSpec) -> bool:
    lowered = (spec.name or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if lowered.startswith("mcp__") or lowered.startswith("mcp-"):
        return bool(spec.read_only)
    return compact in _LOCAL_AGENT_TOOL_NAMES


def _local_agent_task_explicitly_requests_shell(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if _looks_like_update_existing_repo_task(lowered):
        return True
    markers = (
        "bash",
        "shell",
        "terminal",
        "powershell",
        "cmd",
        "git ",
        "gh ",
        "pytest",
        "npm",
        "pnpm",
        "yarn",
        "pip ",
        "python -m",
        "node ",
        "ruff",
        "lint",
        "test",
        "verify",
        "build",
        "run command",
        "execute command",
        "install",
        "dependency",
        "dependencies",
        "commit",
        "push",
        "pull",
        "update repo",
        "update repository",
        "sync repo",
        "sync repository",
        "终端",
        "命令",
        "运行命令",
        "执行命令",
        "测试",
        "验证",
        "跑测试",
        "启动",
        "构建",
        "安装",
        "依赖",
        "拉取",
        "同步仓库",
        "更新仓库",
        "提交",
        "推送",
    )
    return any(marker in lowered for marker in markers)


def prepare_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ctx = build_context(tools)
    return prepare_openai_messages(messages, ctx)


def prepare_openai_messages(messages: list[dict[str, Any]], context: ToolBridgeContext) -> list[dict[str, Any]]:
    if not context.enabled:
        return [message for message in messages if isinstance(message, dict)]
    prompt = build_tool_prompt(context.tools, context.options)
    call_names = _assistant_tool_call_names(messages)
    converted = [_convert_message(message, call_names=call_names, options=context.options) for message in messages if isinstance(message, dict)]
    if not _allows_ds2api_style_progress_passthrough(context):
        task_messages = _messages_for_current_human_task(messages)
        loop_guard = _tool_loop_guard_message(task_messages)
        if loop_guard:
            converted.append({"role": "user", "content": loop_guard})
        repeat_skill_guard = _repeat_skill_guard_message(task_messages)
        if repeat_skill_guard:
            converted.append({"role": "user", "content": repeat_skill_guard})
    task_messages = _messages_for_current_human_task(messages)
    failed_path_guard = _failed_path_tool_result_guard_message(task_messages, context)
    if failed_path_guard:
        converted.append({"role": "user", "content": failed_path_guard})
    if not _allows_ds2api_style_progress_passthrough(context):
        unchanged_read_guard = _unchanged_read_tool_result_guard_message(task_messages, context)
        if unchanged_read_guard:
            converted.append({"role": "user", "content": unchanged_read_guard})
    boundary_guard = _new_task_boundary_guard_message(messages, context)
    if boundary_guard:
        converted.append({"role": "user", "content": boundary_guard})
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
    code_change_rule = (
        "- For implementation or bug-fix tasks, inspect existing files before changing them, prefer Edit/MultiEdit for existing code, "
        "and do not claim completion after only creating an isolated helper/module unless the user explicitly asked for a standalone new file. "
        "If you create a new file, also integrate it from an existing entry point or request another tool to verify where it is used.\n"
        if any(_compact_tool_name(str(function.get("name") or "")) in {"edit", "write", "multiedit", "applypatch"} for function in functions)
        else ""
    )
    return (
        "You are using WebAI Gateway's strict tool bridge. You are allowed to request the downstream client to execute listed tools; "
        "do not confuse this with executing tools inside the web model runtime yourself.\n\n"
        "Available tools (allowed names only):\n"
        f"{tool_manifest}\n\n"
        "Decision rules:\n"
        "- If the task can be answered from the conversation or prior Tool result messages, answer normally.\n"
        "- If a tool is needed, output exactly one DSML tool block and no natural language outside it.\n"
        "- Never use provider-native browsing tags such as <search>. Use only the DSML tool_calls wrapper below.\n"
        "- Never output summaries like 'Assistant requested tool calls'. Those are history text, not the required protocol.\n"
        "- Never say an available tool does not exist or that you lack permission for files, commands, Bash/Git, or project updates. Do not call AskUserQuestion to request permission; Request the listed tool directly and let the downstream client enforce permissions.\n"
        "- Do not call AskUserQuestion for optional next-step or scope selection; continue if the task is clear.\n"
        "- After a Tool result message, use the observation to answer or request another allowed tool. Do not wait, do not claim that you executed a tool yourself, and do not repeat a failed identical call.\n"
        "- If a Tool result says is_error: true, do not treat it as successful data. Choose a different allowed tool/input if recovery is possible; otherwise explain the failure briefly.\n"
        f"{code_change_rule}"
        "- For Glob/file-discovery tools, avoid repository-wide recursive patterns such as **/*, **/*.ext, or **/package.json unless the input also scopes the search to a narrow path. Prefer Read for known files, LS/list tools for directory overviews, or scoped patterns like src/**/*.py.\n"
        "- For public GitHub repository or source-code URLs, prefer machine-readable endpoints such as https://api.github.com/repos/<owner>/<repo>, the GitHub contents API, or raw.githubusercontent.com files instead of interactive HTML pages.\n\n"
        "Required tool-call format:\n"
        "<|DSML|tool_calls>\n"
        "  <|DSML|invoke name=\"tool_name\">\n"
        "    <|DSML|parameter name=\"arg\"><![CDATA[value]]></|DSML|parameter>\n"
        "  </|DSML|invoke>\n"
        "</|DSML|tool_calls>\n"
        "DSML rules: use one <|DSML|tool_calls> root; put each call in <|DSML|invoke name=\"...\">; put every top-level argument in <|DSML|parameter name=\"...\">; wrap all string values in <![CDATA[...]]>; objects use nested XML elements; arrays repeat <item>; do not wrap DSML in markdown fences. The first non-whitespace characters of a tool request must be <|DSML|tool_calls>. Compatibility: canonical <tool_calls>/<invoke>/<parameter> is accepted, but prefer DSML.\n"
        "Never omit the opening <|DSML|tool_calls> tag. Wrong 1 - mixed text after DSML. Wrong 2 - markdown code fences. Wrong 3 - missing opening wrapper.\n"
        f"{_build_ds2api_style_tool_examples(specs)}"
        f"Default maximum tool calls per turn: {cfg.max_calls_per_turn}; read-only tools may use up to {cfg.max_readonly_calls_per_turn}. "
        "All arguments must be inside the input object."
    )


def _build_ds2api_style_tool_examples(specs: list[ToolSpec]) -> str:
    examples: list[str] = []
    basic = _first_ds2api_style_example(specs)
    if basic:
        examples.append("Example A - Single tool:\n" + basic)
    pair = [_ds2api_style_example_for_tool(spec) for spec in specs[:2]]
    pair = [item for item in pair if item]
    if len(pair) >= 2:
        examples.append("Example B - Two tools in parallel:\n" + _render_ds2api_style_tool_block(pair[:2]))
    script = next((item for item in (_ds2api_style_script_example_for_tool(spec) for spec in specs) if item), "")
    if script:
        examples.append("Example C - Tool with long script using CDATA:\n" + script)
    if not examples:
        return ""
    return "\nCorrect DSML examples:\n\n" + "\n\n".join(examples) + "\n\n"


def _first_ds2api_style_example(specs: list[ToolSpec]) -> str:
    for spec in specs:
        rendered = _ds2api_style_example_for_tool(spec)
        if rendered:
            return _render_ds2api_style_tool_block([rendered])
    return ""


def _ds2api_style_example_for_tool(spec: ToolSpec) -> tuple[str, list[tuple[str, str]]] | None:
    compact = _compact_tool_name(spec.name)
    if not spec.name:
        return None
    if compact in {"bash", "executecommand", "execcommand", "exec", "execute", "shell", "terminal", "command", "runcommand"}:
        return (spec.name, [(_shell_command_key(spec), "pwd")])
    if compact in {"read", "readfile", "fileread"}:
        return (spec.name, [(_first_schema_property_name(spec, ("file_path", "path")) or "file_path", "README.md")])
    if compact in {"glob", "grep", "searchfiles", "listfiles", "ls", "listdir", "lsp"} or "search" in compact:
        key = _first_schema_property_name(spec, ("pattern", "query", "path")) or "query"
        value = "**/*.py" if key == "pattern" else "tool call parser"
        return (spec.name, [(key, value)])
    if compact in {"edit", "write", "multiedit", "applypatch"}:
        return (spec.name, [(_first_schema_property_name(spec, ("file_path", "path")) or "file_path", "README.md")])
    return (spec.name, [])


def _ds2api_style_script_example_for_tool(spec: ToolSpec) -> str:
    compact = _compact_tool_name(spec.name)
    if compact not in {"bash", "executecommand", "execcommand", "exec", "execute", "shell", "terminal", "command", "runcommand"}:
        return ""
    script = "cat > /tmp/test_escape.sh <<'EOF'\n#!/bin/bash\necho 'single \"double\"'\necho \"literal dollar: $HOME\"\nEOF\nbash /tmp/test_escape.sh"
    params = [(_shell_command_key(spec), script)]
    if _schema_has_property(spec, "description"):
        params.append(("description", "Test shell escaping"))
    return _render_ds2api_style_tool_block([(spec.name, params)])


def _render_ds2api_style_tool_block(examples: list[tuple[str, list[tuple[str, str]]]]) -> str:
    lines = ["<|DSML|tool_calls>"]
    for name, params in examples:
        lines.append(f'  <|DSML|invoke name="{html.escape(name, quote=True)}">')
        for key, value in params:
            lines.append(
                f'    <|DSML|parameter name="{html.escape(key, quote=True)}"><![CDATA[{value}]]></|DSML|parameter>'
            )
        lines.append("  </|DSML|invoke>")
    lines.append("</|DSML|tool_calls>")
    return "\n".join(lines)


def _first_schema_property_name(spec: ToolSpec, names: tuple[str, ...]) -> str:
    properties = spec.input_schema.get("properties") if isinstance(spec.input_schema, dict) else None
    if not isinstance(properties, dict):
        return ""
    for name in names:
        if name in properties:
            return name
    return ""


def _schema_has_property(spec: ToolSpec, name: str) -> bool:
    properties = spec.input_schema.get("properties") if isinstance(spec.input_schema, dict) else None
    return isinstance(properties, dict) and name in properties


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


def _new_task_boundary_guard_message(messages: list[dict[str, Any]], context: ToolBridgeContext) -> str:
    if not _current_task_starts_after_prior_tool_history(messages):
        return ""
    task = (context.task_text or _effective_human_task_text(messages)).strip()
    if not task:
        return ""
    return (
        "New user task boundary: the latest user request starts a new task after earlier tool history. "
        "Treat previous tool calls, previous tool results, background task ids, and install/run commands as historical context only. "
        "Do not poll TaskOutput, continue prior background jobs, or resume prior setup/install commands unless the latest user request explicitly asks to continue them.\n"
        f"Current user task: {task[:1000]}"
    )


def _tool_loop_guard_message(messages: list[dict[str, Any]]) -> str:
    if not _last_message_is_tool_result(messages):
        return ""
    calls = _recent_tool_call_summaries(messages)
    if len(calls) < 3:
        return ""
    window = calls[-6:]
    grouped: dict[str, list[tuple[str, str]]] = {}
    for name, category, summary in window:
        grouped.setdefault(category, []).append((name, summary))
    repeated = max(grouped.values(), key=len)
    if len(repeated) < 3:
        return ""
    names = ", ".join(sorted({name for name, _ in repeated}))
    examples = "\n".join(f"- {summary}" for _, summary in repeated[-3:])
    return (
        "Tool loop guard: recent tool history contains repeated or equivalent calls "
        f"for {names}.\n"
        f"{examples}\n"
        "If the latest tool results already provide enough evidence, show success, or show no progress, "
        "stop requesting equivalent tools and give the final answer. Only request another tool if it uses "
        "materially different input that is still required to complete the user task."
    )


def _repeat_skill_guard_message(messages: list[dict[str, Any]]) -> str:
    if not (_last_message_is_tool_result(messages) or _last_message_is_skill_injection_text(messages)):
        return ""
    calls = _recent_tool_call_summaries(messages)
    skill_summaries = [summary for name, _, summary in calls[-8:] if _compact_tool_name(name) == "skill"]
    if not skill_summaries:
        return ""
    unique_recent = list(dict.fromkeys(skill_summaries[-3:]))
    examples = "\n".join(f"- {summary}" for summary in unique_recent)
    return (
        "Skill progress guard: the Skill tool has already loaded the requested skill in this active tool loop.\n"
        f"{examples}\n"
        "Do not request the same Skill again. Use the loaded skill instructions already in the conversation, "
        "then continue the original user task with a different allowed tool such as Read, Glob, Grep, Edit, "
        "or Write. If no more tool evidence is needed, provide a substantive final answer instead."
    )


def _failed_path_tool_result_guard_message(messages: list[dict[str, Any]], context: ToolBridgeContext) -> str:
    call = _last_failed_discovery_call(messages)
    if not call:
        return ""
    summary = _tool_loop_summary(call)
    compact = _compact_tool_name(call.name)
    is_read = compact in {"read", "readfile", "fileread"}
    label = "Read call" if is_read else "file-discovery call"
    repeat_instruction = (
        "Do not repeat the same Read input and do not create that same or nearby guessed path with Write. "
        if is_read
        else "Do not repeat the same input and do not create that same guessed path or its children with Write. "
    )
    return (
        f"Tool result guard: the last {label} failed because the file, path, or directory does not exist.\n"
        f"- Failed call: {summary}\n"
        f"{repeat_instruction}"
        f"Use {_discovery_tool_names_for_error(context)} to discover real paths, "
        "or explain the missing file if it blocks the task."
    )


def _unchanged_read_tool_result_guard_message(messages: list[dict[str, Any]], context: ToolBridgeContext) -> str:
    calls = _recent_unchanged_read_calls(messages)
    if not calls:
        return ""
    call = calls[-1]
    summary = _tool_loop_summary(call)
    return (
        "Tool result guard: the last Read result reported unchanged content.\n"
        f"- Reused call: {summary}\n"
        "Do not repeat the same Read input. Use the earlier Read content already present in the conversation, "
        f"choose {_discovery_tool_names_for_error(context)} with materially different input if more context is needed, "
        "use Edit/Write when ready to change code, or answer if the evidence is sufficient."
    )


def _last_tool_result_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "system":
            continue
        content = message.get("content")
        if role == "tool":
            return _content_to_text(content)
        if role == "user" and isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return _content_to_text(block.get("content"))
        return ""
    return ""


def _last_tool_result_record(messages: list[dict[str, Any]]) -> tuple[str, bool]:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "system":
            continue
        results = _message_tool_results(message)
        if results:
            _, text, is_error = results[-1]
            return text, is_error
        return "", False
    return "", False


def _last_message_is_tool_result(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "system":
            continue
        if role == "tool":
            return True
        if role == "user" and _message_is_tool_result(message):
            return True
        content = message.get("content")
        if role == "user" and isinstance(content, list):
            return any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
        return False
    return False


def _last_message_is_skill_injection_text(messages: list[dict[str, Any]]) -> bool:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "system":
            continue
        if role != "user":
            return False
        if _message_is_gateway_control_instruction(message):
            continue
        text = _content_to_text(message.get("content"))
        return _looks_like_skill_injection_text(text) or _message_is_skill_payload_after_tool_result(messages, index)
    return False


def _last_failed_read_call(messages: list[dict[str, Any]]) -> ToolCallDraft | None:
    for call in reversed(_unresolved_failed_discovery_calls(messages)):
        if _compact_tool_name(call.name) in {"read", "readfile", "fileread"}:
            return call
    return None


def _last_failed_discovery_call(messages: list[dict[str, Any]]) -> ToolCallDraft | None:
    calls = _unresolved_failed_discovery_calls(messages)
    return calls[-1] if calls else None


def _unresolved_failed_discovery_calls(messages: list[dict[str, Any]]) -> list[ToolCallDraft]:
    calls_by_id: dict[str, ToolCallDraft] = {}
    created_paths: set[str] = set()
    unresolved: list[ToolCallDraft] = []

    def remember_failed(call: ToolCallDraft) -> None:
        path = _normalize_path_for_compare(_tool_path_from_input(call.input))
        if path:
            unresolved[:] = [
                existing
                for existing in unresolved
                if _normalize_path_for_compare(_tool_path_from_input(existing.input)) != path
            ]
        unresolved.append(call)

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "assistant":
            for call in _message_tool_call_drafts(message):
                if call.id:
                    calls_by_id[call.id] = call
            continue
        results = _message_tool_results(message)
        if role == "user" and not results:
            unresolved = []
            continue
        for tool_call_id, result_text, is_error in results:
            created_path = _created_file_path_from_tool_result(result_text)
            if created_path:
                created_paths.add(_normalize_path_for_compare(created_path))
            call = calls_by_id.get(tool_call_id)
            if call is None:
                continue
            compact = _compact_tool_name(call.name)
            path = _normalize_path_for_compare(_tool_path_from_input(call.input))
            is_discovery = compact in {
                "read",
                "readfile",
                "fileread",
                "glob",
                "grep",
                "ls",
                "listdir",
                "find",
                "search",
            } or "search" in compact
            if is_error and is_discovery and _MISSING_FILE_TOOL_RESULT_RE.search(result_text or ""):
                remember_failed(call)
                continue
            if not is_error and compact in {"write", "writefile"}:
                if path:
                    created_paths.add(path)
                continue
            if not is_error and is_discovery:
                if compact in {"read", "readfile", "fileread"} and path and path in created_paths:
                    continue
                unresolved = []
    return unresolved[-8:]


def _recent_unchanged_read_calls(messages: list[dict[str, Any]]) -> list[ToolCallDraft]:
    calls_by_id: dict[str, ToolCallDraft] = {}
    unchanged: list[ToolCallDraft] = []

    def remember(call: ToolCallDraft) -> None:
        path = _normalize_path_for_compare(_tool_path_from_input(call.input))
        if path:
            unchanged[:] = [
                existing
                for existing in unchanged
                if _normalize_path_for_compare(_tool_path_from_input(existing.input)) != path
            ]
        unchanged.append(call)

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "assistant":
            for call in _message_tool_call_drafts(message):
                if call.id:
                    calls_by_id[call.id] = call
            continue
        for tool_call_id, result_text, is_error in _message_tool_results(message):
            if is_error or not _UNCHANGED_READ_TOOL_RESULT_RE.search(result_text or ""):
                continue
            call = calls_by_id.get(tool_call_id)
            if call is None:
                continue
            if _compact_tool_name(call.name) in {"read", "readfile", "fileread"}:
                remember(call)
    return unchanged[-8:]


def _has_repo_discovery_for_write(messages: list[dict[str, Any]]) -> bool:
    calls_by_id: dict[str, ToolCallDraft] = {}
    created_paths: set[str] = set()
    discovered = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") == "assistant":
            for call in _message_tool_call_drafts(message):
                if call.id:
                    calls_by_id[call.id] = call
            continue
        for tool_call_id, result_text, is_error in _message_tool_results(message):
            created_path = _created_file_path_from_tool_result(result_text)
            if created_path:
                created_paths.add(_normalize_path_for_compare(created_path))
            call = calls_by_id.get(tool_call_id)
            if call is None or is_error:
                continue
            compact = _compact_tool_name(call.name)
            path = _normalize_path_for_compare(_tool_path_from_input(call.input))
            if compact in {"write", "writefile"}:
                if path:
                    created_paths.add(path)
                continue
            if compact in {"glob", "grep", "ls", "listdir", "find", "search"} or "search" in compact:
                discovered = True
                continue
            if compact in {"read", "readfile", "fileread"} and path and path not in created_paths:
                discovered = True
    return discovered


def _message_tool_results(message: dict[str, Any]) -> list[tuple[str, str, bool]]:
    role = str(message.get("role") or "")
    content = message.get("content")
    converted = _converted_tool_result_record(message)
    if converted:
        return [converted]
    if role == "tool":
        text = _content_to_text(content)
        is_error = bool(message.get("is_error")) or bool(_MISSING_FILE_TOOL_RESULT_RE.search(text))
        return [(str(message.get("tool_call_id") or ""), text, is_error)]
    if role == "user" and isinstance(content, list):
        results: list[tuple[str, str, bool]] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = _content_to_text(block.get("content"))
            is_error = bool(block.get("is_error")) or bool(_MISSING_FILE_TOOL_RESULT_RE.search(text))
            results.append((str(block.get("tool_use_id") or ""), text, is_error))
        return results
    return []


def _converted_tool_result_record(message: dict[str, Any]) -> tuple[str, str, bool] | None:
    if str(message.get("role") or "") != "user":
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    match = _CONVERTED_TOOL_RESULT_RE.match(content)
    if not match:
        return None
    body = match.group("body") or ""
    body = re.split(
        r"\n\n(?:Use this tool result to continue the task\.|The tool call failed\.)",
        body,
        maxsplit=1,
    )[0].strip()
    is_error = (match.group("is_error") or "").strip().lower() == "true"
    is_error = is_error or bool(_MISSING_FILE_TOOL_RESULT_RE.search(body))
    return (match.group("id").strip(), body, is_error)


def _created_file_path_from_tool_result(text: str) -> str:
    match = _CREATED_FILE_TOOL_RESULT_RE.search(text or "")
    if not match:
        return ""
    return match.group("path").strip()


def _recent_tool_call_summaries(messages: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    return [
        (call.name, _tool_loop_category(call), _tool_loop_summary(call))
        for call in _recent_tool_call_drafts(messages)
    ]


def _recent_tool_call_drafts(messages: list[dict[str, Any]]) -> list[ToolCallDraft]:
    calls: list[ToolCallDraft] = []
    seen_call_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or str(message.get("role") or "") != "assistant":
            continue
        for call in _message_tool_call_drafts(message):
            calls.append(call)
            if call.id:
                seen_call_ids.add(call.id)
    for message in messages:
        if not isinstance(message, dict):
            continue
        for tool_call_id, result_text, is_error in _message_tool_results(message):
            skill_name = _skill_name_from_tool_result(result_text)
            if not is_error and skill_name:
                calls.append(
                    ToolCallDraft(
                        id=tool_call_id or f"inferred_skill_{len(calls) + 1}",
                        name="Skill",
                        input={"skill": skill_name},
                    )
                )
                continue
            if is_error or (tool_call_id and tool_call_id in seen_call_ids):
                continue
            created_path = _created_file_path_from_tool_result(result_text)
            if created_path:
                calls.append(
                    ToolCallDraft(
                        id=tool_call_id or f"inferred_write_{len(calls) + 1}",
                        name="Write",
                        input={"file_path": created_path},
                    )
                )
    for message in messages:
        if not isinstance(message, dict) or str(message.get("role") or "") != "user":
            continue
        if _message_is_tool_result(message):
            continue
        skill_name = _skill_name_from_injection_text(_content_to_text(message.get("content")))
        if skill_name:
            calls.append(
                ToolCallDraft(
                    id=f"inferred_skill_text_{len(calls) + 1}",
                    name="Skill",
                    input={"skill": skill_name},
                )
            )
    return calls


def _recent_successful_tool_result_records(messages: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    calls_by_id: dict[str, ToolCallDraft] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") == "assistant":
            for call in _message_tool_call_drafts(message):
                if call.id:
                    calls_by_id[call.id] = call
            continue
        for tool_call_id, result_text, is_error in _message_tool_results(message):
            if is_error:
                continue
            call = calls_by_id.get(tool_call_id)
            if call is None:
                continue
            compact_name = _compact_tool_name(call.name)
            result_is_successful = bool(_SUCCESSFUL_TOOL_RESULT_RE.search(result_text or ""))
            if not result_is_successful and not (
                compact_name in {"bash", "shell", "terminal"} and not (result_text or "").strip()
            ):
                continue
            records.append((call.name, _tool_loop_summary(call), result_text or ""))
    return records[-8:]


def _skill_name_from_tool_result(text: str) -> str:
    match = _SKILL_LAUNCH_TOOL_RESULT_RE.search(text or "")
    if not match:
        return ""
    return match.group("skill").strip().strip("`'\"")


def _skill_name_from_injection_text(text: str) -> str:
    if not _looks_like_skill_injection_text(text):
        return ""
    base_match = re.search(r"(?im)^Base directory for this skill:\s*(?P<path>.+?)\s*$", text or "")
    if base_match:
        path = base_match.group("path").strip().rstrip("\\/")
        name = re.split(r"[\\/]+", path)[-1].strip()
        if name:
            return name
    colon_heading_match = _SKILL_DOC_COLON_HEADING_RE.search(text or "")
    if colon_heading_match:
        return colon_heading_match.group("skill").strip().lower()
    heading_match = _SKILL_DOC_HEADING_RE.search(text or "")
    if not heading_match:
        return ""
    heading = heading_match.group(0)
    title = re.sub(r"(?im)^\s*#\s*", "", heading).strip()
    title = re.sub(r"(?i)\bskill\b\s*$", "", title).strip()
    slug = re.sub(r"[^A-Za-z0-9_.:@/-]+", "-", title.lower()).strip("-")
    return slug


def _skill_name_from_input(input_value: dict[str, Any]) -> str:
    for key in ("skill", "skill_name", "skillName", "name"):
        value = input_value.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _message_tool_call_drafts(message: dict[str, Any]) -> list[ToolCallDraft]:
    drafts: list[ToolCallDraft] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for item in tool_calls:
            draft = _normalize_item(item)
            if draft:
                drafts.append(draft)
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                draft = _normalize_item(block)
                if draft:
                    drafts.append(draft)
    elif isinstance(content, str):
        drafts.extend(_converted_tool_call_drafts(content))
    return drafts


def _converted_tool_call_drafts(text: str) -> list[ToolCallDraft]:
    raw = text or ""
    if not (_TOOL_SUMMARY_HEADER_RE.search(raw) and _has_dsml_tool_call_syntax(raw)):
        return []
    names = {
        html.unescape(match.group("name")).strip()
        for match in re.finditer(r"<\|?DSML\|?invoke\b[^>]*\bname=(?P<quote>[\"'])(?P<name>.*?)(?P=quote)", raw, re.IGNORECASE | re.DOTALL)
        if match.group("name").strip()
    }
    if not names:
        return []
    context = ToolBridgeContext(
        enabled=True,
        mode="strict",
        tools=[
            ToolSpec(name=name, description="", input_schema={"type": "object"}, read_only=_is_read_only_name(name))
            for name in sorted(names)
        ],
        options=ToolBridgeConfig(tool_profile="all"),
    )
    result = parse_tool_response(raw, context)
    return list(result.tool_calls)


def _tool_loop_category(call: ToolCallDraft) -> str:
    name = call.name.strip()
    lowered = name.lower()
    if lowered in {"bash", "shell", "terminal"}:
        command = _shell_command_from_input(call.input)
        shell_category = _shell_command_loop_category(command)
        if shell_category:
            return f"{lowered}:{shell_category}"
    return f"{lowered}:{_stable_tool_input(call.input)}"


def _tool_loop_summary(call: ToolCallDraft) -> str:
    if call.name.strip().lower() in {"bash", "shell", "terminal"}:
        command = _shell_command_from_input(call.input)
        if command:
            return f"{call.name} command: {_shorten(command, 180)}"
    return f"{call.name} input: {_shorten(_stable_tool_input(call.input), 180)}"


def _shell_command_from_input(input_value: dict[str, Any]) -> str:
    for key in ("command", "cmd"):
        value = input_value.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _new_tool_call_id() -> str:
    return "call_" + uuid.uuid4().hex


def _shell_command_loop_category(command: str) -> str:
    lowered = re.sub(r"\s+", " ", (command or "").strip().lower())
    if not lowered:
        return ""
    git_markers = ("git ", "git.exe ")
    git_housekeeping = (" fetch", " reset", " clean", " status", " diff", " rev-parse", " log", " show", " branch", " pull")
    if any(marker in f" {lowered}" for marker in git_markers) and any(marker in f" {lowered}" for marker in git_housekeeping):
        return "git-housekeeping"
    first = re.split(r"\s+|&&|\|\||;", lowered, maxsplit=1)[0]
    return first or lowered


def _shell_no_progress_housekeeping_category(command: str) -> str:
    lowered = re.sub(r"\s+", " ", (command or "").strip().lower())
    if not lowered:
        return ""
    padded = f" {lowered} "
    if any(marker in padded for marker in (" git ", " git.exe ")) and any(
        marker in padded for marker in _SHELL_READONLY_GIT_HOUSEKEEPING_MARKERS
    ):
        return "git-readonly-housekeeping"
    if _SHELL_FILE_DISCOVERY_HOUSEKEEPING_RE.search(lowered):
        return "file-discovery-housekeeping"
    return ""


def _shell_command_from_loop_summary(summary: str) -> str:
    match = re.match(r"^[^:\r\n]+ command:\s*(?P<command>.*)$", summary or "")
    return match.group("command").strip() if match else ""


def _stable_tool_input(input_value: dict[str, Any]) -> str:
    try:
        return json.dumps(input_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(input_value)


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
    phases: list[BridgePhase] = []

    def record_phase(name: str, status: str, detail: str = "") -> None:
        phases.append(BridgePhase(name=name, status=status, detail=detail))

    def finish(
        content: str,
        tool_calls: list[ToolCallDraft],
        error: BridgeError | None = None,
        warning: str | None = None,
        raw_content: str | None = None,
    ) -> BridgeResult:
        if error is None and not tool_calls and context.tool_choice_policy.is_required():
            error = BridgeError(
                "tool_choice_violation",
                "tool_choice requires at least one valid tool call.",
                repairable=False,
            )
        phase_names = {phase.name for phase in phases}
        if error is not None and "parse/normalize" not in phase_names:
            record_phase("parse/normalize", "error", error.kind)
            phase_names.add("parse/normalize")
        elif tool_calls and "parse/normalize" not in phase_names:
            record_phase("parse/normalize", "ok", f"call_count={len(tool_calls)}")
            phase_names.add("parse/normalize")
        if error is not None and "validate" not in phase_names:
            record_phase("validate", "error", error.kind)
        elif tool_calls and "validate" not in phase_names:
            record_phase("validate", "ok", f"call_count={len(tool_calls)}")
        return BridgeResult(
            content=content,
            tool_calls=tool_calls,
            error=error,
            warning=warning,
            raw_content=raw,
            phases=list(phases),
        )

    candidates: list[Any] = []
    marker_seen = bool(
        _FENCED_TOOL_RE.search(raw)
        or _XML_TOOL_RE.search(raw)
        or _has_dsml_tool_call_syntax(raw)
        or _LEGACY_FUNCTION_CALLS_RE.search(raw)
        or _TOOL_CODE_RE.search(raw)
    )
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
    used_dsml_tool_calls = False
    if not candidates:
        dsml_candidates = _extract_dsml_tool_call_candidates(raw)
        if dsml_candidates:
            candidates.extend(dsml_candidates)
            used_dsml_tool_calls = True
            malformed_seen = False
    used_xml_function_equals = False
    if not candidates:
        xml_function_candidates = _extract_xml_function_equals_candidates(raw)
        if xml_function_candidates:
            candidates.extend(xml_function_candidates)
            used_xml_function_equals = True
            malformed_seen = False
    stripped = raw.strip()
    used_bare = False
    if not candidates and (stripped.startswith("{") or stripped.startswith("[")):
        item = _loads(stripped)
        if item is None:
            malformed_seen = True
        elif not _looks_like_tool_json_candidate(item):
            record_phase("extract", "ok", "plain_text/no_candidate")
            return finish(content=raw, tool_calls=[])
        else:
            candidates.append(item)
            used_bare = True
    used_embedded_json = False
    if not candidates:
        embedded_candidates = _extract_embedded_tool_json_candidates(raw)
        if embedded_candidates:
            candidates.extend(embedded_candidates)
            used_embedded_json = True
    echoed_tool_history = _looks_like_echoed_tool_history(raw)
    used_summary = False
    if not candidates and not echoed_tool_history:
        summary_candidates = _extract_tool_summary_candidates(raw, context)
        if summary_candidates:
            candidates.extend(summary_candidates)
            used_summary = True
    used_legacy_function_calls = False
    if not candidates:
        legacy_candidates = _extract_legacy_function_call_candidates(raw)
        if legacy_candidates:
            candidates.extend(legacy_candidates)
            used_legacy_function_calls = True
    used_tool_code = False
    if not candidates:
        tool_code_candidates = _extract_tool_code_candidates(raw, context)
        if tool_code_candidates:
            candidates.extend(tool_code_candidates)
            used_tool_code = True
    used_bare_function_call = False
    if not candidates:
        bare_function_candidates = _extract_bare_function_call_candidates(raw, context)
        if bare_function_candidates:
            candidates.extend(bare_function_candidates)
            used_bare_function_call = True
    used_provider_search = False
    if not candidates:
        search_candidates, search_query = _extract_provider_search_candidates(raw, context)
        if search_candidates:
            candidates.extend(search_candidates)
            used_provider_search = True
        elif search_query is not None:
            record_phase("extract", "ok", "plain_text/no_candidate")
            return finish(
                content=_provider_search_fallback(search_query),
                tool_calls=[],
                warning="provider_search_markup_without_search_tool",
            )
    used_unwrapped_shell = False
    if not candidates:
        shell_candidates = _extract_unwrapped_shell_command_candidates(raw, context)
        if shell_candidates:
            candidates.extend(shell_candidates)
            used_unwrapped_shell = True
    if not candidates:
        record_phase("extract", "ok", "plain_text/no_candidate")
        warning = "tool_result_claim_without_tool_call" if _TOOL_RESULT_CLAIM_RE.search(raw) else None
        if _allows_ds2api_style_agent_guard_passthrough(context):
            return finish(content=raw, tool_calls=[], warning=warning, raw_content=raw)
        fenced_shell_commands = _extract_fenced_shell_command_lines(raw)
        if echoed_tool_history:
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "echoed_tool_history_without_tool_json",
                    (
                        "The model echoed previous tool-call history instead of emitting a current "
                        "DSML tool request. Retry with exactly one current DSML <|DSML|tool_calls> block, "
                        "or answer only from actual prior tool results."
                    ),
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        markdown_artifact = _markdown_tool_call_artifact_summary(raw, context)
        if markdown_artifact:
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "tool_call_markdown_without_tool_json",
                    (
                        "The model wrote an allowed tool call inside a normal markdown/code fence "
                        f"without using the required DSML tool protocol ({markdown_artifact}). "
                        "Request the tool with exact DSML, or answer from actual prior tool results without claiming execution."
                    ),
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if _is_allowed_tool_denial(raw, context):
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError("tool_denial_without_call", "模型把下游允许工具误判为不可用", repairable=True),
                warning=warning,
                raw_content=raw,
            )
        if _is_deferred_named_tool_action_without_call(raw, context):
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "deferred_named_tool_action_without_call",
                    "The model said it would use an allowed downstream tool but did not emit the required DSML tool call.",
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if _is_unverified_code_change_completion(raw, context):
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "unverified_code_change_completion",
                    (
                        "The model claimed a local code implementation was complete after only an isolated Write-style "
                        "tool history. Inspect existing files and integrate the change with Read/Grep/Edit/MultiEdit, "
                        "or explain the limitation without claiming completion."
                    ),
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if _is_runaway_plain_text_without_tool_call(raw, context):
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "runaway_plain_text_without_tool_call",
                    (
                        "The model produced a very long plain-text response in a strict tool-bridge turn "
                        "without requesting any allowed tool. It must either request one allowed tool with "
                        "DSML or give a concise final answer from actual prior tool results."
                    ),
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if _is_unexecuted_verification_command_after_write(raw, context):
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "unexecuted_verification_command_after_write",
                    (
                        "The model wrote or edited local files, then provided shell verification commands as plain text "
                        "instead of requesting an allowed downstream tool or clearly stating that verification was not run."
                    ),
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if _is_prose_shell_command_intent_without_call(raw, context):
            shell_error = _hidden_local_agent_shell_tool_error("Bash", context) or BridgeError(
                    "shell_command_without_tool_json",
                    (
                        "The model described running a shell command instead of using an allowed downstream tool. "
                        "Request an available tool with DSML, or ask the user to confirm shell execution "
                        "if shell is required."
                    ),
                repairable=True,
            )
            return finish(
                content=raw,
                tool_calls=[],
                error=shell_error,
                warning=warning,
                raw_content=raw,
            )
        if fenced_shell_commands and (context.has_tool_loop or _looks_like_local_agent_task(context.task_text)):
            shell_error = _hidden_local_agent_shell_tool_error("Bash", context) or BridgeError(
                    "shell_command_without_tool_json",
                    (
                        "The model wrote shell commands in a fenced code block instead of using an allowed "
                        "downstream tool. Request an available tool with DSML, or ask the user "
                        "to confirm shell execution if shell is required."
                    ),
                repairable=True,
            )
            return finish(
                content=raw,
                tool_calls=[],
                error=shell_error,
                warning=warning,
                raw_content=raw,
            )
        if _is_incomplete_fix_stub_without_tool_call(raw, context):
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "incomplete_fix_stub_without_tool_call",
                    "The model produced an incomplete local-code fix stub instead of requesting an allowed downstream tool or giving a complete review.",
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if _is_deferred_local_file_inspection_without_call(raw, context):
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "deferred_tool_action_without_call",
                    "The model said it needed to inspect, review, read, check, list, or search local files but did not request an allowed downstream tool.",
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if _is_unproven_final_answer_without_tool_call(raw, context):
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "unproven_final_answer_without_tool_call",
                    (
                        "The model returned plain text in an active local-agent tool loop, but the text still "
                        "describes a next action or incomplete step. Request an allowed downstream tool with "
                        "DSML, or provide a complete final answer based only on actual prior tool results."
                    ),
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if context.allowed_names and _has_code_change_tool(context) and _DEFERRED_CODE_CHANGE_ACTION_RE.search(raw):
            if _allows_plain_review_or_plan_text(context):
                return finish(
                    content=raw,
                    tool_calls=[],
                    warning=warning,
                    raw_content=raw,
                )
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "deferred_code_change_without_call",
                    "The model promised to implement, modify, write, patch, or update code but did not request an allowed downstream tool. Use an allowed tool call instead of describing the code change.",
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if context.allowed_names and _DEFERRED_TOOL_ACTION_RE.search(raw):
            return finish(
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
        if _is_premature_clarification_without_tool_call(raw, context):
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError(
                    "premature_clarification_without_tool_call",
                    "The model asked the user to provide repository details before using available local tools. Inspect the local path/repo first with Bash/Read/Glob.",
                    repairable=True,
                ),
                warning=warning,
                raw_content=raw,
            )
        if marker_seen or malformed_seen:
            if _allows_ds2api_style_rejected_markup_passthrough(context, raw):
                return finish(content=raw, tool_calls=[], warning=warning, raw_content=raw)
            return finish(
                content=raw,
                tool_calls=[],
                error=BridgeError("malformed_json", "工具调用 JSON 无效", repairable=True),
                warning=warning,
                raw_content=raw,
            )
        return finish(content=raw, tool_calls=[], warning=warning, raw_content=raw)

    record_phase("extract", "ok", f"candidate_count={len(candidates)}")
    normalized, error = _normalize_candidates(candidates, context)
    if error is not None:
        return finish(content=raw, tool_calls=[], error=error, raw_content=raw)
    if (
        used_bare
        or used_embedded_json
        or marker_seen
        or used_dsml_tool_calls
        or used_xml_function_equals
        or used_summary
        or used_legacy_function_calls
        or used_tool_code
        or used_bare_function_call
        or used_provider_search
        or used_unwrapped_shell
    ):
        return finish(content="", tool_calls=normalized, raw_content=raw)
    clean = _FENCED_TOOL_RE.sub("", raw)
    clean = _XML_TOOL_RE.sub("", clean)
    clean = _LEGACY_FUNCTION_CALLS_RE.sub("", clean)
    clean = _TOOL_CODE_RE.sub("", clean)
    return finish(content=clean.strip() if not normalized else "", tool_calls=normalized, raw_content=raw)


def to_openai_tool_calls(
    tool_calls: list[dict[str, Any]] | list[ToolCallDraft],
    context: ToolBridgeContext | None = None,
) -> list[dict[str, Any]]:
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
        args = _normalize_tool_input_for_schema(name, args, context)
        out.append(
            {
                "id": call_id or _new_tool_call_id(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    return out


def _normalize_tool_input_for_schema(name: str, args: Any, context: ToolBridgeContext | None) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    if context is None:
        return args
    schema = _schema_for_tool_name(name, context)
    if schema is None:
        return args
    normalized, changed = _normalize_value_with_schema(args, schema)
    return normalized if changed and isinstance(normalized, dict) else args


def _schema_for_tool_name(name: str, context: ToolBridgeContext) -> dict[str, Any] | None:
    wanted = (name or "").strip().lower()
    if not wanted:
        return None
    for tool in context.tools:
        if tool.name.strip().lower() == wanted and isinstance(tool.input_schema, dict):
            return tool.input_schema
    return None


def _normalize_value_with_schema(value: Any, schema: Any) -> tuple[Any, bool]:
    if value is None or not isinstance(schema, dict) or not schema:
        return value, False
    if _schema_should_coerce_to_string(schema):
        return _stringify_schema_value(value)
    if _looks_like_object_schema(schema):
        if not isinstance(value, dict) or not value:
            return value, False
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        additional = schema.get("additionalProperties")
        changed = False
        out: dict[str, Any] = {}
        for key, current in value.items():
            next_value = current
            field_changed = False
            prop_schema = properties.get(key)
            if prop_schema is not None:
                next_value, field_changed = _normalize_value_with_schema(current, prop_schema)
            elif additional is not None:
                next_value, field_changed = _normalize_value_with_schema(current, additional)
            out[key] = next_value
            changed = changed or field_changed
        return (out, True) if changed else (value, False)
    if _looks_like_array_schema(schema):
        if not isinstance(value, list) or not value:
            return value, False
        items_schema = schema.get("items")
        if items_schema is None:
            return value, False
        changed = False
        out: list[Any] = []
        if isinstance(items_schema, list):
            for index, item in enumerate(value):
                if index >= len(items_schema):
                    out.append(item)
                    continue
                next_value, item_changed = _normalize_value_with_schema(item, items_schema[index])
                out.append(next_value)
                changed = changed or item_changed
        else:
            for item in value:
                next_value, item_changed = _normalize_value_with_schema(item, items_schema)
                out.append(next_value)
                changed = changed or item_changed
        return (out, True) if changed else (value, False)
    return value, False


def _schema_should_coerce_to_string(schema: dict[str, Any]) -> bool:
    if isinstance(schema.get("const"), str):
        return True
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(item, str) for item in enum):
        return True
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type.strip().lower() == "string"
    if isinstance(schema_type, list):
        seen_string = False
        for item in schema_type:
            if not isinstance(item, str):
                return False
            lowered = item.strip().lower()
            if lowered == "string":
                seen_string = True
            elif lowered != "null":
                return False
        return seen_string
    return False


def _looks_like_object_schema(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if isinstance(schema_type, str) and schema_type.strip().lower() == "object":
        return True
    return isinstance(schema.get("properties"), dict) or "additionalProperties" in schema


def _looks_like_array_schema(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if isinstance(schema_type, str) and schema_type.strip().lower() == "array":
        return True
    return "items" in schema


def _stringify_schema_value(value: Any) -> tuple[Any, bool]:
    if value is None or isinstance(value, str):
        return value, False
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True), True
    except (TypeError, ValueError):
        return value, False


def _extract_tool_summary_candidates(text: str, context: ToolBridgeContext) -> list[dict[str, Any]]:
    header_match = _TOOL_SUMMARY_HEADER_RE.search(text or "")
    if not header_match:
        return []
    allowed_lower = {name.lower() for name in context.allowed_names}
    candidates: list[dict[str, Any]] = []
    summary_text = (text or "")[header_match.end() :]
    for match in _TOOL_SUMMARY_CALL_START_RE.finditer(summary_text):
        name = match.group("name").strip()
        if name.lower() not in allowed_lower:
            continue
        input_start = _skip_ws(summary_text, match.end())
        if input_start >= len(summary_text) or summary_text[input_start] != "{":
            continue
        input_end = _balanced_json_object_end(summary_text, input_start)
        if input_end is None:
            continue
        close_paren = _skip_ws(summary_text, input_end)
        if close_paren >= len(summary_text) or summary_text[close_paren] != ")":
            continue
        args = _loads_summary_input(summary_text[input_start:input_end])
        candidates.append({"id": f"toolu_summary_{len(candidates) + 1}", "name": name, "input": args})
    return candidates


def _looks_like_echoed_tool_history(text: str) -> bool:
    return bool(_TOOL_HISTORY_ECHO_RE.search(text or ""))


def looks_like_tool_protocol_output(text: str) -> bool:
    raw = text or ""
    return bool(
        _has_dsml_tool_call_syntax(raw)
        or _LEAKED_TOOL_CALL_ARRAY_RE.search(raw)
        or _LEAKED_TOOL_RESULT_BLOB_RE.search(raw)
    )


def sanitize_leaked_tool_protocol_output(text: str) -> str:
    raw = text or ""
    if not raw:
        return raw
    out = _EMPTY_JSON_FENCE_RE.sub("", raw)
    out = _LEAKED_TOOL_CALL_ARRAY_RE.sub("", out)
    out = _LEAKED_TOOL_RESULT_BLOB_RE.sub("", out)
    out = _strip_dangling_think_suffix(out)
    out = _LEAKED_THINK_TAG_RE.sub("", out)
    out = _LEAKED_META_MARKER_RE.sub("", out)
    out = _strip_leaked_tool_call_wrapper_blocks(out)
    out = _sanitize_leaked_agent_xml_blocks(out)
    return out


def _strip_dangling_think_suffix(text: str) -> str:
    matches = list(_LEAKED_THINK_TAG_RE.finditer(text or ""))
    if not matches:
        return text
    depth = 0
    last_open = -1
    for match in matches:
        tag = match.group(0).lower()
        compact = re.sub(r"\s+", "", tag)
        if compact.startswith("</"):
            if depth > 0:
                depth -= 1
                if depth == 0:
                    last_open = -1
            continue
        if depth == 0:
            last_open = match.start()
        depth += 1
    if depth == 0 or last_open < 0:
        return text
    prefix = text[:last_open]
    return prefix if prefix.strip() else ""


def _strip_leaked_tool_call_wrapper_blocks(text: str) -> str:
    raw = text or ""
    ranges = _find_dsml_tool_call_block_ranges(raw)
    if not ranges:
        return raw
    out: list[str] = []
    pos = 0
    for start, end in ranges:
        if start < pos:
            continue
        out.append(raw[pos:start])
        pos = end
    out.append(raw[pos:])
    return "".join(out)


def _sanitize_leaked_agent_xml_blocks(text: str) -> str:
    out = text or ""

    def replace_complete(match: re.Match[str]) -> str:
        return _LEAKED_AGENT_RESULT_TAG_RE.sub("", match.group("body") or "")

    out = _LEAKED_AGENT_XML_BLOCK_RE.sub(replace_complete, out)
    if _LEAKED_AGENT_WRAPPER_TAG_RE.search(out):
        out = _LEAKED_AGENT_WRAPPER_PLUS_RESULT_OPEN_RE.sub(
            lambda match: _LEAKED_AGENT_RESULT_TAG_RE.sub("", match.group(0)),
            out,
        )
        out = _LEAKED_AGENT_RESULT_PLUS_WRAPPER_CLOSE_RE.sub(
            lambda match: _LEAKED_AGENT_RESULT_TAG_RE.sub("", match.group(0)),
            out,
        )
        out = _LEAKED_AGENT_WRAPPER_TAG_RE.sub("", out)
    return out


def _has_dsml_tool_call_syntax(text: str) -> bool:
    return bool(_extract_dsml_tool_call_blocks(text))


def _extract_dsml_tool_call_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for block in _extract_dsml_tool_call_blocks(text):
        normalized_block = _normalize_dsml_tool_tags(block)
        variants = [normalized_block]
        recovered_block = _sanitize_loose_cdata(normalized_block)
        if recovered_block != normalized_block:
            variants.append(recovered_block)
        for candidate_block in variants:
            block_candidates: list[dict[str, Any]] = []
            for _start, _end, attrs, body in _find_xml_element_blocks(candidate_block, "invoke"):
                invoke_attrs = _legacy_attrs(attrs)
                name = str(invoke_attrs.get("name") or invoke_attrs.get("tool") or invoke_attrs.get("function") or "").strip()
                if not name:
                    continue
                input_value = _xml_invoke_input(body)
                if input_value:
                    if not _tool_call_input_has_meaningful_value(input_value):
                        continue
                elif _xml_invoke_body_is_invalid_without_params(body):
                    continue
                block_candidates.append(
                    {
                        "name": name,
                        "input": input_value,
                    }
                )
            if block_candidates:
                candidates.extend(block_candidates)
                break
    return candidates


def _extract_dsml_tool_call_blocks(text: str) -> list[str]:
    raw = text or ""
    ranges = _find_dsml_tool_call_block_ranges(raw)
    blocks = [raw[start:end] for start, end in ranges]
    blocks.extend(_extract_missing_opening_wrapper_dsml_blocks(raw, ranges))
    return blocks


def _find_dsml_tool_call_block_ranges(text: str) -> list[tuple[int, int]]:
    raw = text or ""
    fenced_ranges = _fenced_code_ranges(raw)
    ranges: list[tuple[int, int]] = []
    pos = 0
    while True:
        start_match = _XML_TOOL_CALLS_OPEN_RE.search(raw, pos)
        if not start_match:
            break
        if _index_in_ranges(start_match.start(), fenced_ranges) or _index_inside_cdata(raw, start_match.start()):
            pos = start_match.end()
            continue
        close_match = _find_regex_outside_cdata(_XML_TOOL_CALLS_CLOSE_RE, raw, start_match.end())
        if close_match is None and "<![CDATA[" in raw[start_match.end() :]:
            close_match = _XML_TOOL_CALLS_CLOSE_RE.search(raw, start_match.end())
        if close_match is None:
            pos = start_match.end()
            continue
        ranges.append((start_match.start(), close_match.end()))
        pos = close_match.end()
    return ranges


def _extract_missing_opening_wrapper_dsml_blocks(text: str, existing_ranges: list[tuple[int, int]]) -> list[str]:
    raw = text or ""
    fenced_ranges = _fenced_code_ranges(raw)
    blocks: list[str] = []
    pos = 0
    while True:
        invoke_match = _XML_INVOKE_OPEN_RE.search(raw, pos)
        if invoke_match is None:
            break
        if (
            _index_in_ranges(invoke_match.start(), fenced_ranges)
            or _index_inside_cdata(raw, invoke_match.start())
            or _index_in_ranges(invoke_match.start(), existing_ranges)
        ):
            pos = invoke_match.end()
            continue
        close_match = _find_regex_outside_cdata(_XML_TOOL_CALLS_CLOSE_RE, raw, invoke_match.end())
        if close_match is None:
            pos = invoke_match.end()
            continue
        if _index_in_ranges(close_match.start(), fenced_ranges):
            pos = close_match.end()
            continue
        open_before = _find_last_tool_calls_open_before(raw, invoke_match.start())
        if open_before is not None and open_before >= pos:
            pos = close_match.end()
            continue
        blocks.append("<tool_calls>" + raw[invoke_match.start() : close_match.end()])
        pos = close_match.end()
    return blocks


def _find_last_tool_calls_open_before(text: str, end: int) -> int | None:
    last: int | None = None
    for match in _XML_TOOL_CALLS_OPEN_RE.finditer((text or "")[: max(end, 0)]):
        if not _index_inside_cdata(text, match.start()):
            last = match.start()
    return last


def _fenced_code_ranges(text: str) -> list[tuple[int, int]]:
    raw = text or ""
    ranges: list[tuple[int, int]] = []
    in_fence = False
    fence_marker = ""
    fence_start = 0
    in_cdata = False
    offset = 0
    for line in raw.splitlines(keepends=True):
        line_start = offset
        line_end = offset + len(line)
        offset = line_end
        if in_cdata or _cdata_starts_before_fence(line):
            in_cdata = _update_cdata_state(in_cdata, line)
            continue
        trimmed = line.lstrip(" \t")
        if not in_fence:
            marker = _fence_open_marker(trimmed)
            if marker:
                in_fence = True
                fence_marker = marker
                fence_start = line_start
            continue
        if _is_fence_close_marker(trimmed, fence_marker):
            ranges.append((fence_start, line_end))
            in_fence = False
            fence_marker = ""
    if in_fence:
        ranges.append((fence_start, len(raw)))
    return ranges


def _without_fenced_code_blocks(text: str) -> str:
    raw = text or ""
    ranges = _fenced_code_ranges(raw)
    if not ranges:
        return raw
    out: list[str] = []
    pos = 0
    for start, end in ranges:
        out.append(raw[pos:start])
        pos = end
    out.append(raw[pos:])
    return "".join(out)


def _cdata_starts_before_fence(line: str) -> bool:
    cdata_idx = line.lower().find("<![cdata[")
    if cdata_idx < 0:
        return False
    fence_idx = _first_fence_marker_index(line)
    return fence_idx < 0 or cdata_idx < fence_idx


def _first_fence_marker_index(line: str) -> int:
    backtick = line.find("```")
    tilde = line.find("~~~")
    if backtick < 0:
        return tilde
    if tilde < 0:
        return backtick
    return min(backtick, tilde)


def _update_cdata_state(in_cdata: bool, line: str) -> bool:
    lower = line.lower()
    pos = 0
    state = in_cdata
    while pos < len(lower):
        if state:
            end = lower.find("]]>", pos)
            if end < 0:
                return True
            pos = end + len("]]>")
            state = False
            continue
        start = lower.find("<![cdata[", pos)
        if start < 0:
            return False
        pos = start + len("<![cdata[")
        state = True
    return state


def _fence_open_marker(line: str) -> str:
    if len(line) < 3 or line[0] not in {"`", "~"}:
        return ""
    char = line[0]
    count = 0
    while count < len(line) and line[count] == char:
        count += 1
    if count < 3:
        return ""
    return char * count


def _is_fence_close_marker(line: str, marker: str) -> bool:
    if not marker or not line or line[0] != marker[0]:
        return False
    count = 0
    while count < len(line) and line[count] == marker[0]:
        count += 1
    if count < len(marker):
        return False
    return line[count:].strip() == ""


def _cdata_ranges(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _CDATA_RE.finditer(text or "")]


def _index_in_ranges(index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= index < end for start, end in ranges)


def _index_inside_cdata(text: str, index: int) -> bool:
    raw = text or ""
    start = raw.rfind("<![CDATA[", 0, index + 1)
    if start < 0:
        return False
    end = raw.rfind("]]>", 0, index + 1)
    return end < start


def _find_regex_outside_cdata(pattern: re.Pattern[str], text: str, start: int = 0) -> re.Match[str] | None:
    pos = start
    raw = text or ""
    while True:
        match = pattern.search(raw, pos)
        if match is None:
            return None
        if not _index_inside_cdata(raw, match.start()):
            return match
        pos = match.end()


def _normalize_dsml_tool_tags(text: str) -> str:
    cdata_ranges = _cdata_ranges(text)

    def replace(match: re.Match[str]) -> str:
        if _index_in_ranges(match.start(), cdata_ranges):
            return match.group(0)
        closing = bool(match.group("closing"))
        body = (match.group("body") or "").strip()
        tag_info = _dsml_local_tag_from_body(body)
        if tag_info is None:
            return match.group(0)
        tag, tag_end = tag_info
        if closing:
            return f"</{tag}>"
        attrs = body[tag_end:].strip()
        attrs = re.sub(r"^[|｜\s]+", "", attrs)
        attrs = re.sub(r"[|｜\s]+$", "", attrs)
        attrs = attrs.strip("|｜ \t\r\n")
        self_close = attrs.endswith("/")
        if self_close:
            attrs = attrs[:-1].rstrip()
        suffix = " /" if self_close else ""
        return f"<{tag}{(' ' + attrs) if attrs else ''}{suffix}>"

    return _XML_ANY_TAG_RE.sub(replace, text or "")


def _dsml_local_tag_from_body(body: str) -> tuple[str, int] | None:
    lowered = (body or "").replace("｜", "|").lower()
    lowered = lowered.replace("｜", "|")
    for tag in ("tool_calls", "invoke", "parameter"):
        for match in re.finditer(re.escape(tag), lowered):
            tail = lowered[match.end() : match.end() + 1]
            if tail and (tail.isalnum() or tail in "_:-"):
                continue
            prefix = lowered[: match.start()]
            noise = prefix.replace("dsml", "").replace("|", "").strip()
            noise = noise.replace("｜", "")
            if noise:
                continue
            return tag, match.end()
    return None


def _find_xml_element_blocks(text: str, tag: str) -> list[tuple[int, int, str, str]]:
    raw = text or ""
    start_re = re.compile(rf"<{re.escape(tag)}\b(?P<attrs>[^>]*)>", re.IGNORECASE | re.DOTALL)
    close_re = re.compile(rf"</{re.escape(tag)}\s*>", re.IGNORECASE | re.DOTALL)
    blocks: list[tuple[int, int, str, str]] = []
    pos = 0
    while True:
        start_match = _find_regex_outside_cdata(start_re, raw, pos)
        if start_match is None:
            break
        close_match = _find_regex_outside_cdata(close_re, raw, start_match.end())
        if close_match is None:
            break
        blocks.append(
            (
                start_match.start(),
                close_match.end(),
                start_match.group("attrs") or "",
                raw[start_match.end() : close_match.start()],
            )
        )
        pos = close_match.end()
    return blocks


def _xml_invoke_input(body: str) -> dict[str, Any]:
    raw = (body or "").strip()
    params: dict[str, Any] = {}
    for _start, _end, attrs, param_body in _find_xml_element_blocks(raw, "parameter"):
        param_attrs = _legacy_attrs(attrs)
        key = str(param_attrs.get("name") or param_attrs.get("key") or "").strip()
        if not key:
            continue
        value = _parse_xml_parameter_value(key, param_body)
        if key in params:
            previous = params[key]
            if isinstance(previous, list):
                previous.append(value)
            else:
                params[key] = [previous, value]
        else:
            params[key] = value
    if params:
        return params
    if raw.startswith("{"):
        parsed = _loads(raw)
        if isinstance(parsed, dict):
            nested = parsed.get("input")
            return nested if isinstance(nested, dict) else parsed
    return {}


def _parse_xml_parameter_value(name: str, body: str) -> Any:
    raw = (body or "").strip()
    cdata = _standalone_cdata_text(raw)
    if cdata is not None:
        if _preserves_cdata_string_parameter(name):
            return cdata
        if _cdata_fragment_looks_explicitly_structured(cdata):
            structured = _parse_structured_xml_fragment(_normalize_cdata_for_structured_parse(cdata))
            if structured is not None:
                return structured
        loose_array = _parse_loose_json_array_value(cdata, name)
        if loose_array is not None:
            return loose_array
        return _parse_scalar_xml_text(cdata)
    structured = _parse_structured_xml_fragment(raw)
    if structured is not None:
        return structured
    text = _strip_xml_tags(raw)
    loose_array = _parse_loose_json_array_value(text, name)
    if loose_array is not None:
        return loose_array
    return _parse_scalar_xml_text(text)


def _sanitize_loose_cdata(text: str) -> str:
    raw = text or ""
    lower = raw.lower()
    open_marker = "<![cdata["
    close_marker = "]]>"
    out: list[str] = []
    changed = False
    pos = 0
    while pos < len(raw):
        start = lower.find(open_marker, pos)
        if start < 0:
            out.append(raw[pos:])
            break
        content_start = start + len(open_marker)
        out.append(raw[pos:start])
        end = lower.find(close_marker, content_start)
        if end >= 0:
            close_end = end + len(close_marker)
            out.append(raw[start:close_end])
            pos = close_end
            continue
        changed = True
        out.append(raw[content_start:])
        pos = len(raw)
    return "".join(out) if changed else raw


def _xml_invoke_body_is_invalid_without_params(body: str) -> bool:
    raw = (body or "").strip()
    if not raw:
        return False
    if _find_xml_element_blocks(raw, "parameter"):
        return False
    if raw.startswith("{"):
        parsed = _loads(raw)
        if not isinstance(parsed, dict):
            return True
        nested = parsed.get("input")
        return nested is not None and not isinstance(nested, dict)
    return True


def _tool_call_input_has_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_tool_call_input_has_meaningful_value(child) for child in value.values())
    if isinstance(value, list):
        return any(_tool_call_input_has_meaningful_value(child) for child in value)
    return True


def _standalone_cdata_text(raw: str) -> str | None:
    text = raw or ""
    stripped = text.strip()
    if not stripped.lower().startswith("<![cdata["):
        return None
    matches = list(_CDATA_RE.finditer(stripped))
    if matches:
        remainder = _CDATA_RE.sub("", stripped).strip()
        if not remainder:
            return "".join(match.group("body") or "" for match in matches)
    if stripped.lower().startswith("<![cdata[") and stripped.endswith("]]>"):
        return stripped[len("<![CDATA[") : -len("]]>")]
    if stripped.lower().startswith("<![cdata[") and "]]>" not in stripped:
        return stripped[len("<![CDATA[") :]
    return None


def _preserves_cdata_string_parameter(name: str) -> bool:
    return (name or "").strip().lower() in {
        "cmd",
        "code",
        "command",
        "content",
        "description",
        "file_content",
        "file_path",
        "header",
        "label",
        "new_string",
        "old_string",
        "path",
        "pattern",
        "prompt",
        "query",
        "question",
        "script",
        "text",
        "url",
    }


def _parse_structured_xml_fragment(raw: str) -> Any | None:
    text = (raw or "").strip()
    if not text or "<" not in text:
        return None
    try:
        root = ET.fromstring(f"<root>{text}</root>")
    except ET.ParseError:
        return None
    children = list(root)
    if not children:
        return None
    if root.text and root.text.strip():
        return None
    if any(child.tail and child.tail.strip() for child in children):
        return None
    if all(_xml_local_name(child.tag).lower() == "item" for child in children):
        return [_parse_etree_node_value(child) for child in children]
    out: dict[str, Any] = {}
    for child in children:
        key = _xml_local_name(child.tag)
        value = _parse_etree_node_value(child)
        if key in out:
            previous = out[key]
            if isinstance(previous, list):
                previous.append(value)
            else:
                out[key] = [previous, value]
        else:
            out[key] = value
    return out


def _normalize_cdata_for_structured_parse(raw: str) -> str:
    return html.unescape(_CDATA_BR_SEPARATOR_RE.sub("\n", (raw or "").strip()))


def _cdata_fragment_looks_explicitly_structured(raw: str) -> bool:
    text = _normalize_cdata_for_structured_parse(raw)
    if not text or "<" not in text or ">" not in text:
        return False
    try:
        root = ET.fromstring(f"<root>{text}</root>")
    except ET.ParseError:
        return False
    children = list(root)
    if len(children) != 1:
        return len(children) > 1
    first = children[0]
    if _xml_local_name(first.tag).lower() == "item":
        return True
    return bool(list(first))


def _xml_local_name(tag: str) -> str:
    value = str(tag or "")
    if "}" in value:
        return value.rsplit("}", 1)[-1]
    return value.split(":", 1)[-1]


def _parse_etree_node_value(node: ET.Element) -> Any:
    children = list(node)
    if not children:
        return _parse_scalar_xml_text(node.text or "")
    out: dict[str, Any] = {}
    text_parts: list[str] = []
    if node.text and node.text.strip():
        text_parts.append(node.text)
    for child in children:
        _append_xml_child_value(out, _xml_local_name(child.tag), _parse_etree_node_value(child))
        if child.tail and child.tail.strip():
            text_parts.append(child.tail)
    if len(out) == 1 and "item" in out:
        item = out["item"]
        return item if isinstance(item, list) else [item]
    text_value = "".join(text_parts)
    if text_value.strip():
        out["_text"] = _parse_scalar_xml_text(text_value)
    return out


def _append_xml_child_value(out: dict[str, Any], key: str, value: Any) -> None:
    if not key:
        return
    if key in out:
        previous = out[key]
        if isinstance(previous, list):
            previous.append(value)
        else:
            out[key] = [previous, value]
    else:
        out[key] = value


def _top_level_xml_children(text: str) -> list[tuple[int, int, str, str, str]]:
    raw = text or ""
    children: list[tuple[int, int, str, str, str]] = []
    pos = 0
    while True:
        start_match = _find_regex_outside_cdata(_XML_CHILD_START_RE, raw, pos)
        if start_match is None:
            break
        tag = start_match.group("tag") or ""
        close_re = re.compile(rf"</{re.escape(tag)}\s*>", re.IGNORECASE | re.DOTALL)
        close_match = _find_regex_outside_cdata(close_re, raw, start_match.end())
        if close_match is None:
            break
        children.append(
            (
                start_match.start(),
                close_match.end(),
                tag,
                start_match.group("attrs") or "",
                raw[start_match.end() : close_match.start()],
            )
        )
        pos = close_match.end()
    return children


def _parse_xml_node_value(tag: str, body: str) -> Any:
    structured = _parse_structured_xml_fragment(body)
    if structured is not None:
        return structured
    return _parse_xml_parameter_value(tag, body)


def _parse_scalar_xml_text(value: str) -> Any:
    text = html.unescape(value or "").strip()
    if not text:
        return ""
    if text.startswith(("{", "[")):
        parsed = _loads(text)
        if parsed is not None:
            return parsed
    if text.lower() in {"true", "false", "null"} or re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", text):
        try:
            return json.loads(text)
        except Exception:
            return text
    return text


def _parse_loose_json_array_value(raw: str, param_name: str) -> list[Any] | None:
    if _preserves_cdata_string_parameter(param_name):
        return None
    text = html.unescape(raw or "").strip()
    if not text:
        return None
    parsed = _parse_loose_array_element_value(text)
    if parsed is not None:
        coerced = _coerce_array_value(parsed, param_name)
        if coerced is not None:
            return coerced
    segments = _split_top_level_json_values(text)
    if not segments:
        return None
    values: list[Any] = []
    for segment in segments:
        parsed_segment = _parse_loose_array_element_value(segment)
        if parsed_segment is None:
            return None
        values.append(parsed_segment)
    return values


def _parse_loose_array_element_value(raw: str) -> Any | None:
    text = html.unescape(raw or "").strip()
    if not text:
        return None
    parsed = _loads(text)
    if parsed is not None:
        return parsed
    repaired = _escape_invalid_json_backslashes(text)
    if repaired != text:
        parsed = _loads(repaired)
        if parsed is not None:
            return parsed
    repaired_loose = _repair_loose_json(text)
    if repaired_loose != text:
        parsed = _loads(repaired_loose)
        if parsed is not None:
            return parsed
    if "<" in text and ">" in text:
        structured = _parse_structured_xml_fragment(text)
        if structured is not None:
            return structured
    return None


_UNQUOTED_JSON_KEY_RE = re.compile(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:')
_MISSING_ARRAY_BRACKETS_RE = re.compile(
    r'(:\s*)(\{(?:[^{}]|\{[^{}]*\})*\}(?:\s*,\s*\{(?:[^{}]|\{[^{}]*\})*\})+)'
)


def _repair_loose_json(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return text
    text = _UNQUOTED_JSON_KEY_RE.sub(r'\1"\2":', text)
    text = _MISSING_ARRAY_BRACKETS_RE.sub(r"\1[\2]", text)
    return text


def _coerce_array_value(value: Any, param_name: str) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if len(value) != 1:
            return None
        if "item" in value:
            item = value["item"]
            nested = _coerce_array_value(item, "")
            return nested if nested is not None else [item]
        if param_name and param_name in value:
            return _coerce_array_value(value[param_name], "")
    return None


def _split_top_level_json_values(raw: str) -> list[str] | None:
    text = (raw or "").strip()
    if not text:
        return None
    values: list[str] = []
    start = 0
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            if depth > 0:
                depth -= 1
        elif char == "," and depth == 0:
            segment = text[start:index].strip()
            if not segment:
                return None
            values.append(segment)
            start = index + 1
    last = text[start:].strip()
    if not last:
        return None
    values.append(last)
    return values if len(values) >= 2 else None


def _extract_legacy_function_call_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for block_match in _LEGACY_FUNCTION_CALLS_RE.finditer(text or ""):
        block = block_match.group("body") or ""
        for invoke in _LEGACY_INVOKE_RE.finditer(block):
            attrs = _legacy_attrs(invoke.group("attrs") or "")
            name = str(attrs.get("name") or attrs.get("tool") or attrs.get("function") or "").strip()
            if not name:
                continue
            input_value = _legacy_invoke_input(invoke.group("body") or "")
            candidates.append(
                {
                    "id": f"toolu_legacy_{len(candidates) + 1}",
                    "name": name,
                    "input": input_value,
                }
            )
    return candidates


def _extract_xml_function_equals_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    raw = text or ""
    tool_blocks = [block_match.group(1) or "" for block_match in _XML_TOOL_RE.finditer(raw)]
    blocks = tool_blocks or [raw]
    for block in blocks:
        for function_match in _XML_FUNCTION_EQUALS_RE.finditer(block):
            name = function_match.group("name").strip()
            input_value = _legacy_invoke_input(function_match.group("body") or "")
            candidates.append(
                {
                    "id": f"toolu_xml_function_{len(candidates) + 1}",
                    "name": name,
                    "input": input_value,
                }
            )
    return candidates


def _extract_tool_code_candidates(text: str, context: ToolBridgeContext) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for block_match in _TOOL_CODE_RE.finditer(text or ""):
        block = block_match.group("body") or ""
        candidates.extend(
            _extract_function_call_line_candidates(
                block,
                context,
                id_prefix="toolu_code",
                start_index=len(candidates) + 1,
            )
        )
    return candidates


def _extract_bare_function_call_candidates(text: str, context: ToolBridgeContext) -> list[dict[str, Any]]:
    raw = text or ""
    if not raw.strip() or "```" in raw:
        return []
    return _extract_function_call_line_candidates(raw, context, id_prefix="toolu_bare")


def _markdown_tool_call_artifact_summary(text: str, context: ToolBridgeContext) -> str:
    if not context.allowed_names:
        return ""
    summaries: list[str] = []
    for match in _FENCED_CODE_BLOCK_RE.finditer(text or ""):
        lang = (match.group("lang") or "").strip().lower()
        if lang == "tool_json":
            continue
        body = (match.group("body") or "").strip()
        if not body:
            continue
        candidates = _extract_function_call_line_candidates(
            body,
            context,
            id_prefix="toolu_markdown_artifact",
            start_index=len(summaries) + 1,
        )
        for candidate in candidates:
            draft = _normalize_item(candidate)
            if draft:
                summaries.append(_tool_loop_summary(draft))
    return "; ".join(summaries[:3])


def _extract_function_call_line_candidates(
    text: str,
    context: ToolBridgeContext,
    *,
    id_prefix: str,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    allowed_lower = {name.lower() for name in context.allowed_names}
    if not allowed_lower:
        return []

    candidates: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        source = line.strip()
        if not source or not _BARE_FUNCTION_CALL_LINE_RE.match(source):
            continue
        parsed = _parse_allowed_function_call_source(source, allowed_lower)
        if parsed is None:
            continue
        name, input_value = parsed
        candidates.append(
            {
                "id": f"{id_prefix}_{start_index + len(candidates)}",
                "name": name,
                "input": input_value,
            }
        )
    return candidates


def _parse_allowed_function_call_source(source: str, allowed_lower: set[str]) -> tuple[str, dict[str, Any]] | None:
    try:
        expr = ast.parse(source, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        return None
    name = expr.func.id.strip()
    if name.lower() not in allowed_lower:
        return None
    if not expr.args and not expr.keywords:
        return name, {}
    if len(expr.args) == 1 and not expr.keywords:
        try:
            value = ast.literal_eval(expr.args[0])
        except (ValueError, TypeError, SyntaxError):
            return None
        if isinstance(value, dict):
            return name, value
        return None
    if expr.args:
        return None

    input_value: dict[str, Any] = {}
    for keyword in expr.keywords:
        if keyword.arg is None:
            return None
        try:
            value = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError):
            return None
        input_value[str(keyword.arg)] = value
    return name, input_value


def _legacy_invoke_input(body: str) -> dict[str, Any]:
    raw = (body or "").strip()
    params = _xml_invoke_input(_normalize_dsml_tool_tags(raw))
    if params:
        return params
    if raw.startswith("{"):
        parsed = _loads(raw)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _legacy_attrs(raw_attrs: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _LEGACY_ATTR_RE.finditer(raw_attrs or ""):
        attrs[match.group("name").lower()] = html.unescape(match.group("value") or "")
    return attrs


def _strip_xml_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _balanced_json_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


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


def _tool_choice_is_none(tool_choice: Any) -> bool:
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() == "none"
    if isinstance(tool_choice, dict):
        return str(tool_choice.get("type") or "").strip().lower() == "none"
    return False


def _messages_have_tool_loop(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if _message_has_tool_interaction(message):
            return True
    return False


def _messages_for_current_human_task(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    start = _current_human_task_start_index(messages)
    if start <= 0:
        return messages
    return messages[start:]


def _current_human_task_start_index(messages: list[Any]) -> int:
    fallback = -1
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or str(message.get("role") or "") != "user":
            continue
        if _message_is_tool_result(message):
            continue
        if _message_is_gateway_control_instruction(message):
            continue
        if _message_is_skill_payload_after_tool_result(messages, index):
            continue
        text = _human_task_text_from_message(message)
        if not text.strip():
            continue
        if fallback < 0:
            fallback = index
        if not _looks_like_continuation_task(text):
            return index
    return fallback


def _message_is_skill_payload_after_tool_result(messages: list[Any], index: int) -> bool:
    message = messages[index]
    text = _content_to_text(message.get("content"))
    if not text.strip():
        return False
    launched_skill = ""
    for prior in range(index - 1, -1, -1):
        previous = messages[prior]
        if not isinstance(previous, dict):
            continue
        if str(previous.get("role") or "") == "system":
            continue
        if not _message_is_tool_result(previous):
            return False
        match = _SKILL_LAUNCH_TOOL_RESULT_RE.search(_content_to_text(previous.get("content")))
        if match:
            launched_skill = match.group("skill") or ""
        break
    if not launched_skill:
        return False
    heading = _first_markdown_heading(text)
    if not heading:
        return False
    return _compact_tool_name(launched_skill) in _compact_tool_name(heading)


def _first_markdown_heading(text: str) -> str:
    match = re.search(r"(?m)^\s{0,3}#{1,3}\s+(?P<head>[^\n#]{1,180})\s*$", text or "")
    return match.group("head").strip() if match else ""


def _current_task_starts_after_prior_tool_history(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    start = _current_human_task_start_index(messages)
    if start <= 0:
        return False
    return any(isinstance(message, dict) and _message_has_tool_interaction(message) for message in messages[:start])


def _message_has_tool_interaction(message: dict[str, Any]) -> bool:
    if message.get("role") == "tool":
        return True
    if isinstance(message.get("tool_calls"), list) and message.get("tool_calls"):
        return True
    if _converted_tool_result_record(message):
        return True
    content = message.get("content")
    if isinstance(content, list):
        return any(isinstance(block, dict) and str(block.get("type") or "") in {"tool_use", "tool_result"} for block in content)
    if isinstance(content, dict):
        return str(content.get("type") or "") in {"tool_use", "tool_result"}
    if isinstance(content, str) and str(message.get("role") or "") == "assistant":
        return bool(_TOOL_SUMMARY_HEADER_RE.search(content) and _has_dsml_tool_call_syntax(content))
    return False


def _message_is_gateway_control_instruction(message: dict[str, Any]) -> bool:
    if str(message.get("role") or "") != "user":
        return False
    if _message_is_tool_result(message):
        return False
    text = _content_to_text(message.get("content")).lstrip()
    return text.startswith(
        (
            "Previous tool JSON",
            "Tool loop guard:",
            "Skill progress guard:",
            "Tool result guard:",
            "New user task boundary:",
        )
    )


def _latest_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or str(message.get("role") or "") != "user":
            continue
        return _content_to_text(message.get("content"))
    return ""


def _latest_human_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or str(message.get("role") or "") != "user":
            continue
        if _message_is_tool_result(message):
            continue
        if _message_is_gateway_control_instruction(message):
            continue
        text = _human_task_text_from_message(message)
        if text.strip():
            return text
    return ""


def _effective_human_task_text(messages: Any) -> str:
    texts = _human_user_texts_newest_first(messages)
    if not texts:
        return ""
    latest = texts[0]
    if not _looks_like_continuation_task(latest):
        return latest
    for text in texts[1:]:
        if text.strip() and not _looks_like_continuation_task(text):
            return f"{text.strip()}\n{latest.strip()}".strip()
    return latest


def _human_user_texts_newest_first(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    texts: list[str] = []
    for message in reversed(messages):
        if not isinstance(message, dict) or str(message.get("role") or "") != "user":
            continue
        if _message_is_tool_result(message):
            continue
        if _message_is_gateway_control_instruction(message):
            continue
        text = _human_task_text_from_message(message)
        if text.strip():
            texts.append(text)
    return texts


def _human_task_text_from_message(message: dict[str, Any]) -> str:
    text = _content_to_text(message.get("content"))
    command_args = _extract_command_args(text)
    if command_args:
        return command_args
    if _looks_like_skill_injection_text(text):
        return _extract_skill_arguments(text)
    return text


def _extract_command_args(text: str) -> str:
    match = _COMMAND_ARGS_RE.search(text or "")
    if not match:
        return ""
    return _clean_task_arguments(match.group("body"))


def _extract_skill_arguments(text: str) -> str:
    match = _SKILL_ARGUMENTS_RE.search(text or "")
    if not match:
        return ""
    return _clean_task_arguments(match.group("body"))


def _clean_task_arguments(text: str) -> str:
    cleaned = html.unescape(str(text or "")).strip()
    if cleaned.lower() in {"", "none", "(none)", "n/a"}:
        return ""
    return cleaned


def _looks_like_skill_injection_text(text: str) -> bool:
    value = text or ""
    stripped = value.lstrip()
    return (
        stripped.startswith("Base directory for this skill:")
        or ("<EXTREMELY-IMPORTANT>" in value and "# Using Skills" in value)
        or ("Skill instructions" in value and "ARGUMENTS:" in value)
        or bool(_SKILL_FRONTMATTER_RE.search(value))
        or bool(_SKILL_DOC_HEADING_RE.search(value) and _SKILL_DOC_MARKER_RE.search(value))
        or bool(_SKILL_DOC_COLON_HEADING_RE.search(value) and _SKILL_DOC_MARKER_RE.search(value))
    )


def _message_is_tool_result(message: dict[str, Any]) -> bool:
    if str(message.get("role") or "") == "tool":
        return True
    if _converted_tool_result_record(message):
        return True
    content = message.get("content")
    if isinstance(content, dict):
        return content.get("type") == "tool_result"
    if isinstance(content, list) and content:
        saw_tool_result = False
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                saw_tool_result = True
                continue
            text = ""
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
            elif item is not None:
                text = str(item).strip()
            if text:
                return False
        return saw_tool_result
    return False


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
    chinese_markers = {
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
    }
    if value in chinese_markers:
        return True
    followup_markers = (
        "可以删掉",
        "可以删除",
        "可以丢弃",
        "可以重置",
        "确认删掉",
        "确认删除",
        "确认丢弃",
        "本地的修改可以删掉",
        "本地修改可以删掉",
        "local changes can be discarded",
        "discard local changes",
        "discard the local changes",
        "okay to delete",
        "ok to delete",
        "yes, discard",
    )
    if (
        len(value) <= 240
        and not _WINDOWS_DRIVE_PATH_RE.search(value)
        and any(marker in value for marker in followup_markers)
    ):
        return True
    if re.fullmatch(
        r"(?:好|好的|可以|确认|继续|接着|然后|往下)"
        r"(?:[，,。.!！?？\s]*(?:继续|执行|处理|推进|改进|修改|修复|落地|实现|完善|优化|监控|查看|检查|搜索|分析|做|跑|进行))*"
        r"[，,。.!！?？\s]*",
        value,
    ):
        return True
    if any(value.startswith(prefix) for prefix in ("继续 ", "接着 ")):
        return True
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
    prefixes = ("继续 ", "接着 ", "continue ", "proceed ", "go on ", "keep going ", "more ", "next ")
    return any(value.startswith(prefix) for prefix in prefixes)


def _looks_like_meta_capability_question(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    english = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    english_patterns = (
        "what can you do",
        "what can u do",
        "what are your capabilities",
        "what capabilities do you have",
        "what capability do you have",
        "what tools do you have",
        "what functions do you have",
        "what features do you have",
        "what can you help with",
        "what do you support",
        "which tools do you have",
        "list your tools",
        "show your tools",
        "your capabilities",
    )
    if any(pattern in english for pattern in english_patterns):
        return True

    compact = re.sub(r"[\s？?！!。,.，、:：；;\"'“”‘’（）()【】\[\]{}<>《》]+", "", lowered)
    subject_markers = ("你", "助手", "模型", "ai", "agent", "claude", "qwen")
    capability_markers = (
        "有什么功能",
        "有哪些功能",
        "有什么能力",
        "有哪些能力",
        "能做什么",
        "可以做什么",
        "会做什么",
        "能干什么",
        "会干什么",
        "支持什么功能",
        "支持哪些功能",
        "有什么工具",
        "有哪些工具",
        "支持什么工具",
        "支持哪些工具",
    )
    if any(subject in compact for subject in subject_markers) and any(marker in compact for marker in capability_markers):
        return True
    return compact in {
        "能做什么",
        "可以做什么",
        "会做什么",
        "能干什么",
        "有什么功能",
        "有哪些功能",
        "有什么能力",
        "有哪些能力",
    }


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
        "修复",
        "改进",
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
        "落地",
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
        "project",
        "local",
        "review",
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
    if any(name.lower() in lowered for name in context.allowed_names):
        return True
    return any(
        _compact_tool_name(name)
        in {
            "agent",
            "task",
            "bash",
            "glob",
            "grep",
            "read",
            "write",
            "edit",
            "multiedit",
            "notebookedit",
            "skill",
            "webfetch",
            "websearch",
        }
        for name in context.allowed_names
    )


def _is_deferred_named_tool_action_without_call(text: str, context: ToolBridgeContext) -> bool:
    if not context.allowed_names:
        return False
    if not (
        context.has_tool_loop
        or _looks_like_local_agent_task(context.task_text)
        or _task_explicitly_allows_mutation(context.task_text)
    ):
        return False
    lowered = _without_fenced_code_blocks(text).lower()
    if _PRIOR_TOOL_RESULT_REFERENCE_RE.search(lowered):
        return False
    action_markers = (
        "next",
        "need",
        "try",
        "attempt",
        "run",
        "use",
        "call",
        "execute",
        "inspect",
        "read",
        "search",
        "check",
        "list",
        "\u4e0b\u4e00\u6b65",
        "\u9700\u8981",
        "\u5c1d\u8bd5",
        "\u6b63\u5728",
        "\u8fd0\u884c",
        "\u4f7f\u7528",
        "\u8c03\u7528",
        "\u6267\u884c",
        "\u67e5\u770b",
        "\u8bfb\u53d6",
        "\u641c\u7d22",
        "\u83b7\u53d6",
        "\u5de5\u5177",
    )
    if not any(marker in lowered for marker in action_markers):
        return False
    for name in context.allowed_names:
        compact = _compact_tool_name(name)
        if compact and re.search(rf"(?<![A-Za-z0-9_]){re.escape(compact)}(?![A-Za-z0-9_])", lowered):
            return True
    return False


def _is_unverified_code_change_completion(text: str, context: ToolBridgeContext) -> bool:
    if not context.has_tool_loop or not context.recent_tool_call_names:
        return False
    if not context.allowed_names or not _has_code_change_tool(context):
        return False
    if not (_looks_like_local_agent_task(context.task_text) or _task_explicitly_allows_mutation(context.task_text)):
        return False
    if _task_explicitly_requests_standalone_new_file(context.task_text):
        return False
    raw = (text or "").strip()
    if not raw or not _CODE_CHANGE_COMPLETION_CLAIM_RE.search(raw):
        return False

    recent = [_compact_tool_name(name) for name in context.recent_tool_call_names]
    has_mutating_history = any(
        name in {"write", "writefile", "edit", "editfile", "multiedit", "applypatch", "patch"}
        for name in recent
    )
    if not has_mutating_history:
        return bool(_FIRST_PERSON_CODE_CHANGE_COMPLETION_CLAIM_RE.search(raw))

    available = {_compact_tool_name(tool.name) for tool in context.tools}
    has_followup_tool = bool(available & {"read", "glob", "grep", "ls", "edit", "multiedit", "applypatch"})
    if not has_followup_tool:
        return False

    has_inspection_history = any(
        name in {"read", "glob", "grep", "ls"} or "search" in name or "find" in name
        for name in recent
    )
    has_integration_history = any(name in {"edit", "editfile", "multiedit", "applypatch", "patch"} for name in recent)
    if has_integration_history:
        return False
    if _STANDALONE_WRITE_COMPLETION_RE.search(raw):
        return True
    return not has_inspection_history


def _is_incomplete_fix_stub_without_tool_call(text: str, context: ToolBridgeContext) -> bool:
    if not context.has_tool_loop or not context.allowed_names:
        return False
    if not (_looks_like_local_agent_task(context.task_text) or _has_code_change_tool(context)):
        return False
    raw = (text or "").strip()
    if not raw or len(raw) > 2000:
        return False
    has_fix_marker = bool(
        _CODE_PATCH_SNIPPET_WITHOUT_TOOL_RE.search(raw)
        or re.search(
            r"\b(?:bug|fix|issue|error|problem|middleware|auth|token)\b|"
            r"(?:\u4fee\u590d|\u95ee\u9898|\u9519\u8bef|\u6545\u969c|\u7f3a\u9677)",
            raw,
            re.IGNORECASE,
        )
    )
    if not has_fix_marker:
        return False
    if len(raw) <= 400 and (raw.endswith((":","\uff1a")) or _INLINE_CODE_FIX_STEP_RE.search(raw)):
        return True
    return bool(_FENCED_CODE_BLOCK_RE.search(raw))


def _is_deferred_local_file_inspection_without_call(text: str, context: ToolBridgeContext) -> bool:
    if not context.has_tool_loop or not context.allowed_names:
        return False
    if not (_looks_like_local_agent_task(context.task_text) or _has_read_like_tool(context)):
        return False
    raw = (text or "").strip()
    if not raw or len(raw) > 1200:
        return False
    if not _has_read_like_tool(context):
        return False
    return bool(_LOCAL_FILE_INSPECTION_INTENT_RE.search(raw))


def _is_unproven_final_answer_without_tool_call(text: str, context: ToolBridgeContext) -> bool:
    if not context.has_tool_loop or not context.allowed_names:
        return False
    if not (_looks_like_local_agent_task(context.task_text) or _has_read_like_tool(context) or _has_code_change_tool(context)):
        return False
    raw = (text or "").strip()
    if not raw or len(raw) > 1200:
        return False
    if _looks_like_complete_final_answer(raw, context):
        return False
    return bool(_UNPROVEN_FINAL_ACTION_RE.search(raw))


def _looks_like_complete_final_answer(text: str, context: ToolBridgeContext) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if _UNPROVEN_FINAL_ACTION_RE.search(raw):
        return False
    if _FINAL_ANSWER_MARKER_RE.search(raw):
        return True
    if _allows_plain_review_or_plan_text(context) and len(raw) >= 160:
        return not _UNPROVEN_FINAL_ACTION_RE.search(raw)
    return False


def _is_runaway_plain_text_without_tool_call(text: str, context: ToolBridgeContext) -> bool:
    if not context.allowed_names:
        return False
    if _allows_plain_review_or_plan_text(context):
        return False
    if len(text or "") < _RUNAWAY_PLAIN_TEXT_MIN_CHARS:
        return False
    return (
        context.has_tool_loop
        or _looks_like_local_agent_task(context.task_text)
        or _task_explicitly_allows_mutation(context.task_text)
    )


def _is_unexecuted_verification_command_after_write(text: str, context: ToolBridgeContext) -> bool:
    if not context.has_tool_loop or not context.allowed_names:
        return False
    if not _recent_tool_history_has_mutation(context):
        return False
    for match in _FENCED_CODE_BLOCK_RE.finditer(text or ""):
        lang = (match.group("lang") or "").strip().lower()
        body = match.group("body") or ""
        if lang in {"bash", "sh", "shell", "zsh", "powershell", "pwsh", "ps1", "cmd", "bat"} and _has_shell_command_line(body):
            return True
    return False


def _recent_tool_history_has_mutation(context: ToolBridgeContext) -> bool:
    return any(
        _compact_tool_name(name) in {"write", "writefile", "edit", "editfile", "multiedit", "applypatch", "patch"}
        for name in context.recent_tool_call_names
    )


def _has_shell_command_line(text: str) -> bool:
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "::", "rem ")):
            continue
        return True
    return False


def _task_explicitly_requests_standalone_new_file(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    markers = (
        "standalone file",
        "single file",
        "new file",
        "create a file",
        "write a file",
        "add a file",
        "新增文件",
        "创建文件",
        "写一个文件",
        "单文件",
        "独立文件",
    )
    return any(marker in lowered for marker in markers)


def _is_premature_clarification_without_tool_call(text: str, context: ToolBridgeContext) -> bool:
    if not context.allowed_names:
        return False
    raw = text or ""
    if not (
        (_CLARIFICATION_REQUEST_RE.search(raw) and _REPO_CLARIFICATION_RE.search(raw))
        or _looks_like_local_scope_selection_prompt(raw)
    ):
        return False
    allowed = {name.strip().lower() for name in context.allowed_names}
    return any(name in allowed for name in {"bash", "read", "glob", "ls", "grep"})


def _looks_like_local_scope_selection_prompt(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or len(raw) > 1200:
        return False
    lowered = raw.lower()
    return bool(
        (
            "target unclear" in lowered
            or "review target unclear" in lowered
            or "pick action" in lowered
            or "specify file" in lowered
            or "specify target" in lowered
            or "specify scope" in lowered
        )
        and (
            "review" in lowered
            or "scan" in lowered
            or "file" in lowered
            or "project" in lowered
            or "repo" in lowered
            or "scope" in lowered
        )
    )


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


def _extract_unwrapped_shell_command_candidates(text: str, context: ToolBridgeContext) -> list[Any]:
    shell_tool = _select_shell_execution_tool(context.tools) or _select_shell_execution_tool(list(context.hidden_tools))
    if not shell_tool:
        return []
    if not (context.has_tool_loop or _looks_like_local_agent_task(context.task_text)):
        return []
    fenced_commands = _extract_fenced_shell_command_lines(text or "")
    commands = fenced_commands or [match.group("command").strip() for match in _SHELL_COMMAND_LINE_RE.finditer(text or "")]
    if not commands:
        return []
    if not fenced_commands and len(commands) == 1 and not _SHELL_COMMAND_INTENT_RE.search(text or ""):
        return []
    return [
        {
            "calls": [
                {
                    "id": "call_web_shell_1",
                    "name": shell_tool.name,
                    "input": {_shell_command_key(shell_tool): " && ".join(commands)},
                }
            ]
        }
    ]


def _extract_fenced_shell_command_lines(text: str) -> list[str]:
    commands: list[str] = []
    for match in _FENCED_CODE_BLOCK_RE.finditer(text or ""):
        lang = (match.group("lang") or "").strip().lower()
        if lang not in {"bash", "sh", "shell", "zsh", "powershell", "pwsh", "ps1", "cmd", "bat", "batch"}:
            continue
        body = match.group("body") or ""
        commands.extend(match.group("command").strip() for match in _SHELL_COMMAND_LINE_RE.finditer(body))
    return commands


def _is_prose_shell_command_intent_without_call(text: str, context: ToolBridgeContext) -> bool:
    if not (context.has_tool_loop or _looks_like_local_agent_task(context.task_text)):
        return False
    return bool(_PROSE_SHELL_COMMAND_INTENT_RE.search(text or ""))


def _has_read_like_tool(context: ToolBridgeContext) -> bool:
    for tool in context.tools:
        name = _compact_tool_name(tool.name)
        if name in {"read", "glob", "grep", "ls", "lsp", "webfetch", "websearch"}:
            return True
        if any(part in name for part in ("search", "find", "list", "read", "fetch", "query")):
            return True
    return False


def build_repair_messages(messages: list[dict[str, Any]], bad_text: str, error: BridgeError | None) -> list[dict[str, Any]]:
    reason = error.message if error else "tool call format is invalid"
    error_kind = error.kind if error else "invalid_tool_call"
    if error_kind in {
        "off_task_environment_configuration_question",
        "off_task_scope_escalation_question",
        "optional_scope_question_without_need",
        "ask_user_question_budget_exceeded",
    }:
        repair_instruction = (
            "Previous tool JSON asked the user an off-task, repeated, or optional clarification question.\n"
            f"Error code: {error_kind}.\n"
            f"Error: {reason}.\n"
            "Do not call AskUserQuestion on this turn. Do not ask the user to authorize unrelated environment setup, "
            "agent settings, restructuring, migration, deletion, or broad project changes that the user did not request. "
            "For focus, scope, priority, or area choices, choose the broad/default option implied by the task and move forward. "
            "Continue the original task using available evidence or a materially necessary allowed tool. If the user asked "
            "for review, audit, analysis, or an improvement plan, provide that substantive answer with no DSML tool block."
        )
        return [
            *messages,
            {"role": "assistant", "content": (bad_text or "")[:4000]},
            {"role": "user", "content": repair_instruction},
        ]
    if error_kind == "repeat_same_ask_user_without_progress":
        repair_instruction = (
            "Previous tool JSON asked the user another question immediately after an AskUserQuestion already received an answer.\n"
            f"Error code: {error_kind}.\n"
            f"Error: {reason}.\n"
            "Do not call AskUserQuestion again on this turn. Use the user's previous answer already in the conversation. "
            "Continue the original task with a materially necessary allowed tool, or provide a substantive final answer "
            "if the gathered evidence is enough. For review, audit, analysis, or improvement-plan tasks, prefer moving "
            "forward with existing evidence instead of asking another clarification question."
        )
        return [
            *messages,
            {"role": "assistant", "content": (bad_text or "")[:4000]},
            {"role": "user", "content": repair_instruction},
        ]
    if error_kind == "successful_tool_misread_as_blocked_question":
        repair_instruction = (
            "Previous tool JSON asked the user for permission or pasted output by claiming the prior tool was blocked, "
            "but the immediately preceding Tool result was successful.\n"
            f"Error code: {error_kind}.\n"
            f"Error: {reason}.\n"
            "Do not call AskUserQuestion for this. Use the successful Tool result already in the conversation. "
            "If the result is empty, treat it as empty output, not a permission failure. Continue the original task "
            "with a materially different allowed tool, or provide a substantive final answer if the evidence is enough."
        )
        return [
            *messages,
            {"role": "assistant", "content": (bad_text or "")[:4000]},
            {"role": "user", "content": repair_instruction},
        ]
    if error_kind == "repeat_same_skill_without_progress":
        repair_instruction = (
            "Previous tool JSON repeated a Skill call that already completed in this conversation.\n"
            f"Error code: {error_kind}.\n"
            f"Error: {reason}.\n"
            "Do not request the same Skill again. The skill result is already available in the conversation. "
            "Continue the original user task with a different allowed tool such as Read, Glob, Grep, Edit, or Write. "
            "If the gathered evidence is already enough, provide a substantive final answer with no DSML tool block. "
            "If more evidence is needed, output exactly one valid DSML tool block using a non-Skill tool."
        )
        return [
            *messages,
            {"role": "assistant", "content": (bad_text or "")[:4000]},
            {"role": "user", "content": repair_instruction},
        ]
    if error_kind == "off_task_environment_configuration_final":
        repair_instruction = (
            "Previous final answer went off task into agent/environment configuration advice.\n"
            f"Error code: {error_kind}.\n"
            f"Error: {reason}.\n"
            "Do not ask the user to configure Claude, Codex, statusLine, settings.json, plugins, hooks, or badges "
            "unless the latest user task explicitly requested that exact configuration work. Continue the original "
            "local-agent task. If more evidence is needed, output exactly one valid DSML tool block using an "
            "allowed tool such as Read, Glob, Grep, LSP, WebFetch, Edit, or Write. If the gathered evidence is enough, "
            "return a substantive final answer grounded in that evidence with no DSML tool block. No manual setup steps."
        )
        return [
            *messages,
            {"role": "assistant", "content": (bad_text or "")[:4000]},
            {"role": "user", "content": repair_instruction},
        ]
    if error_kind == "repeat_discovery_call_without_progress":
        repair_instruction = (
            "Previous tool JSON repeated a discovery call that already returned evidence in this conversation.\n"
            f"Error code: {error_kind}.\n"
            f"Error: {reason}.\n"
            "Do not request the same Glob/Grep/LS/LSP input again unless a file-changing tool has run since then. "
            "Use the earlier result already in the conversation. If more evidence is required, request exactly one "
            "materially different allowed tool/input. If the evidence is enough, provide a substantive final answer "
            "with no DSML tool block."
        )
        return [
            *messages,
            {"role": "assistant", "content": (bad_text or "")[:4000]},
            {"role": "user", "content": repair_instruction},
        ]
    if error_kind == "repeat_read_call_without_progress":
        repair_instruction = (
            "Previous tool JSON repeated a Read call that already returned evidence in this conversation.\n"
            f"Error code: {error_kind}.\n"
            f"Error: {reason}.\n"
            "Do not request the same Read input again unless a file-changing tool has run since then. "
            "Use the earlier Read result already in the conversation. If more evidence is required, request exactly "
            "one materially different file, range, or allowed tool/input. If the evidence is enough, provide a "
            "substantive final answer with no DSML tool block."
        )
        return [
            *messages,
            {"role": "assistant", "content": (bad_text or "")[:4000]},
            {"role": "user", "content": repair_instruction},
        ]
    if error_kind == "repeat_shell_housekeeping_without_progress":
        repair_instruction = (
            "Previous tool JSON repeated shell housekeeping that already returned evidence in this conversation.\n"
            f"Error code: {error_kind}.\n"
            f"Error: {reason}.\n"
            "Do not request another equivalent Bash/shell git status, git diff, git log, file listing, or file discovery "
            "command unless a file-changing tool has run since then. Use the earlier result already in the conversation. "
            "If more evidence is required, request exactly one materially different allowed tool/input. Prefer Read/Grep/Glob "
            "for inspection or Edit/Write for a real change. If the evidence is enough, provide a substantive final answer "
            "with no DSML tool block."
        )
        return [
            *messages,
            {"role": "assistant", "content": (bad_text or "")[:4000]},
            {"role": "user", "content": repair_instruction},
        ]
    repair_instruction = (
        "Previous tool call was invalid.\n"
        f"Error code: {error_kind}.\n"
        f"Error: {reason}.\n"
        "The listed tools are available through the downstream client, not your own runtime. "
        "Do not say tools do not exist. Do not say you cannot access files or commands. "
        "If your previous answer said you would research, inspect, read, search, check, analyze, execute, run, update, pull, reset, or apply something, it was incomplete because it did not request a tool. "
        "没有发起工具调用时，不要说“我先研究/让我查看/正在搜索”。 "
        "If the task needs a listed tool, request a tool using the protocol below.\n"
        "Do not use the literal placeholder name tool_name; replace it with one exact tool name from the allowed tools list.\n"
        "Rewrite only one valid DSML tool block. Do not output explanation text. Required format:\n"
        "<|DSML|tool_calls>\n"
        "  <|DSML|invoke name=\"tool_name\">\n"
        "  </|DSML|invoke>\n"
        "</|DSML|tool_calls>"
    )
    return [
        *messages,
        {"role": "assistant", "content": (bad_text or "")[:4000]},
        {"role": "user", "content": repair_instruction},
    ]


def _normalize_candidates(candidates: list[Any], context: ToolBridgeContext) -> tuple[list[ToolCallDraft], BridgeError | None]:
    out: list[ToolCallDraft] = []
    seen_ids: set[str] = set()
    soft_non_progress_errors: list[BridgeError] = []
    allowed = context.allowed_names
    canonical_by_lower: dict[str, str] = {}
    for tool in context.tools:
        canonical_by_lower.setdefault(tool.name.lower(), tool.name)
    specs_by_name = {tool.name: tool for tool in context.tools}
    candidate_batches = [(candidate, _candidate_items(candidate)) for candidate in candidates]
    candidate_item_count = sum(len(items) for _, items in candidate_batches)
    for candidate, items in candidate_batches:
        if not items:
            return [], BridgeError("empty_tool_call", "工具调用为空")
        for item in items:
            normalized = _normalize_item(item)
            if normalized is None:
                return [], BridgeError("invalid_tool_call", "工具调用必须是对象")
            canonical_name = normalized.name if normalized.name in allowed else canonical_by_lower.get(normalized.name.lower())
            if not canonical_name:
                if _compact_tool_name(normalized.name) in {"write", "writefile"}:
                    failed_read_write_error = _write_after_failed_read_error(normalized, normalized.name, context)
                    if failed_read_write_error:
                        return [], failed_read_write_error
                    failed_path_write_error = _write_after_failed_path_error(normalized, normalized.name, context)
                    if failed_path_write_error:
                        return [], failed_path_write_error
                hidden_shell_replacement = _safe_replacement_for_hidden_readonly_shell_tool(normalized, context)
                if hidden_shell_replacement:
                    normalized = hidden_shell_replacement
                    canonical_name = hidden_shell_replacement.name
                    specs_by_name.setdefault(
                        canonical_name,
                        _tool_by_name(list(context.hidden_tools), canonical_name)
                        or ToolSpec(canonical_name, "", {"type": "object"}, False),
                    )
                else:
                    hidden_shell_error = _hidden_local_agent_shell_tool_error(normalized.name, context)
                    if hidden_shell_error is not None:
                        return [], hidden_shell_error
                if not canonical_name:
                    replacement = _safe_replacement_for_delegation_tool(normalized, context.tools)
                    if not replacement:
                        replacement = _safe_replacement_for_file_read_tool(normalized, context.tools)
                    if not replacement:
                        replacement = _safe_replacement_for_file_discovery_tool(normalized, context.tools)
                    if not replacement:
                        replacement = _safe_replacement_for_cli_tool(normalized, context.tools)
                    if replacement:
                        normalized = replacement
                        canonical_name = replacement.name
                    else:
                        return [], BridgeError("unknown_tool", f"未知工具：{normalized.name}", repairable=True)
            if not isinstance(normalized.input, dict):
                shell_replacement = _safe_replacement_for_shell_string_input(normalized, canonical_name, context.tools)
                if shell_replacement:
                    normalized = shell_replacement
                else:
                    return [], BridgeError("invalid_input", f"工具 {normalized.name} 的 input 必须是对象")
            replacement = _safe_replacement_for_directory_read_tool(normalized, canonical_name, context.tools)
            if replacement:
                normalized = replacement
                canonical_name = replacement.name
            replacement = _safe_replacement_for_glob_input_shape(normalized, canonical_name, context.tools)
            if replacement:
                normalized = replacement
                canonical_name = replacement.name
            replacement = _safe_replacement_for_expensive_glob(normalized, canonical_name, context.tools)
            if replacement:
                normalized = replacement
                canonical_name = replacement.name
            else:
                expensive_glob = _expensive_glob_message(canonical_name, normalized.input)
                if expensive_glob:
                    return [], BridgeError("expensive_tool_input", expensive_glob, repairable=True)
            normalized = _normalize_shell_tool_command(normalized, canonical_name, context.tools)
            replacement = _safe_replacement_for_windows_dir_shell_command(normalized, canonical_name, context.tools)
            if replacement:
                normalized = replacement
                canonical_name = replacement.name
                replacement = _safe_replacement_for_expensive_glob(normalized, canonical_name, context.tools)
                if replacement:
                    normalized = replacement
                    canonical_name = replacement.name
            normalized = _repair_common_tool_input_shape(normalized, canonical_name, specs_by_name)
            normalized = _normalize_ask_user_question_input(normalized, canonical_name)
            required_error = _missing_required_tool_input_error(normalized, canonical_name, specs_by_name)
            if required_error:
                return [], required_error
            off_task_scope_question_error = _off_task_scope_escalation_question_error(
                normalized,
                canonical_name,
                context,
            )
            if off_task_scope_question_error:
                soft_non_progress_errors.append(off_task_scope_question_error)
                continue
            off_task_question_error = _off_task_environment_configuration_question_error(
                normalized,
                canonical_name,
                context,
            )
            if off_task_question_error:
                soft_non_progress_errors.append(off_task_question_error)
                continue
            successful_tool_blocked_question_error = _successful_tool_misread_as_blocked_question_error(
                normalized,
                canonical_name,
                context,
            )
            if successful_tool_blocked_question_error:
                soft_non_progress_errors.append(successful_tool_blocked_question_error)
                continue
            repeat_ask_user_error = _repeat_ask_user_question_without_progress_error(
                normalized,
                canonical_name,
                context,
            )
            if repeat_ask_user_error:
                soft_non_progress_errors.append(repeat_ask_user_error)
                continue
            ask_user_budget_error = _ask_user_question_budget_error(
                normalized,
                canonical_name,
                context,
            )
            if ask_user_budget_error:
                soft_non_progress_errors.append(ask_user_budget_error)
                continue
            optional_scope_question_error = _optional_scope_selection_question_error(
                normalized,
                canonical_name,
                context,
            )
            if optional_scope_question_error:
                soft_non_progress_errors.append(optional_scope_question_error)
                continue
            repeat_skill_error = _repeat_same_skill_without_progress_error(
                normalized,
                canonical_name,
                context,
            )
            if repeat_skill_error:
                soft_non_progress_errors.append(repeat_skill_error)
                continue
            repeat_unchanged_read_error = _repeat_unchanged_read_error(normalized, canonical_name, context)
            if repeat_unchanged_read_error:
                soft_non_progress_errors.append(repeat_unchanged_read_error)
                continue
            repeat_read_error = _repeat_read_call_without_progress_error(
                normalized,
                canonical_name,
                context,
                min_previous_count=1 if candidate_item_count > 1 else 2,
            )
            if repeat_read_error:
                soft_non_progress_errors.append(repeat_read_error)
                continue
            repeat_discovery_error = _repeat_discovery_call_without_progress_error(
                normalized,
                canonical_name,
                context,
            )
            if repeat_discovery_error:
                soft_non_progress_errors.append(repeat_discovery_error)
                continue
            repeat_shell_housekeeping_error = _repeat_shell_housekeeping_without_progress_error(
                normalized,
                canonical_name,
                context,
            )
            if repeat_shell_housekeeping_error:
                soft_non_progress_errors.append(repeat_shell_housekeeping_error)
                continue
            failed_read_write_error = _write_after_failed_read_error(normalized, canonical_name, context)
            if failed_read_write_error:
                return [], failed_read_write_error
            failed_path_write_error = _write_after_failed_path_error(normalized, canonical_name, context)
            if failed_path_write_error:
                return [], failed_path_write_error
            shell_error = _shell_tool_command_error(normalized, canonical_name, context)
            if shell_error:
                return [], shell_error
            call_id = normalized.id or _new_tool_call_id()
            if call_id in seen_ids:
                return [], BridgeError("duplicate_tool_call_id", f"重复的工具调用 id：{call_id}")
            seen_ids.add(call_id)
            out.append(ToolCallDraft(id=call_id, name=canonical_name, input=normalized.input))
    if not out:
        if soft_non_progress_errors:
            return [], soft_non_progress_errors[0]
        return [], BridgeError("empty_tool_call", "工具调用为空")
    max_calls = context.options.max_calls_per_turn
    if all(specs_by_name.get(call.name, ToolSpec(call.name, "", {}, False)).read_only for call in out):
        max_calls = context.options.max_readonly_calls_per_turn
    if len(out) > max_calls:
        return [], BridgeError("too_many_tool_calls", f"本轮工具调用过多：{len(out)} > {max_calls}")
    return out, None


def _hidden_local_agent_shell_tool_error(name: str, context: ToolBridgeContext) -> BridgeError | None:
    if _allows_ds2api_style_shell_passthrough(context):
        return None
    compact = _compact_tool_name(name)
    if not _is_bash_tool_name(name) and compact not in {"shell", "terminal", "powershell", "cmd"}:
        return None
    policy = (context.options.exposure_policy or "safe").strip().lower()
    profile = _resolved_tool_profile_for_task(context.options, context.task_text)
    if profile != "agent" and policy not in {"local-agent", "local_agent", "code-agent", "code_agent"}:
        return None
    return BridgeError(
        "unsafe_local_shell_command",
        "Bash/shell is intentionally hidden for this local-agent task because the user did not explicitly request shell execution. Use Read/Glob/Grep/Edit/Write tools, or ask the user to confirm shell execution.",
        repairable=True,
    )


def _allows_ds2api_style_shell_passthrough(context: ToolBridgeContext) -> bool:
    return _allows_ds2api_style_tool_passthrough(context)


def _allows_ds2api_style_tool_passthrough(context: ToolBridgeContext) -> bool:
    return _configured_tool_profile(context.options) == "all"


def _allows_ds2api_style_progress_passthrough(context: ToolBridgeContext) -> bool:
    return _resolved_tool_profile_for_task(context.options, context.task_text) == "all"


def _allows_ds2api_style_agent_guard_passthrough(context: ToolBridgeContext) -> bool:
    return _resolved_tool_profile_for_task(context.options, context.task_text) == "all"


def _allows_ds2api_style_rejected_markup_passthrough(context: ToolBridgeContext, raw: str) -> bool:
    return _allows_ds2api_style_tool_passthrough(context) and _looks_like_ds2api_xml_tool_markup(raw)


def _looks_like_ds2api_xml_tool_markup(raw: str) -> bool:
    text = raw or ""
    lowered = text.lower()
    return bool(
        _XML_TOOL_CALLS_OPEN_RE.search(text)
        or _XML_TOOL_CALLS_CLOSE_RE.search(text)
        or _XML_INVOKE_OPEN_RE.search(text)
        or "<tool_call" in lowered
        or "<tools" in lowered
    )


def _discovery_tool_names_for_error(context: ToolBridgeContext) -> str:
    available = {_compact_tool_name(tool.name) for tool in context.tools}
    names: list[str] = []
    if "glob" in available:
        names.append("Glob")
    if "grep" in available:
        names.append("Grep")
    if available & {"ls", "listdir", "listdirectory", "dir", "tree"}:
        names.append("LS")
    if available & {"read", "readfile", "fileread"}:
        names.append("Read")
    return _join_tool_names_for_guidance(names) or "an available read-only discovery tool"


def _join_tool_names_for_guidance(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} or {names[1]}"
    return f"{'/'.join(names[:-1])} or {names[-1]}"


def _off_task_environment_configuration_question_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _allows_ds2api_style_agent_guard_passthrough(context):
        return None
    if _compact_tool_name(canonical_name) != "askuserquestion":
        return None
    if not (context.has_tool_loop or _looks_like_local_agent_task(context.task_text)):
        return None
    if _OFF_TASK_ENV_CONFIG_QUESTION_RE.search(context.task_text or ""):
        return None
    question_text = _ask_user_text(_first_ask_user_question(call.input), call.input)
    probe = f"{question_text}\n{_stable_tool_input(call.input)}"
    if not _OFF_TASK_ENV_CONFIG_QUESTION_RE.search(probe):
        return None
    return BridgeError(
        "off_task_environment_configuration_question",
        (
            "The model asked the user to configure agent/environment settings that are unrelated to the "
            "current local-agent task. Continue the original task with an allowed tool, or answer from "
            "the evidence already gathered."
        ),
        repairable=True,
    )


def _off_task_scope_escalation_question_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _allows_ds2api_style_agent_guard_passthrough(context):
        return None
    if _compact_tool_name(canonical_name) != "askuserquestion":
        return None
    if not (context.has_tool_loop or _looks_like_local_agent_task(context.task_text)):
        return None
    if _OFF_TASK_SCOPE_ESCALATION_QUESTION_RE.search(context.task_text or ""):
        return None
    question_text = _ask_user_text(_first_ask_user_question(call.input), call.input)
    probe = f"{question_text}\n{_stable_tool_input(call.input)}"
    if not _OFF_TASK_SCOPE_ESCALATION_QUESTION_RE.search(probe):
        return None
    return BridgeError(
        "off_task_scope_escalation_question",
        (
            "The model asked the user to authorize broad restructuring, migration, deletion, or rewrite work "
            "that the current task did not request. Continue the original task with available evidence, a "
            "materially necessary allowed tool, or a substantive final answer."
        ),
        repairable=True,
    )


def _task_explicitly_allows_ask_user_questions(text: str) -> bool:
    return bool(_EXPLICIT_ASK_USER_TASK_RE.search(text or ""))


def _ask_user_probe_text(call: ToolCallDraft) -> str:
    first_question = _first_ask_user_question(call.input)
    return " ".join(
        part
        for part in (
            _ask_user_header(first_question, call.input),
            _ask_user_text(first_question, call.input),
            _stable_tool_input(call.input),
        )
        if part
    )


def _has_non_question_tool_progress(context: ToolBridgeContext) -> bool:
    return any(_compact_tool_name(name) != "askuserquestion" for name in context.recent_tool_call_names)


def _optional_scope_selection_question_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _allows_ds2api_style_agent_guard_passthrough(context):
        return None
    if _compact_tool_name(canonical_name) != "askuserquestion":
        return None
    if _task_explicitly_allows_ask_user_questions(context.task_text):
        return None
    if not (context.has_tool_loop or _looks_like_local_agent_task(context.task_text)):
        return None
    probe = _ask_user_probe_text(call)
    if not _OPTIONAL_SCOPE_SELECTION_QUESTION_RE.search(probe):
        return None
    if not (
        _has_non_question_tool_progress(context)
        or _looks_like_readonly_review_task(context.task_text)
        or _task_explicitly_allows_mutation(context.task_text)
        or _has_read_like_tool(context)
    ):
        return None
    return BridgeError(
        "optional_scope_question_without_need",
        (
            "The model asked an optional focus, area, or scope selection question even though the local-agent task "
            "can continue with a safe default. Choose the broad/default option implied by the task, use existing "
            "evidence, request a materially necessary non-question tool, or provide a substantive answer."
        ),
        repairable=True,
    )


def _ask_user_question_budget_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _allows_ds2api_style_agent_guard_passthrough(context):
        return None
    if _compact_tool_name(canonical_name) != "askuserquestion":
        return None
    if _task_explicitly_allows_ask_user_questions(context.task_text):
        return None
    if not context.has_tool_loop:
        return None
    if not any(_compact_tool_name(name) == "askuserquestion" for name in context.recent_tool_call_names):
        return None
    return BridgeError(
        "ask_user_question_budget_exceeded",
        (
            "The model already asked the user a question in this local-agent tool loop. Do not ask another "
            "clarification or scope question; continue with the user's existing answer, a materially necessary "
            "non-question tool, or a substantive final answer."
        ),
        repairable=True,
    )


def _repeat_ask_user_question_without_progress_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _allows_ds2api_style_agent_guard_passthrough(context):
        return None
    if _compact_tool_name(canonical_name) != "askuserquestion":
        return None
    if not context.has_tool_loop or not context.recent_tool_call_names:
        return None
    if _compact_tool_name(context.recent_tool_call_names[-1]) != "askuserquestion":
        return None
    return BridgeError(
        "repeat_same_ask_user_without_progress",
        (
            "The model asked another AskUserQuestion immediately after the previous AskUserQuestion already "
            "received an answer. Use the existing answer, request a materially necessary non-question tool, "
            "or provide a substantive final answer."
        ),
        repairable=True,
    )


def _successful_tool_misread_as_blocked_question_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _allows_ds2api_style_agent_guard_passthrough(context):
        return None
    if _compact_tool_name(canonical_name) != "askuserquestion":
        return None
    if not context.has_tool_loop:
        return None
    result_text = context.last_tool_result_text or ""
    immediate_success = bool(
        result_text and not context.last_tool_result_is_error and _SUCCESSFUL_TOOL_RESULT_RE.search(result_text)
    )
    recent_success_summary = (
        context.recent_successful_tool_result_summaries[-1]
        if context.recent_successful_tool_result_summaries
        else ""
    )
    recent_success_name = (
        context.recent_successful_tool_result_names[-1]
        if context.recent_successful_tool_result_names
        else ""
    )
    if not immediate_success and not recent_success_summary:
        return None
    question_text = _ask_user_probe_text(call)
    if not _TOOL_BLOCKED_CLAIM_RE.search(question_text):
        return None
    recent_tool = (
        context.recent_tool_call_names[-1]
        if immediate_success and context.recent_tool_call_names
        else recent_success_name or "the previous successful tool"
    )
    successful_evidence = recent_success_summary or recent_tool
    return BridgeError(
        "successful_tool_misread_as_blocked_question",
        (
            f"The model claimed {recent_tool} was blocked or needed user authorization, but "
            f"{successful_evidence} already produced a successful tool_result in the recent tool loop. "
            "Use the successful result instead of asking the user."
        ),
        repairable=True,
    )


def _repeat_same_skill_without_progress_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _allows_ds2api_style_progress_passthrough(context):
        return None
    if _compact_tool_name(canonical_name) != "skill":
        return None
    if not context.has_tool_loop or not (context.recent_tool_call_summaries or context.recent_skill_names):
        return None
    current_skill = _skill_name_from_input(call.input)
    if current_skill and any(
        _compact_tool_name(skill_name) == _compact_tool_name(current_skill)
        for skill_name in context.recent_skill_names
    ):
        return BridgeError(
            "repeat_same_skill_without_progress",
            (
                f"The model repeated the same Skill call ({current_skill}) in the active tool loop. "
                "Use the skill result already in the conversation and continue the original task with a "
                "different allowed tool or a substantive final answer."
            ),
            repairable=True,
        )
    current = _tool_loop_summary(ToolCallDraft(id=call.id, name=canonical_name, input=call.input))
    if current not in context.recent_tool_call_summaries:
        return None
    return BridgeError(
        "repeat_same_skill_without_progress",
        (
            f"The model repeated the same Skill call ({current}) in the active tool loop. "
            "Use the skill result already in the conversation and continue the original task with a "
            "different allowed tool or a substantive final answer."
        ),
        repairable=True,
    )


def _repeat_unchanged_read_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _allows_ds2api_style_progress_passthrough(context):
        return None
    if _compact_tool_name(canonical_name) not in {"read", "readfile", "fileread"}:
        return None
    read_path = _tool_path_from_input(call.input)
    unchanged_paths = context.recent_unchanged_read_paths or (
        (context.last_unchanged_read_path,) if context.last_unchanged_read_path else ()
    )
    if not context.has_tool_loop or not read_path or not unchanged_paths:
        return None
    matched_index = next(
        (
            index
            for index, unchanged_path in enumerate(unchanged_paths)
            if _normalize_path_for_compare(read_path) == _normalize_path_for_compare(unchanged_path)
        ),
        None,
    )
    if matched_index is None:
        return None
    summaries = context.recent_unchanged_read_summaries or (
        (context.last_unchanged_read_summary,) if context.last_unchanged_read_summary else ()
    )
    previous = (
        summaries[matched_index]
        if matched_index < len(summaries) and summaries[matched_index]
        else unchanged_paths[matched_index]
    )
    return BridgeError(
        "repeat_unchanged_read_without_progress",
        (
            "The model repeated a Read for a file whose latest tool result said the content was unchanged "
            f"({previous}). Use the earlier Read result already in the conversation, choose a materially "
            "different tool/input, edit/write when ready, or answer from the available evidence."
        ),
        repairable=True,
    )


def _repeat_read_call_without_progress_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
    *,
    min_previous_count: int = 2,
) -> BridgeError | None:
    if _allows_ds2api_style_progress_passthrough(context):
        return None
    if _compact_tool_name(canonical_name) not in {"read", "readfile", "fileread"}:
        return None
    if not context.has_tool_loop or not context.recent_tool_call_summaries:
        return None
    if _recent_tool_history_has_mutation(context):
        return None
    current = _tool_loop_summary(ToolCallDraft(id=call.id, name=canonical_name, input=call.input))
    previous_count = sum(1 for summary in context.recent_tool_call_summaries if summary == current)
    if previous_count < min_previous_count:
        return None
    return BridgeError(
        "repeat_read_call_without_progress",
        (
            f"The model repeated the same Read call ({current}) in an active tool loop without any "
            "intervening file change. Use the earlier Read result already in the conversation, choose "
            "a materially different file/range/tool, edit/write when ready, or answer from the available evidence."
        ),
        repairable=True,
    )


def _repeat_discovery_call_without_progress_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _allows_ds2api_style_progress_passthrough(context):
        return None
    if _compact_tool_name(canonical_name) not in _REPEAT_DISCOVERY_TOOL_NAMES:
        return None
    if not context.has_tool_loop or not context.recent_tool_call_summaries:
        return None
    if _recent_tool_history_has_mutation(context):
        return None
    current = _tool_loop_summary(ToolCallDraft(id=call.id, name=canonical_name, input=call.input))
    if current not in context.recent_tool_call_summaries:
        return None
    return BridgeError(
        "repeat_discovery_call_without_progress",
        (
            f"The model repeated the same discovery call ({current}) without any intervening file change. "
            "Use the earlier result already in the conversation, choose a materially different tool/input, "
            "edit/write when ready, or answer from the available evidence."
        ),
        repairable=True,
    )


def _repeat_shell_housekeeping_without_progress_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _allows_ds2api_style_progress_passthrough(context):
        return None
    tool = _tool_by_name(list(context.tools), canonical_name)
    if not _is_shell_execution_tool(tool, canonical_name):
        return None
    if not context.has_tool_loop or not context.recent_tool_call_summaries:
        return None
    if _recent_tool_history_has_mutation(context):
        return None
    command = _shell_command_from_input(call.input)
    category = _shell_no_progress_housekeeping_category(command)
    if not category:
        return None
    current = _tool_loop_summary(ToolCallDraft(id=call.id, name=canonical_name, input=call.input))
    previous_count = 0
    for name, summary in zip(context.recent_tool_call_names, context.recent_tool_call_summaries):
        if _compact_tool_name(name) not in _SHELL_LOOP_TOOL_NAMES:
            continue
        previous_command = _shell_command_from_loop_summary(summary)
        if _shell_no_progress_housekeeping_category(previous_command) == category:
            previous_count += 1
    if previous_count < 2 and current not in context.recent_tool_call_summaries:
        return None
    return BridgeError(
        "repeat_shell_housekeeping_without_progress",
        (
            f"The model repeated shell housekeeping ({current}) in an active tool loop without any "
            "intervening file change. Use the earlier shell result already in the conversation, choose "
            "a materially different allowed tool/input such as Read, Glob, Grep, Edit, or Write, or "
            "provide a substantive final answer if the evidence is enough."
        ),
        repairable=True,
    )


def _write_after_failed_read_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _compact_tool_name(canonical_name) not in {"write", "writefile"}:
        return None
    failed_paths = context.recent_failed_read_paths or (
        (context.last_failed_read_path,) if context.last_failed_read_path else ()
    )
    if not context.has_tool_loop or not failed_paths:
        return None
    if _task_explicitly_requests_standalone_new_file(context.task_text):
        return None
    write_path = _tool_path_from_input(call.input)
    matched_index = next(
        (index for index, failed_path in enumerate(failed_paths) if _paths_match_for_missing_read(write_path, failed_path)),
        None,
    )
    if not write_path or matched_index is None:
        return None
    available = {_compact_tool_name(tool.name) for tool in context.tools}
    if not (available & {"glob", "grep", "ls", "listdir", "read", "readfile"}):
        return None
    failed_summaries = context.recent_failed_read_summaries or (
        (context.last_failed_read_summary,) if context.last_failed_read_summary else ()
    )
    failed = (
        failed_summaries[matched_index]
        if matched_index < len(failed_summaries) and failed_summaries[matched_index]
        else failed_paths[matched_index]
    )
    return BridgeError(
        "write_after_failed_read_without_discovery",
        (
            "The model tried to create a path at or near a previous Read result that said the file does not exist "
            f"({failed}). Use {_discovery_tool_names_for_error(context)} to discover the repository "
            "structure before writing a guessed new module. Only create a standalone new file when the user "
            "explicitly asks for that."
        ),
        repairable=True,
    )


def _write_after_failed_path_error(
    call: ToolCallDraft,
    canonical_name: str,
    context: ToolBridgeContext,
) -> BridgeError | None:
    if _compact_tool_name(canonical_name) not in {"write", "writefile"}:
        return None
    failed_paths = context.recent_failed_discovery_paths or (
        (context.last_failed_discovery_path,) if context.last_failed_discovery_path else ()
    )
    if not context.has_tool_loop or not failed_paths:
        return None
    if _task_explicitly_requests_standalone_new_file(context.task_text):
        return None
    write_path = _tool_path_from_input(call.input)
    matched_index = next(
        (index for index, failed_path in enumerate(failed_paths) if _path_same_or_under(write_path, failed_path)),
        None,
    )
    if not write_path or matched_index is None:
        return None
    available = {_compact_tool_name(tool.name) for tool in context.tools}
    if not (available & {"glob", "grep", "ls", "listdir", "read", "readfile"}):
        return None
    failed_summaries = context.recent_failed_discovery_summaries or (
        (context.last_failed_discovery_summary,) if context.last_failed_discovery_summary else ()
    )
    failed = (
        failed_summaries[matched_index]
        if matched_index < len(failed_summaries) and failed_summaries[matched_index]
        else failed_paths[matched_index]
    )
    return BridgeError(
        "write_after_failed_path_without_discovery",
        (
            "The model tried to create a file at or under a path that the previous tool result said does not exist "
            f"({failed}). Use {_discovery_tool_names_for_error(context)} to discover the "
            "repository structure before writing a guessed new module or test file. Only create a standalone new file "
            "when the user explicitly asks for that."
        ),
        repairable=True,
    )


def _repair_common_tool_input_shape(
    call: ToolCallDraft,
    canonical_name: str,
    specs_by_name: dict[str, ToolSpec],
) -> ToolCallDraft:
    spec = specs_by_name.get(canonical_name)
    schema = spec.input_schema if spec and isinstance(spec.input_schema, dict) else {}
    required = schema.get("required") if isinstance(schema, dict) else None
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(required, list) or not isinstance(properties, dict):
        return call

    updated = dict(call.input)
    changed = False
    updated, schema_alias_changed = _repair_schema_property_aliases(updated, required, properties)
    changed = changed or schema_alias_changed
    if "skill" in required and "skill" not in updated and _schema_property_accepts_string(properties.get("skill")):
        skill = _pop_first_string_alias(updated, ("skill_name", "name", "skillName"))
        if skill:
            updated["skill"] = skill
            changed = True

    if "prompt" in required and "prompt" not in updated and _schema_property_accepts_string(properties.get("prompt")):
        prompt = _pop_first_string_alias(updated, ("focus", "query", "task", "instruction", "instructions", "text"))
        if prompt:
            updated["prompt"] = _shorten(_first_meaningful_line(prompt), 4000)
            changed = True

    if "file_path" in required and "file_path" not in updated and _schema_property_accepts_string(properties.get("file_path")):
        file_path = _pop_first_string_alias(updated, ("path", "file", "filepath", "filePath", "filename"))
        if file_path:
            updated["file_path"] = file_path
            changed = True

    if "path" in required and "path" not in updated and _schema_property_accepts_string(properties.get("path")):
        path = _pop_first_string_alias(updated, ("file_path", "file", "filepath", "filePath", "filename"))
        if path:
            updated["path"] = path
            changed = True

    if "description" in required and "description" not in updated and _schema_property_accepts_string(properties.get("description")):
        description = _description_from_input(updated)
        if description:
            updated["description"] = description
            changed = True

    if "pattern" in required and "pattern" not in updated and _schema_property_accepts_string(properties.get("pattern")):
        pattern = _pattern_from_patterns(updated.get("patterns"))
        if pattern:
            updated["pattern"] = pattern
            updated.pop("patterns", None)
            changed = True

    if not changed:
        return call
    return ToolCallDraft(id=call.id, name=call.name, input=updated)


def _repair_schema_property_aliases(
    input_value: dict[str, Any],
    required: list[Any],
    properties: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    updated = dict(input_value)
    changed = False
    for raw_name in required:
        if not isinstance(raw_name, str) or raw_name in updated:
            continue
        prop_schema = properties.get(raw_name)
        alias_key = _schema_property_alias_key(raw_name, updated, properties)
        if not alias_key:
            continue
        value = updated.get(alias_key)
        if value is None:
            continue
        updated[raw_name] = value
        if alias_key not in properties:
            updated.pop(alias_key, None)
        changed = True
        if prop_schema is not None:
            normalized, normalized_changed = _normalize_value_with_schema(updated[raw_name], prop_schema)
            if normalized_changed:
                updated[raw_name] = normalized
                changed = True
    return updated, changed


def _schema_property_alias_key(
    required_name: str,
    input_value: dict[str, Any],
    properties: dict[str, Any],
) -> str:
    normalized_required = _normalize_schema_property_name(required_name)
    for key in input_value:
        if not isinstance(key, str) or key == required_name:
            continue
        if _normalize_schema_property_name(key) == normalized_required:
            return key
    if not _schema_property_accepts_string(properties.get(required_name)):
        return ""
    for alias in _semantic_schema_property_aliases(required_name):
        if alias in input_value and isinstance(input_value.get(alias), str) and input_value.get(alias).strip():
            return alias
    return ""


def _normalize_schema_property_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _semantic_schema_property_aliases(required_name: str) -> tuple[str, ...]:
    compact = _normalize_schema_property_name(required_name)
    if compact == "subject":
        return ("title", "description", "summary", "name", "task", "task_name", "taskName")
    return ()


def _normalize_ask_user_question_input(call: ToolCallDraft, canonical_name: str) -> ToolCallDraft:
    if _compact_tool_name(canonical_name) != "askuserquestion":
        return call
    input_value = dict(call.input)
    first_question = _first_ask_user_question(input_value)
    question = _ask_user_text(first_question, input_value)
    if not question:
        return call
    header = _ask_user_header(first_question, input_value)
    options = _ask_user_options(first_question, input_value)
    multi_select = _ask_user_multi_select(first_question, input_value)
    normalized_question: dict[str, Any] = {"question": question}
    if header:
        normalized_question["header"] = header
    if options:
        normalized_question["options"] = options
    if multi_select is not None:
        normalized_question["multiSelect"] = multi_select
    return ToolCallDraft(
        id=call.id,
        name=call.name,
        input={"questions": [normalized_question]},
    )


def _first_ask_user_question(input_value: dict[str, Any]) -> Any:
    questions = input_value.get("questions")
    if isinstance(questions, list) and questions:
        return questions[0]
    return None


def _ask_user_text(first_question: Any, input_value: dict[str, Any]) -> str:
    if isinstance(first_question, str) and first_question.strip():
        return first_question.strip()
    if isinstance(first_question, dict):
        for key in ("question", "content", "text", "prompt", "description"):
            value = first_question.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("question", "content", "text", "prompt", "description"):
        value = input_value.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _ask_user_header(first_question: Any, input_value: dict[str, Any]) -> str:
    if isinstance(first_question, dict):
        value = first_question.get("header")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = input_value.get("header")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _ask_user_options(first_question: Any, input_value: dict[str, Any]) -> list[dict[str, str]]:
    raw_options: Any = None
    if isinstance(first_question, dict):
        raw_options = first_question.get("options")
    if raw_options is None:
        raw_options = input_value.get("options")
    return _normalized_ask_user_options(raw_options)


def _ask_user_multi_select(first_question: Any, input_value: dict[str, Any]) -> bool | None:
    if isinstance(first_question, dict) and isinstance(first_question.get("multiSelect"), bool):
        return first_question["multiSelect"]
    if isinstance(input_value.get("multiSelect"), bool):
        return input_value["multiSelect"]
    return None


def _normalized_ask_user_options(raw_options: Any) -> list[dict[str, str]]:
    if not isinstance(raw_options, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw_options[:8]:
        if isinstance(item, str) and item.strip():
            label = _shorten(item.strip(), 48)
            out.append({"label": label, "description": item.strip()})
        elif isinstance(item, dict):
            label_value = item.get("label") or item.get("text") or item.get("value") or item.get("content")
            if not isinstance(label_value, str) or not label_value.strip():
                continue
            description_value = item.get("description")
            label = _shorten(label_value.strip(), 48)
            description = description_value.strip() if isinstance(description_value, str) and description_value.strip() else label
            out.append({"label": label, "description": description})
    return out


def _pop_first_string_alias(input_value: dict[str, Any], aliases: tuple[str, ...]) -> str:
    for key in aliases:
        value = input_value.get(key)
        if isinstance(value, str) and value.strip():
            input_value.pop(key, None)
            return value.strip()
    return ""


def _schema_property_accepts_string(property_schema: Any) -> bool:
    if not isinstance(property_schema, dict):
        return True
    schema_type = property_schema.get("type")
    if isinstance(schema_type, list):
        return "string" in schema_type
    return schema_type in {None, "string"}


def _description_from_input(input_value: dict[str, Any]) -> str:
    for key in ("description", "prompt", "focus", "task", "query", "instruction", "instructions", "text"):
        value = input_value.get(key)
        if isinstance(value, str) and value.strip():
            return _shorten(_first_meaningful_line(value), 120)
    return ""


def _first_meaningful_line(text: str) -> str:
    for line in re.split(r"[\r\n]+", text or ""):
        cleaned = re.sub(r"\s+", " ", line).strip(" -*\t")
        if cleaned:
            return cleaned
    return ""


def _pattern_from_patterns(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, list):
        return ""
    patterns = [item.strip().replace("\\", "/") for item in value if isinstance(item, str) and item.strip()]
    if not patterns:
        return ""
    if len(patterns) == 1:
        return patterns[0]
    common_prefix = _common_glob_prefix(patterns)
    suffixes = [pattern[len(common_prefix) :] for pattern in patterns]
    if common_prefix and all(suffix and "/" not in suffix for suffix in suffixes):
        return f"{common_prefix}{{{','.join(suffixes)}}}"
    return patterns[0]


def _common_glob_prefix(patterns: list[str]) -> str:
    if not patterns:
        return ""
    prefix = patterns[0]
    for pattern in patterns[1:]:
        while prefix and not pattern.startswith(prefix):
            prefix = prefix[:-1]
    slash = prefix.rfind("/")
    if slash < 0:
        return ""
    return prefix[: slash + 1]


def _missing_required_tool_input_error(
    call: ToolCallDraft,
    canonical_name: str,
    specs_by_name: dict[str, ToolSpec],
) -> BridgeError | None:
    spec = specs_by_name.get(canonical_name)
    schema = spec.input_schema if spec and isinstance(spec.input_schema, dict) else {}
    required = schema.get("required") if isinstance(schema, dict) else None
    if not isinstance(required, list):
        return None
    missing = [str(name) for name in required if isinstance(name, str) and name not in call.input]
    if not missing:
        return None
    return BridgeError(
        "missing_required_tool_input",
        f"工具 {canonical_name} 缺少必需参数：{', '.join(missing)}",
        repairable=True,
    )


def _dedupe_tool_specs(tools: list[ToolSpec] | tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
    out: list[ToolSpec] = []
    seen: set[str] = set()
    for tool in tools:
        key = tool.name.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(tool)
    return tuple(out)


def _safe_replacement_for_hidden_readonly_shell_tool(
    call: ToolCallDraft,
    context: ToolBridgeContext,
) -> ToolCallDraft | None:
    if not context.tools:
        return None
    if _hidden_local_agent_shell_tool_error(call.name, context) is None:
        return None
    shell_tool = _hidden_shell_tool_for_call(call.name, context)
    if shell_tool is None:
        return None
    command_key = _shell_command_key(shell_tool)
    if isinstance(call.input, str):
        raw_command = call.input
        updated: dict[str, Any] = {}
    elif isinstance(call.input, dict):
        raw_command = ""
        for key in (command_key, "command", "cmd"):
            value = call.input.get(key)
            if isinstance(value, str) and value.strip():
                raw_command = value
                break
        updated = dict(call.input)
    else:
        return None
    command = _normalize_windows_paths_for_bash(_normalize_shell_command_syntax(raw_command))
    if not command or not _is_readonly_vcs_shell_command(command):
        return None
    updated.pop("command", None)
    updated.pop("cmd", None)
    updated[command_key] = command
    return ToolCallDraft(id=call.id, name=shell_tool.name, input=updated)


def _hidden_shell_tool_for_call(name: str, context: ToolBridgeContext) -> ToolSpec | None:
    candidates = [tool for tool in context.hidden_tools if _is_shell_execution_tool(tool, tool.name)]
    if not candidates:
        return None
    lowered = (name or "").strip().lower()
    compact = _compact_tool_name(name)
    for tool in candidates:
        if tool.name.strip().lower() == lowered or _compact_tool_name(tool.name) == compact:
            return tool
    if _is_bash_tool_name(name):
        bash_tool = _select_bash_tool(list(candidates))
        if bash_tool:
            return bash_tool
    if compact in {"shell", "terminal", "powershell", "cmd"}:
        return candidates[0]
    return None


def _is_readonly_vcs_shell_command(command: str) -> bool:
    segments = [segment.strip() for segment in _SHELL_COMMAND_SPLIT_RE.split(command or "") if segment.strip()]
    if not segments:
        return False
    saw_vcs = False
    for index, segment in enumerate(segments):
        if _shell_segment_has_unsafe_metacharacters(segment):
            return False
        words = _shell_words(segment)
        if not words:
            return False
        if _is_cd_shell_segment(words):
            if saw_vcs or index != 0:
                return False
            continue
        if _is_readonly_git_shell_segment(words):
            saw_vcs = True
            continue
        return False
    return saw_vcs


def _shell_segment_has_unsafe_metacharacters(segment: str) -> bool:
    if re.search(r"(?<!\\)(?:`|\$\()", segment or ""):
        return True
    words = _shell_words(segment)
    return any(word in {"|", "&", ">", ">>", "1>", "1>>", "2>", "2>>", "&>", "&>>"} for word in words)


def _is_cd_shell_segment(words: list[str]) -> bool:
    if not words or words[0].lower() != "cd":
        return False
    return len(words) == 2 and bool(words[1].strip().strip("\"'"))


def _is_readonly_git_shell_segment(words: list[str]) -> bool:
    if not words:
        return False
    normalized_words = ["git", *words[1:]] if words[0].lower() == "git.exe" else words
    if normalized_words[0].lower() != "git":
        return False
    lowered_words = [word.lower() for word in normalized_words]
    if _has_shell_output_redirection(lowered_words):
        return False
    subcommand = _git_subcommand(normalized_words)
    if subcommand not in _READONLY_GIT_SUBCOMMANDS:
        return False
    if any(word == "--output" or word.startswith("--output=") for word in lowered_words):
        return False
    args = _git_args_after_subcommand(normalized_words)
    if subcommand in {"branch", "tag"}:
        return _git_list_args_are_readonly(args)
    if subcommand == "remote":
        return _git_remote_args_are_readonly(args)
    return True


def _git_args_after_subcommand(words: list[str]) -> list[str]:
    index = _git_subcommand_index(words)
    if index < 0:
        return []
    return words[index + 1 :]


def _git_subcommand_index(words: list[str]) -> int:
    index = 1
    while index < len(words):
        word = words[index]
        lowered = word.lower()
        if word in _SHELL_OPTIONS_WITH_VALUES:
            index += 2
            continue
        if lowered.startswith("--git-dir=") or lowered.startswith("--work-tree=") or lowered.startswith("--namespace="):
            index += 1
            continue
        if word.startswith("-"):
            index += 1
            continue
        return index
    return -1


def _git_list_args_are_readonly(args: list[str]) -> bool:
    if not args:
        return True
    blocked = {"-d", "-D", "-f", "-m", "-M", "--delete", "--force", "--move"}
    for arg in args:
        lowered = arg.lower()
        if lowered in blocked or lowered.startswith("--delete=") or lowered.startswith("--force="):
            return False
        if not lowered.startswith("-"):
            return False
    return True


def _git_remote_args_are_readonly(args: list[str]) -> bool:
    if not args:
        return True
    blocked = {"add", "rename", "remove", "rm", "set-head", "set-branches", "set-url", "update", "prune"}
    lowered = [arg.lower() for arg in args]
    if lowered[0] in blocked:
        return False
    return all(arg.startswith("-") for arg in lowered)


def _safe_replacement_for_cli_tool(call: ToolCallDraft, tools: list[ToolSpec]) -> ToolCallDraft | None:
    if not _is_cli_tool_name(call.name):
        return None
    bash_tool = _select_bash_tool(tools)
    if not bash_tool:
        return None
    if isinstance(call.input, dict):
        command = _cli_tool_command(call.name, call.input)
    elif isinstance(call.input, str):
        command = _cli_tool_string_command(call.name, call.input)
    else:
        return None
    if not command:
        return None
    return ToolCallDraft(
        id=call.id,
        name=bash_tool.name,
        input={_shell_command_key(bash_tool): _normalize_windows_paths_for_bash(command)},
    )


def _safe_replacement_for_delegation_tool(call: ToolCallDraft, tools: list[ToolSpec]) -> ToolCallDraft | None:
    if not isinstance(call.input, dict):
        return None
    requested = _compact_tool_name(call.name)
    alias_groups = (
        {"agent", "subagent", "subagenttask", "task", "explore", "explorer"},
        {"todowrite", "todo", "updatetodo", "todo_write"},
    )
    alias_group = next((group for group in alias_groups if requested in group), None)
    if not alias_group:
        return None
    candidates = [_tool for _tool in tools if _compact_tool_name(_tool.name) in alias_group]
    if not candidates:
        return None
    target = _best_delegation_target(call.input, candidates)
    if target is None:
        return None
    return ToolCallDraft(id=call.id, name=target.name, input=dict(call.input))


def _best_delegation_target(input_value: dict[str, Any], candidates: list[ToolSpec]) -> ToolSpec | None:
    scored: list[tuple[int, int, ToolSpec]] = []
    for index, tool in enumerate(candidates):
        score = 0
        schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if properties:
            score += sum(1 for key in input_value if key in properties)
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        if all(isinstance(key, str) and key in input_value for key in required):
            score += 4
        scored.append((score, -index, tool))
    if not scored:
        return None
    scored.sort(key=lambda item: item[:2], reverse=True)
    return scored[0][2]


def _compact_tool_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())


def _has_code_change_tool(context: ToolBridgeContext) -> bool:
    code_change_tools = {
        "write",
        "writefile",
        "edit",
        "editfile",
        "multiedit",
        "notebookedit",
        "applypatch",
        "patch",
        "bash",
    }
    for tool in context.tools:
        compact = _compact_tool_name(tool.name)
        if compact in code_change_tools:
            return True
        if _is_shell_execution_tool(tool, tool.name):
            return True
    return False


def _safe_replacement_for_file_read_tool(call: ToolCallDraft, tools: list[ToolSpec]) -> ToolCallDraft | None:
    if not isinstance(call.input, dict):
        return None
    requested = _compact_tool_name(call.name)
    if requested not in {"readfile", "fileread", "read"}:
        return None
    read_tool = _select_file_read_tool(tools)
    if not read_tool:
        return None
    return ToolCallDraft(id=call.id, name=read_tool.name, input=dict(call.input))


def _select_file_read_tool(tools: list[ToolSpec]) -> ToolSpec | None:
    preferred = [tool for tool in tools if _compact_tool_name(tool.name) in {"read", "readfile", "fileread"}]
    if preferred:
        return preferred[0]
    return None


def _safe_replacement_for_directory_read_tool(
    call: ToolCallDraft,
    canonical_name: str,
    tools: list[ToolSpec],
) -> ToolCallDraft | None:
    if _compact_tool_name(canonical_name) not in {"read", "readfile", "fileread"}:
        return None
    if not isinstance(call.input, dict):
        return None
    path = _tool_path_from_input(call.input)
    if not _read_target_looks_like_directory(path, call.input):
        return None
    glob_tool = _select_glob_tool(tools)
    if not glob_tool:
        return None
    return ToolCallDraft(
        id=call.id,
        name=glob_tool.name,
        input={"path": path.rstrip("\\/") or ".", "pattern": "*"},
    )


def _read_target_looks_like_directory(path: str, input_value: dict[str, Any]) -> bool:
    raw = (path or "").strip().strip("\"'")
    if not raw:
        return False
    if any(key in input_value for key in ("directory", "dir_path", "folder")):
        return True
    if raw in {".", "./", ".\\", "/", "\\"}:
        return True
    if raw.endswith(("/", "\\")):
        return True
    normalized = raw.replace("\\", "/").rstrip("/")
    leaf = normalized.rsplit("/", 1)[-1].lower()
    if not leaf or leaf in {".", ".."}:
        return True
    if leaf in _COMMON_EXTENSIONLESS_FILE_NAMES:
        return False
    if "." in leaf:
        return False
    return bool(re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("/"))


_COMMON_EXTENSIONLESS_FILE_NAMES = {
    "dockerfile",
    "license",
    "makefile",
    "notice",
    "procfile",
    "readme",
}


def _safe_replacement_for_file_discovery_tool(call: ToolCallDraft, tools: list[ToolSpec]) -> ToolCallDraft | None:
    if not _is_file_discovery_alias(call.name) or not isinstance(call.input, dict):
        return None
    glob_tool = _select_glob_tool(tools)
    if not glob_tool:
        return None
    if _is_directory_listing_alias(call.name):
        normalized = _normalized_directory_listing_input(call.input)
        return ToolCallDraft(id=call.id, name=glob_tool.name, input=normalized)
    normalized = _normalized_glob_input(call.input, recursive_alias=True)
    if not normalized.get("pattern"):
        return None
    return ToolCallDraft(id=call.id, name=glob_tool.name, input=normalized)


def _is_file_discovery_alias(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())
    return compact in {"dir", "filesearch", "findfiles", "ls", "list", "listdir", "listdirectory", "listfiles", "searchfiles", "filefinder", "tree"}


def _is_directory_listing_alias(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", (name or "").strip().lower())
    return compact in {"dir", "ls", "list", "listdir", "listdirectory", "tree"}


def _normalized_directory_listing_input(input_value: dict[str, Any]) -> dict[str, Any]:
    scope = _glob_scope(input_value)
    pattern = _glob_pattern(input_value) or "*"
    normalized: dict[str, Any] = {"pattern": pattern}
    if scope:
        normalized["path"] = scope.rstrip("\\/")
    return normalized


def _safe_replacement_for_shell_string_input(call: ToolCallDraft, canonical_name: str, tools: list[ToolSpec]) -> ToolCallDraft | None:
    if not isinstance(call.input, str):
        return None
    shell_tool = _tool_by_name(tools, canonical_name)
    if not _is_shell_execution_tool(shell_tool, canonical_name):
        return None
    return ToolCallDraft(
        id=call.id,
        name=call.name,
        input={_shell_command_key(shell_tool) if shell_tool else "command": _normalize_windows_paths_for_bash(call.input.strip())},
    )


def _normalize_shell_tool_command(call: ToolCallDraft, canonical_name: str, tools: list[ToolSpec]) -> ToolCallDraft:
    shell_tool = _tool_by_name(tools, canonical_name)
    if not _is_shell_execution_tool(shell_tool, canonical_name):
        return call
    command_key = _shell_command_key(shell_tool) if shell_tool else "command"
    command = call.input.get(command_key)
    if not isinstance(command, str):
        return call
    normalized = _normalize_shell_command_syntax(command)
    normalized = _normalize_windows_paths_for_bash(normalized)
    if _is_bash_tool_name(canonical_name):
        normalized = _normalize_cmd_cd_drive_switch_for_bash(normalized)
    if normalized == command:
        return call
    updated = dict(call.input)
    updated[command_key] = normalized
    return ToolCallDraft(id=call.id, name=call.name, input=updated)


def _normalize_shell_command_syntax(command: str) -> str:
    normalized = html.unescape(command or "").strip()
    # Web models often emit shell connectors as entities or duplicate/trailing tokens.
    # Keep command semantics intact; only remove syntax that would create empty segments.
    for _ in range(3):
        previous = normalized
        normalized = re.sub(r"\s*(&&|\|\||;)\s*(?:\1\s*)+", r" \1 ", normalized).strip()
        normalized = re.sub(r"\s*(?:&&|\|\||;)\s*$", "", normalized).strip()
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()
        if normalized == previous:
            break
    return normalized


def _normalize_cmd_cd_drive_switch_for_bash(command: str) -> str:
    return re.sub(r"(?i)(^|(?:&&|\|\||;)\s*)cd\s+/d\s+", r"\1cd ", command or "").strip()


def _safe_replacement_for_windows_dir_shell_command(
    call: ToolCallDraft,
    canonical_name: str,
    tools: list[ToolSpec],
) -> ToolCallDraft | None:
    shell_tool = _tool_by_name(tools, canonical_name)
    if not _is_shell_execution_tool(shell_tool, canonical_name):
        return None
    command_key = _shell_command_key(shell_tool) if shell_tool else "command"
    command = call.input.get(command_key)
    if not isinstance(command, str):
        return None
    replacement = _windows_dir_recursive_replacement(command)
    if not replacement:
        return None

    glob_tool = _select_glob_tool(tools)
    if glob_tool:
        glob_input: dict[str, Any] = {"pattern": replacement["pattern"]}
        if replacement["path"]:
            glob_input["path"] = replacement["path"]
        return ToolCallDraft(id=call.id, name=glob_tool.name, input=glob_input)

    if not _is_bash_tool_name(canonical_name):
        return None
    path = replacement["path"] or "."
    command_path = shlex.quote(path)
    find_command = f"find {command_path} -maxdepth 3 -type f"
    if replacement["pattern"] not in {"*", "**/*"}:
        find_command += f" -name {shlex.quote(replacement['pattern'].removeprefix('**/'))}"
    updated = dict(call.input)
    updated[command_key] = find_command
    return ToolCallDraft(id=call.id, name=call.name, input=updated)


def _windows_dir_recursive_replacement(command: str) -> dict[str, str] | None:
    segments = [segment.strip() for segment in _SHELL_COMMAND_SPLIT_RE.split(command or "") if segment.strip()]
    if len(segments) != 1:
        return None
    words = _shell_words(segments[0])
    if len(words) < 2 or words[0].lower() != "dir":
        return None
    lowered_args = [word.lower() for word in words[1:]]
    if not any(arg in {"/s", "/b"} for arg in lowered_args):
        return None
    non_options = [word for word in words[1:] if not word.startswith(("/", "-"))]
    raw_path = non_options[0] if non_options else "."
    raw_path = raw_path.strip().strip("\"'") or "."
    path = _windows_path_to_bash_path(raw_path)
    pattern = "*"
    if _contains_glob_magic(path):
        scope, embedded_pattern = _split_embedded_glob_path(path)
        path = scope or "."
        pattern = embedded_pattern or "*"
    return {"path": path.rstrip("\\/"), "pattern": pattern.replace("\\", "/") or "*"}


def _shell_tool_command_error(call: ToolCallDraft, canonical_name: str, context: ToolBridgeContext) -> BridgeError | None:
    shell_tool = _tool_by_name(context.tools, canonical_name) or _tool_by_name(list(context.hidden_tools), canonical_name)
    if not _is_shell_execution_tool(shell_tool, canonical_name):
        return None
    command_key = _shell_command_key(shell_tool) if shell_tool else "command"
    command = call.input.get(command_key)
    if not isinstance(command, str) or not command.strip():
        return BridgeError("incomplete_shell_command", "Shell command is empty; provide a complete command string.", repairable=True)
    if _allows_ds2api_style_shell_passthrough(context):
        return None
    return (
        _invalid_shell_dialect_error(command, canonical_name)
        or _invalid_shell_command_artifact_error(command)
        or _incomplete_shell_command_error(command)
        or _unsafe_contextual_shell_command_error(command, context)
    )


def _is_cli_tool_name(name: str) -> bool:
    lowered = (name or "").strip().lower()
    return lowered in {
        "bun",
        "cargo",
        "deno",
        "docker",
        "docker-compose",
        "exec",
        "execute",
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


def _select_shell_execution_tool(tools: list[ToolSpec]) -> ToolSpec | None:
    bash_tool = _select_bash_tool(tools)
    if bash_tool:
        return bash_tool
    for tool in tools:
        if _is_shell_execution_tool(tool, tool.name):
            return tool
    return None


def _tool_by_name(tools: list[ToolSpec], name: str) -> ToolSpec | None:
    return next((tool for tool in tools if tool.name == name), None)


def _is_shell_execution_tool(tool: ToolSpec | None, name: str) -> bool:
    lowered = (name or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if compact in {"bash", "exec", "execute", "shell", "terminal", "command", "runcommand", "powershell", "cmd"}:
        return True
    if not tool or not _has_shell_command_property(tool):
        return False
    description = (tool.description or "").lower()
    return any(marker in description for marker in ("shell", "terminal", "command", "execute", "run command", "bash", "powershell", "cmd"))


def _has_shell_command_property(tool: ToolSpec) -> bool:
    properties = tool.input_schema.get("properties") if isinstance(tool.input_schema, dict) else None
    return isinstance(properties, dict) and any(key in properties for key in ("command", "cmd"))


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


def _cli_tool_string_command(name: str, input_value: str) -> str:
    command_name = name.strip()
    command = input_value.strip()
    if not command:
        return command_name
    if _is_shell_wrapper_tool_name(command_name):
        return command
    return command if _command_starts_with_cli_name(command, command_name) else f"{command_name} {command}"


def _is_shell_wrapper_tool_name(name: str) -> bool:
    lowered = (name or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    return compact in {"exec", "execute", "shell", "terminal", "command", "runcommand"}


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
    return re.sub(r"/+", "/", path.replace("\\", "/"))


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
        if len(words) >= 2 and words[0].lower() == "git" and words[1].lower() == "clone":
            meaningful_args = _meaningful_git_clone_args(words[2:])
            if not meaningful_args:
                return BridgeError(
                    "incomplete_shell_command",
                    "git clone is missing the repository URL. Include the repo URL and destination path when needed.",
                    repairable=True,
                )
    return None


def _invalid_shell_command_artifact_error(command: str) -> BridgeError | None:
    for segment in _SHELL_COMMAND_SPLIT_RE.split(command):
        stripped = segment.strip()
        if not stripped:
            continue
        words = _shell_words(stripped)
        if not words:
            continue
        command_word = words[0].strip()
        if _looks_like_package_requirement_command(command_word):
            return BridgeError(
                "invalid_shell_command_artifact",
                "The Bash command starts with a package requirement token, not an executable command. Do not convert dependency list text into shell execution.",
                repairable=True,
            )
        if _looks_like_code_invocation_command(command_word):
            return BridgeError(
                "invalid_shell_command_artifact",
                "The Bash command starts with source-code syntax, not an executable command. Do not convert test code or snippets into shell execution.",
                repairable=True,
            )
        if len(words) >= 2 and words[0] == "git" and words[1] == "clone":
            meaningful_args = _meaningful_git_clone_args(words[2:])
            if any(_looks_like_placeholder_arg(arg) for arg in meaningful_args):
                return BridgeError(
                    "invalid_shell_command_artifact",
                    "git clone contains a placeholder repository URL. Use a real URL only when the user explicitly asked to clone a repository.",
                    repairable=True,
                )
    return None


def _unsafe_contextual_shell_command_error(command: str, context: ToolBridgeContext) -> BridgeError | None:
    review_error = _unsafe_readonly_review_shell_command_error(command, context)
    if review_error:
        return review_error
    update_clone_error = _unsafe_update_existing_repo_clone_error(command, context)
    if update_clone_error:
        return update_clone_error
    local_error = _unsafe_local_agent_shell_command_error(command, context)
    if local_error:
        return local_error
    unanchored_error = _unsafe_unanchored_shell_command_error(command, context)
    if unanchored_error:
        return unanchored_error
    return None


def _unsafe_update_existing_repo_clone_error(command: str, context: ToolBridgeContext) -> BridgeError | None:
    if not _looks_like_update_existing_repo_task(context.task_text):
        return None
    task_paths = {_normalize_path_for_compare(match.group("path")) for match in _WINDOWS_DRIVE_PATH_RE.finditer(context.task_text)}
    if not task_paths:
        return None
    for segment in _SHELL_COMMAND_SPLIT_RE.split(command):
        words = _shell_words(segment.strip())
        if len(words) >= 2 and words[0] == "git" and words[1] == "clone":
            meaningful_args = _meaningful_git_clone_args(words[2:])
            if len(meaningful_args) >= 2 and _normalize_path_for_compare(meaningful_args[-1]) in task_paths:
                return BridgeError(
                    "unsafe_clone_to_requested_local_path",
                    "git clone targets the local path named by the task. Inspect the existing local repository first with git -C <path> remote -v and git -C <path> status before cloning or replacing it.",
                    repairable=True,
                )
    return None


def _unsafe_local_agent_shell_command_error(command: str, context: ToolBridgeContext) -> BridgeError | None:
    task = context.task_text or ""
    if not task or not _looks_like_local_agent_task(task):
        return None
    for segment in _SHELL_COMMAND_SPLIT_RE.split(command):
        words = _shell_words(segment.strip())
        if not words:
            continue
        if _is_destructive_review_shell_segment(words) and not _task_explicitly_allows_shell_segment(words, task):
            return BridgeError(
                "unsafe_local_shell_command",
                "This local-agent task does not explicitly authorize destructive repository or filesystem shell commands. Use read-only inspection or structured Edit/Write tools first.",
                repairable=False,
            )
        if _is_mutating_review_setup_segment(words) and not _task_explicitly_allows_shell_segment(words, task):
            return BridgeError(
                "unsafe_local_shell_command",
                "This local-agent task does not explicitly authorize dependency installation, environment setup, arbitrary script execution, or auto-fix shell commands. Use structured code-edit tools or ask for confirmation.",
                repairable=False,
            )
        if _is_broad_test_shell_segment(words) and not _task_explicitly_allows_shell_segment(words, task):
            return BridgeError(
                "unsafe_local_shell_command",
                "This local-agent task does not explicitly request broad test execution. Use targeted inspection or ask for verification permission before running test suites.",
                repairable=False,
            )
    return None


def _unsafe_unanchored_shell_command_error(command: str, context: ToolBridgeContext) -> BridgeError | None:
    if not _shell_command_has_high_risk_segment(command):
        return None
    task = (context.task_text or "").strip()
    if _task_explicitly_allows_high_risk_shell(command, task):
        return None
    if task and not _looks_like_continuation_task(task):
        return None
    return BridgeError(
        "unsafe_shell_command_requires_explicit_task",
        "High-risk shell commands such as dependency installation, branch changes, broad test suites, generated test creation, or arbitrary script execution require an explicit current user task. Use read-only inspection commands first, or ask the user to confirm the execution task.",
        repairable=False,
    )


def _unsafe_readonly_review_shell_command_error(command: str, context: ToolBridgeContext) -> BridgeError | None:
    task = context.task_text or ""
    if not _looks_like_readonly_review_task(task):
        return None
    for segment in _SHELL_COMMAND_SPLIT_RE.split(command):
        words = _shell_words(segment.strip())
        if not words:
            continue
        if _is_destructive_review_shell_segment(words) and not _task_explicitly_allows_shell_segment(words, task):
            return BridgeError(
                "unsafe_review_shell_command",
                "This is a read-only review/analysis task. Do not request destructive repository or filesystem commands such as reset, clean, clone, commit, push, rm, or mv unless the user explicitly asks for updates or modifications. Use read-only inspection commands first.",
                repairable=False,
            )
        if _is_mutating_review_setup_segment(words) and not _task_explicitly_allows_shell_segment(words, task):
            return BridgeError(
                "unsafe_review_shell_command",
                "This is a read-only review/analysis task. Do not install dependencies, create virtual environments, or run auto-fix commands unless the user explicitly asks for setup or modifications. Inspect existing files and report findings first.",
                repairable=False,
            )
        if _is_broad_test_shell_segment(words) and not _task_explicitly_allows_shell_segment(words, task):
            return BridgeError(
                "unsafe_review_shell_command",
                "This is a review/analysis task, not a verification task. Do not run broad test suites unless the user explicitly asks for tests or verification. Inspect targeted files and report findings instead.",
                repairable=False,
            )
    return None


def _shell_command_has_high_risk_segment(command: str) -> bool:
    for segment in _SHELL_COMMAND_SPLIT_RE.split(command):
        words = _shell_words(segment.strip())
        if words and _is_high_risk_shell_segment(words):
            return True
    return False


def _task_explicitly_allows_high_risk_shell(command: str, task: str) -> bool:
    task = task or ""
    if not task or _looks_like_continuation_task(task):
        return False
    for segment in _SHELL_COMMAND_SPLIT_RE.split(command):
        words = _shell_words(segment.strip())
        if not words or not _is_high_risk_shell_segment(words):
            continue
        if _task_explicitly_allows_shell_segment(words, task):
            continue
        return False
    return True


def _task_explicitly_allows_shell_segment(words: list[str], task: str) -> bool:
    if _is_broad_test_shell_segment(words):
        return _task_explicitly_allows_tests(task)
    if _is_mutating_review_setup_segment(words):
        return _task_explicitly_allows_shell_setup(task)
    if _is_destructive_review_shell_segment(words):
        return _task_explicitly_allows_destructive_shell_segment(words, task)
    return True


def _task_explicitly_allows_shell_setup(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _SHELL_SETUP_TASK_MARKERS)


def _task_explicitly_allows_destructive_shell_segment(words: list[str], text: str) -> bool:
    lowered = (text or "").lower()
    command = words[0].lower() if words else ""
    if command == "git":
        subcommand = _git_subcommand(words)
        if subcommand == "pull":
            return _looks_like_update_existing_repo_task(text) or any(
                marker in lowered for marker in _SHELL_GIT_UPDATE_TASK_MARKERS
            ) or any(marker in lowered for marker in _SHELL_DESTRUCTIVE_TASK_MARKERS)
        if subcommand in {"checkout", "switch"}:
            return "branch" in lowered or "分支" in lowered or any(marker in lowered for marker in _SHELL_DESTRUCTIVE_TASK_MARKERS)
        if subcommand in {"reset", "clean", "rm", "restore"}:
            return any(marker in lowered for marker in _SHELL_DESTRUCTIVE_TASK_MARKERS)
        if subcommand in {"clone", "commit", "push", "merge", "rebase", "stash", "filter-repo", "filter-branch"}:
            return any(marker in lowered for marker in _SHELL_DESTRUCTIVE_TASK_MARKERS)
    return any(marker in lowered for marker in _SHELL_DESTRUCTIVE_TASK_MARKERS)


def _is_high_risk_shell_segment(words: list[str]) -> bool:
    return (
        _is_destructive_review_shell_segment(words)
        or _is_mutating_review_setup_segment(words)
        or _is_broad_test_shell_segment(words)
    )


def _allows_plain_review_or_plan_text(context: ToolBridgeContext) -> bool:
    task = context.task_text or ""
    if not _looks_like_readonly_review_task(task):
        return False
    return not _task_explicitly_allows_mutation(task)


def _looks_like_readonly_review_task(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _REVIEW_TASK_MARKERS)


def _task_explicitly_allows_mutation(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _MUTATING_TASK_MARKERS)


def _task_explicitly_allows_tests(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _TEST_TASK_MARKERS)


def _is_destructive_review_shell_segment(words: list[str]) -> bool:
    command = words[0].lower()
    if command in _REVIEW_BLOCKED_SHELL_COMMANDS:
        return True
    if command != "git":
        return False
    subcommand = _git_subcommand(words)
    return subcommand in _REVIEW_BLOCKED_GIT_SUBCOMMANDS


def _is_mutating_review_setup_segment(words: list[str]) -> bool:
    if not words:
        return False
    command = words[0].lower()
    lowered_words = [word.lower() for word in words]
    if _has_shell_output_redirection(lowered_words):
        return True
    if command in {"python", "python3", "py", "node", "npx"}:
        return True
    if command in {"tee", "curl", "wget"}:
        return True
    if "sed" in lowered_words and _has_any_word(lowered_words, {"-i", "--in-place"}):
        return True
    if "perl" in lowered_words and any(word.startswith("-pi") for word in lowered_words):
        return True
    if command == "sed" and _has_any_word(lowered_words[1:], {"-i", "--in-place"}):
        return True
    if command == "perl" and any(word.startswith("-pi") for word in lowered_words[1:]):
        return True
    if command == "xargs" and "sed" in lowered_words and _has_any_word(lowered_words, {"-i", "--in-place"}):
        return True
    if command in _REVIEW_INSTALL_COMMANDS:
        return _has_any_word(words[1:], {"install", "add", "sync", "update", "upgrade"})
    if command in _REVIEW_PACKAGE_MANAGER_COMMANDS:
        return len(words) == 1 or _has_any_word(words[1:], {"add", "install", "update", "upgrade"})
    if command in {"python", "python3", "py"} and len(words) >= 3 and words[1] == "-m":
        module = words[2].lower()
        if module == "venv":
            return True
        if module == "pip":
            return _has_any_word(words[3:], {"install", "add", "sync", "update", "upgrade"})
    if command in {"virtualenv", "ruff", "pre-commit"}:
        return command == "virtualenv" or _has_any_word(words[1:], {"--fix", "--unsafe-fixes", "install"})
    return False


def _has_shell_output_redirection(words: list[str]) -> bool:
    for word in words:
        if word in {">", ">>", "1>", "1>>", "2>", "2>>", "&>", "&>>"}:
            return True
        if re.match(r"^(?:[12]?>>?|&>>?).+", word):
            return True
    return False


def _has_any_word(words: list[str], needles: set[str]) -> bool:
    lowered = {word.lower() for word in words}
    return any(needle in lowered for needle in needles)


def _looks_like_package_requirement_command(word: str) -> bool:
    value = (word or "").strip()
    if not value:
        return False
    return bool(re.match(r"^[A-Za-z0-9_.-]+(?:===|==|!=|~=|>=|<=|>|<).+$", value))


def _looks_like_code_invocation_command(word: str) -> bool:
    value = (word or "").strip()
    if not value:
        return False
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_.]*\(.+\)$", value))


def _looks_like_placeholder_arg(arg: str) -> bool:
    value = (arg or "").strip().strip("\"'")
    if not value:
        return False
    lowered = value.lower()
    if re.fullmatch(r"<[^>]+>", value):
        return True
    if re.fullmatch(r"\{[^{}]+\}", value):
        return True
    if re.fullmatch(r"\[[^\[\]]+\]", value):
        return True
    return lowered in {"repository-url", "repo-url", "your-repo-url", "your-repository-url", "example-repo-url"}


def _invalid_shell_dialect_error(command: str, canonical_name: str) -> BridgeError | None:
    if not _is_bash_tool_name(canonical_name):
        return None
    for segment in _SHELL_COMMAND_SPLIT_RE.split(command):
        words = _shell_words(segment.strip())
        if len(words) >= 2 and words[0].lower() == "dir" and _has_any_word(words[1:], {"/s", "/b"}):
            return BridgeError(
                "invalid_shell_dialect",
                "Bash does not support Windows cmd.exe flags such as 'dir /s'. Use a Bash-compatible read-only command such as 'find <path> -maxdepth 3 -type f' or 'rg --files <path>'.",
                repairable=True,
            )
    return None


def _git_subcommand(words: list[str]) -> str:
    index = 1
    while index < len(words):
        word = words[index]
        lowered = word.lower()
        if word in _SHELL_OPTIONS_WITH_VALUES:
            index += 2
            continue
        if lowered.startswith("--git-dir=") or lowered.startswith("--work-tree=") or lowered.startswith("--namespace="):
            index += 1
            continue
        if word.startswith("-"):
            index += 1
            continue
        return lowered
    return ""


def _is_broad_test_shell_segment(words: list[str]) -> bool:
    if not words:
        return False
    command = words[0].lower()
    if command in {"pytest", "py.test"}:
        return True
    if command in {"python", "python3", "py"} and len(words) >= 3 and words[1] == "-m" and words[2].lower() == "pytest":
        return True
    if command in {"npm", "pnpm", "yarn"} and len(words) >= 2:
        return words[1].lower() == "test" or (words[1].lower() == "run" and len(words) >= 3 and words[2].lower() == "test")
    return False


def _looks_like_update_existing_repo_task(text: str) -> bool:
    return _update_existing_repo_path_match(text) is not None


def _update_existing_repo_path_match(text: str) -> re.Match[str] | None:
    raw = text or ""
    for match in _WINDOWS_DRIVE_PATH_RE.finditer(raw):
        if not _windows_path_looks_like_directory(match.group("path")):
            continue
        if _has_update_marker_near(raw, match.start(), match.end()):
            return match
    return None


def _windows_path_looks_like_directory(path: str) -> bool:
    normalized = (path or "").strip().strip("\"'").replace("\\", "/").rstrip("/")
    leaf = normalized.rsplit("/", 1)[-1]
    return not bool(re.search(r"\.[A-Za-z0-9]{1,12}$", leaf))


def _has_update_marker_near(text: str, start: int, end: int) -> bool:
    window = (text or "")[max(0, start - 96) : min(len(text or ""), end + 96)].lower()
    return any(marker in window for marker in ("更新", "update", "pull", "sync", "upgrade"))


def _normalize_path_for_compare(path: str) -> str:
    return (path or "").strip().strip("\"'").replace("\\", "/").rstrip("/").lower()


def _tool_path_from_input(input_value: dict[str, Any]) -> str:
    for key in ("file_path", "filepath", "path", "file", "filename"):
        value = input_value.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _paths_match_for_missing_read(candidate: str, failed_path: str) -> bool:
    left = _normalize_path_for_compare(candidate)
    right = _normalize_path_for_compare(failed_path)
    if not left or not right:
        return False
    if left == right or left.endswith(f"/{right}") or right.endswith(f"/{left}"):
        return True
    failed_parent = right.rsplit("/", 1)[0] if "/" in right else ""
    return bool(failed_parent and _path_same_or_under(left, failed_parent))


def _path_same_or_under(candidate: str, failed_path: str) -> bool:
    left = _normalize_path_for_compare(candidate)
    right = _normalize_path_for_compare(failed_path)
    if not left or not right:
        return False
    if left == right or left.startswith(f"{right}/") or right.endswith(f"/{left}"):
        return True
    tail = right.rsplit("/", 1)[-1]
    return bool(tail and (left == tail or left.startswith(f"{tail}/") or left.endswith(f"/{tail}") or f"/{tail}/" in left))


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
    if not _should_sanitize_repository_wide_glob(scope, pattern):
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


def _safe_replacement_for_glob_input_shape(
    call: ToolCallDraft,
    canonical_name: str,
    tools: list[ToolSpec],
) -> ToolCallDraft | None:
    if not _is_glob_tool_name(canonical_name):
        return None
    normalized = _normalized_glob_input(call.input)
    if normalized == call.input:
        return None
    glob_tool = _tool_by_name(tools, canonical_name)
    return ToolCallDraft(id=call.id, name=glob_tool.name if glob_tool else canonical_name, input=normalized)


def _expensive_glob_message(name: str, input_value: dict[str, Any]) -> str | None:
    if not _is_glob_tool_name(name):
        return None
    pattern = _glob_pattern(input_value)
    if not pattern:
        return None
    if not _is_repository_wide_glob(pattern):
        return None
    scope = _glob_scope(input_value)
    if not _should_sanitize_repository_wide_glob(scope, pattern):
        return None
    return (
        f"Glob pattern {pattern!r} is repository-wide and likely to time out on large workspaces. "
        "Use Read for known files, LS/list tools for directory overviews, or a scoped Glob pattern with a narrow path."
    )


def _should_sanitize_repository_wide_glob(scope: str, pattern: str) -> bool:
    if not scope or _is_rootish_glob_scope(scope):
        return True
    if _is_all_files_glob(pattern):
        return True
    return False


def _is_glob_tool_name(name: str) -> bool:
    lowered = (name or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    return lowered == "glob" or compact.endswith("glob")


def _select_glob_tool(tools: list[ToolSpec]) -> ToolSpec | None:
    return next((tool for tool in tools if _is_glob_tool_name(tool.name)), None)


def _normalized_glob_input(input_value: dict[str, Any], *, recursive_alias: bool = False) -> dict[str, Any]:
    raw_pattern = _glob_pattern(input_value)
    raw_scope = _glob_scope(input_value)
    scope = raw_scope
    pattern = raw_pattern

    if not pattern and raw_scope and _contains_glob_magic(raw_scope):
        scope, pattern = _split_embedded_glob_path(raw_scope)
    elif pattern and not raw_scope and _contains_glob_magic(pattern):
        embedded_scope, embedded_pattern = _split_embedded_glob_path(pattern)
        if embedded_scope and embedded_pattern:
            scope, pattern = embedded_scope, embedded_pattern

    if pattern:
        pattern = _normalize_glob_pattern(pattern, recursive=recursive_alias)

    normalized: dict[str, Any] = {}
    if scope and scope != pattern:
        normalized["path"] = scope.rstrip("\\/")
    if pattern:
        normalized["pattern"] = pattern.replace("\\", "/")

    if not normalized:
        return dict(input_value)
    return normalized


def _contains_glob_magic(value: str) -> bool:
    return any(char in (value or "") for char in ("*", "?", "["))


def _split_embedded_glob_path(value: str) -> tuple[str, str]:
    raw = (value or "").strip()
    first_magic = min((idx for idx in (raw.find("*"), raw.find("?"), raw.find("[")) if idx >= 0), default=-1)
    if first_magic < 0:
        return raw, ""
    slash_index = max(raw.rfind("\\", 0, first_magic), raw.rfind("/", 0, first_magic))
    if slash_index < 0:
        return "", raw
    return raw[:slash_index], raw[slash_index + 1 :]


def _normalize_glob_pattern(pattern: str, *, recursive: bool = False) -> str:
    raw = (pattern or "").strip().replace("\\", "/")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) > 1:
        extensions = []
        for part in parts:
            match = re.fullmatch(r"\*\.([A-Za-z0-9_+-]+)", part)
            if not match:
                return raw
            extensions.append(match.group(1))
        prefix = "**/" if recursive else ""
        return f"{prefix}*.{{{','.join(extensions)}}}"
    if recursive and raw and "/" not in raw and not raw.startswith("**/"):
        return f"**/{raw}"
    return raw


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


def _is_rootish_glob_scope(scope: str) -> bool:
    normalized = (scope or "").replace("\\", "/").strip()
    return normalized in {".", "./", "/", ""}


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
        args = parsed if isinstance(parsed, dict) else args.strip()
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return ToolCallDraft(id=str(item.get("id") or item.get("tool_call_id") or ""), name=name.strip(), input=args)  # type: ignore[arg-type]
    return ToolCallDraft(id=str(item.get("id") or item.get("tool_call_id") or ""), name=name.strip(), input=args)


def _render_dsml_tool_calls_for_prompt(tool_calls: list[Any]) -> str:
    lines = ["<|DSML|tool_calls>"]
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = str(fn.get("name") or item.get("name") or "").strip()
        if not name:
            continue
        args = fn.get("arguments") or item.get("args") or item.get("input") or {}
        if isinstance(args, str):
            parsed = _loads(args)
            args = parsed if isinstance(parsed, dict) else {}
        if not isinstance(args, dict):
            args = {}
        lines.append(f'  <|DSML|invoke name="{html.escape(name, quote=True)}">')
        for key, value in args.items():
            lines.extend(_render_dsml_parameter(str(key), value, indent="    "))
        lines.append("  </|DSML|invoke>")
    lines.append("</|DSML|tool_calls>")
    return "\n".join(lines)


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
        suffix = _render_dsml_tool_calls_for_prompt(message["tool_calls"])
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
