"""Strict, renderer-backed validation for GLM-5.2 SFT JSONL datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

FATAL = "fatal"
WARNING = "warning"

_ROOT_KEYS = {"messages", "tools", "chat_template_kwargs", "meta"}
_MESSAGE_KEYS = {
    "system": {"role", "content"},
    "user": {"role", "content"},
    "assistant": {"role", "content", "reasoning_content", "tool_calls"},
    "tool": {"role", "content", "tool_call_id", "name"},
}
_MEDIA_TYPES = {
    "image",
    "image_url",
    "video",
    "video_url",
    "audio",
    "audio_url",
    "input_audio",
}
_PROTOCOL_MARKERS = (
    "[gMASK]<sop>",
    "[gMASK]",
    "<sop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
    "<tools>",
    "</tools>",
)


class EnvironmentValidationError(RuntimeError):
    """A validation prerequisite is missing or unusable."""


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class Finding:
    """One actionable dataset finding."""

    severity: Literal["fatal", "warning"]
    layer: str
    rule: str
    line: int
    record_id: str
    path: str
    message: str
    fix: str

    def render(self) -> str:
        return (
            f"[{self.layer}:{self.rule}] line {self.line} id={self.record_id} "
            f"path={self.path}\n"
            f"  现象: {self.message}\n"
            f"  修复: {self.fix}"
        )


@dataclass
class ValidationReport:
    """Complete result of validating one JSONL file."""

    data_path: str
    total_records: int = 0
    rendered_records: int = 0
    trainable_records: int = 0
    truncated_records: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def fatal_count(self) -> int:
        return sum(item.severity == FATAL for item in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(item.severity == WARNING for item in self.findings)

    @property
    def exit_code(self) -> int:
        if self.fatal_count or self.trainable_records == 0:
            return 1
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_path": self.data_path,
            "status": "failed" if self.exit_code else "passed",
            "total_records": self.total_records,
            "rendered_records": self.rendered_records,
            "trainable_records": self.trainable_records,
            "truncated_records": self.truncated_records,
            "fatal_count": self.fatal_count,
            "warning_count": self.warning_count,
            "findings": [asdict(item) for item in self.findings],
        }


class _RecordValidator:
    def __init__(self, report: ValidationReport, line: int, record: dict[str, Any]):
        self.report = report
        self.line = line
        self.record = record
        meta = record.get("meta")
        raw_id = None
        if isinstance(meta, Mapping):
            raw_id = meta.get("id", meta.get("source_id"))
        self.record_id = str(raw_id) if raw_id is not None else f"line-{line}"
        self.valid = True
        self.tool_names: set[str] = set()

    def add(
        self,
        severity: Literal["fatal", "warning"],
        layer: str,
        rule: str,
        path: str,
        message: str,
        fix: str,
    ) -> None:
        self.report.findings.append(
            Finding(
                severity=severity,
                layer=layer,
                rule=rule,
                line=self.line,
                record_id=self.record_id,
                path=path,
                message=message,
                fix=fix,
            )
        )
        if severity == FATAL:
            self.valid = False

    def fatal(self, layer: str, rule: str, path: str, message: str, fix: str) -> None:
        self.add(FATAL, layer, rule, path, message, fix)

    def warning(self, layer: str, rule: str, path: str, message: str, fix: str) -> None:
        self.add(WARNING, layer, rule, path, message, fix)

    def exact_keys(
        self,
        value: Mapping[str, Any],
        *,
        allowed: set[str],
        required: set[str],
        path: str,
    ) -> None:
        unknown = sorted(set(value) - allowed)
        missing = sorted(required - set(value))
        if unknown:
            self.fatal(
                "L1",
                "unknown-field",
                path,
                f"存在未声明字段: {unknown}",
                "删除未知字段；非训练元数据放入顶层 meta。",
            )
        if missing:
            self.fatal(
                "L1",
                "missing-field",
                path,
                f"缺少必填字段: {missing}",
                "补齐数据契约要求的字段。",
            )

    def protocol_strings(self, value: Any, path: str) -> None:
        if isinstance(value, str):
            marker = next((item for item in _PROTOCOL_MARKERS if item in value), None)
            if marker is not None:
                self.fatal(
                    "L1",
                    "protocol-injection",
                    path,
                    f"原始数据包含 renderer 保留标记 {marker!r}",
                    "移除手写协议标记，thinking/tool 边界必须由结构化字段生成。",
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self.protocol_strings(item, f"{path}[{index}]")
        elif isinstance(value, Mapping):
            for key, item in value.items():
                self.protocol_strings(str(key), f"{path}.<key>")
                self.protocol_strings(item, f"{path}.{key}")

    def content(self, value: Any, *, role: str, path: str) -> tuple[bool, list[str]]:
        """Validate content and return thinking signal plus tool references."""
        self.protocol_strings(value, path)
        if isinstance(value, str):
            return False, []
        if not isinstance(value, list) or not value:
            self.fatal(
                "L1",
                "content-shape",
                path,
                "content 必须是字符串或非空 content-part 数组",
                "改为字符串，或提供合法的 text/thinking/media/tool result parts。",
            )
            return False, []

        thinking = False
        references: list[str] = []
        if role == "tool" and isinstance(value[0], Mapping):
            first = value[0]
            if first.get("type") == "tool_reference":
                for index, item in enumerate(value):
                    item_path = f"{path}[{index}]"
                    if not isinstance(item, Mapping):
                        self.fatal(
                            "L1",
                            "tool-reference-shape",
                            item_path,
                            "tool_reference 数组混入了非对象元素",
                            "每个元素使用 {type: tool_reference, name: ...}。",
                        )
                        continue
                    self.exact_keys(
                        item,
                        allowed={"type", "name"},
                        required={"type", "name"},
                        path=item_path,
                    )
                    name = item.get("name")
                    if (
                        item.get("type") != "tool_reference"
                        or not isinstance(name, str)
                        or not name
                    ):
                        self.fatal(
                            "L1",
                            "tool-reference-shape",
                            item_path,
                            "tool_reference 需要非空字符串 name",
                            "修正 type/name，并确保引用名称已在 tools 中声明。",
                        )
                    else:
                        references.append(name)
                return False, references
            if "output" in first:
                for index, item in enumerate(value):
                    item_path = f"{path}[{index}]"
                    if not isinstance(item, Mapping):
                        self.fatal(
                            "L1",
                            "tool-output-shape",
                            item_path,
                            "结构化 tool output 数组混入了非对象元素",
                            "每个元素仅提供字符串 output 字段。",
                        )
                        continue
                    self.exact_keys(
                        item,
                        allowed={"output"},
                        required={"output"},
                        path=item_path,
                    )
                    if not isinstance(item.get("output"), str):
                        self.fatal(
                            "L1",
                            "tool-output-shape",
                            f"{item_path}.output",
                            "tool output 必须是字符串",
                            "在写入 JSONL 前把输出规范化为字符串。",
                        )
                return False, []

        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            if isinstance(item, str):
                continue
            if not isinstance(item, Mapping):
                self.fatal(
                    "L1",
                    "content-part-shape",
                    item_path,
                    "content part 必须是字符串或对象",
                    "删除该元素或转换为合法 content part。",
                )
                continue
            part_type = item.get("type")
            if part_type == "text":
                self.exact_keys(
                    item,
                    allowed={"type", "text"},
                    required={"type", "text"},
                    path=item_path,
                )
                if not isinstance(item.get("text"), str):
                    self.fatal(
                        "L1",
                        "text-part-shape",
                        f"{item_path}.text",
                        "text part 的 text 必须是字符串",
                        "把 text 转换为字符串。",
                    )
            elif part_type == "thinking":
                thinking = True
                self.exact_keys(
                    item,
                    allowed={"type", "thinking"},
                    required={"type", "thinking"},
                    path=item_path,
                )
                if role != "assistant":
                    self.fatal(
                        "L1",
                        "thinking-role",
                        item_path,
                        "ThinkingPart 只能出现在 assistant 消息中",
                        "移动到对应 assistant 消息或删除。",
                    )
                if not isinstance(item.get("thinking"), str):
                    self.fatal(
                        "L1",
                        "thinking-shape",
                        f"{item_path}.thinking",
                        "ThinkingPart.thinking 必须是字符串",
                        "把推理内容转换为字符串。",
                    )
            elif part_type in _MEDIA_TYPES:
                continue
            else:
                self.fatal(
                    "L1",
                    "content-part-type",
                    item_path,
                    f"不支持的 content part type: {part_type!r}",
                    "使用 text/thinking 或 GLM-5.2 支持的 media part。",
                )
        return thinking, references

    def tools(self) -> None:
        raw_tools = self.record.get("tools")
        if raw_tools is None:
            return
        if not isinstance(raw_tools, list) or not raw_tools:
            self.fatal(
                "L1",
                "tools-shape",
                "tools",
                "tools 存在时必须是非空数组",
                "删除空 tools 字段，或提供至少一个合法工具定义。",
            )
            return
        for index, raw_tool in enumerate(raw_tools):
            path = f"tools[{index}]"
            if not isinstance(raw_tool, Mapping):
                self.fatal(
                    "L1",
                    "tool-definition-shape",
                    path,
                    "工具定义必须是对象",
                    "使用 OpenAI wrapped 或 Tinker bare ToolSpec。",
                )
                continue
            tool = raw_tool
            if "function" in raw_tool:
                self.exact_keys(
                    raw_tool,
                    allowed={"type", "function"},
                    required={"type", "function"},
                    path=path,
                )
                if raw_tool.get("type") != "function" or not isinstance(
                    raw_tool.get("function"), Mapping
                ):
                    self.fatal(
                        "L1",
                        "tool-definition-shape",
                        path,
                        "wrapped 工具需要 type=function 和对象 function",
                        "修正 OpenAI function tool 包装。",
                    )
                    continue
                tool = cast(Mapping[str, Any], raw_tool["function"])
                path = f"{path}.function"
            self.exact_keys(
                tool,
                allowed={
                    "name",
                    "description",
                    "parameters",
                    "defer_loading",
                    "strict",
                },
                required={"name"},
                path=path,
            )
            self.protocol_strings(tool, path)
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                self.fatal(
                    "L1",
                    "tool-name",
                    f"{path}.name",
                    "工具 name 必须是非空字符串",
                    "提供稳定且唯一的函数名。",
                )
            elif name in self.tool_names:
                self.fatal(
                    "L1",
                    "duplicate-tool",
                    f"{path}.name",
                    f"工具 {name!r} 重复声明",
                    "每个工具名称只保留一个定义。",
                )
            else:
                self.tool_names.add(name)
            if "description" in tool and not isinstance(tool["description"], str):
                self.fatal(
                    "L1",
                    "tool-description",
                    f"{path}.description",
                    "工具 description 必须是字符串",
                    "把 description 转换为字符串。",
                )
            if "parameters" in tool and not isinstance(tool["parameters"], Mapping):
                self.fatal(
                    "L1",
                    "tool-parameters",
                    f"{path}.parameters",
                    "工具 parameters 必须是 JSON Schema 对象",
                    "提供对象而不是字符串或数组。",
                )
            for flag in ("defer_loading", "strict"):
                if flag in tool and not isinstance(tool[flag], bool):
                    self.fatal(
                        "L1",
                        "tool-flag",
                        f"{path}.{flag}",
                        f"{flag} 必须是布尔值",
                        "使用 true 或 false。",
                    )

    def tool_call(self, raw: Any, path: str) -> tuple[str | None, str | None]:
        if not isinstance(raw, Mapping):
            self.fatal(
                "L1",
                "tool-call-shape",
                path,
                "tool_call 必须是对象",
                "使用 OpenAI function-call 对象。",
            )
            return None, None
        self.exact_keys(
            raw,
            allowed={"type", "id", "function"},
            required={"function"},
            path=path,
        )
        if "type" in raw and raw["type"] != "function":
            self.fatal(
                "L1",
                "tool-call-type",
                f"{path}.type",
                "tool_call.type 必须为 function",
                "改为 function 或删除可选 type 字段。",
            )
        call_id = raw.get("id")
        if call_id is not None and not isinstance(call_id, str):
            self.fatal(
                "L1",
                "tool-call-id",
                f"{path}.id",
                "tool_call.id 必须是字符串",
                "把 id 转换为字符串。",
            )
            call_id = None
        function = raw.get("function")
        if not isinstance(function, Mapping):
            self.fatal(
                "L1",
                "tool-function-shape",
                f"{path}.function",
                "tool_call.function 必须是对象",
                "提供 name 和 arguments。",
            )
            return cast(str | None, call_id), None
        self.exact_keys(
            function,
            allowed={"name", "arguments"},
            required={"name", "arguments"},
            path=f"{path}.function",
        )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            self.fatal(
                "L1",
                "tool-call-name",
                f"{path}.function.name",
                "tool call name 必须是非空字符串",
                "提供已声明工具的名称。",
            )
            name = None
        elif name:
            self.protocol_strings(name, f"{path}.function.name")
        arguments = function.get("arguments")
        decoded = arguments
        if isinstance(arguments, str):
            try:
                decoded = {} if arguments == "" else json.loads(arguments)
            except json.JSONDecodeError as exc:
                self.fatal(
                    "L1",
                    "tool-arguments-json",
                    f"{path}.function.arguments",
                    f"arguments 不是合法 JSON: {exc.msg}",
                    "写入合法 JSON 对象字符串，或直接使用 JSON 对象。",
                )
                decoded = None
        if not isinstance(decoded, Mapping):
            self.fatal(
                "L1",
                "tool-arguments-object",
                f"{path}.function.arguments",
                "arguments 必须是 JSON 对象或可解码为对象的字符串",
                "不要使用数组、标量或双重编码 JSON。",
            )
        else:
            self.protocol_strings(decoded, f"{path}.function.arguments")
        return cast(str | None, call_id), cast(str | None, name)

    def messages(self) -> None:
        raw_messages = self.record.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            self.fatal(
                "L1",
                "messages-shape",
                "messages",
                "messages 必须是非空数组",
                "每条记录至少提供一条合法消息。",
            )
            return

        pending: list[tuple[str | None, str | None, str]] = []
        assistant_count = 0
        for index, raw_message in enumerate(raw_messages):
            path = f"messages[{index}]"
            if not isinstance(raw_message, Mapping):
                self.fatal(
                    "L1",
                    "message-shape",
                    path,
                    "消息必须是对象",
                    "提供 role 和 content 字段。",
                )
                continue
            role = raw_message.get("role")
            if role not in _MESSAGE_KEYS:
                self.fatal(
                    "L1",
                    "role",
                    f"{path}.role",
                    f"未知 role: {role!r}",
                    "仅使用 system/user/assistant/tool。",
                )
                continue
            role = cast(str, role)
            self.exact_keys(
                raw_message,
                allowed=_MESSAGE_KEYS[role],
                required={"role", "content"},
                path=path,
            )

            if pending and role != "tool":
                self.fatal(
                    "L1",
                    "tool-pairing",
                    path,
                    f"仍缺少 {len(pending)} 条 tool 响应，下一条却是 {role}",
                    "每个 assistant tool_call 后立即提供等量 tool 消息。",
                )
                pending.clear()
            if role == "tool":
                if not pending:
                    self.fatal(
                        "L1",
                        "tool-pairing",
                        path,
                        "tool 消息前没有待响应的 assistant tool_call",
                        "删除孤立 tool 消息或补齐前置 tool_call。",
                    )
                else:
                    expected_id, expected_name, _ = pending.pop(0)
                    actual_id = raw_message.get("tool_call_id")
                    actual_name = raw_message.get("name")
                    if actual_id is not None and not isinstance(actual_id, str):
                        self.fatal(
                            "L1",
                            "tool-response-id",
                            f"{path}.tool_call_id",
                            "tool_call_id 必须是字符串",
                            "使用对应 tool_call 的字符串 id。",
                        )
                    elif expected_id and actual_id and expected_id != actual_id:
                        self.fatal(
                            "L1",
                            "tool-response-id",
                            f"{path}.tool_call_id",
                            f"响应 id {actual_id!r} 不匹配调用 id {expected_id!r}",
                            "按调用顺序填写一致的 tool_call_id。",
                        )
                    if actual_name is not None and not isinstance(actual_name, str):
                        self.fatal(
                            "L1",
                            "tool-response-name",
                            f"{path}.name",
                            "tool 响应 name 必须是字符串",
                            "使用对应工具名称。",
                        )
                    elif expected_name and actual_name and expected_name != actual_name:
                        self.fatal(
                            "L1",
                            "tool-response-name",
                            f"{path}.name",
                            f"响应工具 {actual_name!r} 不匹配调用 {expected_name!r}",
                            "按调用顺序填写一致的工具名称。",
                        )

            thinking, references = self.content(
                raw_message.get("content"), role=role, path=f"{path}.content"
            )
            for name in references:
                if name not in self.tool_names:
                    self.fatal(
                        "L1",
                        "tool-reference-name",
                        f"{path}.content",
                        f"tool_reference {name!r} 未在 tools 中声明",
                        "添加匹配工具定义或修正引用名称。",
                    )

            if role == "assistant":
                assistant_count += 1
                legacy_reasoning = raw_message.get("reasoning_content")
                if "reasoning_content" in raw_message and not isinstance(
                    legacy_reasoning, str
                ):
                    self.fatal(
                        "L1",
                        "reasoning-shape",
                        f"{path}.reasoning_content",
                        "reasoning_content 必须是字符串；null 也不允许",
                        "改为字符串；无推理时删除该字段。",
                    )
                elif isinstance(legacy_reasoning, str):
                    self.protocol_strings(legacy_reasoning, f"{path}.reasoning_content")
                if thinking and "reasoning_content" in raw_message:
                    self.fatal(
                        "L1",
                        "reasoning-source",
                        path,
                        "同一消息同时使用 ThinkingPart 和 reasoning_content",
                        "只保留一种；新数据优先使用 ThinkingPart。",
                    )
                calls = raw_message.get("tool_calls")
                if calls is not None:
                    if not isinstance(calls, list) or not calls:
                        self.fatal(
                            "L1",
                            "tool-calls-shape",
                            f"{path}.tool_calls",
                            "tool_calls 存在时必须是非空数组",
                            "删除空字段或提供合法调用。",
                        )
                    else:
                        for call_index, call in enumerate(calls):
                            call_path = f"{path}.tool_calls[{call_index}]"
                            call_id, name = self.tool_call(call, call_path)
                            pending.append((call_id, name, call_path))
                            if name and name not in self.tool_names:
                                self.fatal(
                                    "L1",
                                    "undeclared-tool-call",
                                    f"{call_path}.function.name",
                                    f"调用的工具 {name!r} 未在 tools 中声明",
                                    "添加匹配工具定义或修正调用名称。",
                                )

        if pending:
            self.fatal(
                "L1",
                "tool-pairing",
                "messages",
                f"记录结束时仍缺少 {len(pending)} 条 tool 响应",
                "在记录末尾补齐与 tool_calls 一一对应的 tool 消息。",
            )
        if assistant_count == 0:
            self.fatal(
                "L1",
                "no-assistant",
                "messages",
                "记录不包含 assistant 消息，因此没有监督目标",
                "加入至少一条需要训练的 assistant 回复。",
            )

    def kwargs(self) -> None:
        kwargs = self.record.get("chat_template_kwargs")
        if kwargs is None:
            return
        if not isinstance(kwargs, Mapping):
            self.fatal(
                "L1",
                "template-kwargs-shape",
                "chat_template_kwargs",
                "chat_template_kwargs 必须是对象",
                "提供 enable_thinking/reasoning_effort，或删除该字段。",
            )
            return
        self.exact_keys(
            kwargs,
            allowed={"enable_thinking", "reasoning_effort"},
            required=set(),
            path="chat_template_kwargs",
        )
        if "enable_thinking" in kwargs and not isinstance(
            kwargs["enable_thinking"], bool
        ):
            self.fatal(
                "L1",
                "enable-thinking",
                "chat_template_kwargs.enable_thinking",
                "enable_thinking 必须是布尔值",
                "使用 true 或 false。",
            )
        effort = kwargs.get("reasoning_effort")
        if effort is not None and effort not in {"high", "max"}:
            self.fatal(
                "L1",
                "reasoning-effort",
                "chat_template_kwargs.reasoning_effort",
                f"不支持 reasoning_effort={effort!r}",
                "使用 high 或 max。",
            )
        if kwargs.get("enable_thinking") is False:
            messages = self.record.get("messages", [])
            if isinstance(messages, list):
                for index, message in enumerate(messages):
                    if (
                        not isinstance(message, Mapping)
                        or message.get("role") != "assistant"
                    ):
                        continue
                    content = message.get("content")
                    structured = isinstance(content, list) and any(
                        isinstance(part, Mapping) and part.get("type") == "thinking"
                        for part in content
                    )
                    if "reasoning_content" in message or structured:
                        self.fatal(
                            "L1",
                            "disabled-thinking-data",
                            f"messages[{index}]",
                            "enable_thinking=false 的记录仍包含推理内容",
                            "删除推理字段，或启用 thinking renderer。",
                        )

    def run(self) -> bool:
        self.exact_keys(
            self.record,
            allowed=_ROOT_KEYS,
            required={"messages"},
            path="$",
        )
        meta = self.record.get("meta")
        if meta is not None and not isinstance(meta, Mapping):
            self.fatal(
                "L1",
                "meta-shape",
                "meta",
                "meta 必须是对象",
                "将非训练字段放进 JSON 对象。",
            )
        self.tools()
        self.messages()
        self.kwargs()
        return self.valid


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _load_records(
    path: Path, report: ValidationReport
) -> list[tuple[int, dict[str, Any]]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EnvironmentValidationError(f"无法读取数据文件 {path}: {exc}") from exc
    if raw.startswith(b"\xef\xbb\xbf"):
        report.findings.append(
            Finding(
                FATAL,
                "L0",
                "utf8-bom",
                1,
                "line-1",
                "$",
                "文件包含 UTF-8 BOM",
                "以 UTF-8（无 BOM）重新保存文件。",
            )
        )
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        line_number = raw[: exc.start].count(b"\n") + 1
        report.findings.append(
            Finding(
                FATAL,
                "L0",
                "utf8",
                line_number,
                f"line-{line_number}",
                "$",
                f"文件不是合法 UTF-8: {exc.reason}",
                "以 UTF-8（无 BOM）重新编码文件。",
            )
        )
        return []
    if not text:
        report.findings.append(
            Finding(
                FATAL,
                "L0",
                "empty-file",
                0,
                "file",
                "$",
                "数据文件为空",
                "写入至少一条 JSONL 记录。",
            )
        )
        return []

    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            report.findings.append(
                Finding(
                    FATAL,
                    "L0",
                    "blank-line",
                    line_number,
                    f"line-{line_number}",
                    "$",
                    "JSONL 中存在空行",
                    "删除空行；每一行必须恰好是一条 JSON 对象。",
                )
            )
            continue
        report.total_records += 1
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except _DuplicateKeyError as exc:
            report.findings.append(
                Finding(
                    FATAL,
                    "L0",
                    "duplicate-json-key",
                    line_number,
                    f"line-{line_number}",
                    "$",
                    f"JSON 对象包含重复 key {str(exc)!r}",
                    "删除重复 key，避免解析器静默覆盖数据。",
                )
            )
            continue
        except json.JSONDecodeError as exc:
            report.findings.append(
                Finding(
                    FATAL,
                    "L0",
                    "json",
                    line_number,
                    f"line-{line_number}",
                    "$",
                    f"JSON 解析失败（列 {exc.colno}）: {exc.msg}",
                    "修复该行 JSON 语法。",
                )
            )
            continue
        if not isinstance(value, dict):
            report.findings.append(
                Finding(
                    FATAL,
                    "L0",
                    "record-object",
                    line_number,
                    f"line-{line_number}",
                    "$",
                    "每行顶层值必须是 JSON 对象",
                    "用包含 messages 的对象包裹记录。",
                )
            )
            continue
        records.append((line_number, value))
    return records


def _load_tokenizer(model_name: str) -> Any:
    try:
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:
        raise EnvironmentValidationError(
            "缺少 tinker-cookbook；安装 mindlab-toolkit[glm52-renderer] 或 [test]"
        ) from exc
    try:
        return get_tokenizer(model_name)
    except Exception as exc:
        raise EnvironmentValidationError(
            f"无法加载 tokenizer {model_name!r}: {type(exc).__name__}: {exc}"
        ) from exc


def _tinker_messages(raw_messages: list[dict[str, Any]]) -> list[Any]:
    from tinker_cookbook.renderers import ToolCall

    converted: list[Any] = []
    for message_index, raw in enumerate(raw_messages):
        message = dict(raw)
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            converted_calls = []
            for call_index, call in enumerate(calls):
                function = call["function"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    decoded = {} if arguments == "" else json.loads(arguments)
                else:
                    decoded = arguments
                converted_calls.append(
                    ToolCall(
                        id=str(call.get("id", f"call-{message_index}-{call_index}")),
                        function=ToolCall.FunctionBody(
                            name=function["name"],
                            arguments=json.dumps(decoded, ensure_ascii=False),
                        ),
                    )
                )
            message["tool_calls"] = converted_calls
        converted.append(message)
    return converted


def _render_record(
    record: dict[str, Any], tokenizer: Any, renderer_cache: dict[tuple[bool, str], Any]
) -> tuple[list[int], list[float]]:
    from tinker_cookbook.renderers import TrainOnWhat

    from mint.renderers import GLM52DisableThinkingRenderer, GLM52Renderer

    kwargs = record.get("chat_template_kwargs") or {}
    enable_thinking = kwargs.get("enable_thinking", True)
    effort = kwargs.get("reasoning_effort", "max")
    cache_key = (enable_thinking, effort)
    renderer = renderer_cache.get(cache_key)
    if renderer is None:
        renderer = (
            GLM52Renderer(tokenizer, reasoning_effort=effort)
            if enable_thinking
            else GLM52DisableThinkingRenderer(tokenizer)
        )
        renderer_cache[cache_key] = renderer
    prefix = renderer.create_conversation_prefix_with_tools(record.get("tools") or [])
    messages = [*prefix, *_tinker_messages(record["messages"])]
    model_input, weights = renderer.build_supervised_example(
        messages, TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    return model_input.to_ints(), [float(value) for value in weights.tolist()]


def _canonical_digest(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_jsonl(
    data_path: str | os.PathLike[str],
    *,
    max_seq_len: int,
    model_name: str = "zai-org/GLM-5.2",
    tokenizer: Any | None = None,
    supervised_fraction_low: float = 0.02,
    supervised_fraction_high: float = 0.98,
) -> ValidationReport:
    """Run fail-closed structural and real-render validation for a JSONL file."""
    if max_seq_len < 2:
        raise EnvironmentValidationError("max_seq_len 必须 >= 2")
    if not 0 <= supervised_fraction_low <= supervised_fraction_high <= 1:
        raise EnvironmentValidationError("监督占比阈值必须满足 0 <= low <= high <= 1")
    path = Path(data_path)
    if not path.is_file():
        raise EnvironmentValidationError(f"数据文件不存在: {path}")

    report = ValidationReport(data_path=str(path))
    records = _load_records(path, report)
    structurally_valid: list[tuple[int, dict[str, Any], _RecordValidator]] = []
    seen: dict[str, int] = {}
    for line, record in records:
        validator = _RecordValidator(report, line, record)
        if validator.run():
            structurally_valid.append((line, record, validator))
        digest = _canonical_digest(record)
        if digest in seen:
            validator.warning(
                "L6",
                "duplicate-record",
                "$",
                f"记录与第 {seen[digest]} 行完全重复",
                "确认是否需要去重，避免过采样或数据泄漏。",
            )
        else:
            seen[digest] = line

    if not structurally_valid:
        return report
    if tokenizer is None:
        tokenizer = _load_tokenizer(model_name)

    renderer_cache: dict[tuple[bool, str], Any] = {}
    for line, record, validator in structurally_valid:
        try:
            tokens, weights = _render_record(record, tokenizer, renderer_cache)
        # Renderer/tokenizer implementations expose several exception families;
        # this per-record boundary must convert all data-dependent failures into
        # findings while still allowing BaseException interrupts to propagate.
        except Exception as exc:  # noqa: BLE001
            validator.fatal(
                "L2",
                "renderer",
                "$",
                f"真实 GLM-5.2 renderer 拒绝该记录: {type(exc).__name__}: {exc}",
                "按 renderer 错误修正消息、工具或 thinking 结构。",
            )
            continue
        report.rendered_records += 1
        if len(tokens) != len(weights):
            validator.fatal(
                "L4",
                "token-weight-length",
                "$",
                f"token 数 {len(tokens)} 与 weight 数 {len(weights)} 不一致",
                "这是 renderer 契约错误；停止训练并提交最小复现。",
            )
            continue
        if any(weight not in {0.0, 1.0} for weight in weights):
            validator.fatal(
                "L4",
                "non-binary-weight",
                "$",
                "监督权重包含 0/1 之外的值",
                "这是 renderer 契约错误；停止训练并提交最小复现。",
            )
            continue

        effective_weights = weights
        if len(tokens) > max_seq_len:
            report.truncated_records += 1
            validator.warning(
                "L5",
                "right-truncation",
                "$",
                f"序列长度 {len(tokens)} 超过上限 {max_seq_len}，将右截断",
                "缩短样本，或确认训练使用完全相同的 max_seq_len。",
            )
            effective_weights = weights[:max_seq_len]
        supervised = sum(weight > 0 for weight in effective_weights)
        if supervised == 0:
            validator.fatal(
                "L5",
                "no-supervision",
                "$",
                "完整渲染或右截断后没有任何监督 token",
                "保留 assistant 目标并缩短其前文，确保目标落在长度上限内。",
            )
            continue
        fraction = supervised / len(effective_weights)
        if fraction < supervised_fraction_low or fraction > supervised_fraction_high:
            validator.warning(
                "L6",
                "supervised-fraction",
                "$",
                f"监督 token 占比 {fraction:.4f} 超出建议区间 "
                f"[{supervised_fraction_low:.4f}, {supervised_fraction_high:.4f}]",
                "检查对话长度和 assistant mask；确认异常占比符合训练意图。",
            )
        if len(tokens) >= int(max_seq_len * 0.95) and len(tokens) <= max_seq_len:
            validator.warning(
                "L6",
                "near-length-limit",
                "$",
                f"序列长度 {len(tokens)} 已接近上限 {max_seq_len}",
                "预留模板变化余量，或固定并审计训练长度配置。",
            )
        if validator.valid:
            report.trainable_records += 1
    return report


def _write_report(path: str, report: ValidationReport | Mapping[str, Any]) -> None:
    payload = report.to_dict() if isinstance(report, ValidationReport) else dict(report)
    try:
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise EnvironmentValidationError(f"无法写入报告 {path}: {exc}") from exc


def _print_report(report: ValidationReport, max_examples_per_rule: int) -> None:
    grouped = Counter(
        (item.severity, item.layer, item.rule) for item in report.findings
    )
    for severity in (FATAL, WARNING):
        title = "致命违规 FATAL" if severity == FATAL else "警告 WARNING"
        selected = [item for item in report.findings if item.severity == severity]
        if not selected:
            continue
        print(f"\n===== {title} ({len(selected)}) =====")
        emitted: Counter[tuple[str, str]] = Counter()
        for item in selected:
            key = (item.layer, item.rule)
            if emitted[key] >= max_examples_per_rule:
                continue
            print("\n" + item.render())
            emitted[key] += 1
        for (item_severity, layer, rule), count in sorted(grouped.items()):
            if item_severity == severity and count > max_examples_per_rule:
                print(
                    f"\n[{layer}:{rule}] 另有 {count - max_examples_per_rule} 条未显示"
                )

    print("\n" + "=" * 60)
    print(f"[validate] 记录总数: {report.total_records}")
    print(f"[validate] 完成真实渲染: {report.rendered_records}")
    print(f"[validate] 最终可训练: {report.trainable_records}")
    print(f"[validate] 发生右截断: {report.truncated_records}")
    print(f"[validate] 致命问题: {report.fatal_count}")
    print(f"[validate] 警告: {report.warning_count}")
    print("=" * 60)
    if report.exit_code:
        print("[validate] 校验失败：数据不可进入训练。")
    else:
        print("[validate] 校验通过。")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLM-5.2 SFT JSONL 严格验证器")
    parser.add_argument("--data", required=True, help="待验证 JSONL 文件")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="训练序列长度上限；未提供且无 MINT_MAX_SEQ_LEN 时 fail closed",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MINT_BASE_MODEL", "zai-org/GLM-5.2"),
        help="用于真实 renderer 校验的 tokenizer 名称",
    )
    parser.add_argument("--report", help="写入机器可读 JSON 报告")
    parser.add_argument("--frac-low", type=float, default=0.02)
    parser.add_argument("--frac-high", type=float, default=0.98)
    parser.add_argument("--max-examples-per-rule", type=int, default=10)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_seq_len is None and os.environ.get("MINT_MAX_SEQ_LEN"):
        try:
            args.max_seq_len = int(os.environ["MINT_MAX_SEQ_LEN"])
        except ValueError:
            message = (
                f"MINT_MAX_SEQ_LEN 必须是整数，得到 {os.environ['MINT_MAX_SEQ_LEN']!r}"
            )
            print(f"[validate] 环境错误(exit 2): {message}", file=sys.stderr)
            if args.report:
                try:
                    _write_report(
                        args.report,
                        {"status": "environment_error", "error": message},
                    )
                except EnvironmentValidationError as exc:
                    print(f"[validate] 环境错误(exit 2): {exc}", file=sys.stderr)
            return 2
    if args.max_seq_len is None:
        message = (
            "未提供 --max-seq-len 且 MINT_MAX_SEQ_LEN 未设置；必须与训练使用相同上限"
        )
        print(f"[validate] 环境错误(exit 2): {message}", file=sys.stderr)
        if args.report:
            try:
                _write_report(
                    args.report, {"status": "environment_error", "error": message}
                )
            except EnvironmentValidationError as exc:
                print(f"[validate] 环境错误(exit 2): {exc}", file=sys.stderr)
        return 2
    try:
        report = validate_jsonl(
            args.data,
            max_seq_len=args.max_seq_len,
            model_name=args.model_name,
            supervised_fraction_low=args.frac_low,
            supervised_fraction_high=args.frac_high,
        )
        if args.report:
            _write_report(args.report, report)
    except EnvironmentValidationError as exc:
        print(f"[validate] 环境错误(exit 2): {exc}", file=sys.stderr)
        if args.report:
            try:
                _write_report(
                    args.report,
                    {"status": "environment_error", "error": str(exc)},
                )
            except EnvironmentValidationError as report_exc:
                print(f"[validate] 环境错误(exit 2): {report_exc}", file=sys.stderr)
        return 2
    _print_report(report, max(1, args.max_examples_per_rule))
    return report.exit_code


__all__ = [
    "EnvironmentValidationError",
    "Finding",
    "ValidationReport",
    "main",
    "parse_args",
    "validate_jsonl",
]


if __name__ == "__main__":
    raise SystemExit(main())
