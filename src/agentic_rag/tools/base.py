"""Tool = hàm Python + mô tả (schema) để LLM biết khi nào/cách gọi. LLM chỉ ĐỀ NGHỊ gọi; code của ta mới là bên thực thi,
nên đây là chỗ kiểm tra tham số, phân quyền, xin xác nhận trước khi chạy."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Callable

from ..llm.base import ToolCall, ToolSpec

if TYPE_CHECKING:  # tránh import vòng
    from ..agent.evidence import EvidenceStore


@dataclass
class ToolContext:
    """Ngữ cảnh của MỘT lượt hỏi. user_id đến từ phiên đăng nhập, KHÔNG phải từ LLM."""
    evidence: "EvidenceStore"
    settings: Any
    today: date
    user_id: str | None = None
    confirm: Callable[[str, dict], bool] | None = None  # human-in-the-loop cho tool ghi
    extras: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    content: str


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[[dict, ToolContext], "str | ToolResult"]
    tags: tuple[str, ...] = ()
    read_only: bool = True
    requires_confirmation: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


_PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def _coerce(v: Any, t: str | None) -> Any:
    """LLM hay gửi "5" thay vì 5, hoặc mảng dạng chuỗi JSON: sửa nhẹ trước khi kiểm tra kiểu."""
    if isinstance(v, str):
        s = v.strip()
        if t == "integer" and re.fullmatch(r"-?\d+", s):
            return int(s)
        if t == "number" and re.fullmatch(r"-?\d+(\.\d+)?", s):
            return float(s)
        if t == "boolean" and s.lower() in ("true", "false"):
            return s.lower() == "true"
        if t in ("array", "object") and s[:1] in "[{":
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return v
    return v


def validate_args(schema: dict, args: Any) -> tuple[dict, str | None]:
    if not isinstance(args, dict):
        return {}, "arguments phải là một object"
    props, clean = schema.get("properties", {}), {}
    for k, v in args.items():
        if k not in props:
            continue  # bỏ tham số lạ: LLM không thể tự chèn tham số (vd student_id) vào tool
        spec = props[k]
        t = spec.get("type")
        v = _coerce(v, t)
        if t in _PY and (not isinstance(v, _PY[t]) or (t in ("integer", "number") and isinstance(v, bool))):
            return clean, f"'{k}' phải có kiểu {t}"
        if "enum" in spec and v not in spec["enum"]:
            return clean, f"'{k}' phải thuộc {spec['enum']}"
        clean[k] = v
    missing = [r for r in schema.get("required", []) if r not in clean]
    return clean, (f"thiếu tham số bắt buộc: {missing}" if missing else None)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool trùng tên: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def subset(self, names: list[str] | None = None, tags: list[str] | None = None) -> "ToolRegistry":
        """Tập con tool - nền tảng để sau này tách agent chuyên biệt (multi-agent)."""
        return ToolRegistry([t for t in self._tools.values()
                             if (names is None or t.name in names) and (tags is None or set(tags) & set(t.tags))])

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in self._tools.values())

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(False, f"Công cụ '{call.name}' không tồn tại. Công cụ hiện có: {', '.join(self._tools)}")
        args, err = validate_args(tool.parameters, call.arguments)
        if err:
            return ToolResult(False, f"Tham số không hợp lệ cho {call.name}: {err}")
        if tool.requires_confirmation and not (ctx.confirm and ctx.confirm(tool.name, args)):
            return ToolResult(False, "Hành động này cần người dùng xác nhận nhưng chưa được xác nhận nên KHÔNG được thực hiện.")
        try:
            out = tool.func(args, ctx)
        except Exception as e:  # noqa: BLE001 - trả lỗi về cho agent để nó tự sửa
            return ToolResult(False, f"Lỗi khi chạy {call.name}: {type(e).__name__}: {e}")
        return out if isinstance(out, ToolResult) else ToolResult(True, str(out))
