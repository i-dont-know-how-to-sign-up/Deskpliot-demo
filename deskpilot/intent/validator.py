from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tools.tool_registry import ToolParameter, ToolSpec
from .schemas import SlotValidationResult


@dataclass(frozen=True)
class SlotValidator:
    """Validate and coerce tool-call slots against a tool schema."""

    def validate(self, tool: ToolSpec | dict[str, Any] | None, arguments: dict[str, Any]) -> SlotValidationResult:
        if tool is None:
            return SlotValidationResult(ok=False, error="Unknown tool.")

        normalized_args: dict[str, Any] = {}
        missing_slots: list[str] = []
        warnings: list[str] = []

        # 按工具 schema 逐项校验，先补默认值，再检查必填参数和类型。
        parameters = self._iter_parameters(tool)
        for parameter in parameters:
            raw_value = arguments.get(parameter.name, None)
            if raw_value in (None, ""):
                if parameter.required and parameter.default is None:
                    missing_slots.append(parameter.name)
                    continue
                normalized_args[parameter.name] = parameter.default
                continue
            # 只做最小必要的类型转换，尽量让模型输出能落到工具真实签名上。
            coerced, warning, ok = self._coerce_value(parameter.type, raw_value)
            if not ok:
                return SlotValidationResult(
                    ok=False,
                    arguments=normalized_args,
                    missing_slots=missing_slots,
                    warnings=warnings,
                    error=f"Invalid value for slot '{parameter.name}'",
                )
            normalized_args[parameter.name] = coerced
            if warning:
                warnings.append(f"{parameter.name}: {warning}")

        # 额外参数先保留，便于工具自身扩展或后续调试排查。
        known_keys = {parameter.name for parameter in parameters}
        for key, value in arguments.items():
            if key not in normalized_args and key not in known_keys:
                normalized_args[key] = value

        if missing_slots:
            return SlotValidationResult(
                ok=False,
                arguments=normalized_args,
                missing_slots=missing_slots,
                warnings=warnings,
                error="Missing required slots.",
            )

        return SlotValidationResult(ok=True, arguments=normalized_args, warnings=warnings)

    def _coerce_value(self, declared_type: str, value: Any) -> tuple[Any, str, bool]:
        declared = str(declared_type or "").lower()
        if "string|array" in declared:
            if isinstance(value, list):
                return value, "", True
            return str(value), "", True
        if "array" in declared:
            if isinstance(value, list):
                return value, "", True
            return [value], "coerced scalar to array", True
        if "boolean" in declared or declared == "bool":
            if isinstance(value, bool):
                return value, "", True
            if isinstance(value, (int, float)):
                return bool(value), "coerced numeric to boolean", True
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True, "coerced text to boolean", True
            if text in {"0", "false", "no", "off"}:
                return False, "coerced text to boolean", True
            return value, "", False
        if "integer" in declared or declared in {"int"}:
            try:
                return int(value), "", True
            except (TypeError, ValueError):
                return value, "", False
        if "number" in declared or "float" in declared:
            try:
                return float(value), "", True
            except (TypeError, ValueError):
                return value, "", False
        if "object" in declared or "dict" in declared:
            if isinstance(value, dict):
                return value, "", True
            return value, "", False
        return str(value), "", True

    def _iter_parameters(self, tool: ToolSpec | dict[str, Any]) -> list[ToolParameter]:
        if isinstance(tool, ToolSpec):
            return list(tool.parameters)
        parameters = tool.get("parameters", []) if isinstance(tool, dict) else []
        normalized: list[ToolParameter] = []
        for parameter in parameters:
            if isinstance(parameter, ToolParameter):
                normalized.append(parameter)
                continue
            if isinstance(parameter, dict):
                normalized.append(
                    ToolParameter(
                        name=str(parameter.get("name", "")),
                        type=str(parameter.get("type", "string")),
                        required=bool(parameter.get("required", True)),
                        description=str(parameter.get("description", "")),
                        default=parameter.get("default"),
                    )
                )
        return normalized
