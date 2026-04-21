"""
Pluggable tool registry for the generator agent (process-local, MCP-like metadata).

Each registered tool has a stable key (lowercase_snake), human-readable docs for prompts,
and a handler that returns a partial state update dict (same contract as other agent nodes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .agent_config import BUILTIN_TOOL_NAMES

ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]

_ROUTING_RESERVED = frozenset({"answer", "registered", "no_tool"})


@dataclass
class RegisteredToolSpec:
    """Metadata + callable for a registered tool (built-in or plugin)."""

    name: str  # lowercase_snake, unique, must not shadow a reserved built-in key
    display_name: str
    description: str
    parameter_docs: str
    handler: ToolHandler = field(repr=False)
    examples: List[str] = field(default_factory=list)
    usage_guidance: str = ""


class ToolRegistry:
    """Singleton-style registry; use get_tool_registry() for default instance."""

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredToolSpec] = {}

    def register(self, spec: RegisteredToolSpec, *, allow_builtin_name: bool = False) -> None:
        key = spec.name.strip().lower()
        if not key:
            raise ValueError("tool name must be non-empty")
        if key in _ROUTING_RESERVED:
            raise ValueError(f"tool name {key!r} is reserved for routing / JSON protocol")
        if not allow_builtin_name and key in BUILTIN_TOOL_NAMES:
            raise ValueError(
                f"tool name {key!r} is a built-in key; user plugins must use a different name"
            )
        if key in self._tools:
            raise ValueError(f"duplicate tool registration: {key!r}")
        self._tools[key] = RegisteredToolSpec(
            name=key,
            display_name=spec.display_name,
            description=spec.description,
            parameter_docs=spec.parameter_docs,
            handler=spec.handler,
            examples=list(spec.examples),
            usage_guidance=str(getattr(spec, "usage_guidance", None) or ""),
        )

    def is_registered(self, name: str) -> bool:
        return name.strip().lower() in self._tools

    def get(self, name: str) -> Optional[RegisteredToolSpec]:
        return self._tools.get(name.strip().lower())

    def list_plugin_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def unregister(self, name: str) -> None:
        self._tools.pop(name.strip().lower(), None)

    def clear(self) -> None:
        self._tools.clear()


_DEFAULT_REGISTRY = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    return _DEFAULT_REGISTRY


def register_tool(spec: RegisteredToolSpec) -> None:
    """Register a user/plugin tool on the process-wide default registry."""
    _DEFAULT_REGISTRY.register(spec, allow_builtin_name=False)
