from __future__ import annotations

from enum import Enum


class AgentToolMode(str, Enum):
    NO_TOOL = "no_tool"
    KB_ONLY = "kb_only"
    WEB_ONLY = "web_only"
    KB_AND_WEB = "kb_and_web"
