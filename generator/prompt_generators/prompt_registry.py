from __future__ import annotations

from abc import ABC, abstractmethod

PROMPT_REGISTRY: dict[str, dict[str, object]] = {}


class BasePromptStrategy(ABC):
    @abstractmethod
    def generate(self, op: str) -> str:
        raise NotImplementedError


def register_prompt(language: str, strategy_name: str):
    def decorator(cls):
        PROMPT_REGISTRY.setdefault(language, {})[strategy_name] = cls()
        return cls

    return decorator
