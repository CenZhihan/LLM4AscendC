from abc import ABC, abstractmethod

PROMPT_REGISTRY = {}

class BasePromptStrategy(ABC):
    @abstractmethod
    def generate(self, op) -> str:
        pass

def register_prompt(language: str, strategy_name: str):
    def decorator(cls):
        PROMPT_REGISTRY.setdefault(language, {})[strategy_name] = cls()
        return cls
    return decorator

def get_prompt_strategy(language: str, strategy_name: str) -> BasePromptStrategy:
    """Get a registered prompt strategy."""
    if language not in PROMPT_REGISTRY:
        raise ValueError(f"Language '{language}' not registered")
    if strategy_name not in PROMPT_REGISTRY[language]:
        raise ValueError(f"Strategy '{strategy_name}' not registered for language '{language}'")
    return PROMPT_REGISTRY[language][strategy_name]
