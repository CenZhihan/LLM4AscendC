"""Prompt generation strategies."""
from .prompt_registry import register_prompt, BasePromptStrategy, get_prompt_strategy, PROMPT_REGISTRY

# 导入所有策略模块以触发注册
from . import ascendc_add_shot
from . import ascendc_add_shot_with_code
from . import ascendc_add_shot_with_doc
from . import ascendc_selected_shot
from . import cuda_add_shot
from . import cuda_selected_shot

__all__ = ['register_prompt', 'BasePromptStrategy', 'get_prompt_strategy', 'PROMPT_REGISTRY']