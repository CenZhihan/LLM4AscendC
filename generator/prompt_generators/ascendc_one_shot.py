from __future__ import annotations

from generator.prompt_generators.prompt_registry import BasePromptStrategy, register_prompt
from generator.prompt_generators.prompt_utils import ascendc_template, read_relavant_files


@register_prompt("ascendc", "one_shot")
class AscendcOneShotPromptStrategy(BasePromptStrategy):
    """Few-shot example fixed to leaky_relu (snapshot under generation/prompts/)."""

    def generate(self, op: str) -> str:
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files("ascendc", op, "leaky_relu")
        return ascendc_template(arc_src, example_arch_src, example_new_arch_src, op, "leaky_relu")
