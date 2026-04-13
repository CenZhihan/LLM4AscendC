from __future__ import annotations

from generation.prompt_generators.prompt_registry import BasePromptStrategy, register_prompt
from generation.prompt_generators.prompt_utils import ascendc_template
from generation.repo_root import REPO_ROOT
from generation.text_utils import read_file
from vendor.mkb.dataset import dataset


@register_prompt("ascendc", "none")
class AscendcNoShotPromptStrategy(BasePromptStrategy):
    def generate(self, op: str) -> str:
        category = dataset[op]["category"]
        arch_path = REPO_ROOT / "vendor" / "mkb" / "reference" / category / f"{op}.py"
        if not arch_path.is_file():
            raise FileNotFoundError(f"Reference architecture file not found: {arch_path}")
        arc_src = read_file(str(arch_path))
        return ascendc_template(arc_src, "", "", op, op)
