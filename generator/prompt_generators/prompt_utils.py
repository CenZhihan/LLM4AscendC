from __future__ import annotations

import os

from generator.repo_root import REPO_ROOT
from generator.text_utils import read_file, underscore_to_pascalcase
from vendor.mkb.dataset import dataset

ASCENDC_PROBLEM_STATEMENT = (
    "You are an expert in writing custom AscendC kernels to optimize PyTorch architectures "
    "by replacing specific operators for performance gains.\n"
)
ASCENDC_PROBLEM_INSTRUCTION = """
Your task: Replace relevant PyTorch operators in the architecture named Model with custom AscendC kernels. Generate an optimized version named ModelNew, including the six Python strings listed above. Just output the code, no other text, and NO testing code!\n
"""

ASCENDC_OUTPUT_FORMAT_NO_EXAMPLE = """
Output format (strict, no examples given here—follow these rules only):

1) You must output exactly one code block: start with a line ```python and end with ```. Do NOT output any prose, description, ellipsis (e.g. "、、、"), or markdown before or after this block. Do NOT put any non-Python line inside the block.

2) The block must be valid, executable Python: use 4 spaces for indentation (no tabs); no syntax errors; no trailing comment lines that look like prose. Use only ASCII in the entire block (no Chinese characters or fullwidth punctuation such as 。、（，）；write all comments and docstrings in English). The entire block will be run as Python to extract the six variables.

3) The block must define exactly these six string variables (e.g. using r\"\"\" or r''' for multi-line strings):
- project_json_src: JSON string for the operator project. The parsed JSON must have a top-level key \"op\" (required by the Ascend build toolchain). Under \"op\" put operator metadata (name, inputs, outputs, kernel/tiling/sources, etc.). Do not use other top-level shapes like \"op_name\" or \"operator\" alone—the root must contain \"op\".
- host_tiling_src: Tiling header (.h) content
- host_operator_src: Host-side operator (.cpp) content
- kernel_src: AscendC kernel (.cpp) content (kernel name must match the given operator name)
- python_bind_src: Python binding code
- model_src: PyTorch model code that uses the custom op (class ModelNew)

4) Naming: kernel and operator names must use the given name (e.g. {op_name}) and PascalCase in JSON/host (e.g. {op_pascal}).
"""


def read_relavant_files(language: str, op: str, example: str) -> tuple[str, str, str]:
    category = dataset[op]["category"]
    prompts_dir = REPO_ROOT / "generator" / "prompts"
    example_arch_path = prompts_dir / f"cuda_model_{example}.py"
    example_new_arch_path = prompts_dir / f"{language}_new_model_{example}.py"
    new_arch_path = REPO_ROOT / "vendor" / "mkb" / "reference" / category / f"{op}.py"

    if not example_arch_path.is_file():
        raise FileNotFoundError(f"Example architecture file not found: {example_arch_path}")
    if not example_new_arch_path.is_file():
        raise FileNotFoundError(f"Example new architecture file not found: {example_new_arch_path}")
    if not new_arch_path.is_file():
        raise FileNotFoundError(f"Reference architecture file not found: {new_arch_path}")

    example_arch = read_file(str(example_arch_path))
    example_new_arch = read_file(str(example_new_arch_path))
    arch = read_file(str(new_arch_path))
    return arch, example_arch, example_new_arch


def ascendc_template(
    arc_src: str,
    example_arch_src: str,
    example_new_arch_src: str,
    op: str,
    example_op: str,
) -> str:
    op = op + "_custom"
    example_op = example_op + "_custom"
    prompt = ASCENDC_PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
    Here is an example to illustrate the expected transformation using custom AscendC operators. **Original architecture with kernel name `{example_op}`:**\n
    ```python \n
    {example_arch_src}
    ``` \n
    Transformed version using custom AscendC kernels:
    This transformation includes six embedded Python strings: `project_json_src`, `host_tiling_src`, `host_operator_src`, `kernel_src`, `python_bind_src` and `model_src`.
    The kernel function name in `kernel_src` must exactly match the provided kernel name. The operator definition in `project_json_src` and `host_operator_src` should also correspond to the kernel name, but follow PascalCase naming:
    ```python
    {example_new_arch_src}
    ``` \n
    """
    else:
        prompt += ASCENDC_OUTPUT_FORMAT_NO_EXAMPLE.format(
            op_name=op,
            op_pascal=underscore_to_pascalcase(op),
        )

    prompt += f"""
    Now, you are given the following architecture with kernel name {op}(PascalCase: {underscore_to_pascalcase(op)}): \n
    ```python
    {arc_src}
    ```
        """
    prompt += ASCENDC_PROBLEM_INSTRUCTION
    return prompt
