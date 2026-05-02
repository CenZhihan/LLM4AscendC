#!/usr/bin/env python3
"""Assemble the prompt for a given operator using the same logic as prompt_utils."""
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from generator.text_utils import read_file, underscore_to_pascalcase
from generator.prompt_generators.prompt_utils import (
    ASCENDC_PROBLEM_STATEMENT,
    ASCENDC_OUTPUT_FORMAT_NO_EXAMPLE,
)

def build_prompt(op: str, category: str = "activation"):
    op_custom = op + "_custom"
    op_pascal = underscore_to_pascalcase(op_custom)

    # Read reference implementation
    ref_path = os.path.join(REPO_ROOT, "vendor", "mkb", "reference", category, f"{op}.py")
    arc_src = read_file(ref_path)

    # Read example files
    prompts_dir = os.path.join(REPO_ROOT, "generator", "prompts")
    example_arch = read_file(os.path.join(prompts_dir, "cuda_model_leaky_relu.py"))
    example_new_arch = read_file(os.path.join(prompts_dir, "ascendc_new_model_leaky_relu.py"))

    example_op = "leaky_relu"

    prompt = ASCENDC_PROBLEM_STATEMENT

    prompt += f"""
Here is an example to illustrate the expected transformation using custom AscendC operators. **Original architecture with kernel name `{example_op}_custom`:**

```python

{example_arch}
```

Transformed version using custom AscendC kernels:
This transformation includes six embedded Python strings: `project_json_src`, `host_tiling_src`, `host_operator_src`, `kernel_src`, `python_bind_src` and `model_src`.
The kernel function name in `kernel_src` must exactly match the provided kernel name. The operator definition in `project_json_src` and `host_operator_src` should also correspond to the kernel name, but follow PascalCase naming:

```python
{example_new_arch}
```

"""

    prompt += f"""
Now, you are given the following architecture with kernel name {op_custom} (PascalCase: {op_pascal}):

```python
{arc_src}
```

"""
    prompt += """
Your task: Replace relevant PyTorch operators in the architecture named Model with custom AscendC kernels. Generate an optimized version named ModelNew, including the six Python strings listed above. Just output the code, no other text, and NO testing code!

Output format (strict, no examples given here—follow these rules only):

1) You must output exactly one code block: start with a line ```python and end with ```. Do NOT output any prose, description, ellipsis (e.g. "..."), or markdown before or after this block. Do NOT put any non-Python line inside the block.

2) The block must be valid, executable Python: use 4 spaces for indentation (no tabs); no syntax errors; no trailing comment lines that look like prose. Use only ASCII in the entire block (no Chinese characters or fullwidth punctuation such as 。、（，）；write all comments and docstrings in English). The entire block will be run as Python to extract the six variables.

3) The block must define exactly these six string variables (e.g. using r\"\"\" or r''' for multi-line strings):
- project_json_src: JSON string for the operator project. The parsed JSON must have a top-level key \"op\" (required by the Ascend build toolchain). Under \"op\" put operator metadata (name, inputs, outputs, kernel/tiling/sources, etc.). Do not use other top-level shapes like \"op_name\" or \"operator\" alone—the root must contain \"op\".
- host_tiling_src: Tiling header (.h) content
- host_operator_src: Host-side operator (.cpp) content
- kernel_src: AscendC kernel (.cpp) content (kernel name must match the given operator name)
- python_bind_src: Python binding code
- model_src: PyTorch model code that uses the custom op (class ModelNew)

4) Naming: kernel and operator names must use the given name (e.g. """ + op_custom + """) and PascalCase in JSON/host (e.g. """ + op_pascal + """).
"""

    return prompt

if __name__ == "__main__":
    op = sys.argv[1] if len(sys.argv) > 1 else "relu"
    category = sys.argv[2] if len(sys.argv) > 2 else "activation"
    print(build_prompt(op, category))
