"""Prompt builder for CUDA-Agent-Ops-6K fused tasks (six-string txt bundle, no eval_src)."""

from __future__ import annotations

import json
from typing import Any, Dict

from generator.prompt_generators.prompt_utils import (
    ASCENDC_OUTPUT_FORMAT_NO_EXAMPLE,
    ASCENDC_PROBLEM_INSTRUCTION,
    ASCENDC_PROBLEM_STATEMENT,
    load_ascendc_example_pair,
)
from generator.text_utils import underscore_to_pascalcase

_SUPPORTED_STRATEGIES = frozenset({"one_shot", "none"})


def build_cuda_agent_fused_prompt(
    strategy_name: str,
    row: Dict[str, Any],
    op_key: str,
) -> str:
    """
    Build AscendC agent prompt from one CUDA-Agent-Ops-6K jsonl row.

    Output contract: exactly six string variables (project_json_src ... model_src).
    Do not emit eval_src; evaluation loads golden PyTorch from the same dataset row at harness time.

    ``strategy_name`` (CUDA-Agent entry only): ``one_shot`` inserts the leaky_relu few-shot pair — same files as
    single-operator ``ascendc`` + ``one_shot``. ``none`` skips that block but keeps the fused task + format rules.
    """
    if strategy_name not in _SUPPORTED_STRATEGIES:
        raise ValueError(
            f"CUDA-Agent fused prompts currently support strategies {_SUPPORTED_STRATEGIES!r}; "
            f"got {strategy_name!r}"
        )

    ref_code = str(row.get("code") or "")
    ops_raw = row.get("ops")
    if isinstance(ops_raw, str):
        try:
            ops_list = json.loads(ops_raw)
        except json.JSONDecodeError as e:
            raise ValueError("row['ops'] must be valid JSON array string") from e
    elif isinstance(ops_raw, list):
        ops_list = ops_raw
    else:
        raise ValueError("row['ops'] must be a JSON string or list")

    data_source = str(row.get("data_source") or "unknown")
    op_name = f"{op_key}_custom"
    op_pascal = underscore_to_pascalcase(op_name)

    ops_bulleted = "\n".join(f"  - {x}" for x in ops_list)

    use_few_shot = strategy_name == "one_shot"

    prompt = ASCENDC_PROBLEM_STATEMENT
    if use_few_shot:
        example_arch_src, example_new_arch_src = load_ascendc_example_pair(
            example="leaky_relu",
            language="ascendc",
        )
        ex_display = "leaky_relu_custom"
        prompt += f"""
Here is an **example** (same files as single-operator ``ascendc`` strategy ``one_shot``: ``cuda_model_leaky_relu.py`` + ``ascendc_new_model_leaky_relu.py``). **Original toy architecture (kernel name `{ex_display}`):**
```python
{example_arch_src}
```
Transformed version using custom AscendC kernels (six embedded Python strings: `project_json_src`, `host_tiling_src`, `host_operator_src`, `kernel_src`, `python_bind_src`, `model_src`):
```python
{example_new_arch_src}
```
Your answer for the **CUDA-Agent task below** must use the **same assignment style** as this example:
six variables assigned to triple-quoted Python strings (no raw-string prefix before the opening quote).
Do not use section headings like === project_json_src === instead of real assignments.

"""

    prompt += f"""
You are implementing a **fused** custom AscendC operator for a multi-op PyTorch reference program.
Task id (use as kernel/operator naming stem): `{op_key}`.
Dataset field `data_source`: {data_source!r}.

**Fused API list (replace / fuse these in the reference forward into one or more custom ops as you see fit):**
{ops_bulleted}

**Reference PyTorch program (class `Model` is the golden behavior; you will provide `ModelNew` in model_src only):**
```python
{ref_code}
```

**Constraints:**
- Implement a single fused custom op (or a small set) on NPU that matches the reference `Model` outputs
  for the same inputs (numerical match is verified by the offline harness using the same `code` as golden).
- Output **exactly six** Python string variables: `project_json_src`, `host_tiling_src`, `host_operator_src`,
  `kernel_src`, `python_bind_src`, `model_src`. Do **not** define `eval_src`.
- Use kernel/JSON/operator names derived from `{op_name}` and PascalCase `{op_pascal}` in project JSON.
"""
    prompt += ASCENDC_OUTPUT_FORMAT_NO_EXAMPLE.format(
        op_name=op_name,
        op_pascal=op_pascal,
    )
    prompt += ASCENDC_PROBLEM_INSTRUCTION
    return prompt
