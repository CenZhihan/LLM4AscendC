"""Single-op LLM call: stream completion to op.txt / op_cot.txt using XI_AI_* client."""
from __future__ import annotations

import os
from typing import Any

from generation import gen_config
from generation.llm_config import get_xi_model_name


def generate_and_write_single(
    prompt: str,
    client: Any,
    out_dir: str,
    op: str,
    model: str | None = None,
) -> None:
    model = model or get_xi_model_name()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=gen_config.temperature,
        n=gen_config.num_completions,
        top_p=gen_config.top_p,
    )
    reasoning_content = ""
    answer_content = ""
    is_answering = False
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
            answer_content += delta.content
    if reasoning_content:
        with open(os.path.join(out_dir, f"{op}_cot.txt"), "w", encoding="utf-8") as out_file:
            out_file.write(reasoning_content)
    with open(os.path.join(out_dir, f"{op}.txt"), "w", encoding="utf-8") as out_file:
        out_file.write(answer_content)
