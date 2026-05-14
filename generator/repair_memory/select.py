from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .llm_util import openai_client_from_llm_config


SYSTEM = """You pick past repair memories that help the current AscendC kernel repair task.
Each manifest line is one memory (tab-separated fields: id, op, category, tool_mode, tier, anchor, summary).
Return a JSON object ONLY of the form: {"memory_ids": ["..."]} with at most N ids from the manifest.

Selection policy (prefer recall over empty):
- If the manifest contains entries whose anchor or summary plausibly matches the current failure
  (same error class, overlapping log phrases, same toolchain stage such as txt bundle / CPack /
  opbuild / tiling / NPU runtime), include their ids even when the originating op differs.
- Prefer diverse, complementary hints (e.g. one about missing txt blocks and one about CPack)
  when both are relevant.
- Return {"memory_ids": []} only when every manifest line is clearly unrelated to the query
  repair_context and operator context, or the manifest is empty.
- Do not fabricate ids: every id MUST appear exactly as id=... on a manifest line."""


def select_memory_ids(
    *,
    llm_config: Dict[str, Any],
    manifest_text: str,
    query_text: str,
    max_n: int = 5,
) -> List[str]:
    if not (manifest_text or "").strip():
        return []
    client = openai_client_from_llm_config(llm_config)
    model = llm_config.get("model") or ""
    user = (
        f"Select up to {max_n} memory ids.\n\n"
        f"=== Manifest ===\n{manifest_text}\n\n=== Current query ===\n{query_text[:12000]}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM.replace("N", str(max_n))},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return []

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    ids = obj.get("memory_ids")
    if not isinstance(ids, list):
        return []
    out: List[str] = []
    for x in ids:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
        if len(out) >= max_n:
            break
    return out
