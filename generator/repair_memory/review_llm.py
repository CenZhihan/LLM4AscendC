from __future__ import annotations

import json
import re
from typing import Any, Dict

from .llm_util import openai_client_from_llm_config


def _has_template_parts(s: str) -> bool:
    t = (s or "").strip()
    if len(t) < 30:
        return False
    # expect 当 / 不要 / 应 or English fallback
    ok_cn = ("当" in t) and ("不要" in t) and ("应" in t)
    ok_en = ("when" in t.lower()) and ("do not" in t.lower() or "don't" in t.lower()) and ("instead" in t.lower())
    return ok_cn or ok_en


def generate_repair_natural_language(
    *,
    llm_config: Dict[str, Any],
    tier: str,
    prev_summary: str,
    curr_summary: str,
) -> str:
    client = openai_client_from_llm_config(llm_config)
    model = llm_config.get("model") or ""
    prompt = (
        "Write ONE short repair memory in Chinese using this exact pattern:\n"
        "当 <触发条件，含错误类型或日志要点> 时，不要 <无效做法> ，应 <有效做法> 。\n"
        f"Tier={tier} (A=objective compile/correctness improved; B=pipeline stage progressed).\n\n"
        f"Previous attempt signals:\n{prev_summary[:6000]}\n\n"
        f"Current attempt signals:\n{curr_summary[:6000]}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output only the single conditional sentence, no quotes."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""
    if not _has_template_parts(text):
        return ""
    return text[:1200]
