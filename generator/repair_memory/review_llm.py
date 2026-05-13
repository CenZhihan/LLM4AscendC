from __future__ import annotations

from typing import Any, Dict

from .llm_util import openai_client_from_llm_config


def _has_template_parts(s: str) -> bool:
    t = (s or "").strip()
    if len(t) < 30:
        return False
    tl = t.lower().strip()
    # English conditional: When ... do not / don't ... instead ...
    has_when = tl.startswith("when ") or " when " in tl
    has_neg = "do not " in tl or "don't " in tl or "dont " in tl
    has_instead = "instead " in tl
    return bool(has_when and has_neg and has_instead)


def generate_repair_natural_language(
    *,
    llm_config: Dict[str, Any],
    tier: str,
    prev_summary: str,
    curr_summary: str,
    code_diff_text: str = "",
) -> str:
    client = openai_client_from_llm_config(llm_config)
    model = llm_config.get("model") or ""
    diff_section = ""
    if (code_diff_text or "").strip():
        diff_section = (
            "\nUnified diff between adjacent attempts (generated operator txt; use for API/symbol/structure fixes):\n"
            f"{code_diff_text.strip()}\n"
        )
    prompt = (
        "Write exactly ONE short repair memory in **English**, as a single conditional sentence, using this pattern:\n"
        "When <concrete trigger, including error type or log gist>, do not <ineffective approach>; instead <effective approach>.\n"
        "Ground the trigger and the fix in the log anchors above and in the unified diff when present: "
        "if the diff shows a clear change (e.g. wrong API or symbol → correct one), name it explicitly in the sentence.\n"
        f"Tier={tier} (A = objective compile/correctness improved; B = pipeline stage progressed).\n\n"
        f"Previous attempt signals:\n{prev_summary[:6000]}\n\n"
        f"Current attempt signals:\n{curr_summary[:6000]}\n"
        f"{diff_section}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You output only that single English conditional sentence. "
                        "No quotes, no bullet points, no second sentence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""
    if not _has_template_parts(text):
        return ""
    return text[:1200]
