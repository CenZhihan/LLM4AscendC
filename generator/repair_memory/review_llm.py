from __future__ import annotations

from typing import Any, Dict

from .llm_util import assistant_message_text, openai_client_from_llm_config

_COT_PREFIXES = (
    "we need to",
    "we are given",
    "following the pattern",
    "write exactly one",
)


def _has_template_parts(s: str) -> bool:
    t = (s or "").strip()
    if len(t) < 30:
        return False
    tl = t.lower()
    if not tl.startswith("when "):
        return False
    if any(tl.startswith(p) for p in _COT_PREFIXES):
        return False
    if "following the pattern" in tl:
        return False
    has_neg = "do not " in tl or "don't " in tl or "dont " in tl
    has_instead = "instead " in tl
    if not (has_neg and has_instead):
        return False
    if len(t) > 400 and "; instead" not in tl:
        return False
    return True


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
        "When <concrete trigger, including error type or log gist>, do not <ineffective approach>; instead <effective approach>.\n\n"
        "Rules:\n"
        "- Use root_cause / root_cause_anchor and log_excerpt for the trigger when compile/API errors are present "
        "(e.g. C++ 'error: no member named', 'cannot convert', kernel compilation errors).\n"
        "- symptom / symptom_anchor (CPack, INSTALL, opbuild) is often a downstream effect; do not treat packaging "
        "alone as the root fix unless no compile error exists in the signals.\n"
        "- Name specific APIs, types, or symbols from the logs or diff when available.\n"
        "- Output ONLY that one sentence. No meta commentary (do not write 'We need to...' or restate these instructions).\n"
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
                        "You output only that single English conditional sentence starting with 'When '. "
                        "No quotes, no bullet points, no second sentence, no planning text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        text = assistant_message_text(resp.choices[0].message)
    except Exception:
        return ""
    if not _has_template_parts(text):
        return ""
    return text[:1200]
