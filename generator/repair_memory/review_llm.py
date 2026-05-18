from __future__ import annotations

from typing import Any, Dict

from .llm_util import assistant_message_text, openai_client_from_llm_config

_COT_PREFIXES = (
    "we need to",
    "we are given",
    "following the pattern",
    "write exactly one",
)

# Post-filter: context-sensitive host APIs must name where they apply (not prompt-specific).
_CONTEXT_SENSITIVE_MARKERS = (
    "getinputshape",
    "getoutputshape",
    "storageshape",
    "gert::shape",
    "getoriginshape",
    "setdimnum",
)

_WHERE_MARKERS = (
    "tilingfunc",
    "infershape",
    "tilingcontext",
    "infershapecontext",
    "op_host",
    "host_operator",
    "host tiling",
    "kernel",
    "python_bind",
    "txt bundle",
    "parse_txt",
    "opbuild",
    "cpack",
    "cmake",
    "eval",
    "correctness",
    "compile",
    "507035",
    "507034",
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


def _has_scenario_scope(s: str) -> bool:
    """Reject host/API advice that does not state where it applies (function, stage, or artifact)."""
    tl = (s or "").lower()
    if any(m in tl for m in _CONTEXT_SENSITIVE_MARKERS):
        return any(m in tl for m in _WHERE_MARKERS)
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
        "When <trigger + where it applies>, do not <ineffective approach>; instead <effective approach>.\n\n"
        "Rules:\n"
        "- The **When** clause must pin the **applicable scenario** (where/when this advice holds): e.g. failure stage "
        "(txt bundle, opbuild, eval), host function or file (TilingFunc, InferShape, kernel), or symptom vs root-cause type. "
        "Do not write advice that could be reused blindly on unrelated stages or code paths.\n"
        "- Prefer root_cause / log_excerpt for compile/API triggers; treat CPack/INSTALL/opbuild as symptom unless logs "
        "show no earlier compile error.\n"
        "- Name concrete APIs, types, or symbols from the logs or diff when available.\n"
        "- Example (scope matters): bad — \"When GetInputShape returns Shape*, do not use StorageShape*; instead use Shape*.\" "
        "good — \"When opbuild fails in InferShape with Shape* vs StorageShape* mismatch, do not use StorageShape* from "
        "GetInputShape; instead use gert::Shape* and SetDim.\" (same API, different host functions — state which one.)\n"
        "- Output ONLY that one sentence. No meta commentary.\n"
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
                        "The When clause must state where the advice applies, not only what error text appeared. "
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
    if not _has_scenario_scope(text):
        return ""
    return text[:1200]
