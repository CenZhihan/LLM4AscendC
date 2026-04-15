"""Unified OpenAI-compatible client for AscendC generation (direct + Agent).

Resolution order:
1) If USE_API_CONFIG=1 (or true): load generator/local_api_config.py (copy from local_api_config.example.py).
2) Else: environment variables XI_AI_API_KEY, XI_AI_BASE_URL, XI_AI_MODEL.
"""
from __future__ import annotations

import os


def _load_file_api_config() -> dict[str, str] | None:
    if os.environ.get("USE_API_CONFIG", "").strip().lower() not in ("1", "true"):
        return None
    try:
        from generator import local_api_config as lac
    except ImportError as e:
        raise SystemExit(
            "USE_API_CONFIG=1 is set but generator/local_api_config.py could not be imported.\n"
            "Copy generator/local_api_config.example.py to generator/local_api_config.py "
            "and fill in XI_AI_API_KEY (and optionally XI_AI_BASE_URL, XI_AI_MODEL).\n"
            f"Import error: {e}"
        ) from e

    key = (
        (getattr(lac, "XI_AI_API_KEY", None) or getattr(lac, "OPENAI_API_KEY", None) or "")
    ).strip()
    base = (
        (getattr(lac, "XI_AI_BASE_URL", None) or getattr(lac, "OPENAI_API_BASE", None) or "")
    ).strip()
    model = (
        (getattr(lac, "XI_AI_MODEL", None) or getattr(lac, "MODEL", None) or "gpt-5")
    ).strip()
    if not key:
        raise SystemExit(
            "generator/local_api_config.py: set XI_AI_API_KEY or OPENAI_API_KEY (non-empty)."
        )
    if not base:
        base = "https://api-2.xi-ai.cn/v1"
    return {
        "api_key": key,
        "base_url": base,
        "model": model,
    }


def get_llm_config_from_env() -> dict[str, str]:
    file_cfg = _load_file_api_config()
    if file_cfg is not None:
        return file_cfg

    api_key = os.getenv("XI_AI_API_KEY")
    if not api_key or not str(api_key).strip():
        raise SystemExit(
            "Missing XI_AI_API_KEY. Either:\n"
            "  export USE_API_CONFIG=1\n"
            "  and create generator/local_api_config.py from local_api_config.example.py\n"
            "or:\n"
            "  export XI_AI_API_KEY=your_key\n"
            "  export XI_AI_BASE_URL=https://api-2.xi-ai.cn/v1   # optional\n"
            "  export XI_AI_MODEL=gpt-5   # optional\n"
        )
    base_url = os.getenv("XI_AI_BASE_URL", "https://api-2.xi-ai.cn/v1").strip()
    model_name = os.getenv("XI_AI_MODEL", "gpt-5").strip()
    return {
        "api_key": api_key.strip(),
        "base_url": base_url,
        "model": model_name,
    }


def get_xi_openai_client():
    from openai import OpenAI

    cfg = get_llm_config_from_env()
    return OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=1_000_000,
        max_retries=3,
    )


def get_xi_model_name() -> str:
    return get_llm_config_from_env()["model"]
