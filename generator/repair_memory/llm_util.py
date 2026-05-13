from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI


def openai_client_from_llm_config(llm_config: Dict[str, Any]) -> OpenAI:
    return OpenAI(
        api_key=llm_config["api_key"],
        base_url=(llm_config.get("base_url") or "").strip() or None,
    )
