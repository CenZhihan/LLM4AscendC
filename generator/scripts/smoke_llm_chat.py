#!/usr/bin/env python3
"""
最小聊天补全冒烟：使用与 Agent 相同的 generator/local_api_config.py（XI_AI_*）。

用法（在 LLM4AscendC 仓库根目录）：
  python3 generator/scripts/smoke_llm_chat.py
  python3 generator/scripts/smoke_llm_chat.py --model deepseek-v4-flash
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import APIStatusError, OpenAI

from generator.agent.agent_config import get_llm_config_compatible


def _mask_key(key: str) -> str:
    k = (key or "").strip()
    if len(k) <= 12:
        return "***"
    return f"{k[:6]}...{k[-4:]}"


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke test OpenAI-compatible chat.completions")
    p.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="覆盖 local_api_config 中的 XI_AI_MODEL（默认 deepseek-v4-flash）",
    )
    args = p.parse_args()

    try:
        cfg = get_llm_config_compatible(cli_model=args.model)
    except Exception as e:
        print(f"[config] 加载失败: {e}")
        return 2

    api_key = cfg["api_key"]
    base_url = cfg["base_url"]
    model = cfg["model"]
    print("[config] base_url:", base_url)
    print("[config] model   :", model)
    print("[config] api_key :", _mask_key(api_key))

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "user", "content": "Reply with exactly one word: pong"},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            # reasoning 类模型可能在正文前占用较多 token，过小会得到 content 为空且 finish_reason=length
            max_tokens=2048,
        )
    except APIStatusError as e:
        print("[call] HTTP", getattr(e, "status_code", None), repr(e))
        body = getattr(e, "response", None)
        if body is not None:
            try:
                txt = body.text
            except Exception:
                txt = None
            if txt:
                print("[call] response body:", txt[:4000])
        traceback.print_exc()
        return 1
    except Exception as e:
        print("[call] 失败:", repr(e))
        traceback.print_exc()
        return 1

    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    print("[ok] finish_reason:", getattr(choice, "finish_reason", None))
    print("[ok] content:", content[:2000] if content else "(empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
