#!/usr/bin/env python3
"""
验证 DeepSeek-V4-Flash 在「非流式 / 流式」下 message 字段与选条解析行为。

背景（与主生成管道对齐）：
- 主 agent（answer / choose_tool）使用 **streaming**，最终答案在 **delta.content** 累积。
- 选条曾用 **非流式** + 只读 ``message.content``：DeepSeek 常为 **空**，而 CoT 在 ``reasoning_content``；
  且整段 CoT 里会复述带 ``memory_ids`` 的示例 JSON，**贪婪** ``\{[\\s\\S]*\\}`` 会误匹配。
- 现 ``select_repair_memories``：**流式** + ``raw_decode`` 扫描取 **最后一个** 含 ``memory_ids`` 的 JSON。

用法：
  cd LLM4AscendC
  python generator/scripts/tmp_verify_assistant_message_fields.py --model deepseek-v4-flash
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_local_api = REPO_ROOT / "generator" / "local_api_config.py"
if _local_api.is_file() and os.environ.get("USE_API_CONFIG", "").strip().lower() not in (
    "1",
    "true",
):
    os.environ["USE_API_CONFIG"] = "1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="",
        help="覆盖配置文件中的模型名（建议 deepseek-v4-flash）",
    )
    args = parser.parse_args()

    from generator.llm_config import get_llm_config_from_env
    from generator.repair_memory.llm_util import (
        assistant_message_text,
        chat_completion_stream_content_reasoning,
        openai_client_from_llm_config,
    )
    from generator.repair_memory.select import parse_memory_selection_output, select_repair_memories

    cfg = dict(get_llm_config_from_env())
    if args.model.strip():
        cfg["model"] = args.model.strip()

    client = openai_client_from_llm_config(cfg)
    model = cfg.get("model") or ""

    system = (
        "Return a JSON object ONLY of this shape (two keys), no markdown fences:\n"
        '{"memory_ids": [], "selection_rationale": "smoke"}'
    )
    user = (
        "Manifest line: id=dummy-op-uuid\top=x\tcategory=activation\t"
        "tool_mode=no_tool\ttier=A\tanchor=txt\tsummary=test.\n"
        "Pick zero or one id; output JSON only."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    print("=== config ===")
    print("model:", model)
    bu = str(cfg.get("base_url") or "")
    print("base_url:", bu[:88] + ("..." if len(bu) > 88 else ""))

    # --- 1) 非流式：观察 content / reasoning（历史上选条只读 content 会挂）---
    print("\n=== A) non-stream chat.completions ===")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            stream=False,
        )
    except Exception as e:
        print("API error:", repr(e))
        return 2

    msg = resp.choices[0].message
    c = (getattr(msg, "content", None) or "").strip()
    r = (getattr(msg, "reasoning_content", None) or "").strip()
    print("len(message.content):", len(c))
    print("len(message.reasoning_content):", len(r))
    merged_ns = assistant_message_text(msg)
    print("len(assistant_message_text):", len(merged_ns))
    p_ns = parse_memory_selection_output(merged_ns, max_n=5)
    print("parse_ok (non-stream merged):", p_ns.get("parse_ok"), "parse_error:", repr(p_ns.get("parse_error")))

    # --- 2) 流式：与主 agent / 当前 select 一致 ---
    print("\n=== B) stream (repair_memory select path) ===")
    try:
        content_s, reasoning_s = chat_completion_stream_content_reasoning(
            client,
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
    except Exception as e:
        print("stream API error:", repr(e))
        return 2

    print("len(stream content):", len(content_s))
    print("len(stream reasoning):", len(reasoning_s))
    raw_stream = (content_s or "").strip() or (reasoning_s or "").strip()
    print("len(raw for parse):", len(raw_stream))
    if len(raw_stream) <= 600:
        print("raw preview:", raw_stream)
    else:
        print("raw head:", raw_stream[:400], "\n... [snip] ...\n", "raw tail:", raw_stream[-400:])

    p_st = parse_memory_selection_output(raw_stream, max_n=5)
    print("parse_ok (stream):", p_st.get("parse_ok"))
    print("memory_ids:", p_st.get("memory_ids"))
    print("selection_rationale:", repr(p_st.get("selection_rationale")))
    print("parse_error:", repr(p_st.get("parse_error")))

    # --- 3) 端到端：select_repair_memories（含真实 manifest 形状）---
    print("\n=== C) select_repair_memories end-to-end ===")
    manifest = (
        "id=11111111-1111-1111-1111-111111111111\top=elu\tcategory=activation\t"
        "tool_mode=no_tool\ttier=B\tanchor=CMake Error\tsummary=CPack fix hint.\n"
        "id=22222222-2222-2222-2222-222222222222\top=relu\tcategory=activation\t"
        "tool_mode=no_tool\ttier=A\tanchor=txt bundle\tsummary=txt blocks hint."
    )
    query = (
        "op=relu\ncategory=activation\ntool_mode=no_tool\neval_mode=full\nattempt_id=2\n"
        "repair_context:\nValueError: txt bundle missing blocks: ['kernel_src']\n"
    )
    try:
        sel = select_repair_memories(
            llm_config=cfg,
            manifest_text=manifest,
            query_text=query,
            max_n=2,
        )
    except Exception as e:
        print("select_repair_memories error:", repr(e))
        return 2

    print("parse_ok:", sel.get("parse_ok"))
    print("parse_error:", repr(sel.get("parse_error")))
    print("memory_ids:", sel.get("memory_ids"))
    print("selection_rationale:", repr((sel.get("selection_rationale") or "")[:500]))
    raw_out = (sel.get("raw_model_output") or "")[:800]
    print("raw_model_output (first 800 chars):", raw_out if raw_out else "(empty)")

    ok_stream = bool(p_st.get("parse_ok"))
    ok_e2e = bool(sel.get("parse_ok"))
    if ok_stream and ok_e2e:
        print("\nOK: 流式 + raw_decode 选条路径在 deepseek-v4-flash 下可用。")
        return 0
    if ok_e2e:
        print("\nPARTIAL: 端到端 OK，但 B 阶段解析未过（可对比 raw）。")
        return 0
    if ok_stream:
        print("\nPARTIAL: B 阶段 OK，端到端未过（检查 manifest / 配额）。")
        return 0

    print("\nFAIL: 请检查上方 len(content) vs len(reasoning) 及 parse_error。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
