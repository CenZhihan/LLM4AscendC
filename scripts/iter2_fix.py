#!/usr/bin/env python3
"""Iter2 fix script: feeds eval errors back to model to generate fixed kernels."""
import sys, os, json, re, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))

from openai import OpenAI
from generator.agent.agent_config import get_llm_config_compatible

def fix_kernel(op, tag, error_info, retries=2):
    cfg = get_llm_config_compatible()
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    model = cfg["model"]

    # Read original prompt
    prompt_path = f"/tmp/prompt_{op}.txt"
    if os.path.exists(prompt_path):
        with open(prompt_path) as f:
            base_prompt = f.read()
        lines = base_prompt.split("\n")
        if lines[0].startswith("[INFO]"):
            base_prompt = "\n".join(lines[1:])
    else:
        print(f"NO PROMPT for {op}")
        return False

    # Read original code
    txt_path = os.path.join(REPO_ROOT, "output", "claudeCase", tag, f"{op}.txt")
    original_code = ""
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            original_code = f.read()

    fix_prompt = f"""{base_prompt}

IMPORTANT: The PREVIOUS version of this kernel FAILED evaluation. Here are the errors:

{error_info}

Please fix ALL issues in the kernel code above. Pay attention to:
1. Include paths must use regular quotes (""), not escaped quotes
2. Do NOT use `TBuf` with `TPipe::InitBuffer` - `InitBuffer` only supports `TQue` queues
3. Ensure all Ascend C APIs are used correctly with proper parameters
4. Make sure the kernel is numerically stable and doesn't crash at runtime
5. For activation ops like relu/gelu use the simple element-wise operations

Output the complete fixed code with all 6 variables in the same format as before.
"""

    for attempt in range(retries + 1):
        print(f"[{op}] iter2 fix attempt {attempt+1}...")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.1,
                timeout=300,
            )
            content = resp.choices[0].message.content
            if not content:
                continue

            # Extract code block
            text = content.strip()
            code = text
            m = re.search(r'```python\s*\n(.*?)\n\s*```', text, re.DOTALL)
            if m:
                code = m.group(1).strip()
            else:
                m = re.search(r'```\s*\n(.*?)\n\s*```', text, re.DOTALL)
                if m:
                    code = m.group(1).strip()

            # Validate
            local_vars = {}
            exec(code, {}, local_vars)
            expected = ['project_json_src','host_tiling_src','host_operator_src','kernel_src','python_bind_src','model_src']
            missing = [v for v in expected if v not in local_vars]
            if missing:
                print(f"[{op}] Missing vars: {missing}")
                continue

            # Save
            with open(txt_path, 'w') as f:
                f.write(code)
            print(f"[{op}] Fixed kernel saved ({len(code)} chars)")

            # Update trace
            trace_path = txt_path.replace(".txt", "_trace.json")
            if os.path.exists(trace_path):
                with open(trace_path) as f:
                    t = json.load(f)
                t["iterations"]["iter2"] = {
                    "tool_calls": [],
                    "feedback_from_iter1": error_info[:500],
                    "generation": code,
                    "eval": {}
                }
                with open(trace_path, 'w') as f:
                    json.dump(t, f, indent=2)

            return True
        except Exception as e:
            print(f"[{op}] Error: {e}")
            if attempt < retries:
                time.sleep(5)

    return False

if __name__ == "__main__":
    op = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "no_tool_baseline"
    error_file = sys.argv[3] if len(sys.argv) > 3 else ""

    if error_file and os.path.exists(error_file):
        with open(error_file) as f:
            error_info = f.read()
    else:
        # Read error from trace
        trace_path = f"/root/LLM4AscendC/output/claudeCase/{tag}/{op}_trace.json"
        if os.path.exists(trace_path):
            with open(trace_path) as f:
                t = json.load(f)
            error_info = t.get("iterations", {}).get("iter1", {}).get("eval", {}).get("error_summary", "unknown error")
        else:
            error_info = "Build or precision error"

    fix_kernel(op, tag, error_info)
