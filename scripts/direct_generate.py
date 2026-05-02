#!/usr/bin/env python3
"""
Direct kernel generator using model API.
Bypasses subagent overhead for no-tool experiments.
Usage: python3 scripts/direct_generate.py <op> <experiment_tag> [category]
"""
import sys, os, json, re, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))

from openai import OpenAI
from generator.agent.agent_config import get_llm_config_compatible

def load_prompt(op):
    p = os.path.join(REPO_ROOT, "tmp", f"prompt_{op}.txt")
    if not os.path.exists(p):
        alt = f"/tmp/prompt_{op}.txt"
        if os.path.exists(alt):
            p = alt
    if not os.path.exists(p):
        # Generate prompt
        from scripts.build_prompt import build_prompt
        prompt = build_prompt(op)
    else:
        with open(p) as f:
            content = f.read()
        # Strip [INFO] line if present
        lines = content.split("\n")
        if lines[0].startswith("[INFO]"):
            content = "\n".join(lines[1:])
        prompt = content
    return prompt

def extract_code_block(text):
    """Extract the Python code block from model output."""
    text = text.strip()
    # Remove thinking/reasoning content if present
    if "<｜end▁of▁thinking｜>" in text:
        text = text.split(" response")[-1]
    # Find ```python ... ``` block
    m = re.search(r'```python\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try without language tag
    m = re.search(r'```\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text

def extract_txt_block(text):
    """Extract txt content (already stripped or raw)."""
    code = extract_code_block(text)
    return code

def validate_code(content):
    """Verify 6 variables exist."""
    local_vars = {}
    try:
        exec(content, {}, local_vars)
        expected = ['project_json_src','host_tiling_src','host_operator_src','kernel_src','python_bind_src','model_src']
        missing = [v for v in expected if v not in local_vars]
        return len(missing) == 0, missing, local_vars
    except Exception as e:
        return False, str(e), {}

def save_files(op, tag, content, trace_data=None):
    output_dir = os.path.join(REPO_ROOT, "output", "claudeCase", tag)
    os.makedirs(output_dir, exist_ok=True)

    # Stripped content for storage
    stripped = content.strip()
    if stripped.startswith("```python"):
        stripped = stripped[len("```python"):].strip()
    if stripped.endswith("```"):
        stripped = stripped[:-len("```")].strip()

    # Save txt
    txt_path = os.path.join(output_dir, f"{op}.txt")
    with open(txt_path, 'w') as f:
        f.write(stripped)

    # Save trace
    if trace_data is None:
        trace_data = {
            "experiment_tag": tag,
            "enabled_tools": [],
            "operator": op,
            "category": "activation",
            "difficulty": "simple",
            "round_mode": "single_round",
            "iterations": {
                "iter1": {
                    "tool_calls": [],
                    "generation": stripped,
                    "eval": {}
                }
            }
        }
    trace_path = os.path.join(output_dir, f"{op}_trace.json")
    with open(trace_path, 'w') as f:
        json.dump(trace_data, f, indent=2)

    return txt_path, trace_path

def generate(op, tag, retries=2):
    cfg = get_llm_config_compatible()
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    model = cfg["model"]

    prompt = load_prompt(op)

    for attempt in range(retries + 1):
        print(f"[{op}] Generating (attempt {attempt+1})...")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                stream=False,
                timeout=300,
            )
            content = resp.choices[0].message.content
            if not content:
                print(f"[{op}] Empty response")
                continue

            code = extract_txt_block(content)
            ok, info, _ = validate_code(code)
            if not ok:
                print(f"[{op}] Validation failed: {info}")
                continue

            txt_path, trace_path = save_files(op, tag, code)
            print(f"[{op}] OK - saved to {txt_path}")
            return True
        except Exception as e:
            print(f"[{op}] Error: {e}")
            if attempt < retries:
                time.sleep(5)
    print(f"[{op}] FAILED after {retries+1} attempts")
    return False

if __name__ == "__main__":
    op = sys.argv[1] if len(sys.argv) > 1 else "relu"
    tag = sys.argv[2] if len(sys.argv) > 2 else "no_tool_single_round"
    generate(op, tag)
