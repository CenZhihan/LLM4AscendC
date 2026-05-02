#!/usr/bin/env python3
"""
Tool-based kernel generation experiment.
Handles tool orchestration, kernel generation, and result saving for tool-enabled experiments.

Usage: python3 scripts/tool_experiment.py <op> <experiment_tag> --tools tool1,tool2 [...]
"""
import sys, os, json, re, time
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))

from openai import OpenAI
from generator.agent.agent_config import get_llm_config_compatible

# Tool definitions for orchestrator prompt
TOOL_DEFINITIONS = {
    "code_search_snippet": {
        "name": "code_search_snippet",
        "display_name": "Code Search Snippet",
        "description": "Retrieve relevant Ascend C code snippets from curated asc-devkit sources",
        "usage_guidance": "Use for finding code examples of specific Ascend C APIs or operator patterns",
    },
    "api_lookup": {
        "name": "api_lookup",
        "display_name": "API Lookup",
        "description": "Query API signature for a specific Ascend C API symbol",
        "usage_guidance": "Use when you need exact API signature including parameter types and order",
    },
    "api_constraint": {
        "name": "api_constraint",
        "display_name": "API Constraint Check",
        "description": "Check usage constraints for a specific Ascend C API",
        "usage_guidance": "Use after api_lookup to check restrictions on API usage",
    },
    "kb_shell_search": {
        "name": "kb_shell_search",
        "display_name": "KB Shell Search",
        "description": "Search Knowledge/ directory for local documentation",
        "usage_guidance": "Use for finding local Ascend C documentation and best practices",
    },
    "ascend_search": {
        "name": "ascend_search",
        "display_name": "Ascend Docs Search",
        "description": "Search online Ascend documentation",
        "usage_guidance": "Use for finding online documentation for Ascend C APIs",
    },
    "ascend_fetch": {
        "name": "ascend_fetch",
        "display_name": "Ascend Docs Fetch",
        "description": "Fetch content from a specific Ascend documentation URL",
        "usage_guidance": "Use after ascend_search to fetch detailed content from a URL",
    },
    "npu_arch": {
        "name": "npu_arch",
        "display_name": "NPU Architecture Query",
        "description": "Query NPU chip architecture capabilities and limits",
        "usage_guidance": "Use when hardware-specific constraints are needed for kernel design",
    },
}

def build_orchestrator_prompt(op, category, tools_enabled, existing_results="", round_count=0):
    """Build tool selection prompt similar to choose_tool.py logic."""
    tool_descs = []
    for t in tools_enabled:
        if t in TOOL_DEFINITIONS:
            d = TOOL_DEFINITIONS[t]
            tool_descs.append(f"### `{t}` — {d['display_name']}\nSummary: {d['description']}.\n{d['usage_guidance']}")

    tools_str = "\n\n".join(tool_descs)
    tool_keys = ", ".join(f"`{t}`" for t in tools_enabled)
    rounds_left = max(0, 5 - round_count)

    prompt = f"""Task: Generate AscendC kernel for {op} operator (category: {category}).

You are a **tool orchestrator**. Pick the next tool to call to gather evidence before code generation.
Rounds remaining: {rounds_left} of 5.

Available tools: {tool_keys}, or `ANSWER`

{tools_str}

Output exactly one JSON object (no markdown fences):
{{"tool": "<key>", "query": "<question>", "args": null, "thinking": {{"goal": "...", "missing_info": "...", "why_tool": "...", "expected_output": "..."}}}}
Or to finish: {{"tool": "ANSWER", "query": "", "args": null}}
"""
    if existing_results:
        prompt += f"\nAlready retrieved:\n{existing_results}\n"
    return prompt

def call_tool(tool_name, query, args=None):
    """Call a specific tool and return results."""
    if tool_name == "code_search_snippet":
        from generator.agent.retrievers.code_search_snippet_retriever import CodeSearchSnippetRetriever
        retriever = CodeSearchSnippetRetriever()
        results = retriever.retrieve(query, top_k=3)
        return "\n\n".join(str(r) for r in results) if results else "No results found"

    elif tool_name in ("api_lookup", "api_constraint"):
        from generator.agent.retrievers.api_doc_retriever import ApiDocRetriever
        retriever = ApiDocRetriever()
        if tool_name == "api_lookup":
            results = retriever.lookup_signature(query)
        else:
            results = retriever.check_constraints(query)
        return str(results) if results else "No results found"

    elif tool_name == "kb_shell_search":
        from generator.agent.retrievers.kb_shell_search_retriever import KbShellSearchRetriever
        retriever = KbShellSearchRetriever()
        results = retriever.retrieve(query, top_k=3)
        return "\n\n".join(str(r) for r in results) if results else "No results found"

    elif tool_name == "npu_arch":
        from generator.agent.nodes.npu_arch import query_npu_arch
        return str(query_npu_arch(query))

    elif tool_name == "ascend_search":
        from generator.agent.nodes.ascend_search import ascend_search
        return str(ascend_search(query))

    elif tool_name == "ascend_fetch":
        from generator.agent.nodes.ascend_fetch import ascend_fetch
        return str(ascend_fetch(query))

    return f"Tool {tool_name} not available in this script"

def generate_kernel(op, category, prompt, context=""):
    """Call model API to generate kernel code."""
    cfg = get_llm_config_compatible()
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    model = cfg["model"]

    full_prompt = prompt
    if context:
        full_prompt += f"\n\nRetrieved evidence:\n{context}\n"

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.1,
        timeout=300,
    )
    return resp.choices[0].message.content

def extract_code_block(text):
    """Extract Python code block from model output."""
    text = text.strip()
    if "```python" in text:
        m = re.search(r'```python\s*\n(.*?)\n\s*```', text, re.DOTALL)
        if m:
            return m.group(1).strip()
    m = re.search(r'```\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text

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

def save_results(op, tag, code, tool_calls, generation_text, round_mode="two_round", category="activation"):
    """Save txt and trace files."""
    outdir = os.path.join(REPO_ROOT, "output", "claudeCase", tag)
    os.makedirs(outdir, exist_ok=True)

    stripped = code.strip()

    txt_path = os.path.join(outdir, f"{op}.txt")
    with open(txt_path, 'w') as f:
        f.write(stripped)

    trace = {
        "experiment_tag": tag,
        "enabled_tools": list(tool_calls.keys()) if isinstance(tool_calls, dict) else [],
        "operator": op,
        "category": category,
        "difficulty": "simple",
        "round_mode": round_mode,
        "iterations": {
            "iter1": {
                "tool_calls": tool_calls if isinstance(tool_calls, list) else [],
                "generation": generation_text or stripped,
                "eval": {}
            }
        }
    }
    if round_mode == "two_round":
        trace["iterations"]["iter2"] = None

    trace_path = os.path.join(outdir, f"{op}_trace.json")
    with open(trace_path, 'w') as f:
        json.dump(trace, f, indent=2)

    return txt_path, trace_path

def run_experiment(op, tag, tools_enabled, round_mode="two_round", category="activation"):
    print(f"\n{'='*60}")
    print(f"Experiment: {tag}, Operator: {op}")
    print(f"Tools: {tools_enabled}, Round mode: {round_mode}")
    print(f"{'='*60}")

    # Read base prompt
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    from build_prompt import build_prompt
    base_prompt = build_prompt(op, category)

    # Tool orchestration loop
    tool_calls_log = []
    results_summary = {}
    context = ""
    round_count = 0
    max_rounds = 5

    if not tools_enabled:
        # No tools - generate directly
        print(f"[{op}] No tools enabled, generating directly...")
        raw = generate_kernel(op, category, base_prompt)
        code = extract_code_block(raw)
        ok, info, _ = validate_code(code)
        if ok:
            save_results(op, tag, code, [], raw, round_mode)
            print(f"[{op}] OK")
            return True
        else:
            print(f"[{op}] Generation failed: {info}")
            return False

    cfg = get_llm_config_compatible()
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    model = cfg["model"]

    while round_count < max_rounds:
        orchestrator_prompt = build_orchestrator_prompt(op, category, tools_enabled, context, round_count)

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": orchestrator_prompt}],
            temperature=0.1,
            timeout=120,
        )
        choice_str = resp.choices[0].message.content.strip()

        try:
            choice = json.loads(choice_str)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            m = re.search(r'\{[^{}]*\}', choice_str, re.DOTALL)
            if m:
                choice = json.loads(m.group())
            else:
                print(f"[{op}] Failed to parse orchestrator response")
                break

        tool = choice.get("tool", "ANSWER")
        query = choice.get("query", "")

        if tool == "ANSWER":
            print(f"[{op}] Orchestrator chose ANSWER (round {round_count})")
            break

        print(f"[{op}] Round {round_count}: calling {tool}('{query[:80]}')")
        result = call_tool(tool, query)

        tool_calls_log.append({"tool": tool, "query": query, "result_summary": str(result)[:200]})
        context += f"\n[{tool}] {query}:\n{str(result)[:500]}\n"
        round_count += 1

    # Generate kernel with collected evidence
    print(f"[{op}] Generating kernel with {len(tool_calls_log)} tool results...")
    raw = generate_kernel(op, category, base_prompt, context)
    code = extract_code_block(raw)

    ok, info, _ = validate_code(code)
    if ok:
        save_results(op, tag, code, tool_calls_log, raw, round_mode)
        print(f"[{op}] OK - saved")
        return True
    else:
        # Try again without validation (maybe we extracted wrong)
        print(f"[{op}] Validation: {info}")
        save_results(op, tag, code, tool_calls_log, raw, round_mode)
        return False

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("op", help="Operator name")
    p.add_argument("tag", help="Experiment tag")
    p.add_argument("--tools", default="", help="Comma-separated tool keys")
    p.add_argument("--single-round", action="store_true", help="Single round mode")
    p.add_argument("--category", default="activation")
    args = p.parse_args()

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    round_mode = "single_round" if args.single_round else "two_round"

    run_experiment(args.op, args.tag, tools, round_mode, args.category)
