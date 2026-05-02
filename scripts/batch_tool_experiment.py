#!/usr/bin/env python3
"""Batch tool experiment runner - calls tools directly, generates kernels, no orchestrator overhead."""
import sys, os, json, re, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from openai import OpenAI
from generator.agent.agent_config import get_llm_config_compatible

OPS = ["elu","gelu","hardsigmoid","hardtanh","leaky_relu","log_softmax","min_gpt_new_gelu","relu","selu","softmax","softplus","swish"]

def call_tool(tool_name, query):
    if tool_name == "code_search_snippet":
        from generator.agent.retrievers.code_search_snippet_retriever import CodeSearchSnippetRetriever
        retriever = CodeSearchSnippetRetriever()
        results = retriever.retrieve(query, top_k=3)
        return "\n\n".join(str(r) for r in results) if results else ""
    elif tool_name == "api_lookup":
        from generator.agent.retrievers.api_doc_retriever import ApiDocRetriever
        retriever = ApiDocRetriever()
        results = retriever.lookup_signature(query)
        return str(results) if results else ""
    elif tool_name == "api_constraint":
        from generator.agent.retrievers.api_doc_retriever import ApiDocRetriever
        retriever = ApiDocRetriever()
        try:
            results = retriever.check_constraints(query, call_context={})
        except Exception:
            results = retriever.lookup_signature(query)
        return str(results) if results else ""
    elif tool_name == "kb_shell_search":
        from generator.agent.retrievers.kb_shell_search import KBShellSearchRetriever
        retriever = KBShellSearchRetriever()
        result = retriever.search(category="all", query=query)
        return str(result) if result else ""
    elif tool_name == "npu_arch":
        from generator.agent.nodes.npu_arch import query_npu_arch
        return str(query_npu_arch(query))
    elif tool_name == "ascend_search":
        from generator.agent.retrievers.ascend_docs_search_retriever import AscendDocsSearchRetriever
        retriever = AscendDocsSearchRetriever()
        results = retriever.search(keyword=query)
        return str(results) if results else ""
    elif tool_name == "ascend_fetch":
        from generator.agent.retrievers.ascend_docs_fetch_retriever import AscendDocsFetchRetriever
        retriever = AscendDocsFetchRetriever()
        result = retriever.fetch(url=query, extract_code=True)
        return str(result) if result else ""
    return ""

def generate_kernel(op, category, prompt, context=""):
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
    local_vars = {}
    try:
        exec(content, {}, local_vars)
        expected = ['project_json_src','host_tiling_src','host_operator_src','kernel_src','python_bind_src','model_src']
        missing = [v for v in expected if v not in local_vars]
        return len(missing) == 0, missing, local_vars
    except Exception as e:
        return False, str(e), {}

def save_results(op, tag, code, tool_evidence, raw_output, tool_keys):
    outdir = os.path.join(REPO_ROOT, "output", "claudeCase", tag)
    os.makedirs(outdir, exist_ok=True)

    stripped = code.strip()
    # Fix r-prefix
    stripped = re.sub(r'=\s*r"""', '= """', stripped)
    stripped = re.sub(r"=\s*r'''", "= '''", stripped)

    txt_path = os.path.join(outdir, f"{op}.txt")
    with open(txt_path, 'w') as f:
        f.write(stripped)

    trace = {
        "experiment_tag": tag,
        "enabled_tools": tool_keys,
        "operator": op,
        "category": category if 'category' in dir() else "activation",
        "difficulty": "simple",
        "round_mode": "single_round",
        "iterations": {
            "iter1": {
                "tool_calls": [{"tool": t, "query": q, "result_length": len(r)} for t, q, r in tool_evidence],
                "generation": raw_output or stripped,
                "eval": {}
            }
        }
    }
    trace_path = os.path.join(outdir, f"{op}_trace.json")
    with open(trace_path, 'w') as f:
        json.dump(trace, f, indent=2)
    return txt_path, trace_path

def gather_evidence(op, tools):
    """Call each tool with a relevant query and return (tool, query, result) list."""
    evidence = []
    queries = {
        "code_search_snippet": f"{op} AscendC kernel implementation",
        "api_lookup": f"{op}",
        "api_constraint": f"{op}",
        "kb_shell_search": f"{op} AscendC kernel activation",
        "npu_arch": "Ascend910b tiling UB budget",
        "ascend_search": f"AscendC {op} kernel operator",
        "ascend_fetch": "",
    }
    for t in tools:
        q = queries.get(t, f"{op} AscendC")
        if t == "ascend_fetch":
            continue  # requires a URL from ascend_search
        print(f"  [{t}] query='{q}'")
        try:
            result = call_tool(t, q)
            if result:
                evidence.append((t, q, result))
        except Exception as e:
            print(f"  [{t}] ERROR: {e}")
    return evidence

def run_experiment(tag, tools, category="activation"):
    print(f"\n{'='*60}")
    print(f"Experiment: {tag}")
    print(f"Tools: {tools}")
    print(f"{'='*60}")

    from build_prompt import build_prompt

    results = {"compiled": 0, "total": 0, "details": {}}

    for op in OPS:
        print(f"\n--- {op} ---")
        base_prompt = build_prompt(op, category)

        evidence = gather_evidence(op, tools)

        # Format context
        context_parts = []
        for t, q, r in evidence:
            context_parts.append(f"[{t}] query='{q}':\n{r[:1500]}")
        context = "\n\n".join(context_parts)

        print(f"  Generating kernel with {len(evidence)} tool results...")
        raw = generate_kernel(op, category, base_prompt, context)
        code = extract_code_block(raw)

        ok, info, _ = validate_code(code)
        if not ok:
            print(f"  Validation failed: {info}")
            # Try raw output
            code = extract_code_block(raw)

        save_results(op, tag, code, evidence, raw, list(tools))
        print(f"  Saved to output/claudeCase/{tag}/{op}.txt")
        results["total"] += 1

    print(f"\n{'='*60}")
    print(f"Experiment {tag} complete: {results}")
    print(f"{'='*60}")

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    tools = [t.strip() for t in sys.argv[2].split(",")] if len(sys.argv) > 2 else []
    if not tag:
        print("Usage: python3 scripts/batch_tool_experiment.py <tag> <tool1,tool2,...>")
        sys.exit(1)
    run_experiment(tag, tools)
