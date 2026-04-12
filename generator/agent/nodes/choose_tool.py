"""
Tool selection node for generator agent.

Let LLM choose the next action: KB, WEB, CODE_RAG, or ANSWER.
"""
from typing import Dict, Any, Tuple

from langchain_core.messages import HumanMessage

from ..agent_state import GeneratorAgentState, MAX_QUERY_ROUNDS
from ..agent_config import (
    AgentToolMode,
    ToolType,
    NO_TOOL,
    has_kb,
    has_web,
    has_code_rag,
    has_env_check_env,
    has_env_check_npu,
    has_env_check_api,
)


def _openai_completion(
    client,
    model: str,
    messages: list,
    stream: bool = False,
) -> Tuple[str, str]:
    """
    Call OpenAI-compatible API for completion.

    Args:
        client: OpenAI client instance
        model: Model name
        messages: Message list
        stream: Whether to stream

    Returns:
        Tuple of (content, reasoning)
    """
    if not stream:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
        )
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        reasoning = (
            getattr(msg, "reasoning_content", None)
            or (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
            or ""
        )
        if reasoning and not isinstance(reasoning, str):
            reasoning = ""
        return content, (reasoning if isinstance(reasoning, str) else "")

    # Stream mode
    stream_resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    content_parts: list = []
    reasoning_parts: list = []
    for chunk in stream_resp:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        r = (
            getattr(delta, "reasoning_content", None)
            or (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
        )
        if r:
            reasoning_parts.append(r)
    return "".join(content_parts).strip(), "".join(reasoning_parts)


def _extract_user_question(state: GeneratorAgentState) -> str:
    """Extract user question from messages."""
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    return user_msgs[0].content if user_msgs else state["messages"][-1].content


def _summarize_existing_results(state: GeneratorAgentState) -> str:
    """Summarize existing retrieval results."""
    kb_results = state.get("kb_results", [])
    web_results = state.get("web_results", [])
    code_rag_results = state.get("code_rag_results", [])
    env_check_results = state.get("env_check_results", [])

    existing = ""
    if kb_results:
        existing += "已查知识库结果（节选）：\n" + "\n".join(kb_results[:2]) + "\n\n"
    if web_results:
        existing += "已搜网页结果（节选）：\n" + "\n".join(web_results[:2]) + "\n\n"
    if code_rag_results:
        existing += "已检索代码结果（节选）：\n" + "\n".join(code_rag_results[:1]) + "\n\n"
    if env_check_results:
        existing += "环境检查结果（节选）：\n" + "\n".join(env_check_results[:1]) + "\n\n"
    return existing


def _build_tool_selection_prompt(
    user_question: str,
    existing_results: str,
    tool_mode: AgentToolMode,
    round_count: int,
) -> str:
    """Build prompt for tool selection."""

    # Build tools description dynamically
    tools_desc: list = []
    if has_kb(tool_mode):
        tools_desc.append("知识库查询（KB）- 查询华为 Ascend C API 文档")
    if has_web(tool_mode):
        tools_desc.append("网页搜索（WEB）- 搜索相关技术文档和博客")
    if has_code_rag(tool_mode):
        tools_desc.append("代码检索（CODE_RAG）- 检索 Ascend C 代码库中的相似实现")
    if has_env_check_env(tool_mode):
        tools_desc.append("环境检查（ENV_CHECK_ENV）- 检查 CANN 环境配置及 API 兼容性")
    if has_env_check_npu(tool_mode):
        tools_desc.append("NPU查询（ENV_CHECK_NPU）- 查询 NPU 设备状态和资源使用")
    if has_env_check_api(tool_mode):
        tools_desc.append("API检查（ENV_CHECK_API）- 验证 Ascend C API 是否在头文件中存在")

    if not tools_desc:
        return ""  # No tools, will return ANSWER

    tools_line = "、".join(tools_desc)
    at_max = round_count >= MAX_QUERY_ROUNDS

    # Build few-shot examples dynamically (order matches tools_desc)
    examples: list = []
    example_idx = 1
    for tool_type in [ToolType.KB, ToolType.WEB, ToolType.CODE_RAG,
                      ToolType.ENV_CHECK_ENV, ToolType.ENV_CHECK_NPU, ToolType.ENV_CHECK_API]:
        if tool_type in tool_mode:
            tool_name = tool_type.value.upper()
            if tool_type == ToolType.KB:
                examples.append(f"示例{example_idx}（查知识库）：\nKB\nAscend C GELU kernel implementation")
            elif tool_type == ToolType.WEB:
                examples.append(f"示例{example_idx}（网页搜索）：\nWEB\nAscend C custom operator tutorial")
            elif tool_type == ToolType.CODE_RAG:
                examples.append(f"示例{example_idx}（代码检索）：\nCODE_RAG\nAscend C softmax kernel example")
            elif tool_type == ToolType.ENV_CHECK_ENV:
                examples.append(f"示例{example_idx}（环境检查）：\nENV_CHECK_ENV\ncheck CANN environment")
            elif tool_type == ToolType.ENV_CHECK_NPU:
                examples.append(f"示例{example_idx}（NPU查询）：\nENV_CHECK_NPU\nquery npu device status")
            elif tool_type == ToolType.ENV_CHECK_API:
                examples.append(f"示例{example_idx}（API检查）：\nENV_CHECK_API\ncheck if AscendC::DataCopy exists")
            example_idx += 1
    examples.append("示例（直接回答）：\nANSWER")
    few_shot = "\n\n".join(examples)

    # Build prompt
    prompt = (
        f"用户问题：\n{user_question}\n\n"
        f"当前可选工具：{tools_line}，或直接回答（ANSWER）。根据需要选择合适的工具。\n"
        "规则：只输出两行。第一行为动作（KB/WEB/CODE_RAG/ENV_CHECK_ENV/ENV_CHECK_NPU/ENV_CHECK_API/ANSWER）；"
        "若选 KB/WEB/CODE_RAG/ENV_CHECK_*/ANSWER，第二行必须是一句完整的英文查询。\n"
    )

    if existing_results:
        prompt += f"\n已有检索内容：\n{existing_results}\n"

    if at_max:
        prompt += f"\n已达最大查询轮数（{MAX_QUERY_ROUNDS}），本回合只能选 ANSWER。\n"

    prompt += f"\n格式参考：\n{few_shot}\n\n"
    prompt += "请按上述格式输出（第一行动作，第二行查询）："

    return prompt


def choose_tool_node(
    state: GeneratorAgentState,
    client,
    model: str,
    tool_mode: AgentToolMode,
) -> Dict[str, Any]:
    """
    Tool selection node: let LLM choose next action.

    Args:
        state: Current agent state
        client: OpenAI client
        model: Model name
        tool_mode: Enabled tool mode (FrozenSet[ToolType])

    Returns:
        Dict with next_action and current_query
    """
    user_question = _extract_user_question(state)
    existing_results = _summarize_existing_results(state)
    round_count = state.get("query_round_count", 0)

    # No tools enabled -> direct answer
    if tool_mode == NO_TOOL:
        return {"next_action": "ANSWER", "current_query": ""}

    # Build and call LLM
    prompt = _build_tool_selection_prompt(user_question, existing_results, tool_mode, round_count)
    if not prompt:
        return {"next_action": "ANSWER", "current_query": ""}

    raw_response, _ = _openai_completion(client, model, [{"role": "user", "content": prompt}], stream=False)

    # Parse response
    lines = [ln.strip() for ln in raw_response.splitlines() if ln.strip()]
    first = (lines[0].upper() if lines else "") or "ANSWER"
    query_line = lines[1] if len(lines) > 1 else ""

    # Determine next action
    at_max = round_count >= MAX_QUERY_ROUNDS
    if at_max:
        next_action = "ANSWER"
        current_query = ""
    elif "ANSWER" in first or first == "ANSWER":
        next_action = "ANSWER"
        current_query = ""
    elif "CODE_RAG" in first and has_code_rag(tool_mode):
        next_action = "CODE_RAG"
        current_query = query_line or user_question
    elif "ENV_CHECK_API" in first and has_env_check_api(tool_mode):
        next_action = "ENV_CHECK_API"
        current_query = query_line or user_question
    elif "ENV_CHECK_NPU" in first and has_env_check_npu(tool_mode):
        next_action = "ENV_CHECK_NPU"
        current_query = query_line or user_question
    elif "ENV_CHECK_ENV" in first and has_env_check_env(tool_mode):
        next_action = "ENV_CHECK_ENV"
        current_query = query_line or user_question
    elif "KB" in first and has_kb(tool_mode):
        next_action = "KB"
        current_query = query_line or user_question
    elif "WEB" in first and has_web(tool_mode):
        next_action = "WEB"
        current_query = query_line or user_question
    else:
        # Fallback: choose first available tool
        if has_kb(tool_mode):
            next_action = "KB"
        elif has_web(tool_mode):
            next_action = "WEB"
        elif has_code_rag(tool_mode):
            next_action = "CODE_RAG"
        elif has_env_check_env(tool_mode):
            next_action = "ENV_CHECK_ENV"
        elif has_env_check_npu(tool_mode):
            next_action = "ENV_CHECK_NPU"
        elif has_env_check_api(tool_mode):
            next_action = "ENV_CHECK_API"
        else:
            next_action = "ANSWER"
        current_query = query_line or user_question

    print(f"[choose_tool] 模型回复: {raw_response[:100]!r} -> next_action={next_action}")
    return {"next_action": next_action, "current_query": current_query}