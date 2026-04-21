"""
LangGraph nodes for generator agent.

Each node handles a specific action in the agent workflow:
- choose_tool: Tool selection (KB/WEB/CODE_RAG/ENV_CHECK_*/ANSWER + new tools)
- kb_query: KB knowledge base query
- web_search: Web search
- code_rag: Code RAG retrieval
- code_search_snippet: Restricted snippet retrieval from CANN skills / asc-devkit
- env_check_env: Environment overview check
- env_check_npu: NPU device query
- env_check_api: API compatibility check
- npu_arch: NPU architecture query
- tiling_calc: Tiling calculation
- tiling_validate: Tiling validation
- api_lookup: API signature lookup
- api_constraint: API constraint check
- api_alternative: API alternative finder
- code_style: Code style check
- security: Security pattern check
- kb_shell_search: Knowledge base shell search
- answer: Final answer generation
"""
from .choose_tool import choose_tool_node
from .kb_query import kb_query_node
from .web_search import web_search_node
from .code_rag import code_rag_node
from .code_search_snippet import code_search_snippet_node
from .env_check import env_check_env_node, env_check_npu_node, env_check_api_node
from .npu_arch import npu_arch_node
from .tiling_calc import tiling_calc_node
from .tiling_validate import tiling_validate_node
from .api_lookup import api_lookup_node
from .api_constraint import api_constraint_node
from .api_alternative import api_alternative_node
from .code_style_check import code_style_node
from .security_check import security_check_node
from .kb_shell_search import kb_shell_search_node
from .ascend_search import ascend_search_node
from .ascend_fetch import ascend_fetch_node
from .registered_tool import registered_tool_dispatch_node, tool_dispatch_node
from .answer import answer_node

__all__ = [
    'choose_tool_node',
    'kb_query_node',
    'web_search_node',
    'code_rag_node',
    'code_search_snippet_node',
    'env_check_env_node',
    'env_check_npu_node',
    'env_check_api_node',
    'npu_arch_node',
    'tiling_calc_node',
    'tiling_validate_node',
    'api_lookup_node',
    'api_constraint_node',
    'api_alternative_node',
    'code_style_node',
    'security_check_node',
    'kb_shell_search_node',
    'ascend_search_node',
    'ascend_fetch_node',
    'registered_tool_dispatch_node',
    'tool_dispatch_node',
    'answer_node',
]