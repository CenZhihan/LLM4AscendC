"""
Retriever wrappers for KB, Web, Code RAG, Environment Check, and new tools.

Provides unified retrieval interfaces for the generator agent.
"""
from .kb_retriever import KBRetriever, query_knowledge
from .web_retriever import WebRetriever, web_search
from .code_retriever import CodeRetriever, query_code_rag
from .code_search_snippet_retriever import CodeSearchSnippetRetriever, CodeSearchSnippetResult, RetrievalUnit, UnifiedSearchResult
from .env_checker import (
    EnvCheckRetriever,
    check_env,
    query_npu_devices,
    check_api_exists,
    EnvCheckResult,
    NpuDeviceResult,
    ApiCheckResult,
    check_env_convenience,
)
from .npu_arch_retriever import NpuArchRetriever, ChipSpecResult
from .tiling_retriever import (
    TilingRetriever,
    TilingParamsResult,
    TilingValidationResult,
    compute_tiling_params,
    validate_tiling_params,
)
from .api_doc_retriever import (
    ApiDocRetriever,
    ApiSignatureResult,
    ApiConstraintResult,
    ApiAlternativeResult,
)
from .code_quality_retriever import (
    CodeQualityRetriever,
    CodingStyleResult,
    SecurityCheckResult,
    check_coding_style,
    check_security_patterns,
)
from .kb_shell_search import (
    KBShellSearchRetriever,
    KBShellSearchResult,
    search_kb,
)
from .ascend_docs_search_retriever import AscendDocsSearchRetriever
from .ascend_docs_fetch_retriever import AscendDocsFetchRetriever

__all__ = [
    'KBRetriever', 'query_knowledge',
    'WebRetriever', 'web_search',
    'CodeRetriever', 'query_code_rag',
    'CodeSearchSnippetRetriever', 'CodeSearchSnippetResult', 'RetrievalUnit', 'UnifiedSearchResult',
    'EnvCheckRetriever',
    'check_env',
    'query_npu_devices',
    'check_api_exists',
    'EnvCheckResult',
    'NpuDeviceResult',
    'ApiCheckResult',
    'check_env_convenience',
    'NpuArchRetriever', 'ChipSpecResult',
    'TilingRetriever', 'TilingParamsResult', 'TilingValidationResult',
    'compute_tiling_params', 'validate_tiling_params',
    'ApiDocRetriever', 'ApiSignatureResult', 'ApiConstraintResult', 'ApiAlternativeResult',
    'CodeQualityRetriever', 'CodingStyleResult', 'SecurityCheckResult',
    'check_coding_style', 'check_security_patterns',
    'KBShellSearchRetriever', 'KBShellSearchResult', 'search_kb',
    'AscendDocsSearchRetriever', 'AscendDocsFetchRetriever',
]