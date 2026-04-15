"""
Generator configuration for LLM-based operator generation.

Separate from tools/common/env.py which handles NPU/CANN environment.
"""
import os
from pathlib import Path

# Project paths
_project_root = os.path.dirname(os.path.abspath(__file__))
_generator_root = os.path.dirname(_project_root)
REPO_ROOT = Path(_generator_root)  # Repo root (LLM4AscendC) for path resolution
ref_impl_base_path = os.path.join(_generator_root, 'vendor/mkb/reference')

# LLM configuration
max_tokens = 8192
temperature = 0.0
top_p = 1.0
num_completions = 1
seed_num = 1024

# RAG configuration - index stored in generator/rag/
rag_index_path = os.path.join(_project_root, 'rag/index')
rag_embedding_model = os.path.join(_project_root, 'models/BAAI/bge-m3')  # 模型目录: generator/models/bge-m3
rag_top_k = 3
rag_max_chars = 8000  # Retrieved content max characters
rag_code_dir = os.path.join(_project_root, 'ascendCode')  # Code library dir (user prepares)
rag_file_extensions = ['.cpp', '.h']  # File types to index

# Ascend C API reference path (for prompt generation)
ascendc_api_reference_path = os.path.join(_project_root, 'ascendc_api_reference.md')  # Optional: API docs file

# Default model from api_config or environment
def get_default_model():
    """Get default model name from config or environment.

    Priority:
    1. USE_API_CONFIG=1 + generation/local_api_config.py
    2. USE_API_CONFIG=1 + api_config.py (repo root)
    3. DEFAULT_MODEL env var
    4. Hardcoded default: deepseek-chat
    """
    if os.environ.get('USE_API_CONFIG'):
        import importlib.util
        # Try generation/local_api_config.py first
        local_config_path = os.path.join(_generator_root, 'generation', 'local_api_config.py')
        if os.path.exists(local_config_path):
            try:
                spec = importlib.util.spec_from_file_location("local_api_config", local_config_path)
                lac = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(lac)
                model = (getattr(lac, 'XI_AI_MODEL', None) or getattr(lac, 'MODEL', None) or '').strip()
                if model:
                    return model
            except Exception:
                pass
        # Fallback: api_config.py at repo root
        api_config_path = os.path.join(_generator_root, 'api_config.py')
        if os.path.exists(api_config_path):
            try:
                spec = importlib.util.spec_from_file_location("api_config", api_config_path)
                api_config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(api_config)
                return getattr(api_config, 'MODEL', 'deepseek-chat')
            except Exception:
                pass
    return os.environ.get('DEFAULT_MODEL', 'deepseek-chat')


# ===== Agent Configuration =====
# KB (Knowledge Base) configuration
agent_kb_persist_dir = os.path.join(_project_root, 'agent/chroma_db')
agent_kb_collection_name = os.environ.get("KB_COLLECTION", "ascend_c_knowledge")

# Web search configuration
agent_web_max_results = 8
agent_web_max_fetch_urls = 5
agent_web_fetch_timeout = 8.0
agent_web_max_chars = 4000

# Code RAG configuration (inherits from RAG config above)
agent_code_rag_top_k = rag_top_k
agent_code_rag_max_chars = rag_max_chars

# Agent workflow configuration
agent_max_query_rounds = 3  # Maximum query rounds before forced ANSWER