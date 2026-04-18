import os
import sys
import importlib.util
# 添加项目根目录到 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import re
import torch
from generator.config import ref_impl_base_path
from generator.dataset import dataset

def _load_file_api_config():
    """加载本地 API 配置。优先 generator/local_api_config.py，其次兼容 generation/local_api_config.py。"""
    if os.environ.get("USE_API_CONFIG", "").strip().lower() not in ("1", "true"):
        return None
    candidates = [
        os.path.join(_project_root, "generator", "local_api_config.py"),
        os.path.join(_project_root, "generation", "local_api_config.py"),
    ]
    config_path = next((p for p in candidates if os.path.exists(p)), None)
    if not config_path:
        return None
    try:
        spec = importlib.util.spec_from_file_location("local_api_config", config_path)
        lac = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lac)
    except Exception:
        return None
    key = (getattr(lac, "XI_AI_API_KEY", None) or getattr(lac, "OPENAI_API_KEY", None) or "").strip()
    base = (getattr(lac, "XI_AI_BASE_URL", None) or getattr(lac, "OPENAI_API_BASE", None) or "").strip()
    model = (getattr(lac, "XI_AI_MODEL", None) or getattr(lac, "MODEL", None) or "").strip()
    return (key, base, model) if key else None


def _load_openai_api_config():
    """若设置 USE_API_CONFIG=1 且存在 api_config.py，则返回 (api_key, base_url, model)；否则返回 None。"""
    if os.environ.get("USE_API_CONFIG", "").strip().lower() not in ("1", "true"):
        return None
    try:
        import api_config as ac
        if getattr(ac, "OPENAI_API_KEY", "").strip():
            return (
                ac.OPENAI_API_KEY.strip(),
                (getattr(ac, "OPENAI_API_BASE", None) or "").strip() or "https://api.openai.com/v1",
                (getattr(ac, "MODEL", None) or "").strip() or "gpt-5",
            )
    except ImportError:
        pass
    return None


def get_default_model_from_config():
    """从配置文件获取默认模型名。优先 local_api_config.py，其次 api_config.py。"""
    cfg = _load_file_api_config()
    if cfg and cfg[2]:
        return cfg[2]
    cfg = _load_openai_api_config()
    return cfg[2] if cfg else None


def get_client(model):
    """获取 OpenAI 兼容 API 客户端。

    配置优先级：
    1. USE_API_CONFIG=1 + generator/local_api_config.py（或兼容 generation/local_api_config.py）
    2. USE_API_CONFIG=1 + api_config.py（仅 gpt 模型）
    3. 模型前缀专用环境变量（DEEPSEEK_API_KEY / DASHSCOPE_API_KEY 等）
    """
    from openai import OpenAI

    # 优先级 1: generation/local_api_config.py — 统一 OpenAI 兼容端点
    file_cfg = _load_file_api_config()
    if file_cfg and file_cfg[0] and file_cfg[1]:
        api_key, base_url, _ = file_cfg
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=10000000,
            max_retries=3,
        )

    # 按模型前缀使用专用端点
    if model.startswith('deepseek'):
        DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
        client = OpenAI(
            api_key=DEEPSEEK_KEY,
            base_url="https://api.deepseek.com",
            timeout=10000000,
            max_retries=3,
        )
    elif model.startswith('qwen'):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=10000000,
            max_retries=3,
        )
    elif model.startswith('zhipu') or model.startswith('glm'):
        api_key = os.environ.get("ZHIPU_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            timeout=10000000,
            max_retries=3,
        )
    elif model.startswith('gpt'):
        # 优先级 2: api_config.py
        cfg = _load_openai_api_config()
        if cfg is not None:
            api_key, base_url, _ = cfg
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=10000000,
                max_retries=3,
            )
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=10000000,
                max_retries=3,
            )
    else:
        api_key = os.environ.get("OPEN_ROUNTER_KEY")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return client

def get_ref_src_path(op):
    return os.path.join(ref_impl_base_path, dataset[op]['category'], f'{op}.py')


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code_block = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code_block.startswith(code_type):
                code = code_block[len(code_type) :].strip()

        return code, f'```{code_block}```'

def underscore_to_pascalcase(underscore_str):
    """
    Convert underscore-separated string to PascalCase.
    
    Args:
        underscore_str (str): Input string with underscores (e.g., "vector_add")
        
    Returns:
        str: PascalCase version (e.g., "VectorAdd")
    """
    if not underscore_str:  # Handle empty string
        return ""
    
    parts = underscore_str.split('_')
    # Capitalize the first letter of each part and join
    return ''.join(word.capitalize() for word in parts if word)
