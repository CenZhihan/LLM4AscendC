# 复制为 local_api_config.py 并填写（local_api_config.py 已加入 .gitignore，勿提交密钥）
#
# Agent 入口 ``generator/scripts/generation/generate_agent.py`` 仅从本文件读取
# XI_AI_API_KEY / XI_AI_BASE_URL / XI_AI_MODEL（不再使用 USE_API_CONFIG 或 XI_* 环境变量）。
# 命令行 ``--model`` 可覆盖本文件中的模型名。
#
# 其他脚本（如 tools/generate_ascendc_operators.py）若仍使用 generator/llm_config，请见其说明。

# 推荐：与本仓库生成逻辑默认命名一致
XI_AI_API_KEY = ""
XI_AI_BASE_URL = ""  # 例如 https://api-2.xi-ai.cn/v1
XI_AI_MODEL = "gpt-5"

# 若你更习惯 OpenAI 风格变量名，也可只填下面两项（key 必填其一）：
# OPENAI_API_KEY = ""
# OPENAI_API_BASE = ""
# MODEL = "gpt-5"
