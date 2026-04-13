# 复制为 local_api_config.py 并填写（local_api_config.py 已加入 .gitignore，勿提交密钥）
#
# 使用方式：
#   export USE_API_CONFIG=1
#   python3 tools/generate_ascendc_operators.py ...
#
# 未设置 USE_API_CONFIG=1 时，仍从环境变量 XI_AI_API_KEY / XI_AI_BASE_URL / XI_AI_MODEL 读取。

# 推荐：与本仓库生成逻辑默认命名一致
XI_AI_API_KEY = ""
XI_AI_BASE_URL = ""  # 例如 https://api-2.xi-ai.cn/v1
XI_AI_MODEL = "gpt-5"

# 若你更习惯 OpenAI 风格变量名，也可只填下面两项（key 必填其一）：
# OPENAI_API_KEY = ""
# OPENAI_API_BASE = ""
# MODEL = "gpt-5"
