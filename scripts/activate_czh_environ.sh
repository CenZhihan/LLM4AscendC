#!/usr/bin/env bash
# 激活本项目隔离 conda 环境（路径均在 cenzhihan 目录下）
unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY no_proxy NO_PROXY
# CANN set_env.sh 会读取 PYTHONPATH；勿 unset（bash set -u / nounset 下会报 unbound variable）
export PYTHONPATH=""
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CZH_ENV_BIN="${ROOT}/miniconda3/envs/czh_environ/bin"

# CANN 工具链（编译/评测子进程仍可通过 build_subprocess_env 使用 python3.11）
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

source "${ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ROOT}/miniconda3/envs/czh_environ"
source "${ROOT}/scripts/setup_pip_mirror.sh"

# Agent / repair-memory / pip 必须用 conda 3.10；set_env 会把 python3.11 放到 PATH 最前
export PATH="${CZH_ENV_BIN}:${PATH}"
export CZH_CONDA_PYTHON="${CZH_ENV_BIN}/python"

export LLM4ASCENDC_ROOT="${ROOT}"
cd "${ROOT}"

# 若之后又 source 了 ascend set_env.sh，在 tmux 命令末尾再执行一次：
#   export PATH="${CZH_ENV_BIN}:$PATH"
