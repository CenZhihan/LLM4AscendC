#!/usr/bin/env bash
# 激活本项目隔离 conda 环境（路径均在 cenzhihan 目录下）
unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY no_proxy NO_PROXY
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate "${ROOT}/miniconda3/envs/czh_environ"
source "${ROOT}/scripts/setup_pip_mirror.sh"
export LLM4ASCENDC_ROOT="${ROOT}"
cd "${ROOT}"
