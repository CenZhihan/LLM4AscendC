#!/usr/bin/env bash
# 项目内 pip 镜像（不影响系统/他人）；source 本文件后再 pip install
unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY no_proxy NO_PROXY
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
# Hugging Face 模型下载镜像（sentence-transformers / llama-index 等）
export HF_ENDPOINT="https://hf-mirror.com"
export HUGGINGFACE_HUB_ENDPOINT="https://hf-mirror.com"
