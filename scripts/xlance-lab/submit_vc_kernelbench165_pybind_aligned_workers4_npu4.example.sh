#!/usr/bin/env bash
# 集群提交示例（按你们 vc 集群实际改分区/资源）。
# 与此前 kernelbench165 任务格式一致，改为：4 workers + 4 NPU、仅评测 pybind_reference_aligned 下 64 个 txt。
#
# 使用前请确认镜像、分区 pdgpu-sjtu-ai、内存/CPU 是否满足并行编译；若 OOM 可加大 -m。
#
# 直接执行（前台交任务）：
#   bash scripts/xlance-lab/submit_vc_kernelbench165_pybind_aligned_workers4_npu4.example.sh

set -euo pipefail

# 与此前单卡任务一致：分区/镜像/仓库路径；4 路并行编译建议内存不低于 128G（可按队列改 -m）。
vc submit -p pdgpu-sjtu-ai \
  -i hub.szaic.com/sjtu-base/sjtu_base-pytorch-for-ascend:cann8.1rc1-torch2.1.0-py3.10 \
  -n 1 -c 8 -m 128G -g 4 \
  -d "/aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC" \
  -j kernelbench165-pybind-aligned-w4-npu4 \
  --cmd "cd /aistor/sjtu/hpc_stor01/home/cenzhihan/LLM_bench/LLM4AscendC && bash scripts/xlance-lab/run_kernelbench165_pybind_aligned_workers4_npu4.sh"
