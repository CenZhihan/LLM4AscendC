#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/aistor/sjtu/hpc_stor01/home/liuxiang/LLM4NPU/LLM4AscendC/ascend_custom_opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/aistor/sjtu/hpc_stor01/home/liuxiang/LLM4NPU/LLM4AscendC/ascend_custom_opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
