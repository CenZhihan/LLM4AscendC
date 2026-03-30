all: elu_custom0
elu_custom0:
	cd /root/LLM4AscendC/artifacts/elu/workspace/EluCustom/build_out/op_kernel/EluCustom_ascend910b/kernel_0 && bash /root/LLM4AscendC/artifacts/elu/workspace/EluCustom/build_out/op_kernel/EluCustom_ascend910b/bin_param/EluCustom-elu_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)