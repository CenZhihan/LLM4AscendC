all: hardtanh_custom0
hardtanh_custom0:
	cd /root/LLM4AscendC/artifacts/hardtanh/workspace/HardtanhCustom/build_out/op_kernel/HardtanhCustom_ascend910b/kernel_0 && bash /root/LLM4AscendC/artifacts/hardtanh/workspace/HardtanhCustom/build_out/op_kernel/HardtanhCustom_ascend910b/bin_param/HardtanhCustom-hardtanh_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)