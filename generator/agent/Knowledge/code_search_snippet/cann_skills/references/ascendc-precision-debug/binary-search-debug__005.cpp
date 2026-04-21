// 使用 CPU/Python 计算参考值
// Python: numpy.exp(x)
half exp_ref = /* 从外部获取的参考值 */;
half exp_npu = Exp(input);

printf("exp comparison: NPU=%.6f, Ref=%.6f, Diff=%.2e\n",
       static_cast<float>(exp_npu),
       static_cast<float>(exp_ref),
       static_cast<float>(abs(exp_npu - exp_ref)));
