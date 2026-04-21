auto holder = gert::TilingContextFaker()
                  .SetOpType("BatchMatMulV3")
                  .NodeIoNum(2, 1)
                  .IrInstanceNum({1, 1})
                  .InputShapes({&x1_shape, &x2_shape})
                  .OutputShapes(output_shapes_ref)
                  .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},    // ← 必须！
                              {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                              {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                              {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                  .NodeInputTd(0, param.input_dtype, param.x1_ori_format, param.x1_format)  // ← 必须！
                  .NodeInputTd(1, param.input_dtype, param.x2_ori_format, param.x2_format)
                  .NodeOutputTd(0, param.y_dtype, param.y_ori_format, param.y_format)
                  .Build();
