auto holder = gert::InferShapeContextFaker()
                  .NodeIoNum(2, 1)
                  .IrInstanceNum({1, 1})
                  .InputShapes({&x1_shape, &x2_shape})
                  .OutputShapes({&output_shape})
                  .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                              {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})  // ← 必须！
                  .Build();
