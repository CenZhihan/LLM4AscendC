TEST_F(TEST_ADD_UT, FLOAT_TENSOR_ADD_TENSOR_SUCC) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};

    float input1[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float input2[6] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    float output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    float expected[6] = {7.0, 7.0, 7.0, 7.0, 7.0, 7.0};
    EXPECT_EQ(CompareResult(output, expected, 6), true);
}
