# Memory矢量计算算子样例介绍

## 概述

本样例集介绍了Memory矢量计算算子不同特性的典型用法，给出了对应的端到端实现。

## 算子开发样例

| 目录名称                                                                                                                 | 功能描述                                                                        | 
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------| 
| [abs_duplicate](./abs_duplicate/) | 本样例介绍无DataCopyPad的非对齐abs_duplicate算子实现，采用核函数<<<>>>调用，有效降低调度开销，实现高效的算子执行 |
| [abs_gather_mask](./abs_gather_mask) | 本样例介绍无DataCopyPad的非对齐abs_gather_mask算子实现，采用核函数<<<>>>调用，有效降低调度开销，实现高效的算子执行 |
| [abs_pad](./abs_pad/) | 本样例介绍基于基础API实现abs_pad样例，展示了通过Pad一次性清零处理冗余数据 |
| [abs_unpad](./abs_unpad/) | 本样例介绍基于基础API实现abs_unpad样例，展示了通过UnPad去除冗余数据的方法 |
| [reduce_min](./reduce_min/) | 本样例介绍无DataCopyPad的非对齐ReduceMin算子核函数直调方法，采用核函数<<<>>>调用，有效降低调度开销，实现高效的算子执行 |
| [block_reduce_max](./block_reduce_max)                                                                               | 本样例基于BlockReduceMax实现最大值归约，可用于对每个datablock内所有元素求最大值                         |
| [compare](./compare)                                                                                             | 本样例基于Compare逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0                                                                                                                    |
| [compare_result_stored_in_a_register](./compare_result_stored_in_a_register)                                     | 本样例基于Compare（结果存入寄存器）接口进行比较操作，可用于逐元素比较两个tensor大小（比较后的结果为真，则输出结果的对应比特位为1，否则为0），并将计算结果存入寄存器中                                                                                   |
| [compares](./compares)                                                                               | 本样例基于Compares接口进行比较操作，可用于一个tensor逐元素与一个标量比较大小（比较后的结果为真，则输出结果的对应比特位为1，否则为0）                                                                                              |
| [compares_flexible_scalar_argument_position](./compares_flexible_scalar_argument_position)           | 本样例利用Ascend 950PR/Ascend 950DT产品的新增特性，实现了具有灵活标量位置能力的ComparesFlexibleScalarArgumentPosition算子，该算子实现与Compares相同，特殊在于该接口还支持标量参数位置在前和在后两种场景，其中标量输入还支持配置LocalTensor单点元素 |
| [gather_mask_built_in_fixed_mode](./gather_mask_built_in_fixed_mode)                                             | 本样例基于GatherMask实现以内置固定模式对应的二进制为gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中                                                                                                       |
| [gather_mask_custom_mode](./gather_mask_custom_mode)                                                             | 本样例基于GatherMask基础API的用户自定义模式接口实现数据聚合，可用于以用户输入的Tensor数值对应的二进制为gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中                                                                         |
| [select](./select)                                                                                               | 本样例基于Select完成选择操作，可用于给定两个源操作数src0和src1，根据selMask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst                                                                                            |
| [select_flexible_scalar_argument_position](./select_flexible_scalar_argument_position)     | 本样例基于Select实现对于给定的两个源操作数src0和scalar标量，根据selMask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。选择的规则为：当selMask的比特位是1时，从src0中选取，比特位是0时选取scalar标量                                                                                                                                                                             |
| [create_vec_index](./create_vec_index) | 本样例基于CreateVecIndex实现创建指定起始值的向量索引的方法 |
| [brcb](./brcb) | 本样例基于Brcb实现数据填充，可用于输入一个tensor，每一次取输入张量中的8个数填充到结果张量的8个datablock中 |
| [duplicate](./duplicate) | 本样例基于Duplicate实现数据填充，可用于将一个变量或立即数复制多次并填充到向量中 |
| [gather](./gather)         | 本样例基于Gather实现对于给定的输入张量和一个地址偏移量，根据偏移地址将输入张量按元素收集到结果张量中 |
| [gatherb](./gatherb) | 本样例基于Gatherb实现对于给定的输入张量和一个地址偏移量，根据偏移地址按照DataBlock的粒度将输入张量按元素收集到结果张量中     |
| [get_reduce_repeat_max_min_spr](./get_reduce_repeat_max_min_spr)                                                               | 本样例介绍基础api GetReduceRepeatMaxMinSpr的调用，GetReduceRepeatMaxMinSpr的功能是获取ReduceMax、ReduceMin连续场景下的最大/最小值以及相应的索引值            |
| [get_reduce_repeat_sum_spr](./get_reduce_repeat_sum_spr)                                                                                         | 本样例介绍基础api GetReduceRepeatSumSpr的调用。GetReduceRepeatSumSpr的功能是获取ReduceSum接口的计算结果 |
| [pair_reduce_sum](./pair_reduce_sum)                                                                                           | 本样例基于PairReduceSum实现求和归约，可用于相邻两个（奇数下标和偶数下标）元素求和                  |
| [mrg_sort](./mrg_sort) | 本样例基于MrgSort基础API实现将已排好序的最多4条队列，合并成1条队列，结果按照score域由大到小排序 |
| [mrg_sort4](./mrg_sort4) | 本样例介绍基础api MrgSort4的调用，该api的功能：将已经排好序的最多4条Region Proposals队列，排列合并成1条队列，结果按照score域由大到小排序 |
| [proposal_concat](./proposal_concat) | 本样例介绍基础api ProposalConcat的调用，该api的功能：将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposal的对应位置里 |
| [proposal_extract](./proposal_extract) | 本样例介绍基础api ProposalExtract的调用，该api的功能：与ProposalConcat相反，从Region Proposal内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列 |
| [rp_sort16](./rp_sort16) | 本样例介绍基础api RpSort16的调用，该api的功能：根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals |
| [sort32](./sort32) | 本样例基于Sort32实现排序操作，该接口一次迭代可以完成32个数的排序 |
| [reduce_max_computation_of_the_first_n_data_elements](./reduce_max_computation_of_the_first_n_data_elements)         | 本样例基于ReduceMax的tensor前n个数据计算接口实现最大值归约，可用于指定从输入tensor的前n个数据中计算找出最大值及最大值对应的索引位置              |
| [reduce_max_high_dimensional_tensor_sharding_computation](./reduce_max_high_dimensional_tensor_sharding_computation) | 本样例基于ReduceMax的tensor高维切分计算接口实现最大值归约，可用于从所有的输入数据中找出最大值及最大值对应的索引位置，使用mask用于控制每次迭代内参与计算的元素。            |
| [reduce_min_computation_of_the_first_n_data_elements](./reduce_min_computation_of_the_first_n_data_elements)         | 本样例基于ReduceMin的tensor前n个数据计算接口实现最小值归约，可用于指定从输入tensor的前n个数据中找出最小值及最小值对应的索引值              |
| [reduce_min_high_dimensional_tensor_sharding_computation](./reduce_min_high_dimensional_tensor_sharding_computation) | 本样例基于ReduceMin的tensor高维切分计算接口实现最小值归约，可用于从所有的输入数据中找出最小值及最小值对应的索引位置，使用mask用于控制每次迭代内参与计算的元素。            |
| [reduce_sum_computation_of_the_first_n_data_elements](./reduce_sum_computation_of_the_first_n_data_elements)         | 本样例基于ReduceSum的tensor前n个数据计算接口实现求和归约，可用于指定对输入tensor的前n个数据求和                 |
| [reduce_sum_high_dimensional_tensor_sharding_computation](./reduce_sum_high_dimensional_tensor_sharding_computation)     | 本样例基于ReduceSum的tensor高维切分计算接口实现求和归约，可用于对所有的输入数据求和，使用mask用于控制每次迭代内参与计算的元素                |
| [repeat_reduce_sum](./repeat_reduce_sum)                                                                             | 本样例基于RepeatReduceSum实现求和归约，可用于每个repeat内所有数据求和                   |
| [scatter_950](./scatter_950) | 本样例基于Scatter指令实现数据离散，可用于根据一个输入张量、一个目的地址偏移张量和偏移地址，将输入张量分散到结果张量中。                                                                                              |
| [trans_data_to_5hd](./trans_data_to_5hd) | 本样例基于TransDataTo5HD实现数据格式转换，可用于NCHW格式转换成NC1HWC0格式，特别的也可以用于二维矩阵数据块的转置 |
| [transpose_common](./transpose_common) | 本样例基于Transpose实现普通转置，适用于对16*16的二维矩阵数据块进行转置 |
| [transpose_enhanced](./transpose_enhanced) | 本样例基于Transpose实现增强转置，适用于对16*16的二维矩阵数据块进行转置，也可用于[N,C,H,W]与[N,H,W,C]互相转换 |
| [whole_reduce_max](./whole_reduce_max)                                                                               | 本样例基于WholeReduceMax实现最大值归约，可用于对每个repeat内所有数据求最大值以及其索引index，返回的索引值为每个repeat内部索引                  |
| [whole_reduce_min](./whole_reduce_min)                                                                               | 本样例基于WholeReduceMin实现获得每个repeat内所有数据的最小值及其索引index的功能，返回的索引值为每个repeat内部索引                  |
| [whole_reduce_sum](./whole_reduce_sum)                                                                               | 本样例基于WholeReduceSum实现对每个repeat内所有数据求和的功能                    |
| [whole_reduce_sum_unalign](./whole_reduce_sum_unalign/) |本样例介绍非对齐WholeReduceSum算子的核函数直调方法，采用核函数<<<>>>调用，有效降低调度开销，实现高效的算子执行 |