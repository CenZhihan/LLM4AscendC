# Memory矢量计算算子样例介绍

## 概述

本样例集介绍了Memory矢量计算算子不同特性的典型用法，给出了对应的端到端实现。

## 算子开发样例

| 目录名称                                                                                                                 | 功能描述                                                                        | 
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------| 
| [block_reduce_max](./block_reduce_max)                                                                               | 本样例基于BlockReduceMax实现最大值归约，可用于对每个datablock内所有元素求最大值                         |
| [compare](./compare)                                                                                             | 本样例基于Compare逐元素比较两个tensor大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0                                                                                                                    |
| [compare_result_stored_in_a_register](./compare_result_stored_in_a_register)                                     | 本样例基于Compare（结果存入寄存器）接口进行比较操作，可用于逐元素比较两个tensor大小（比较后的结果为真，则输出结果的对应比特位为1，否则为0），并将计算结果存入寄存器中                                                                                   |
| [compares](./compares)                                                                               | 本样例基于Compares接口进行比较操作，可用于一个tensor逐元素与一个标量比较大小（比较后的结果为真，则输出结果的对应比特位为1，否则为0）                                                                                              |
| [compares_flexible_scalar_argument_position](./compares_flexible_scalar_argument_position)           | 本样例利用Ascend 950PR/Ascend 950DT产品的新增特性，实现了具有灵活标量位置能力的ComparesFlexibleScalarArgumentPosition算子，该算子实现与Compares相同，特殊在于该接口还支持标量参数位置在前和在后两种场景，其中标量输入还支持配置LocalTensor单点元素 |
| [select](./select)                                                                                               | 本样例基于Select完成选择操作，可用于给定两个源操作数src0和src1，根据selMask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst                                                                                            |
| [select_flexible_scalar_argument_position](./select_flexible_scalar_argument_position)     | 本样例基于Select实现对于给定的两个源操作数src0和scalar标量，根据selMask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。选择的规则为：当selMask的比特位是1时，从src0中选取，比特位是0时选取scalar标量                                                                                                                                                                             |
| [create_vec_index](./create_vec_index) | 本样例介绍了调用CreateVecIndex实现创建指定起始值的向量索引的方法 |
| [brcb](./brcb) | 本样例基于Brcb实现数据填充，可用于每次取输入张量中的8个数填充到结果张量的8个datablock中 |
| [duplicate](./duplicate) | 本样例基于Duplicate实现数据填充，可用于将一个变量或立即数复制多次并填充到向量中 |
| [gather](./gather)         | 本样例基于GatherMask、Gather、Gatherb等接口完成多种场景模式下的数据选择功能，实现从源操作数中选取元素写入目的操作数。 |
| [mrg_sort](./mrg_sort) | 本样例基于Sort32和MrgSort基础API实现将已排好序的最多4条队列，合并成1条队列，结果按照score域由大到小排序 |
| [mrg_sort4](./mrg_sort4) | 本样例介绍基础api MrgSort4的调用，该api的功能：将已经排好序的最多4条Region Proposals队列，排列合并成1条队列，结果按照score域由大到小排序 |
| [proposal_concat](./proposal_concat) | 本样例介绍基础api ProposalConcat的调用，该api的功能：将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposal的对应位置里 |
| [proposal_extract](./proposal_extract) | 本样例介绍基础api ProposalExtract的调用，该api的功能：与ProposalConcat相反，从Region Proposal内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列 |
| [rp_sort16](./rp_sort16) | 本样例介绍基础api RpSort16的调用，该api的功能：根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals |
| [reduce_computation](./reduce_computation)         | 本样例基于ReduceMax/ReduceMin/ReduceSum接口实现归约计算              |
| [repeat_reduce_sum](./repeat_reduce_sum)                                                                             | 本样例基于RepeatReduceSum实现求和归约，可用于每个repeat内所有数据求和                   |
| [scatter_950](./scatter_950) | 本样例基于Scatter指令实现数据离散，可用于根据一个输入张量、一个目的地址偏移张量和偏移地址，将输入张量分散到结果张量中。                                                                                              |
| [trans_data_to_5hd](./trans_data_to_5hd) | 本样例基于TransDataTo5HD实现数据格式转换，可用于NCHW格式转换成NC1HWC0格式，特别的也可以用于二维矩阵数据块的转置 |
| [transpose](./transpose_common) | 本样例基于Transpose实现普通转置和增强转置，适用于对16*16的二维矩阵数据块进行转置，也适用于[N,C,H,W]与[N,H,W,C]四维矩阵互相转换 |
| [whole_reduce_min_max_sum](./whole_reduce_min_max_sum)                                                                               | 本样例介绍归约类接口在多种场景下的使用方法，包括WholeReduceMax、WholeReduceMin、WholeReduceSum、RepeatReduceSum，以及WholeReduceMax/Min配合GetReduceRepeatMaxMinSpr获取全局极值及索引的使用方法。                  |