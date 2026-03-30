
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV2D_AVG_POOL_SIGMOID_SUM_CUSTOM_H_
#define ACLNN_CONV2D_AVG_POOL_SIGMOID_SUM_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConv2dAvgPoolSigmoidSumCustomGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * convBias : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv2dAvgPoolSigmoidSumCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *convBias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConv2dAvgPoolSigmoidSumCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv2dAvgPoolSigmoidSumCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
