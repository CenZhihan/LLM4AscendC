
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV2D_DIVIDE_LEAKY_RELU_CUSTOM_H_
#define ACLNN_CONV2D_DIVIDE_LEAKY_RELU_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConv2dDivideLeakyReluCustomGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * bias : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv2dDivideLeakyReluCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConv2dDivideLeakyReluCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv2dDivideLeakyReluCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
