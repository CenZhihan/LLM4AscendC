
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV_STANDARD2D_SQUARE_INPUT_SQUARE_KERNEL_CUSTOM_H_
#define ACLNN_CONV_STANDARD2D_SQUARE_INPUT_SQUARE_KERNEL_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConvStandard2dSquareInputSquareKernelCustomGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvStandard2dSquareInputSquareKernelCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConvStandard2dSquareInputSquareKernelCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvStandard2dSquareInputSquareKernelCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
