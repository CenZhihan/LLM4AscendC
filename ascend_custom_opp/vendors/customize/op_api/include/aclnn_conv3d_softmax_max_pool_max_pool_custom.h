
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV3D_SOFTMAX_MAX_POOL_MAX_POOL_CUSTOM_H_
#define ACLNN_CONV3D_SOFTMAX_MAX_POOL_MAX_POOL_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConv3dSoftmaxMaxPoolMaxPoolCustomGetWorkspaceSize
 * parameters :
 * x : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv3dSoftmaxMaxPoolMaxPoolCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConv3dSoftmaxMaxPoolMaxPoolCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv3dSoftmaxMaxPoolMaxPoolCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
