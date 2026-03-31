
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV2D_GROUP_NORM_SCALE_MAX_POOL_CLAMP_CUSTOM_H_
#define ACLNN_CONV2D_GROUP_NORM_SCALE_MAX_POOL_CLAMP_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConv2dGroupNormScaleMaxPoolClampCustomGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * bias : required
 * gnGamma : required
 * gnBeta : required
 * scale : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv2dGroupNormScaleMaxPoolClampCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *gnGamma,
    const aclTensor *gnBeta,
    const aclTensor *scale,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConv2dGroupNormScaleMaxPoolClampCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv2dGroupNormScaleMaxPoolClampCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
