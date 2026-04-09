
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LAYER_NORM_CUSTOM_H_
#define ACLNN_LAYER_NORM_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLayerNormCustomGetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * beta : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLayerNormCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *beta,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLayerNormCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLayerNormCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
