
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INPLACE_UPDATE_CUSTOM_H_
#define ACLNN_INPLACE_UPDATE_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInplaceUpdateCustomGetWorkspaceSize
 * parameters :
 * x : required
 * idx : required
 * value : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInplaceUpdateCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *idx,
    const aclTensor *value,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInplaceUpdateCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInplaceUpdateCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
