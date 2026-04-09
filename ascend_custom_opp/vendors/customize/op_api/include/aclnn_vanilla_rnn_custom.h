
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_VANILLA_RNN_CUSTOM_H_
#define ACLNN_VANILLA_RNN_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnVanillaRnnCustomGetWorkspaceSize
 * parameters :
 * x : required
 * h : required
 * weight : required
 * bias : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVanillaRnnCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *h,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnVanillaRnnCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVanillaRnnCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
