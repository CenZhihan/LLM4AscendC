
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LE_NET5CUSTOM_H_
#define ACLNN_LE_NET5CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLeNet5CustomGetWorkspaceSize
 * parameters :
 * x : required
 * conv1W : required
 * conv1B : required
 * conv2W : required
 * conv2B : required
 * fc1W : required
 * fc1B : required
 * fc2W : required
 * fc2B : required
 * fc3W : required
 * fc3B : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLeNet5CustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *conv1W,
    const aclTensor *conv1B,
    const aclTensor *conv2W,
    const aclTensor *conv2B,
    const aclTensor *fc1W,
    const aclTensor *fc1B,
    const aclTensor *fc2W,
    const aclTensor *fc2B,
    const aclTensor *fc3W,
    const aclTensor *fc3B,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLeNet5Custom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLeNet5Custom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
