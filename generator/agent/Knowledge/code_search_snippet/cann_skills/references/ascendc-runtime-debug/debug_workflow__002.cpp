// 在 aclnn 调用后立即检查
aclnnStatus status = aclnnXxxGetWorkspaceSize(...);
if (status != ACLNN_SUCCESS) {
    const char* error_msg = aclGetRecentErrMsg();
    printf("Error: %s\n", error_msg);
    // 根据错误码进入对应处理流程
}
