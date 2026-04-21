if (status != ACLNN_SUCCESS) {
    const char* error_msg = aclGetRecentErrMsg();
    printf("Error code: %d, Message: %s\n", status, error_msg);
}
