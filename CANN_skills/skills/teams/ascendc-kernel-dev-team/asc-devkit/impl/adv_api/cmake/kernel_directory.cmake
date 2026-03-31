set(ASCENDC_API_PATH @INSTALL_LIBRARY_DIR@)
set(ASCENDC_INSTALL_BASE_PATH @CMAKE_INSTALL_PREFIX@/${ASCENDC_API_PATH})

file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/tikcpp/tikcfw)
file(CREATE_LINK ../../../asc/include/adv_api ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/lib SYMBOLIC)
file(CREATE_LINK ../../../asc/impl/adv_api/detail ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/impl SYMBOLIC)

file(CREATE_LINK ../../ascendc/include/highlevel_api/lib ${ASCENDC_INSTALL_BASE_PATH}/tikcpp/tikcfw/lib SYMBOLIC)
file(CREATE_LINK ../../ascendc/include/highlevel_api/kernel_tiling ${ASCENDC_INSTALL_BASE_PATH}/tikcpp/tikcfw/kernel_tiling SYMBOLIC)

