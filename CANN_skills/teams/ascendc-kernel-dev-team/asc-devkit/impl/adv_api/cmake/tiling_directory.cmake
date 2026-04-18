set(ASCENDC_INSTALL_BASE_PATH @CMAKE_INSTALL_PREFIX@/lib)
file(MAKE_DIRECTORY  ${ASCENDC_INSTALL_BASE_PATH}/tikcpp)

file(CREATE_LINK ../../../asc/include/tiling ${ASCENDC_INSTALL_BASE_PATH}/ascendc/include/highlevel_api/tiling SYMBOLIC)

file(CREATE_LINK ../ascendc/include/highlevel_api/tiling ${ASCENDC_INSTALL_BASE_PATH}/tikcpp/tiling SYMBOLIC)
