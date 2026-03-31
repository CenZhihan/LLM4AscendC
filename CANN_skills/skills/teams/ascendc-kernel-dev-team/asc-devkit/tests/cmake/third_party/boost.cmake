set(BOOST_NAME "boost")
set(BOOST_DOWNLOAD_PATH ${CANN_3RD_LIB_PATH}/boost)
set(BOOST_LOCAL_TAR "${CANN_3RD_LIB_PATH}/boost_1_87_0.tar.gz")  # 本地 tar.gz 包路径

# 判断本地已解压的源码路径
if (EXISTS "${CANN_3RD_LIB_PATH}/boost/boost/config.hpp")
    set(BOOST_SRC_PATH ${CANN_3RD_LIB_PATH}/boost)
else()
    set(BOOST_SRC_PATH ${CANN_3RD_LIB_PATH}/boost-1.87.0)
endif()

# 新增：检查本地 tar.gz 包是否存在
if (EXISTS "${BOOST_LOCAL_TAR}")
    message(STATUS "Found local boost tar.gz: ${BOOST_LOCAL_TAR}, extracting...")
    
    # 创建目标目录（如果不存在）
    file(MAKE_DIRECTORY "${BOOST_SRC_PATH}")
    
    # 解压本地的 tar.gz 包
    execute_process(
        COMMAND tar xzf "${BOOST_LOCAL_TAR}" -C "${BOOST_SRC_PATH}" --strip-components=1
        RESULT_VARIABLE EXTRACT_RESULT
        ERROR_VARIABLE EXTRACT_ERROR
    )
    
    if(NOT EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract local boost tar.gz: ${EXTRACT_ERROR}")
    endif()
    
    message(STATUS "Local boost tar.gz extracted successfully to ${BOOST_SRC_PATH}")

# 其次：检查本地源码目录是否存在
elseif (NOT EXISTS "${BOOST_SRC_PATH}/boost/config.hpp")
    set(BOOST_URL "https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/boost_1_87_0.tar.gz")
    message(STATUS "Downloading ${BOOST_NAME} from ${BOOST_URL}")

    include(FetchContent)
    FetchContent_Declare(
        ${BOOST_NAME}
        URL ${BOOST_URL}
        URL_HASH SHA256=f55c340aa49763b1925ccf02b2e83f35fdcf634c9d5164a2acb87540173c741d
        DOWNLOAD_DIR ${BOOST_DOWNLOAD_PATH}
        SOURCE_DIR ${BOOST_SRC_PATH}  # 直接解压到此目录
        TLS_VERIFY OFF
    )
    FetchContent_MakeAvailable(${BOOST_NAME})
endif()