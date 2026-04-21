target_compile_options(kernel_name PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3101>
)
