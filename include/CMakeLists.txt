enable_language(CUDA)
project(n_body_include CUDA)
file(GLOB CUH_HEADERS "./*.cuh")
add_library(${PROJECT_NAME} INTERFACE
    hello_module.cu
    ${CUH_HEADERS}
    )