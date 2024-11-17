enable_language(CUDA)
project(n_body_include CUDA)
add_library(${PROJECT_NAME} INTERFACE hello_module.cu n_body_class.cuh)