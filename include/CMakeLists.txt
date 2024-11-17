enable_language(CUDA)
project(hello_library CUDA)
add_library(${PROJECT_NAME} INTERFACE hello_module.cu)