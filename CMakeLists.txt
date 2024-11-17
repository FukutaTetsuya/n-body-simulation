cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(parallel_hello CUDA)
add_executable(${PROJECT_NAME} main.cu)
add_compile_options($<COMPILE_LANGUAGE:CUDA>-arch=sm_89)

add_subdirectory(include)
target_link_libraries(${PROJECT_NAME} PUBLIC hello_library)
target_include_directories(${PROJECT_NAME} PUBLIC include)
