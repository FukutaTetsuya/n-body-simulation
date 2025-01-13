cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(parallel_n_body CUDA)
add_executable(${PROJECT_NAME} main.cu)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_89,code=sm_89;-lcurand)
#add_compile_options($<COMPILE_LANGUAGE:CUDA>-arch=sm_89)
#add_compile_options($<COMPILE_LANGUAGE:CUDA>-lcurand)

add_subdirectory(include)
target_link_libraries(${PROJECT_NAME} PUBLIC n_body_include)
target_include_directories(${PROJECT_NAME} PUBLIC include)

set(CMAKE_CXX_FLAGS "-Wall")