#include <stdio.h>

__host__ void hello() {
    printf("hello from module\n");
}
__global__ void parallel_hello(){
    printf("parallel hello from %d %d\n", blockIdx.x, threadIdx.x);
    return;
}
__host__ void hello_caller() {
    parallel_hello<<<2,2>>>();
    cudaDeviceSynchronize();
}