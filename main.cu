#include<stdio.h>
#include"hello_module.cu"

int main(){
    printf("hello, world! from main\n");
    hello();
    hello_caller();
    return 0;
}