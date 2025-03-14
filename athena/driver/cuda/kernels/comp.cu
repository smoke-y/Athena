#define THREADS 16
#define TILE_WIDTH 32
#define EXPORT extern "C" __host__ __attribute__((visibility("default")))

#include <stdio.h>
void printGPUBuff(const float *ptr, const unsigned x, const unsigned y){
    printf("[DBG PRINT BUFF]\n");
    unsigned size = sizeof(float) * x * y;
    float *mem = (float*)malloc(size);
    cudaMemcpy(mem, ptr, size, cudaMemcpyDeviceToHost);
    for(int i=0; i<y; i++){
        for(int j=0; j<x; j++){
            printf("%f ", mem[i*x + j]);
        };
        printf("\n");
    }
    free(mem);
    printf("-------\n");
}

#include "driver.cu"
#include "com.cu"
#include "mul.cu"
#include "uno.cu"