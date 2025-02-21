#include <stdio.h>

#define BIN_TEMPLATE(OP, NAME)                                   \
    __global__ void NAME(const float *a, const float *b, float *c, const unsigned X, const unsigned Y){                                                 \
        const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;  \
        const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;  \
        if(x < X && y < Y){                                      \
            const unsigned id = y*X + x;                         \
            c[id] = a[id] OP b[id];                              \
        };                                                       \
    };                                                           \

BIN_TEMPLATE(+, add)
BIN_TEMPLATE(-, sub)
BIN_TEMPLATE(*, mul)
BIN_TEMPLATE(/, div)

void printInfo(char *buff){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sprintf(buff, "Device name: %s\nMem clock rate(KHz): %d\nMem bus width: %d\nPeak mem bandwidth(GB/s): %f\n", prop.name, prop.memoryClockRate, prop.memoryBusWidth, 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

#if(TEST)

#include <math.h>

void printMatrix(float *a, int x, int y){
    for(int i=0; i<y; i++){
        for(int j=0; j<x; j++){
            printf("%f ", a[i*x + j]);
        };
        printf("\n");
    };
};
int main(){
    char buff[100];
    printInfo(buff);
    printf("%s\n", buff);
    const unsigned X = 8;
    const unsigned Y = 8;
    const float xVal = 10.0;
    const float yVal = 5.0;
    float ah[Y][X] = {0.0};
    float bh[Y][X] = {0.0};
    float ch[Y][X] = {0.0};
    for(int i=0; i<Y; i++){
        for(int j=0; j<X; j++){
            ah[i][j] = xVal;
            bh[i][j] = yVal;
        };
    };

    float *a, *b, *c;
    unsigned long size = X*Y*sizeof(float);
    cudaMalloc(&a, size);
    cudaMalloc(&b, size);
    cudaMalloc(&c, size);
    cudaMemcpy(a, ah, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, bh, size, cudaMemcpyHostToDevice);

    dim3 gridSize(2, 2);
    dim3 blockSize(X/2, Y/2);

    printf("%f + %f =\n", xVal, yVal);
    add<<<gridSize, blockSize>>>(a, b, c, X, Y);
    cudaMemcpy(ch, c, size, cudaMemcpyDeviceToHost);
    printMatrix((float*)ch, X, Y);

    printf("%f - %f =\n", xVal, yVal);
    sub<<<gridSize, blockSize>>>(a, b, c, X, Y);
    cudaMemcpy(ch, c, size, cudaMemcpyDeviceToHost);
    printMatrix((float*)ch, X, Y);

    printf("%f * %f =\n", xVal, yVal);
    mul<<<gridSize, blockSize>>>(a, b, c, X, Y);
    cudaMemcpy(ch, c, size, cudaMemcpyDeviceToHost);
    printMatrix((float*)ch, X, Y);

    printf("%f / %f =\n", xVal, yVal);
    div<<<gridSize, blockSize>>>(a, b, c, X, Y);
    cudaMemcpy(ch, c, size, cudaMemcpyDeviceToHost);
    printMatrix((float*)ch, X, Y);
}
#endif