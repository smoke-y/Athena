#include <stdio.h>

#define BIN_TEMPLATE(OP, KERNEL, NAME)                           \
    __global__ void KERNEL(const float *a, const float *b, float *c, const unsigned X, const unsigned Y){\
        const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;  \
        const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;  \
        if(x < X && y < Y){                                      \
            const unsigned id = y*X + x;                         \
            c[id] = a[id] OP b[id];                              \
        };                                                       \
    };                                                           \
    EXPORT void NAME(const float *a, const float *b, float *c, const unsigned X, const unsigned Y){\
        dim3 block(THREADS, THREADS);                            \
        dim3 grid((X+THREADS+1)/THREADS, (Y+THREADS+1)/THREADS); \
        KERNEL<<<grid, block>>>(a,b,c,X,Y);                      \
    };                                                           \
    

#define BIN_SCAL_TEMPLATE(OP, KERNEL, NAME)                      \
    __global__ void KERNEL(const float *a, float *b, const float scalar, const unsigned X, const unsigned Y){\
        const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;  \
        const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;  \
        if(x < X && y < Y){                                      \
            const unsigned id = y*X + x;                         \
            b[id] = a[id] OP scalar;                             \
        };                                                       \
    };                                                           \
    EXPORT void NAME(const float *a, float *b, const float scalar, const unsigned X, const unsigned Y){\
        dim3 block(THREADS, THREADS);                            \
        dim3 grid((X+THREADS+1)/THREADS, (Y+THREADS+1)/THREADS); \
        KERNEL<<<grid, block>>>(a,b,scalar,X,Y);                 \
    };                                                           \

BIN_TEMPLATE(+, addKernel, add)
BIN_TEMPLATE(-, subKernel, sub)
BIN_TEMPLATE(*, mulKernel, mul)
BIN_TEMPLATE(/, divKernel, divnotstd)

BIN_SCAL_TEMPLATE(+, addsKernel, adds)
BIN_SCAL_TEMPLATE(*, mulsKernel, muls)


__global__ void addtKernel(const float *a, float *b, const unsigned X, const unsigned Y){
    const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x < X && y < Y){
        const unsigned id = y*X + x;
        b[id] += a[0];
    }
}
__global__ void powKernel(const float *a, float *b, const unsigned exponent, const unsigned X, const unsigned Y){
    const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x < X && y < Y){
        const unsigned id = y*X + x;
        b[id] = pow(a[id], exponent);
    }
}
EXPORT void addt(const float *a, float *b, const unsigned X, const unsigned Y){
    dim3 block(THREADS, THREADS);
    dim3 grid((X+THREADS+1)/THREADS, (Y+THREADS+1)/THREADS);
    addtKernel<<<grid, block>>>(a,b,X,Y);
}
EXPORT void pownotstd(const float *a, float *b, const unsigned exponent, const unsigned X, const unsigned Y){
    dim3 block(THREADS, THREADS);
    dim3 grid((X+THREADS+1)/THREADS, (Y+THREADS+1)/THREADS);
    powKernel<<<grid, block>>>(a,b,exponent,X,Y);
}

void getInfo(char *buff){
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
    getInfo(buff);
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