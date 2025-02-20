#include "basic.cu"

#define BIN_TEMPLATE(OP, NAME)                              \
    __global__ void NAME(const float *a, const float *b, float *c, const u32 X, const u32 Y){                                                 \
        const u32 x = threadIdx.x + blockIdx.x*blockDim.x;  \
        const u32 y = threadIdx.y + blockIdx.y*blockDim.y;  \
        if(x < X && y < Y){                                 \
            const u32 id = y*X + x;                         \
            c[id] = a[id] OP b[id];                         \
        };                                                  \
    };                                                      \

BIN_TEMPLATE(+, add)
BIN_TEMPLATE(-, sub)
BIN_TEMPLATE(*, mul)
BIN_TEMPLATE(/, div)


#if(TEST)
void printMatrix(float *a, int x, int y){
    for(int i=0; i<y; i++){
        for(int j=0; j<x; j++){
            printf("%f ", a[i*x + j]);
        };
        printf("\n");
    };
};
int main(){
    const u32 X = 8;
    const u32 Y = 8;
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
    u64 size = X*Y*sizeof(float);
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