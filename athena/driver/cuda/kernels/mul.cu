__global__ void dotKernel_slow(const float *a, const float *b, float *c, const unsigned int X1, const unsigned int X2, const unsigned int Y1){
    const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x < X2 && y < Y1){
        float acum = 0.0;
        for(unsigned int i=0; i<X1; i++){
            acum += a[y*X1 + i] * b[i*X2 + x];
        };
        c[y*X2+x] = acum;
    };
};

__global__ void dotKernel(const float *a, const float *b, float *c, const unsigned int X1, const unsigned int X2, const unsigned int Y1){
    const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    tileA[y][x] = 0.0;
    tileB[y][x] = 0.0;
    float acum  = 0.0;

    for(unsigned k=0; k < ((X1-1)/TILE_WIDTH)+1; k++){
        if(y < Y1 && k*TILE_WIDTH + x < X2) tileA[y][x] = a[y*X1 + k*TILE_WIDTH + x];
        if(k*TILE_WIDTH+y < Y1 && x < X2) tileB[y][x] = b[(k*TILE_WIDTH + y)*X2 + x];
        __syncthreads();
        for(unsigned j=0; j < TILE_WIDTH; j++) acum += tileA[y][j] * tileB[j][x];
        __syncthreads();
    };
    if(y < Y1 && x < X2) c[y*X2 + x] = acum;
};

EXPORT void dot(const float *a, const float *b, float *c, const unsigned int X1, const unsigned int X2, const unsigned int Y1){
    dim3 block(THREADS, THREADS);
    dim3 grid((max(X1,X2)+THREADS+1)/THREADS, (max(Y1,X1)+THREADS+1)/THREADS);
    dotKernel<<<grid, block>>>(a,b,c,X1,X2,Y1);
}

#if(TEST)

#include <math.h>
#include <stdio.h>

void printMatrix(float *a, int x, int y){
    for(int i=0; i<y; i++){
        for(int j=0; j<x; j++){
            printf("%f ", a[i*x + j]);
        };
        printf("\n");
    };
};
int main(){
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

    printf("[slow]%f @ %f =\n", xVal, yVal);
    mul_slow<<<gridSize, blockSize>>>(a, b, c, X, Y, Y);
    cudaMemcpy(ch, c, size, cudaMemcpyDeviceToHost);
    printMatrix((float*)ch, X, Y);

    printf("[fast]%f @ %f =\n", xVal, yVal);
    mul<<<gridSize, blockSize>>>(a, b, c, X, Y, Y);
    cudaMemcpy(ch, c, size, cudaMemcpyDeviceToHost);
    printMatrix((float*)ch, X, Y);
};

#endif