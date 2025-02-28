__global__ void negKernel(const float *a, float *b, const unsigned X, const unsigned Y){
    const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x < X && y < Y){
        const unsigned index = y*X + x;
        b[index] = -1 * a[index];
    }
}
__global__ void transKernel(const float *a, float *b, const unsigned X, const unsigned Y){
    const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x < X && y < Y){
        b[x*X + y] = a[y*X + x];
    };
}
__global__ void expKernel(const float *a, float *b, const unsigned X, const unsigned Y){
    const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x < X && y < Y){
        const unsigned index = y*X + x;
        b[index] = exp(a[index]);
    };
}
__global__ void sumKernel(const float *a, float *b, const unsigned X, const unsigned Y){
    extern __shared__ float partialSum[];
    const unsigned x = threadIdx.x;
    const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;
    const unsigned index = y*X + x;
    if (x < X && y < Y) {
        partialSum[index] = a[index];
    } else {
        partialSum[index] = 0.0f;
    }
    for(unsigned stride = blockDim.x >> 1; stride > 0; stride >>=1){
        __syncthreads();
        if(x < stride){
            partialSum[index] += partialSum[index + stride];
        }
    }
    if (x == 0 && X % 2 != 0) {
        partialSum[index] += partialSum[index + X - 1];
    }
    __syncthreads();
    if (x == 0 && y < Y) {
        b[y] = partialSum[y * X];
    }
}

EXPORT void neg(const float *a, float *b, const unsigned X, const unsigned Y){
    dim3 block(THREADS, THREADS);
    dim3 grid((X+THREADS+1)/THREADS, (Y+THREADS+1)/THREADS);
    negKernel<<<grid, block>>>(a,b,X,Y);
}
EXPORT void trans(const float *a, float *b, const unsigned X, const unsigned Y){
    dim3 block(THREADS, THREADS);
    dim3 grid((X+THREADS+1)/THREADS, (Y+THREADS+1)/THREADS);
    transKernel<<<grid, block>>>(a,b,X,Y);
}
EXPORT void expnotstd(const float *a, float *b, const unsigned X, const unsigned Y){
    dim3 block(THREADS, THREADS);
    dim3 grid((X+THREADS+1)/THREADS, (Y+THREADS+1)/THREADS);
    expKernel<<<grid, block>>>(a,b,X,Y);
}
EXPORT void sum(const float *a, float *b, const unsigned X, const unsigned Y){
    dim3 block(X, 1);
    dim3 grid(1, Y);
    sumKernel<<<grid, block, X*Y*sizeof(float)>>>(a,b,X,Y);
}