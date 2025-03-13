__global__ void negKernel(const float *a, float *b, const unsigned X, const unsigned Y){
    const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x < X && y < Y){
        const unsigned index = y*X + x;
        b[index] = -1 * a[index];
    }
}
__global__ void expKernel(const float *a, float *b, const unsigned X, const unsigned Y){
    const unsigned x = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned y = threadIdx.y + blockIdx.y*blockDim.y;
    if(x < X && y < Y){
        const unsigned index = y*X + x;
        b[index] = expf(a[index]);
    };
}

EXPORT void neg(const float *a, float *b, const unsigned X, const unsigned Y){
    dim3 block(THREADS, THREADS);
    dim3 grid((X+THREADS+1)/THREADS, (Y+THREADS+1)/THREADS);
    negKernel<<<grid, block>>>(a,b,X,Y);
}
EXPORT void expnotstd(const float *a, float *b, const unsigned X, const unsigned Y){
    dim3 block(THREADS, THREADS);
    dim3 grid((X+THREADS+1)/THREADS, (Y+THREADS+1)/THREADS);
    expKernel<<<grid, block>>>(a,b,X,Y);
}
__global__ void transKernel(const float *a, float *b, const unsigned X, const unsigned Y) {
    const unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < X && y < Y) {
        b[x * Y + y] = a[y * X + x]; 
    }
}

EXPORT void trans(const float *a, float *b, const unsigned X, const unsigned Y) {
    dim3 block(THREADS, THREADS);
    dim3 grid((X + THREADS - 1) / THREADS, (Y + THREADS - 1) / THREADS);
    transKernel<<<grid, block>>>(a, b, X, Y);
}