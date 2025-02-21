EXPORT float* allocObj(float* arr, unsigned count){
    float *mem;
    const unsigned size = count*sizeof(float);
    cudaMalloc(&mem, size);
    cudaMemcpy(mem, arr, size, cudaMemcpyHostToDevice);
    return mem;
};
EXPORT float* allocNum(unsigned count, float num){
    float *hostMem = (float*)malloc(sizeof(float)*count);
    for(int i=0; i<count; i++) hostMem[i] = num;
    float *deviceMem = allocObj(hostMem, count);
    free(hostMem);
    return deviceMem;
};
EXPORT void load(float *dst, float *src, unsigned count){cudaMemcpy(dst, src, count*sizeof(float), cudaMemcpyHostToDevice);};
EXPORT void numpy(float *dst, float *src, unsigned count){cudaMemcpy(dst, src, count*sizeof(float), cudaMemcpyDeviceToHost);};
EXPORT void fill(float *dst, unsigned count, float num){
    float *hostMem = (float*)malloc(sizeof(float)*count);
    for(int i=0; i<count; i++) hostMem[i] = num;
    cudaMemcpy(dst, hostMem, count*sizeof(float), cudaMemcpyHostToDevice);
}
EXPORT void freeMem(void *ptr){cudaFree(ptr);};