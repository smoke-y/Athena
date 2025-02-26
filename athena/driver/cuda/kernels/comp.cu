#define EXPORT extern "C" __host__ __attribute__((visibility("default")))
#define THREADS 16

#include "driver.cu"
#include "com.cu"
#include "mul.cu"