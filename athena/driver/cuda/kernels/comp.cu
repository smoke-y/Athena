#define THREADS 16
#define TILE_WIDTH 32
#define EXPORT extern "C" __host__ __attribute__((visibility("default")))

#include "driver.cu"
#include "com.cu"
#include "mul.cu"
#include "uno.cu"