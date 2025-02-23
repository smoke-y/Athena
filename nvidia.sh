if [ ! -d "bin/" ]; then
    mkdir bin/
fi

nvcc -shared -c athena/driver/cuda/kernels/comp.cu -o bin/kernel.so --expt-relaxed-constexpr --extended-lambda