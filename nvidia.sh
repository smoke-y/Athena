if [ ! -d "bin/" ]; then
    mkdir bin/
fi

nvcc -c -o bin/kernel.o athena/driver/cuda/kernels/comp.cu -Xcompiler -fPIC
nvcc -shared -o bin/kernel.so bin/kernel.o