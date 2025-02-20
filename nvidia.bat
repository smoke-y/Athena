@echo off

if not exist bin\ (
    mkdir bin\
)

nvcc athena/driver/cuda/kernels/comp.cu --shared -o bin\kernel.dll