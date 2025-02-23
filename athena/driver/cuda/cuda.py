from ..driver import *
import numpy as np
import ctypes

class CudaDriver(Singleton, Driver):
    def __init__(self) -> None:
        super().__init__()
        self.chunkSize = 4096
        self.dll = ctypes.CDLL("bin/kernel.so")
        self.dll.allocNum.restype = ctypes.c_void_p
        self.dll.allocObj.restype = ctypes.c_void_p
        self.dll.allocChunk.restype = ctypes.c_void_p
        self.chunks = [self.dll.allocChunk(self.chunkSize)]
        self.offs   = [0.0]
    def __del__(self) -> None:
        for chunk in self.chunks: self.dll.freeMem(ctypes.c_int64(chunk))
        for i in range(self.sshapeLen): self.dll.freeMem(ctypes.c_int64(self._mem[i]))
    def allocObj(self, arr, tens) -> None:
        tens.id = len(self._mem)
        self._mem.append(obj:=self.dll.allocObj(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size))
    def allocNum(self, num, shape, tens) -> None:
        tens.id = len(self._mem)
        self._mem.append(self.dll.allocNum(int(np.prod(shape)), ctypes.c_float(num)))
    def allocTmp(self, value: float, shape: tuple) -> None:
        count = int(np.prod(shape))
        for i in range(len(self.chunks)):
            if self.chunkSize - self.offs[i] <= count:
                off = self.offs[i]
                self.offs[i] += count*ctypes.sizeof(ctypes.c_float)
                self._mem.append(off:=self.chunks[i] + off)
                self.dll.fill(off, count, value)
                return off
        self.chunks.append(off:=self.dll.allocChunk(self.chunkSize))
        self.offs.append(count*ctypes.sizeof(ctypes.c_float))
        self._mem.append(off)
        self.dll.fill(off, count, value)
        return off
    def numpy(self, id: int, out: np.ndarray) -> None:
        self.dll.numpy(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(self._mem[id]),
            out.size)
    def add(self, lhs, rhs, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.add(
            ctypes.c_int64(self._mem[lhs.id]),
            ctypes.c_int64(self._mem[rhs.id]),
            ctypes.c_int64(self._mem[out.id]),
            lhs.shape[-1], lhs.shape[-2])