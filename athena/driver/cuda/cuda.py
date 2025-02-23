from ..driver import *
import numpy as np
import ctypes

class CudaDriver(Singleton, Driver):
    def __init__(self) -> None:
        super().__init__()
        self.chunkSize = 4096
        self.dll = ctypes.cdll.LoadLibrary("bin/kernel.so")
        self.dll.allocNum.restype = ctypes.c_void_p
        self.chunks = [self.dll.allocChunk(self.chunkSize)]
        self.offs   = [0.0]
    def __del__(self) -> None:
        for chunk in self.chunks: self.dll.freeMem(chunk)
        for i in range(self.sshapeLen): self.dll.freeMem(ctypes.c_int64(self._mem[i]))
    def compile(self) -> None:
        for i in range(len(self.static)):
            obj = self.static[i]
            if len(obj) == 2:
                arr = obj[1]
                self._mem.append(self.dll.allocObj(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size))
            else: self._mem.append(self.dll.allocNum(int(np.prod(obj[2])), ctypes.c_float(obj[1])))
            obj[0].id = i
        x = len(self.static)
        self.sshapeLen = x
        for i in range(len(self.temp)):
            self.temp[i][0].id = i + x
            self._mem.append(0.0)
        del self.static, self.temp
    def allocTemp(self, count: int) -> float:
        for i in range(len(self.chunks)):
            if self.chunkSize - self.offs[i] <= count:
                off = self.offs[i]
                self.offs[i] += count
                return self.chunks[i] + off
        self.chunks.append(off:=self.dll.allocChunk(self.chunkSize))
        self.offs.append(count)
        return off
    def numpy(self, id: int, out: np.ndarray) -> None:
        self.dll.numpy(
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(self._mem[id]),
            out.size)
    def add(self, lhs, rhs, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTemp(out.numel())
        self.dll.add(
            ctypes.pointer(ctypes.c_float(self._mem[lhs.id])),
            ctypes.pointer(ctypes.c_float(self._mem[rhs.id])),
            ctypes.pointer(ctypes.c_float(self._mem[out.id])),
            lhs.shape[-1], lhs.shape[-2])