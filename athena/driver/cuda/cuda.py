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
    def load(self, id, data: np.ndarray) -> None:
        self.dll.load(
            ctypes.c_int64(self._mem[id]),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            data.size)
    def add(self, lhs, rhs, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.add(
            ctypes.c_int64(self._mem[lhs.id]),
            ctypes.c_int64(self._mem[rhs.id]),
            ctypes.c_int64(self._mem[out.id]),
            lhs.shape[-1], lhs.shape[-2])
    def sub(self, lhs, rhs, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.sub(
            ctypes.c_int64(self._mem[lhs.id]),
            ctypes.c_int64(self._mem[rhs.id]),
            ctypes.c_int64(self._mem[out.id]),
            lhs.shape[-1], lhs.shape[-2])
    def mul(self, lhs, rhs, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.mul(
            ctypes.c_int64(self._mem[lhs.id]),
            ctypes.c_int64(self._mem[rhs.id]),
            ctypes.c_int64(self._mem[out.id]),
            lhs.shape[-1], lhs.shape[-2])
    def div(self, lhs, rhs, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.divnotstd(
            ctypes.c_int64(self._mem[lhs.id]),
            ctypes.c_int64(self._mem[rhs.id]),
            ctypes.c_int64(self._mem[out.id]),
            lhs.shape[-1], lhs.shape[-2])
    def dot(self, lhs, rhs, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.dot(
            ctypes.c_int64(self._mem[lhs.id]),
            ctypes.c_int64(self._mem[rhs.id]),
            ctypes.c_int64(self._mem[out.id]),
            lhs.shape[-2], rhs.shape[-2], lhs.shape[-1])
    def adds(self, src, scalar, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.adds(
            ctypes.c_int64(self._mem[src.id]),
            ctypes.c_int64(self._mem[out.id]),
            ctypes.c_float(scalar),
            src.shape[-1], src.shape[-2])
    def muls(self, src, scalar, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.muls(
            ctypes.c_int64(self._mem[src.id]),
            ctypes.c_int64(self._mem[out.id]),
            ctypes.c_float(scalar),
            src.shape[-1], src.shape[-2])
    def addt(self, src, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.addt(
            ctypes.c_int64(self._mem[src.id]),
            ctypes.c_int64(self._mem[out.id]),
            src.shape[-1], src.shape[-2])
    def pow(self, src, p, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.pownotstd(
            ctypes.c_int64(self._mem[src.id]),
            ctypes.c_int64(self._mem[out.id]),
            ctypes.c_float(p),
            src.shape[-1], src.shape[-2])
    def neg(self, src, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.neg(
            ctypes.c_int64(self._mem[src.id]),
            ctypes.c_int64(self._mem[out.id]),
            src.shape[-1], src.shape[-2])
    def trans(self, src, out) -> None:
        if not out.sshape: self._mem[out.id] = self.allocTmp(0, out.shape)
        self.dll.trans(
            ctypes.c_int64(self._mem[src.id]),
            ctypes.c_int64(self._mem[out.id]),
            src.shape[-1], src.shape[-2])
    def fill(self, src, value) -> None:
        if not src.sshape: self._mem[src.id] = self.allocTmp(0, src.shape)
        self.dll.fill(
            ctypes.c_int64(self._mem[src.id]),
            ctypes.c_float(value),
            src.shape[-1] * src.shape[-2])
    def exp(self, src, dst) -> None:
        if not src.sshape: self._mem[src.id] = self.allocTmp(0, src.shape)
        self.dll.expnotstd(
            ctypes.c_int64(self._mem[src.id]),
            ctypes.c_int64(self._mem[dst.id]),
            src.shape[-1], src.shape[-2])
    def sum(self, src, out) -> None:
        if not src.sshape: self._mem[src.id] = self.allocTmp(0, out.shape)
        self.dll.sum(
            ctypes.c_int64(self._mem[src.id]),
            ctypes.c_int64(self._mem[out.id]),
            src.shape[-1], src.shape[-2]
        )