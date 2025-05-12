from .driver import *
import numpy as np

class NumpyDriver(Driver):
    def __init__(self) -> None: super().__init__()
    def allocObj(self, obj: np.ndarray, tens) -> None:
        tens.id = len(self._mem)
        self._mem.append(obj)
    def allocNum(self, num: float, shape: tuple, tens) -> None:
        tens.id = len(self._mem)
        self._mem.append(np.full(shape, num))
    def allocTmp(self, num: float, shape: tuple) -> None: self._mem.append(np.full(shape, num))
    def load(self, id: int, data: np.ndarray) -> None: self._mem[id][:] = data
    def numpy(self, id: int, shape: tuple) -> np.ndarray: return self._mem[id]
    def fill(self, src, value: float) -> None: self._mem[src.id].fill(value)
    def add(self, lhs, rhs, out) -> None: self._mem[out.id] = np.add(self._mem[lhs.id], self._mem[rhs.id])
    def sub(self, lhs, rhs, out) -> None: self._mem[out.id] = np.subtract(self._mem[lhs.id], self._mem[rhs.id])
    def mul(self, lhs, rhs, out) -> None: self._mem[out.id] = np.multiply(self._mem[lhs.id], self._mem[rhs.id])
    def dot(self, lhs, rhs, out) -> None: self._mem[out.id] = np.dot(self._mem[lhs.id], self._mem[rhs.id])
    def div(self, lhs, rhs, out) -> None: self._mem[out.id] = np.divide(self._mem[lhs.id], self._mem[rhs.id])
    def trans(self, src, out) -> None: self._mem[out.id] = np.transpose(self._mem[src.id])
    def addt(self, src, out) -> None: self._mem[out.id] = np.add(self._mem[out.id], self._mem[src.id])
    def sum(self, src, out) -> None: self._mem[out.id] = np.sum(self._mem[src.id])
    def pow(self, src, p, out) -> None: self._mem[out.id] = np.power(self._mem[src.id], p)
    def adds(self, src, scalar, out) -> None: self._mem[out.id] = np.add(self._mem[src.id], scalar)
    def muls(self, src, scalar, out) -> None: self._mem[out.id] = np.multiply(self._mem[src.id], scalar)
    def exp(self, src, out) -> None: self._mem[out.id] = np.exp(self._mem[src.id])
    def neg(self, src, out) -> None: self._mem[out.id] = np.negative(self._mem[src.id])
    def conv1D(self, src, kernel, out) -> None:
        kernel = self._mem[kernel.id]
        src    = self._mem[src.id]
        kernelLen = len(kernel)
        srcLen = len(src)
        x = 0
        res = []
        while True:
            if x + kernelLen > srcLen: break
            chunk = src[x:x+kernelLen]
            x += 1
            res.append((kernel * chunk).sum())
        self._mem[out.id] = np.array(res)
    def conv1dback(self, srcT, kernelT, outgradT) -> None:
        kernel = self._mem[kernelT.id]
        src = self._mem[srcT.id]
        outgrad = self._mem[outgradT.id]
        outsize = len(outgrad)
        res = []
        for i in range(len(kernel)):
            res.append((outgrad * src[i:i+outsize]).sum())
        self._mem[outgradT.id] = np.array(res)
    def conv2d(self, src, kernel, out) -> None:
        kernel = self._mem[kernel.id]
        src = self._mem[src.id]
        kernelLen = kernel.shape
        srcLen = src.shape
        outH = srcLen[0] - kernelLen[0] + 1
        outW = srcLen[1] - kernelLen[1] + 1
        output = np.zeros((outH, outW))
        for i in range(outH):
            for j in range(outW):
                output[i, j] = (kernel * src[i:i+kernelLen[0], j:j+kernelLen[1]]).sum()
        self._mem[out.id] = output
    def conv2dback(self, srcT, kernelT, outgradT) -> None:
        kernel = self._mem[kernelT.id]
        src = self._mem[srcT.id]
        outgrad = self._mem[outgradT.id]
        kernelLen = kernel.shape
        out = np.zeros((kernelLen[0], kernelLen[1]))
        for i in range(kernelLen[0]):
            for j in range(kernelLen[1]):
                out[i, j] = (src[i:i+kernelLen[0], j:j+kernelLen[1]] * outgrad[i, j]).sum()
        self._mem[outgradT.id] = out
    def passComplete(self) -> None: self._mem = self._mem[:self.sshapeLen]
