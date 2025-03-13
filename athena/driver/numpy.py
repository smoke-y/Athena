from .driver import *
import numpy as np

class NumpyDriver(Singleton, Driver):
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
    def passComplete(self) -> None: self._mem = self._mem[:self.sshapeLen]