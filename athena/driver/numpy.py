from driver.driver import *
import numpy as np

class NumpyDriver(Singleton, Driver):
    def __init__(self) -> None:
        super().__init__()
        self._mem.append(None)       #temp tensor
    def compile(self) -> None: pass
    def allocateObj(self, obj: np.ndarray) -> int:
        self._mem.append(obj)
        return len(self._mem) - 1
    def allocateNum(self, num: float, shape: tuple) -> int:
        self._mem.append(np.full(shape, num))
        return len(self._mem) - 1
    def allocateTemp(self, num: float, shape: tuple) -> int:
        self._mem[0] = np.full(shape, num)
        return 0
    def numpy(self, id: int, out: np.ndarray) -> None: out[:] = self._mem[id]
    def fill(self, id: int, value: float) -> None: self._mem[id].fill(value)
    def add(self, lhs, rhs, out) -> None: self._mem[out.id][:] = np.add(self._mem[lhs.id], self._mem[rhs.id])
    def sub(self, lhs, rhs, out) -> None: self._mem[out.id][:] = np.subtract(self._mem[lhs.id], self._mem[rhs.id])
    def mul(self, lhs, rhs, out) -> None: self._mem[out.id][:] = np.multiply(self._mem[lhs.id], self._mem[rhs.id])
    def cpy(self, src, out) -> None: self._mem[out.id][:] = self._mem[src.id]