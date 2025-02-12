from driver.driver import *
import numpy as np

class NumpyDriver(Singleton, Driver):
    def __init__(self) -> None:
        super().__init__()
        self._mem = []
        self.static = []
        self.temp = []
        self.sshapeLen = 0
    def compile(self) -> None:
        for i in range(len(self.static)):
            obj = self.static[i]
            if len(obj) == 2: self._mem.append(np.array(obj[1]))
            else: self._mem.append(np.full(obj[2], obj[1]))
            obj[0].id = i
        x = len(self.static)
        self.sshapeLen = x
        for i in range(len(self.temp)):
            obj = self.temp[i]
            if len(obj) == 2: self._mem.append(np.array(obj[1]))
            else: self._mem.append(np.full(obj[2], obj[1]))
            obj[0].id = i + x
        del self.static, self.temp
    def allocateObj(self, obj: np.ndarray, sshape: bool, tens) -> None:
        self.static.append([tens, obj]) if sshape else self.temp.append([tens, obj])
    def allocateNum(self, num: float, shape: tuple, sshape: bool, tens) -> None:
        self.static.append([tens, num, shape]) if sshape else self.temp.append([tens, num, shape])
    def numpy(self, id: int, out: np.ndarray) -> None: out[:] = self._mem[id]
    def fill(self, id: int, value: float) -> None: self._mem[id].fill(value)
    def add(self, lhs, rhs, out) -> None: self._mem[out.id] = np.add(self._mem[lhs.id], self._mem[rhs.id])
    def sub(self, lhs, rhs, out) -> None: self._mem[out.id] = np.subtract(self._mem[lhs.id], self._mem[rhs.id])
    def mul(self, lhs, rhs, out) -> None: self._mem[out.id] = np.multiply(self._mem[lhs.id], self._mem[rhs.id])
    def dot(self, lhs, rhs, out) -> None: self._mem[out.id] = np.dot(self._mem[lhs.id], self._mem[rhs.id])
    def div(self, lhs, rhs, out) -> None: self._mem[out.id] = np.divide(self._mem[lhs.id], self._mem[rhs.id])
    def trans(self, src, out) -> None: self._mem[out.id] = np.transpose(self._mem[src.id])
    def reTemp(self, value, shape) -> None: self._mem[0] = np.full(shape, value if value is not None else 0)
    def addt(self, src, out) -> None: self._mem[out.id] = np.add(self._mem[out.id], self._mem[src.id])
    def sum(self, src, out) -> None: self._mem[out.id] = np.sum(self._mem[src.id])
    def pow(self, src, p, out) -> None: self._mem[out.id] = np.power(self._mem[src.id], p)
    def adds(self, src, scalar, out) -> None: self._mem[out.id] = np.add(self._mem[src.id], scalar)
    def muls(self, src, scalar, out) -> None: self._mem[out.id] = np.multiply(self._mem[src.id], scalar)
    def fpassComplete(self) -> None: self._mem = self._mem[:self.sshapeLen]