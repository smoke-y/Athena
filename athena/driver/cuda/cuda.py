from numpy.ctypeslib import ndpointer
from ..driver import *
import numpy as np
import ctypes

class CudaDriver(Singleton, Driver):
    def __init__(self) -> None:
        super().__init__()
        self.dll = ctypes.cdll.LoadLibrary("bin/kernel.dll")
    def compile(self) -> None:
        for i in range(len(self.static)):
            obj = self.static[i]
            if len(obj) == 2:
                arr = obj[1]
                self._mem.append(self.dll.allocObj(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size))
            else: self._mem.append(self.dll.allocNum(int(np.prod(obj[2])), obj[1]))
            obj[0].id = i
        x = len(self.static)
        self.sshapeLen = x
        for i in range(len(self.temp)):
            obj = self.temp[i]
            if len(obj) == 2:
                arr = obj[1]
                self._mem.append(self.dll.allocObj(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size))
            else: self._mem.append(self.dll.allocNum(int(np.prod(obj[2])), obj[1]))
            obj[0].id = i + x
        del self.static, self.temp