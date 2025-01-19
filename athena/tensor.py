from __future__ import annotations
from program import *
from typing import Union
from ops import *
import numpy as np

class DType:
    int8  = 0
    int16 = 1
    int32 = 2
    int64 = 3
    flt16 = 4
    flt32 = 5
    flt64 = 6

def broadcast(lhs: Tensor, rhs: Union[float, int, Tensor]) -> Tensor:
    if type(rhs) != Tensor: return Tensor(None, shape=lhs.shape, num=rhs)
    return rhs
    #TODO - broadcast when rhs is also a tensor

class Tensor:
    def __init__(self, data: Union[np.ndarray, list, tuple], shape: tuple = None, num: float = None, dtype: DType = DType.flt32, requireGrad: bool = True, temp: bool = False) -> None:
        if data:
            if type(data) != np.ndarray: data = np.array(data)
            if shape is not None: data = data.reshape(shape)
            self.shape = data.shape
            self.id = PROG.driver.allocateObj(data)
        else:
            assert shape is not None, "shape and data can't be None"
            num = 0 if num is None else num
            self.id = PROG.driver.allocateNum(num, shape) if temp == False else PROG.driver.allocateTemp(num, shape)
            self.shape = shape
        if requireGrad: self.grad = Tensor(None, shape=self.shape, dtype=dtype, requireGrad=False)
        else: self.grad = None
        #TODO - type casting
        self.data = None
    def __repr__(self) -> str: return f"<Tensor {self.shape} @ {hex(id(self))}>"
    def _fill(self, value: float) -> None: PROG.driver.fill(self.id, value)
    def dim(self) -> int: return len(self.shape)
    def numpy(self) -> np.ndarray:
        if self.data is None: self.data = np.zeros(self.shape)
        PROG.driver.numpy(self.id, self.data) 
        return self.data
    def __add__(self, rhs: Tensor) -> Tensor:
        rhs = broadcast(self, rhs)
        PROG.f(Add(self, rhs, t := Tensor(None, self.shape)))
        PROG.ba([
            Add(self.grad, t.grad, self.grad),
            Add(rhs.grad, t.grad, rhs.grad)
        ])
        return t
    def __sub__(self, rhs: Tensor) -> Tensor:
        rhs = broadcast(self, rhs)
        PROG.f(Sub(self, rhs, t := Tensor(None, self.shape)))
        PROG.ba([
            Add(self.grad, t.grad, self.grad),
            Sub(rhs.grad, t.grad, rhs.grad)
        ])
        return t
    def __mul__(self, rhs: Tensor) -> Tensor:
        rhs = broadcast(self, rhs)
        PROG.f(Mul(self, rhs, t := Tensor(None, self.shape)))
        PROG.ba([
            Cpy(rhs, self.grad),
            Cpy(self, rhs.grad)
        ])
        return t
    
t = Tensor([1,3,3])
d = Tensor([1,3,3])
z = t * d
PROG.compile()
PROG.forward()
PROG.backward(z)
print(z.numpy())
print(t.grad.numpy())