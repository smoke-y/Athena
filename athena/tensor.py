from __future__ import annotations
from typing import Union
from .program import *
from .ops import *
import numpy as np

class DType:
    int8  = 0
    int16 = 1
    int32 = 2
    int64 = 3
    flt16 = 4
    flt32 = 5
    flt64 = 6

DELTA = 1e-6

def broadcast(lhs: Tensor, rhs: Union[float, int, Tensor]) -> Tensor:
    if type(rhs) != Tensor: return Tensor(None, shape=lhs.shape, num=rhs)
    if type(rhs) == Tensor and type(lhs) == Tensor:
        if rhs.dim() > 1 and lhs.dim() > 1:
            if rhs.shape != lhs.shape: raise RuntimeError(f"{rhs.shape} cannot be operated with {lhs.shape}")
    return rhs
    #TODO - broadcast when rhs is also a tensor

class Tensor:
    def __init__(self, data: Union[np.ndarray, list, tuple], shape: tuple = None, num: float = None, dtype: DType = DType.flt32, requireGrad: bool = True, sshape: bool = False) -> None:
        self.id = 0
        if data is not None:
            if type(data) != np.ndarray: data = np.array(data)
            if shape is not None: data = data.reshape(shape)
            self.shape = data.shape
            PROG.driver.allocateObj(data, sshape, self)
        else:
            assert shape is not None, "shape and data can't be None"
            PROG.driver.allocateNum(0 if num is None else num, shape, sshape, self)
            self.shape = shape
        if requireGrad: self.grad = Tensor(None, shape=self.shape, dtype=dtype, requireGrad=False)
        else: self.grad = None
        #TODO - type casting
        self.data = None
    @staticmethod
    def rand(shape: tuple, sshape: bool = False) -> Tensor: return Tensor(data=np.random.randn(*shape), sshape=sshape)
    def __repr__(self) -> str: return f"<Tensor {self.shape} @ {self.id}>"
    def _fill(self, value: float) -> None: PROG.driver.fill(self.id, value)
    def dim(self) -> int: return len(self.shape)
    def numpy(self) -> np.ndarray:
        if self.data is None: self.data = np.zeros(self.shape)
        PROG.driver.numpy(self.id, self.data) 
        return self.data
    def sum(self) -> Tensor:
        PROG.f(Sum(self, t := Tensor(None, tuple(list(self.shape[:-2])+[1]), requireGrad=self.grad != None)))
        PROG.ba([AddT(t, self)])
        return t
    def __add__(self, rhs: Tensor) -> Tensor:
        if type(rhs) in [float, int]:
            PROG.f(AddS(self, rhs, t := Tensor(None, self.shape)))
            PROG.ba([Add(self.grad, t.grad, self.grad)])
            return t
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
            Add(rhs, self.grad, self.grad),
            Add(self, rhs.grad, rhs.grad),
        ])
        return t
    def __truediv__(self, rhs: Union[float, int]) -> Tensor:
        rhs = broadcast(self, rhs)
        PROG.f(Div(self, rhs, t := Tensor(None, self.shape)))
        tmp = Tensor(None, self.shape, num=1)
        PROG.ba([
            AddS(rhs, DELTA, rhs),
            Div(tmp, self, tmp),
            Mul(tmp, t.grad, tmp),
            Add(tmp, self.grad, self.grad)
        ])
        return t
    def __matmul__(self, rhs: Tensor) -> Tensor:
        assert type(rhs) == Tensor, "Can only dot with another tensor"
        if self.dim() == 1 and rhs.dim() == 1: assert self.shape[-1] == rhs.shape[-1], f"Cannot dot {self.shape} and {rhs.shape}"
        elif self.dim() > 1 and rhs.dim() == 2: assert self.shape[-1] == rhs.shape[-2], f"Cannot dot {self.shape} and {rhs.shape}"
        else: raise ArithmeticError(f"Cannot dot {self.shape} and {rhs.shape}")
        targetShape = tuple(list(self.shape[:-1]) + [rhs.shape[-1]])
        PROG.f(Dot(self, rhs, t := Tensor(None, targetShape)))
        shape = rhs.shape
        tmp = Tensor(None, shape=shape[:-2] + (shape[-1], shape[-2]))
        shape = self.shape
        tmp2 = Tensor(None, shape=shape[:-2] + (shape[-1], shape[-2]))
        PROG.ba([
            Trans(rhs, tmp),
            Dot(t.grad, tmp, tmp),
            Add(self.grad, tmp, self.grad),
            Trans(self, tmp2),
            Dot(tmp2, t.grad, tmp2),
            Add(rhs.grad, tmp2, rhs.grad)
        ])
        return t
    def __pow__(self, rhs: Union[float, int]) -> Tensor:
        PROG.f(Pow(self, rhs, t := Tensor(None, self.shape)))
        PROG.ba([
            Pow(self, rhs-1, tmp := Tensor(None, self.shape)),
            MulS(tmp, rhs, tmp),
            Mul(tmp, t.grad, tmp),
            Add(tmp, self.grad, self.grad)
        ])
        return t