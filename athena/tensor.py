from __future__ import annotations
from typing import Union
from .driver.numpy import NumpyDriver
from .program import *
from .ops import *
import numpy as np

def broadcast(lhs: Tensor, rhs: Union[float, int, Tensor]) -> Tensor:
    if type(rhs) != Tensor: return Tensor(None, shape=lhs.shape, num=rhs)
    if type(rhs) == Tensor and type(lhs) == Tensor:
        if rhs.dim() > 1 and lhs.dim() > 1:
            if rhs.shape != lhs.shape: raise RuntimeError(f"{rhs.shape} cannot be operated with {lhs.shape}")
    return rhs
    #TODO - broadcast when rhs is also a tensor

class Tensor:
    def __init__(self, data: Union[np.ndarray, list, tuple], shape: tuple = None, num: float = None, requireGrad: bool = True, sshape: bool = False) -> None:
        self.id = 0
        if data is not None:
            data = np.array(data, dtype=np.float32)
            if shape is not None: data = data.reshape(shape)
            self.shape = data.shape
            PROG.driver.allocObj(data, self)
            self.sshape = True
        else:
            assert shape is not None, "shape and data can't be None"
            if sshape: PROG.driver.allocNum(0 if num is None else num, shape, self)
            else:
                PROG.driver.allocTmpComp(self)
                PROG.f(AllocTmp(self, shape, 0 if num is None else num))
            self.sshape = sshape
            self.shape = shape
        if requireGrad: self.grad = Tensor(None, shape=self.shape, requireGrad=False, sshape=sshape)
        else: self.grad = None
    @staticmethod
    def rand(shape: tuple, sshape: bool = False) -> Tensor: return Tensor(data=np.random.randn(*shape), sshape=sshape)
    def __repr__(self) -> str: return f"<Tensor {self.shape} @ {self.id}>"
    def _fill(self, value: float) -> None: PROG.driver.fill(self, float(value))
    def dim(self) -> int: return len(self.shape)
    def numpy(self) -> np.ndarray: return PROG.driver.numpy(self.id, self.shape)
    def sum(self) -> Tensor:
        PROG.f(Sum(self, t := Tensor(None, tuple(list(self.shape[:-1])+[1]))))
        PROG.ba([AddT(t.grad, self.grad)])
        return t
    def numel(self) -> Tensor:
        elem = 1
        for i in self.shape: elem *= i
        return elem
    def __add__(self, rhs: Tensor) -> Tensor:
        if type(rhs) in [float, int]:
            PROG.f(AddS(self, rhs, t := Tensor(None, self.shape)))
            PROG.ba([Add(self.grad, t.grad, self.grad)])
            return t
        rhs = broadcast(self, rhs)
        #TODO: requireGrad for every latent tensors
        PROG.f(Add(self, rhs, t := Tensor(None, self.shape, requireGrad=self.grad is not None)))
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
        if type(rhs) in [float, int]:
            PROG.f(MulS(self, rhs, t := Tensor(None, self.shape)))
            PROG.ba([
                MulS(t.grad, rhs, tmp:=Tensor(None, self.shape)),
                Add(tmp, self.grad, self.grad),
            ])
            return t
        rhs = broadcast(self, rhs)
        PROG.f(Mul(self, rhs, t := Tensor(None, self.shape)))
        PROG.ba([
            Mul(t.grad, rhs, tmp:=Tensor(None, self.shape)),
            Add(tmp, self.grad, self.grad),
            Mul(t.grad, self, tmp:=Tensor(None, self.shape)),
            Add(tmp, rhs.grad, rhs.grad),
        ])
        return t
    def __truediv__(self, rhs: Union[float, int]) -> Tensor:
        rhs = broadcast(self, rhs)
        PROG.f(Div(self, rhs, t := Tensor(None, self.shape)))
        tmp = Tensor(None, self.shape, num=1)
        PROG.ba([
            Div(tmp, rhs, tmp),
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
        tmp = Tensor(None, shape=shape[:-2] + (shape[-1], shape[-2]), requireGrad=False)
        tmp1 = Tensor(None, shape=self.shape, requireGrad=False)
        shape = self.shape
        tmp2 = Tensor(None, shape=shape[:-2] + (shape[-1], shape[-2]), requireGrad=False)
        tmp3 = Tensor(None, shape=rhs.shape, requireGrad=False)
        PROG.ba([
            Trans(rhs, tmp),
            Dot(t.grad, tmp, tmp1),
            Add(self.grad, tmp1, self.grad),
            Trans(self, tmp2),
            Dot(tmp2, t.grad, tmp3),
            Add(rhs.grad, tmp3, rhs.grad),
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
    def __neg__(self) -> Tensor:
        PROG.f(Neg(self, t := Tensor(None, self.shape)))
        PROG.ba([
            Neg(t.grad, tmp := Tensor(None, t.shape)),
            Add(tmp, self.grad, self.grad)
        ])
        return t
    def trans(self) -> Tensor:
        shape = self.shape
        PROG.f(Trans(self, t := Tensor(None, shape=shape[:-2] + (shape[-1], shape[-2]))))
        PROG.ba([
            Trans(t.grad, tmp := Tensor(None, self.shape)),
            Add(t.grad, tmp, tmp)
        ])
        return t
    def load(self, data: Union[np.ndarray, list, tuple]) -> None:
        data = np.array(data,dtype=np.float32)
        data = data.reshape(self.shape)
        PROG.driver.load(self.id, data)