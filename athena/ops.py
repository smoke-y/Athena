from __future__ import annotations
from .program import *

class BinOp:
    def __init__(self, lhs, rhs, out) -> None: self.lhs, self.rhs, self.out = lhs, rhs, out
    def forward(self) -> None: raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def __repr__(self) -> str: return f"{self.__class__.__name__}: {self.lhs.__repr__()}, {self.rhs.__repr__()} -> {self.out.__repr__()}"

class UnOp:
    def __init__(self, src, out) -> None: self.src, self.out = src, out
    def forward(self) -> None: raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def __repr__(self) -> str: return f"{self.__class__.__name__}: {self.src.__repr__()} -> {self.out.__repr__()}"

class AllocTmp:
    def __init__(self, tens, shape: tuple, value: float) -> None: self.shape, self.value, self.tens = shape, value, tens
    def forward(self) -> None: PROG.driver.allocTmp(self.value, self.shape)
    def __repr__(self) -> str: return f"AllocTmp: {self.shape}, {self.value} -> {self.tens.id}"
class DBG:
    def __init__(self, tens) -> None: self.tens = tens
    def forward(self) -> None: print(f"{self.tens.id} -> ({self.tens.shape}\n", self.tens.numpy(), "\n----------")
    def __repr__(self) -> str: return f"DBG: {self.tens.id}"

class Add(BinOp):
    def forward(self) -> None: PROG.driver.add(self.lhs, self.rhs, self.out)
class Sub(BinOp):
    def forward(self) -> None: PROG.driver.sub(self.lhs, self.rhs, self.out)
class Mul(BinOp):
    def forward(self) -> None: PROG.driver.mul(self.lhs, self.rhs, self.out)
class Div(BinOp):
    def forward(self) -> None: PROG.driver.div(self.lhs, self.rhs, self.out)
class Dot(BinOp):
    def forward(self) -> None: PROG.driver.dot(self.lhs, self.rhs, self.out)

class Pow(UnOp):
    def __init__(self, src, pow, out) -> None:
        super().__init__(src, out)
        self.pow = pow
    def forward(self) -> None: PROG.driver.pow(self.src, self.pow, self.out)
class AddS(UnOp):
    def __init__(self, src, scalar, out) -> None:
        super().__init__(src, out)
        self.scalar = scalar
    def forward(self) -> None: PROG.driver.adds(self.src, self.scalar, self.out)
class MulS(UnOp):
    def __init__(self, src, scalar, out) -> None:
        super().__init__(src, out)
        self.scalar = scalar
    def forward(self) -> None: PROG.driver.muls(self.src, self.scalar, self.out)
class AddT(UnOp):
    def forward(self) -> None: PROG.driver.addt(self.src, self.out)
class Sum(UnOp):
    def forward(self) -> None: PROG.driver.sum(self.src, self.out)
class Trans(UnOp):
    def forward(self) -> None: PROG.driver.trans(self.src, self.out)
class Exp(UnOp):
    def forward(self) -> None: PROG.driver.exp(self.src, self.out)
class Neg(UnOp):
    def forward(self) -> None: PROG.driver.neg(self.src, self.out)
class Fill(UnOp):
    def forward(self) -> None: PROG.driver.fill(self.src, self.out)

class Conv1D(BinOp):
    def forward(self) -> None: PROG.driver.conv1D(self.lhs, self.rhs, self.out)
class Conv1DBack(BinOp):
    def forward(self) -> None: PROG.driver.conv1dback(self.lhs, self.lhs.grad, self.rhs, self.rhs.grad, self.out)
class Conv2D(BinOp):
    def forward(self) -> None: PROG.driver.conv2d(self.lhs, self.rhs, self.out)
class Conv2dBack(BinOp):
    def forward(self) -> None: PROG.driver.conv2dback(self.lhs, self.lhs.grad, self.rhs, self.rhs.grad, self.out)
