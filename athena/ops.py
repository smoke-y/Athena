from __future__ import annotations
from .program import *

class BinOp:
    def __init__(self, lhs, rhs, out) -> None: self.lhs, self.rhs, self.out = lhs, rhs, out
    def forward(self) -> None: raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def __repr__(self) -> str: return f"{self.__class__.__name__}: {self.lhs.__repr__()}, {self.rhs.__repr__()}"

class UnOp:
    def __init__(self, src, out) -> None: self.src, self.out = src, out
    def forward(self) -> None: raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def __repr__(self) -> str: return f"{self.__class__.__name__}: {self.out.__repr__()}"

class Add(BinOp):
    def forward(self) -> None:
        PROG.driver.add(self.lhs, self.rhs, self.out)
        self.out.shape = self.lhs.shape
class Sub(BinOp):
    def forward(self) -> None:
        PROG.driver.sub(self.lhs, self.rhs, self.out)
        self.out.shape = self.lhs.shape
class Mul(BinOp):
    def forward(self) -> None:
        PROG.driver.mul(self.lhs, self.rhs, self.out)
        self.out.shape = self.lhs.shape
class Div(BinOp):
    def forward(self) -> None:
        PROG.driver.div(self.lhs, self.rhs, self.out)
        self.out.shape = self.lhs.shape
class Dot(BinOp):
    def forward(self) -> None:
        PROG.driver.dot(self.lhs, self.rhs, self.out)
        self.out.shape = self.lhs.shape[:-1] + [self.rhs.shape[-1]]

class Pow(UnOp):
    def __init__(self, src, pow, out) -> None:
        super().__init__(src, out)
        self.pow = pow
    def forward(self) -> None:
        PROG.driver.pow(self.src, self.pow, self.out)
        self.out.shape = self.src.shape
class AddS(UnOp):
    def __init__(self, src, scalar, out) -> None:
        super().__init__(src, out)
        self.scalar = scalar
    def forward(self) -> None:
        PROG.driver.adds(self.src, self.scalar, self.out)
        self.out.shape = self.src.shape
class MulS(UnOp):
    def __init__(self, src, scalar, out) -> None:
        super().__init__(src, out)
        self.scalar = scalar
    def forward(self) -> None:
        PROG.driver.muls(self.src, self.scalar, self.out)
        self.out.shape = self.src.shape
class AddT(UnOp):
    def forward(self) -> None:
        PROG.driver.addt(self.src, self.out)
        self.out.shape = self.src.shape
class Sum(UnOp):
    def forward(self) -> None: PROG.driver.sum(self.src, self.out)
class Trans(UnOp):
    def forward(self) -> None:
        PROG.driver.trans(self.src, self.out)
        shape = self.src.shape
        self.out.shape = shape[:-2] + [shape[-1], shape[-2]]
class Exp(UnOp):
    def forward(self) -> None:
        PROG.driver.exp(self.src, self.out)
        self.out.shape = self.src.shape
class Neg(UnOp):
    def forward(self) -> None:
        PROG.driver.neg(self.src, self.out)
        self.out.shape = self.src.shape
class Fill(UnOp):
    def forward(self) -> None: PROG.driver.fill(self.src.id, self.out)