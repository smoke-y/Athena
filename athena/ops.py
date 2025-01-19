from __future__ import annotations
from program import PROG

class BinOp:
    def __init__(self, lhs, rhs, out) -> None: self.lhs, self.rhs, self.out = lhs, rhs, out
    def forward(self) -> None: raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def __repr__(self) -> str: return f"{self.__class__.__name__} {self.lhs.__repr__()}, {self.rhs.__repr__()}"

class UnOp:
    def __init__(self, out) -> None: self.out = out
    def forward(self) -> None: raise NotImplementedError(f"forward not implemented for {self.__class__.__name__}")
    def __repr__(self) -> str: return f"{self.__class__.__name__} {self.out.__repr__()}"

class Add(BinOp):
    def forward(self) -> None: PROG.driver.add(self.lhs, self.rhs, self.out)
class Sub(BinOp):
    def forward(self) -> None: PROG.driver.sub(self.lhs, self.rhs, self.out)
class Mul(BinOp):
    def forward(self) -> None: PROG.driver.mul(self.lhs, self.rhs, self.out)

class Cpy(UnOp):
    def __init__(self, src, out) -> None:
        super().__init__(out)
        self.src = src
    def forward(self) -> None: PROG.driver.cpy(self.src, self.out)