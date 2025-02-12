from __future__ import annotations
from driver.numpy import NumpyDriver

class Program:
    def __init__(self) -> None:
        self.forwardSet, self.backwardSet, self.driver  = [], [], NumpyDriver()
    def f(self, op) -> None: self.forwardSet.append(op)
    def ba(self, ops: list) -> None: self.backwardSet = self.backwardSet + ops
    def compile(self) -> None: self.driver.compile()
    def forward(self) -> None:
        for i in self.forwardSet: i.forward()
    def backward(self, z) -> None:
        z.grad._fill(1)
        for i in self.backwardSet: i.forward()
    def fpassComplete(self) -> None: self.driver.fpassComplete()

PROG: Program = Program()