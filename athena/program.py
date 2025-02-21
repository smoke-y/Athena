from __future__ import annotations
from .driver.numpy import NumpyDriver

class Program:
    def __init__(self) -> None:
        self.forwardSet, self.backwardSet, self.optimizerSet, self.driver  = [], [], [], None
    def f(self, op) -> None: self.forwardSet.append(op)
    def fa(self, ops: list) -> None: self.forwardSet = self.forwardSet + ops
    def ba(self, ops: list) -> None: self.backwardSet.append(ops)
    def oa(self, ops: list) -> None: self.optimizerSet = self.optimizerSet + ops
    def compile(self) -> None:
        self.driver.compile()
        self.backwardSet = self.backwardSet[::-1]
        backwardPass = []
        for i in self.backwardSet: backwardPass.extend(i)
        self.backwardSet = backwardPass
    def printForward(self) -> None:
        for i in self.forwardSet: print(i)
    def printBackward(self) -> None:
        for i in self.backwardSet: print(i)
    def forward(self) -> None:
        for i in self.forwardSet: i.forward()
    def backward(self, z) -> None:
        z.grad._fill(1)
        for i in self.backwardSet: i.forward()
    def fpassComplete(self) -> None: self.driver.fpassComplete()

PROG: Program = Program()