from __future__ import annotations

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
    def printOptimize(self) -> None:
        for i in self.optimizerSet: print(i)
    def forward(self) -> None:
        for i in self.forwardSet: i.forward()
    def backward(self, z) -> None:
        z.grad._fill(1.0)
        for i in self.backwardSet: i.forward()
    def passComplete(self) -> None: self.driver.passComplete()
    def reset(self) -> None:
        self.forwardSet, self.backwardSet, self.optimizerSet = [], [], []

PROG: Program = Program()
